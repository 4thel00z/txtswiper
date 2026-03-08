const std = @import("std");

// ── BinaryReader ────────────────────────────────────────────────────────────
//
// Cursor-based reader for Tesseract binary model data. All multi-byte integers
// are little-endian. Strings are prefixed by a u32 length.

pub const BinaryReader = struct {
    data: []const u8,
    pos: usize,

    pub fn init(data: []const u8) BinaryReader {
        return .{ .data = data, .pos = 0 };
    }

    pub fn remaining(self: BinaryReader) usize {
        return if (self.pos <= self.data.len) self.data.len - self.pos else 0;
    }

    pub fn readU8(self: *BinaryReader) !u8 {
        if (self.pos + 1 > self.data.len) return error.UnexpectedEof;
        const val = self.data[self.pos];
        self.pos += 1;
        return val;
    }

    pub fn readI8(self: *BinaryReader) !i8 {
        return @bitCast(try self.readU8());
    }

    pub fn readI32(self: *BinaryReader) !i32 {
        if (self.pos + 4 > self.data.len) return error.UnexpectedEof;
        const val = std.mem.readInt(i32, self.data[self.pos..][0..4], .little);
        self.pos += 4;
        return val;
    }

    pub fn readU32(self: *BinaryReader) !u32 {
        if (self.pos + 4 > self.data.len) return error.UnexpectedEof;
        const val = std.mem.readInt(u32, self.data[self.pos..][0..4], .little);
        self.pos += 4;
        return val;
    }

    pub fn readF32(self: *BinaryReader) !f32 {
        if (self.pos + 4 > self.data.len) return error.UnexpectedEof;
        const val = @as(f32, @bitCast(std.mem.readInt(u32, self.data[self.pos..][0..4], .little)));
        self.pos += 4;
        return val;
    }

    pub fn readF64(self: *BinaryReader) !f64 {
        if (self.pos + 8 > self.data.len) return error.UnexpectedEof;
        const val = @as(f64, @bitCast(std.mem.readInt(u64, self.data[self.pos..][0..8], .little)));
        self.pos += 8;
        return val;
    }

    /// Read a Tesseract-format string: u32 length followed by length bytes.
    /// Returns an owned copy allocated with the given allocator.
    pub fn readString(self: *BinaryReader, allocator: std.mem.Allocator) ![]const u8 {
        const len = try self.readU32();
        if (self.pos + len > self.data.len) return error.UnexpectedEof;
        const result = try allocator.dupe(u8, self.data[self.pos .. self.pos + len]);
        self.pos += len;
        return result;
    }

    /// Return a borrowed slice of `n` bytes and advance the cursor.
    pub fn readBytes(self: *BinaryReader, n: usize) ![]const u8 {
        if (self.pos + n > self.data.len) return error.UnexpectedEof;
        const result = self.data[self.pos .. self.pos + n];
        self.pos += n;
        return result;
    }

    /// Skip `n` bytes.
    pub fn skip(self: *BinaryReader, n: usize) !void {
        if (self.pos + n > self.data.len) return error.UnexpectedEof;
        self.pos += n;
    }
};

// ── SIMD-accelerated dot product functions ──────────────────────────────────

/// Float32 dot product using Zig @Vector SIMD where possible.
/// Processes in chunks of 8 using @Vector(8, f32), then handles
/// any remainder elements with a scalar loop.
/// Asserts that a.len == b.len.
pub fn dot_product(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    const vec_len = 8;
    const chunks = a.len / vec_len;

    var acc: @Vector(vec_len, f32) = @splat(0.0);

    for (0..chunks) |i| {
        const offset = i * vec_len;
        const va: @Vector(vec_len, f32) = a[offset..][0..vec_len].*;
        const vb: @Vector(vec_len, f32) = b[offset..][0..vec_len].*;
        acc += va * vb;
    }

    // Reduce the SIMD accumulator to a single f32.
    var sum: f32 = @reduce(.Add, acc);

    // Handle remaining elements that don't fill a full SIMD vector.
    const tail_start = chunks * vec_len;
    for (tail_start..a.len) |i| {
        sum += a[i] * b[i];
    }

    return sum;
}

/// Int8 dot product returning an i32 accumulator.
/// Used for quantized weight matrices.
/// Widens to i32 before multiplication to avoid overflow.
pub fn dot_product_i8(a: []const i8, b: []const i8) i32 {
    std.debug.assert(a.len == b.len);

    const vec_len = 16;
    const chunks = a.len / vec_len;

    var acc: @Vector(vec_len, i32) = @splat(@as(i32, 0));

    for (0..chunks) |i| {
        const offset = i * vec_len;
        const va_i8: @Vector(vec_len, i8) = a[offset..][0..vec_len].*;
        const vb_i8: @Vector(vec_len, i8) = b[offset..][0..vec_len].*;

        // Widen to i32 before multiply to avoid overflow.
        const va: @Vector(vec_len, i32) = va_i8;
        const vb: @Vector(vec_len, i32) = vb_i8;
        acc += va * vb;
    }

    var sum: i32 = @reduce(.Add, acc);

    // Handle remaining elements.
    const tail_start = chunks * vec_len;
    for (tail_start..a.len) |i| {
        sum += @as(i32, a[i]) * @as(i32, b[i]);
    }

    return sum;
}

// ── WeightMatrix ─────────────────────────────────────────────────────────────

/// A weight matrix that supports both float32 and int8 quantized storage.
/// Layout matches Tesseract: [num_outputs][num_inputs + 1] where +1 is bias in last column.
pub const WeightMatrix = struct {
    num_outputs: usize,
    num_inputs: usize,
    /// Float weights: row-major [num_outputs * (num_inputs + 1)]
    wf: ?[]f32,
    /// Int8 weights: row-major [num_outputs * (num_inputs + 1)]
    wi: ?[]i8,
    /// Per-output scale factors for int8 mode (length: num_outputs)
    scales: ?[]f64,
    allocator: std.mem.Allocator,

    /// Allocate a float-mode weight matrix, zero-initialized.
    pub fn initFloat(allocator: std.mem.Allocator, num_outputs: usize, num_inputs: usize) !WeightMatrix {
        const total = num_outputs * (num_inputs + 1);
        const wf = try allocator.alloc(f32, total);
        @memset(wf, 0.0);
        return WeightMatrix{
            .num_outputs = num_outputs,
            .num_inputs = num_inputs,
            .wf = wf,
            .wi = null,
            .scales = null,
            .allocator = allocator,
        };
    }

    /// Free all allocations.
    pub fn deinit(self: *WeightMatrix) void {
        if (self.wf) |wf| self.allocator.free(wf);
        if (self.wi) |wi| self.allocator.free(wi);
        if (self.scales) |s| self.allocator.free(s);
        self.wf = null;
        self.wi = null;
        self.scales = null;
    }

    /// Number of columns per row (num_inputs + 1, including bias).
    pub fn cols(self: WeightMatrix) usize {
        return self.num_inputs + 1;
    }

    /// Set float weight at [row][col].
    pub fn set(self: WeightMatrix, row: usize, col: usize, val: f32) void {
        self.wf.?[row * self.cols() + col] = val;
    }

    /// Get float weight at [row][col].
    pub fn get(self: WeightMatrix, row: usize, col: usize) f32 {
        return self.wf.?[row * self.cols() + col];
    }

    /// Set bias for output row (last column).
    pub fn setBias(self: WeightMatrix, row: usize, val: f32) void {
        self.set(row, self.num_inputs, val);
    }

    /// Float matrix-vector multiply: output = W * input + bias.
    /// Uses SIMD dot_product for the weight*input portion, then adds bias.
    pub fn matVecFloat(self: WeightMatrix, input: []const f32, output: []f32) void {
        std.debug.assert(input.len == self.num_inputs);
        std.debug.assert(output.len == self.num_outputs);

        const stride = self.cols();
        const wf = self.wf.?;

        for (0..self.num_outputs) |row| {
            const row_start = row * stride;
            const weights_row = wf[row_start .. row_start + self.num_inputs];
            const bias = wf[row_start + self.num_inputs];
            output[row] = dot_product(weights_row, input) + bias;
        }
    }

    /// Quantize float weights to int8 with per-row scaling.
    /// Returns a new WeightMatrix in int8 mode.
    pub fn quantize(self: WeightMatrix, allocator: std.mem.Allocator) !WeightMatrix {
        const stride = self.cols();
        const total = self.num_outputs * stride;
        const wf = self.wf.?;

        const wi = try allocator.alloc(i8, total);
        const scales_arr = try allocator.alloc(f64, self.num_outputs);

        for (0..self.num_outputs) |row| {
            const row_start = row * stride;
            const row_slice = wf[row_start .. row_start + stride];

            // Find max absolute value in this row.
            var max_abs: f64 = 0.0;
            for (row_slice) |v| {
                const abs_v: f64 = @abs(@as(f64, v));
                if (abs_v > max_abs) max_abs = abs_v;
            }

            const scale: f64 = if (max_abs > 0.0) max_abs / 127.0 else 1.0;
            scales_arr[row] = scale;

            // Quantize each weight in the row.
            for (0..stride) |col| {
                const val: f64 = @as(f64, row_slice[col]);
                const quantized = @round(val / scale);
                const clamped = std.math.clamp(quantized, -127.0, 127.0);
                wi[row_start + col] = @intFromFloat(clamped);
            }
        }

        return WeightMatrix{
            .num_outputs = self.num_outputs,
            .num_inputs = self.num_inputs,
            .wf = null,
            .wi = wi,
            .scales = scales_arr,
            .allocator = allocator,
        };
    }

    // ── Deserialization mode flags (matching Tesseract weightmatrix.cpp) ────
    const kInt8Flag: u8 = 0x01;
    const kDoubleFlag: u8 = 0x80;

    /// Deserialize a WeightMatrix from Tesseract binary format.
    /// Handles both float (f64 -> f32) and int8 quantized modes.
    ///
    /// Binary layout:
    ///   u8 mode_flags
    ///   If float mode (kDoubleFlag set, kInt8Flag not set):
    ///     GENERIC_2D_ARRAY<double>: i32 dim1, i32 dim2, f64 empty, f64[dim1*dim2]
    ///   If int8 mode (both kDoubleFlag and kInt8Flag set):
    ///     GENERIC_2D_ARRAY<int8>: i32 dim1, i32 dim2, i8 empty, i8[dim1*dim2]
    ///     u32 scales_count, f64[scales_count] scales
    pub fn deserialize(allocator: std.mem.Allocator, reader: *BinaryReader) !WeightMatrix {
        const mode = try reader.readU8();

        const is_int8 = (mode & kInt8Flag) != 0;
        const is_double = (mode & kDoubleFlag) != 0;

        if (!is_double) {
            // Old float format -- not supported for inference-only.
            return error.UnsupportedOldFormat;
        }

        if (is_int8) {
            // ── Int8 mode ──
            // Read GENERIC_2D_ARRAY<int8_t>
            const dim1 = try reader.readI32();
            const dim2 = try reader.readI32();
            if (dim1 <= 0 or dim2 <= 0) return error.InvalidDimensions;
            const num_outputs: usize = @intCast(dim1);
            const num_cols: usize = @intCast(dim2); // includes bias column
            const num_inputs = num_cols - 1;

            _ = try reader.readI8(); // empty value (ignored)

            const total = num_outputs * num_cols;
            const wi = try allocator.alloc(i8, total);
            errdefer allocator.free(wi);

            // Read raw int8 data (row-major, matching our layout)
            const raw = try reader.readBytes(total);
            @memcpy(wi, @as([*]const i8, @ptrCast(raw.ptr))[0..total]);

            // Read scales array
            const scales_count = try reader.readU32();
            const scales_arr = try allocator.alloc(f64, num_outputs);
            errdefer allocator.free(scales_arr);

            // Read scale values from file. Tesseract stores scale * 127,
            // and at runtime divides by 127 to get the actual scale.
            for (0..scales_count) |i| {
                const val = try reader.readF64();
                if (i < num_outputs) {
                    scales_arr[i] = val / 127.0;
                }
            }
            // If scales_count > num_outputs, we already consumed all bytes.
            // If scales_count < num_outputs, pad remaining with 1.0.
            if (scales_count < num_outputs) {
                for (scales_count..num_outputs) |i| {
                    scales_arr[i] = 1.0;
                }
            }

            return WeightMatrix{
                .num_outputs = num_outputs,
                .num_inputs = num_inputs,
                .wf = null,
                .wi = wi,
                .scales = scales_arr,
                .allocator = allocator,
            };
        } else {
            // ── Float mode ──
            // Read GENERIC_2D_ARRAY<double>
            const dim1 = try reader.readI32();
            const dim2 = try reader.readI32();
            if (dim1 <= 0 or dim2 <= 0) return error.InvalidDimensions;
            const num_outputs: usize = @intCast(dim1);
            const num_cols: usize = @intCast(dim2); // includes bias column
            const num_inputs = num_cols - 1;

            _ = try reader.readF64(); // empty value (ignored)

            const total = num_outputs * num_cols;
            const wf = try allocator.alloc(f32, total);
            errdefer allocator.free(wf);

            // Read f64 values and convert to f32 (row-major, matching our layout)
            for (0..total) |i| {
                const val = try reader.readF64();
                wf[i] = @floatCast(val);
            }

            return WeightMatrix{
                .num_outputs = num_outputs,
                .num_inputs = num_inputs,
                .wf = wf,
                .wi = null,
                .scales = null,
                .allocator = allocator,
            };
        }
    }

    /// Quantized matrix-vector multiply using int8 weights.
    /// Quantizes the input vector (plus implicit 1.0 for bias) to i8,
    /// performs integer dot products on the full row (including bias column),
    /// then scales the result back to f32.
    pub fn matVecI8(self: WeightMatrix, input: []const f32, output: []f32) void {
        std.debug.assert(input.len == self.num_inputs);
        std.debug.assert(output.len == self.num_outputs);

        const stride = self.cols();
        const wi = self.wi.?;
        const scales_arr = self.scales.?;

        // Find max absolute value across input and the implicit 1.0 for bias.
        var input_max_abs: f64 = 1.0; // at least 1.0 for the bias input
        for (input) |v| {
            const abs_v: f64 = @abs(@as(f64, v));
            if (abs_v > input_max_abs) input_max_abs = abs_v;
        }

        const input_scale: f64 = if (input_max_abs > 0.0) input_max_abs / 127.0 else 1.0;

        // Quantize input vector to i8, with 1.0 appended for the bias column.
        var qi_buf: [4096]i8 = undefined;
        const qi: []i8 = qi_buf[0..stride];

        for (0..self.num_inputs) |i| {
            const val: f64 = @as(f64, input[i]);
            const quantized = @round(val / input_scale);
            const clamped = std.math.clamp(quantized, -127.0, 127.0);
            qi[i] = @intFromFloat(clamped);
        }
        // Bias column: quantize 1.0
        const bias_input_q = @round(1.0 / input_scale);
        const bias_input_clamped = std.math.clamp(bias_input_q, -127.0, 127.0);
        qi[self.num_inputs] = @intFromFloat(bias_input_clamped);

        for (0..self.num_outputs) |row| {
            const row_start = row * stride;
            const weights_row = wi[row_start .. row_start + stride];

            // Integer dot product of full row (weights + bias) with quantized input (+ bias input).
            const i32_sum = dot_product_i8(weights_row, qi);

            // Scale back to float.
            const row_scale = scales_arr[row];
            output[row] = @floatCast(@as(f64, @floatFromInt(i32_sum)) * row_scale * input_scale);
        }
    }
};

// ── Tests ────────────────────────────────────────────────────────────────────

test "dot product of two identical unit vectors" {
    // [1,0,0,0] . [1,0,0,0] = 1
    const a = [_]f32{ 1, 0, 0, 0 };
    const b = [_]f32{ 1, 0, 0, 0 };
    try std.testing.expectEqual(@as(f32, 1.0), dot_product(&a, &b));
}

test "dot product orthogonal = 0" {
    // [1,0,0,0] . [0,1,0,0] = 0
    const a = [_]f32{ 1, 0, 0, 0 };
    const b = [_]f32{ 0, 1, 0, 0 };
    try std.testing.expectEqual(@as(f32, 0.0), dot_product(&a, &b));
}

test "dot product with known result" {
    // [1,2,3,4] . [5,6,7,8] = 70
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    try std.testing.expectEqual(@as(f32, 70.0), dot_product(&a, &b));
}

test "dot product longer vector uses SIMD" {
    // 64-element vectors: a = [0,1,2,...,63], b = [1,1,...,1]
    // dot product = sum of 0..63 = 2016
    var a: [64]f32 = undefined;
    var b: [64]f32 = undefined;
    for (0..64) |i| {
        a[i] = @floatFromInt(i);
        b[i] = 1.0;
    }
    try std.testing.expectEqual(@as(f32, 2016.0), dot_product(&a, &b));
}

test "dot product empty vectors" {
    // [] . [] = 0
    const a: []const f32 = &.{};
    const b: []const f32 = &.{};
    try std.testing.expectEqual(@as(f32, 0.0), dot_product(a, b));
}

test "dot product int8 quantized" {
    // [127, 0, -127, 64] . [127, 127, 0, 64] = 16129 + 0 + 0 + 4096 = 20225
    const a = [_]i8{ 127, 0, -127, 64 };
    const b = [_]i8{ 127, 127, 0, 64 };
    try std.testing.expectEqual(@as(i32, 20225), dot_product_i8(&a, &b));
}

test "dot product int8 longer" {
    // 32-element test: a[i] = i, b[i] = 1 -> sum of 0..31 = 496
    var a: [32]i8 = undefined;
    var b: [32]i8 = undefined;
    for (0..32) |i| {
        a[i] = @intCast(i);
        b[i] = 1;
    }
    try std.testing.expectEqual(@as(i32, 496), dot_product_i8(&a, &b));
}

// ── WeightMatrix Tests ──────────────────────────────────────────────────────

test "WeightMatrix float matmul identity-like" {
    const allocator = std.testing.allocator;
    var wm = try WeightMatrix.initFloat(allocator, 2, 2);
    defer wm.deinit();

    // Row 0: [1, 0, bias=0.5] -> output = 1*3 + 0*7 + 0.5 = 3.5
    wm.set(0, 0, 1.0);
    wm.set(0, 1, 0.0);
    wm.setBias(0, 0.5);

    // Row 1: [0, 1, bias=0.0] -> output = 0*3 + 1*7 + 0.0 = 7.0
    wm.set(1, 0, 0.0);
    wm.set(1, 1, 1.0);
    wm.setBias(1, 0.0);

    const input = [_]f32{ 3.0, 7.0 };
    var output: [2]f32 = undefined;
    wm.matVecFloat(&input, &output);

    try std.testing.expectApproxEqAbs(@as(f32, 3.5), output[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), output[1], 1e-5);
}

test "WeightMatrix float matmul 3x4" {
    const allocator = std.testing.allocator;
    // 3 outputs, 4 inputs -> exercises SIMD path (4 inputs)
    var wm = try WeightMatrix.initFloat(allocator, 3, 4);
    defer wm.deinit();

    // Row 0: [1, 2, 3, 4, bias=1]
    wm.set(0, 0, 1.0);
    wm.set(0, 1, 2.0);
    wm.set(0, 2, 3.0);
    wm.set(0, 3, 4.0);
    wm.setBias(0, 1.0);

    // Row 1: [0.5, 0.5, 0.5, 0.5, bias=-1]
    wm.set(1, 0, 0.5);
    wm.set(1, 1, 0.5);
    wm.set(1, 2, 0.5);
    wm.set(1, 3, 0.5);
    wm.setBias(1, -1.0);

    // Row 2: [-1, 1, -1, 1, bias=0]
    wm.set(2, 0, -1.0);
    wm.set(2, 1, 1.0);
    wm.set(2, 2, -1.0);
    wm.set(2, 3, 1.0);
    wm.setBias(2, 0.0);

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output: [3]f32 = undefined;
    wm.matVecFloat(&input, &output);

    // Row 0: 1*1 + 2*2 + 3*3 + 4*4 + 1 = 1+4+9+16+1 = 31
    try std.testing.expectApproxEqAbs(@as(f32, 31.0), output[0], 1e-5);
    // Row 1: 0.5*1 + 0.5*2 + 0.5*3 + 0.5*4 - 1 = 5.0 - 1 = 4.0
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), output[1], 1e-5);
    // Row 2: -1*1 + 1*2 + -1*3 + 1*4 + 0 = -1+2-3+4 = 2
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output[2], 1e-5);
}

test "WeightMatrix quantize and matVecI8 close to float" {
    const allocator = std.testing.allocator;
    var wm = try WeightMatrix.initFloat(allocator, 2, 3);
    defer wm.deinit();

    // Row 0: [2.5, -1.0, 0.5, bias=0.3]
    wm.set(0, 0, 2.5);
    wm.set(0, 1, -1.0);
    wm.set(0, 2, 0.5);
    wm.setBias(0, 0.3);

    // Row 1: [1.0, 1.0, 1.0, bias=-0.5]
    wm.set(1, 0, 1.0);
    wm.set(1, 1, 1.0);
    wm.set(1, 2, 1.0);
    wm.setBias(1, -0.5);

    const input = [_]f32{ 1.0, 2.0, 3.0 };

    // Float result
    var float_output: [2]f32 = undefined;
    wm.matVecFloat(&input, &float_output);

    // Quantize and compute i8 result
    var qwm = try wm.quantize(allocator);
    defer qwm.deinit();

    var i8_output: [2]f32 = undefined;
    qwm.matVecI8(&input, &i8_output);

    // Tolerance for quantization error
    const quant_tol: f32 = 0.15;
    try std.testing.expectApproxEqAbs(float_output[0], i8_output[0], quant_tol);
    try std.testing.expectApproxEqAbs(float_output[1], i8_output[1], quant_tol);
}

test "WeightMatrix quantize preserves zero weights" {
    const allocator = std.testing.allocator;
    // Sparse matrix with mostly zeros
    var wm = try WeightMatrix.initFloat(allocator, 2, 4);
    defer wm.deinit();

    // Row 0: only one non-zero weight
    wm.set(0, 2, 3.0);
    wm.setBias(0, 0.0);

    // Row 1: only bias is non-zero
    wm.setBias(1, 1.5);

    var qwm = try wm.quantize(allocator);
    defer qwm.deinit();

    const wi = qwm.wi.?;
    const stride = qwm.cols();

    // Row 0: positions 0,1,3 should be zero; position 2 should be 127 (max)
    try std.testing.expectEqual(@as(i8, 0), wi[0 * stride + 0]);
    try std.testing.expectEqual(@as(i8, 0), wi[0 * stride + 1]);
    try std.testing.expectEqual(@as(i8, 127), wi[0 * stride + 2]);
    try std.testing.expectEqual(@as(i8, 0), wi[0 * stride + 3]);
    // Bias of row 0 is 0
    try std.testing.expectEqual(@as(i8, 0), wi[0 * stride + 4]);

    // Row 1: all weights should be zero, bias should be 127
    try std.testing.expectEqual(@as(i8, 0), wi[1 * stride + 0]);
    try std.testing.expectEqual(@as(i8, 0), wi[1 * stride + 1]);
    try std.testing.expectEqual(@as(i8, 0), wi[1 * stride + 2]);
    try std.testing.expectEqual(@as(i8, 0), wi[1 * stride + 3]);
    try std.testing.expectEqual(@as(i8, 127), wi[1 * stride + 4]);
}

test "WeightMatrix get/set roundtrip" {
    const allocator = std.testing.allocator;
    var wm = try WeightMatrix.initFloat(allocator, 3, 5);
    defer wm.deinit();

    // Set some values and read them back.
    wm.set(0, 0, 1.5);
    wm.set(1, 3, -2.7);
    wm.set(2, 4, 99.9);
    wm.setBias(0, 0.1);
    wm.setBias(2, -0.5);

    try std.testing.expectEqual(@as(f32, 1.5), wm.get(0, 0));
    try std.testing.expectEqual(@as(f32, -2.7), wm.get(1, 3));
    try std.testing.expectEqual(@as(f32, 99.9), wm.get(2, 4));
    try std.testing.expectEqual(@as(f32, 0.1), wm.get(0, 5)); // bias col
    try std.testing.expectEqual(@as(f32, -0.5), wm.get(2, 5)); // bias col

    // Verify zero-init for untouched cells.
    try std.testing.expectEqual(@as(f32, 0.0), wm.get(0, 1));
    try std.testing.expectEqual(@as(f32, 0.0), wm.get(1, 0));

    // Verify cols() returns num_inputs + 1.
    try std.testing.expectEqual(@as(usize, 6), wm.cols());
}

// ── WeightMatrix Deserialization Tests ──────────────────────────────────────

test "WeightMatrix deserialize float mode" {
    const allocator = std.testing.allocator;

    // Build a synthetic float-mode WeightMatrix in Tesseract binary format:
    //   mode = kDoubleFlag (0x80)
    //   dim1 = 2 (num_outputs)
    //   dim2 = 3 (num_inputs + 1 for bias = 2 inputs + 1)
    //   empty = 0.0 (f64)
    //   data = 6 doubles (row-major):
    //     row0: [1.0, 2.0, 0.5]   (weights + bias)
    //     row1: [3.0, 4.0, -1.0]  (weights + bias)
    var buf: [1 + 4 + 4 + 8 + 6 * 8]u8 = undefined;
    var pos: usize = 0;

    // mode
    buf[pos] = 0x80;
    pos += 1;

    // dim1 = 2
    std.mem.writeInt(i32, buf[pos..][0..4], 2, .little);
    pos += 4;
    // dim2 = 3
    std.mem.writeInt(i32, buf[pos..][0..4], 3, .little);
    pos += 4;

    // empty value (f64 0.0)
    @as(*align(1) f64, @ptrCast(buf[pos .. pos + 8])).*  = 0.0;
    pos += 8;

    // Row 0: [1.0, 2.0, 0.5]
    const vals = [_]f64{ 1.0, 2.0, 0.5, 3.0, 4.0, -1.0 };
    for (vals) |v| {
        @as(*align(1) f64, @ptrCast(buf[pos .. pos + 8])).* = v;
        pos += 8;
    }

    var reader = BinaryReader.init(&buf);
    var wm = try WeightMatrix.deserialize(allocator, &reader);
    defer wm.deinit();

    try std.testing.expectEqual(@as(usize, 2), wm.num_outputs);
    try std.testing.expectEqual(@as(usize, 2), wm.num_inputs);
    try std.testing.expect(wm.wf != null);
    try std.testing.expect(wm.wi == null);

    // Check weights
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), wm.get(0, 0), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), wm.get(0, 1), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), wm.get(0, 2), 1e-5); // bias
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), wm.get(1, 0), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), wm.get(1, 1), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), wm.get(1, 2), 1e-5); // bias

    // Verify matVecFloat works with the deserialized matrix
    const input = [_]f32{ 1.0, 1.0 };
    var output: [2]f32 = undefined;
    wm.matVecFloat(&input, &output);
    // row0: 1*1 + 2*1 + 0.5 = 3.5
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), output[0], 1e-5);
    // row1: 3*1 + 4*1 + (-1.0) = 6.0
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), output[1], 1e-5);

    // All bytes consumed
    try std.testing.expectEqual(@as(usize, 0), reader.remaining());
}

test "WeightMatrix deserialize int8 mode" {
    const allocator = std.testing.allocator;

    // Build a synthetic int8-mode WeightMatrix in Tesseract binary format:
    //   mode = kDoubleFlag | kInt8Flag (0x81)
    //   GENERIC_2D_ARRAY<int8>: dim1=2, dim2=3, empty=0, data=6 bytes
    //   scales_count=2, scales=[2.54, 1.27] (stored as scale * 127)
    const buf_size = 1 + 4 + 4 + 1 + 6 + 4 + 2 * 8;
    var buf: [buf_size]u8 = undefined;
    var pos: usize = 0;

    // mode = 0x81
    buf[pos] = 0x81;
    pos += 1;

    // dim1 = 2
    std.mem.writeInt(i32, buf[pos..][0..4], 2, .little);
    pos += 4;
    // dim2 = 3
    std.mem.writeInt(i32, buf[pos..][0..4], 3, .little);
    pos += 4;

    // empty value (i8 = 0)
    buf[pos] = 0;
    pos += 1;

    // Int8 data: row0 = [127, -64, 10], row1 = [0, 100, -127]
    const i8_vals = [_]i8{ 127, -64, 10, 0, 100, -127 };
    for (i8_vals) |v| {
        buf[pos] = @bitCast(v);
        pos += 1;
    }

    // scales_count = 2
    std.mem.writeInt(u32, buf[pos..][0..4], 2, .little);
    pos += 4;

    // scales: stored as scale * 127 in the file
    // scale[0] = 0.02 -> file stores 0.02 * 127 = 2.54
    // scale[1] = 0.01 -> file stores 0.01 * 127 = 1.27
    @as(*align(1) f64, @ptrCast(buf[pos .. pos + 8])).* = 2.54;
    pos += 8;
    @as(*align(1) f64, @ptrCast(buf[pos .. pos + 8])).* = 1.27;
    pos += 8;

    var reader = BinaryReader.init(&buf);
    var wm = try WeightMatrix.deserialize(allocator, &reader);
    defer wm.deinit();

    try std.testing.expectEqual(@as(usize, 2), wm.num_outputs);
    try std.testing.expectEqual(@as(usize, 2), wm.num_inputs);
    try std.testing.expect(wm.wf == null);
    try std.testing.expect(wm.wi != null);
    try std.testing.expect(wm.scales != null);

    // Check int8 weights
    const wi = wm.wi.?;
    try std.testing.expectEqual(@as(i8, 127), wi[0]);
    try std.testing.expectEqual(@as(i8, -64), wi[1]);
    try std.testing.expectEqual(@as(i8, 10), wi[2]);
    try std.testing.expectEqual(@as(i8, 0), wi[3]);
    try std.testing.expectEqual(@as(i8, 100), wi[4]);
    try std.testing.expectEqual(@as(i8, -127), wi[5]);

    // Check scales (file value / 127)
    const scales_arr = wm.scales.?;
    try std.testing.expectApproxEqAbs(@as(f64, 0.02), scales_arr[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.01), scales_arr[1], 1e-6);

    // All bytes consumed
    try std.testing.expectEqual(@as(usize, 0), reader.remaining());
}

test "BinaryReader basic operations" {
    var buf: [21]u8 = undefined;

    // Byte 0: u8 = 0xFF
    buf[0] = 0xFF;
    // Bytes 1-4: i32 = -42
    std.mem.writeInt(i32, buf[1..5], -42, .little);
    // Bytes 5-8: u32 = 1000
    std.mem.writeInt(u32, buf[5..9], 1000, .little);
    // Bytes 9-16: f64 = 3.14
    @as(*align(1) f64, @ptrCast(buf[9..17])).* = 3.14;
    // Bytes 17-20: string: u32 len=2 + "AB"
    std.mem.writeInt(u32, buf[17..21], 2, .little);

    // Extend buffer to include the string data
    var full_buf: [23]u8 = undefined;
    @memcpy(full_buf[0..21], &buf);
    full_buf[21] = 'A';
    full_buf[22] = 'B';

    var reader = BinaryReader.init(&full_buf);

    try std.testing.expectEqual(@as(usize, 23), reader.remaining());

    const byte = try reader.readU8();
    try std.testing.expectEqual(@as(u8, 0xFF), byte);

    const i32_val = try reader.readI32();
    try std.testing.expectEqual(@as(i32, -42), i32_val);

    const u32_val = try reader.readU32();
    try std.testing.expectEqual(@as(u32, 1000), u32_val);

    const f64_val = try reader.readF64();
    try std.testing.expectApproxEqAbs(@as(f64, 3.14), f64_val, 1e-10);

    const str = try reader.readString(std.testing.allocator);
    defer std.testing.allocator.free(str);
    try std.testing.expectEqualStrings("AB", str);

    try std.testing.expectEqual(@as(usize, 0), reader.remaining());
}

test "BinaryReader eof error" {
    const buf = [_]u8{ 0x01 };
    var reader = BinaryReader.init(&buf);
    _ = try reader.readU8();
    // Now at end, further reads should fail
    try std.testing.expectError(error.UnexpectedEof, reader.readU8());
    try std.testing.expectError(error.UnexpectedEof, reader.readI32());
}
