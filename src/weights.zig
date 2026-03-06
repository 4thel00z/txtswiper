const std = @import("std");

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
