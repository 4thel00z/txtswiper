const std = @import("std");
const math = std.math;

// ── NetworkIO: Activation Data Container ─────────────────────────────────────
//
// Stores a 2D array of shape [width][num_features] in either float32 or int8
// mode. Width corresponds to the number of timesteps (positions along the text
// line). num_features is the number of channels/features at each position.
//
// This is the core data container used by ALL neural network layers to pass
// activations between them.

pub const NetworkIO = struct {
    /// Float32 activation data, row-major: [width * num_features]
    f_data: ?[]f32,
    /// Int8 activation data for quantized mode: [width * num_features]
    i_data: ?[]i8,
    width_: usize,
    num_features_: usize,
    int_mode_: bool,
    allocator: std.mem.Allocator,

    /// Allocate a NetworkIO with the given dimensions and mode.
    /// The appropriate buffer (f_data or i_data) is allocated and zero-initialized.
    pub fn init(allocator: std.mem.Allocator, w: usize, num_features: usize, int_mode: bool) !NetworkIO {
        const total = w * num_features;
        if (int_mode) {
            const buf = try allocator.alloc(i8, total);
            @memset(buf, 0);
            return NetworkIO{
                .f_data = null,
                .i_data = buf,
                .width_ = w,
                .num_features_ = num_features,
                .int_mode_ = true,
                .allocator = allocator,
            };
        } else {
            const buf = try allocator.alloc(f32, total);
            @memset(buf, 0.0);
            return NetworkIO{
                .f_data = buf,
                .i_data = null,
                .width_ = w,
                .num_features_ = num_features,
                .int_mode_ = false,
                .allocator = allocator,
            };
        }
    }

    /// Free the allocated buffer.
    pub fn deinit(self: *NetworkIO) void {
        if (self.f_data) |buf| self.allocator.free(buf);
        if (self.i_data) |buf| self.allocator.free(buf);
        self.f_data = null;
        self.i_data = null;
    }

    /// Return the width (number of timesteps).
    pub fn width(self: NetworkIO) usize {
        return self.width_;
    }

    /// Return the number of features per timestep.
    pub fn numFeatures(self: NetworkIO) usize {
        return self.num_features_;
    }

    /// Write float data into timestep t.
    /// If int_mode is active, the data is quantized from f32 to i8 by scaling
    /// by 127 and clipping to [-127, 127].
    pub fn writeTimeStep(self: *NetworkIO, t: usize, data: []const f32) void {
        std.debug.assert(data.len == self.num_features_);
        std.debug.assert(t < self.width_);

        const offset = t * self.num_features_;

        if (self.int_mode_) {
            const buf = self.i_data.?;
            for (0..self.num_features_) |j| {
                const scaled = data[j] * 127.0;
                const clamped = math.clamp(scaled, -127.0, 127.0);
                buf[offset + j] = @intFromFloat(@round(clamped));
            }
        } else {
            const buf = self.f_data.?;
            @memcpy(buf[offset .. offset + self.num_features_], data);
        }
    }

    /// Read data from timestep t into out.
    /// If int_mode is active, the i8 data is converted back to f32 by dividing
    /// by 127.0.
    pub fn readTimeStep(self: *const NetworkIO, t: usize, out: []f32) void {
        std.debug.assert(out.len == self.num_features_);
        std.debug.assert(t < self.width_);

        const offset = t * self.num_features_;

        if (self.int_mode_) {
            const buf = self.i_data.?;
            for (0..self.num_features_) |j| {
                out[j] = @as(f32, @floatFromInt(buf[offset + j])) / 127.0;
            }
        } else {
            const buf = self.f_data.?;
            @memcpy(out, buf[offset .. offset + self.num_features_]);
        }
    }

    /// Return a slice to the float data at timestep t.
    /// Asserts that int_mode is not active.
    pub fn f(self: *NetworkIO, t: usize) []f32 {
        std.debug.assert(!self.int_mode_);
        std.debug.assert(t < self.width_);
        const offset = t * self.num_features_;
        return self.f_data.?[offset .. offset + self.num_features_];
    }

    /// Return a slice to the int8 data at timestep t.
    /// Asserts that int_mode is active.
    pub fn i(self: *NetworkIO, t: usize) []i8 {
        std.debug.assert(self.int_mode_);
        std.debug.assert(t < self.width_);
        const offset = t * self.num_features_;
        return self.i_data.?[offset .. offset + self.num_features_];
    }

    /// Resize the buffer to accommodate new_width * new_features.
    /// Only grows the allocation, never shrinks. Updates dimensions and zeros
    /// the entire buffer.
    pub fn resize(self: *NetworkIO, new_width: usize, new_features: usize) !void {
        const new_total = new_width * new_features;
        const old_total = self.width_ * self.num_features_;

        if (self.int_mode_) {
            if (new_total > old_total) {
                if (self.i_data) |buf| self.allocator.free(buf);
                self.i_data = try self.allocator.alloc(i8, new_total);
            }
            @memset(self.i_data.?, 0);
        } else {
            if (new_total > old_total) {
                if (self.f_data) |buf| self.allocator.free(buf);
                self.f_data = try self.allocator.alloc(f32, new_total);
            }
            @memset(self.f_data.?, 0.0);
        }

        self.width_ = new_width;
        self.num_features_ = new_features;
    }

    /// Copy one timestep from src into self at dest_t.
    /// Both must have the same num_features and the same mode.
    pub fn copyTimeStepFrom(self: *NetworkIO, dest_t: usize, src: *const NetworkIO, src_t: usize) void {
        std.debug.assert(self.num_features_ == src.num_features_);
        std.debug.assert(self.int_mode_ == src.int_mode_);
        std.debug.assert(dest_t < self.width_);
        std.debug.assert(src_t < src.width_);

        const nf = self.num_features_;
        const dst_offset = dest_t * nf;
        const src_offset = src_t * nf;

        if (self.int_mode_) {
            @memcpy(
                self.i_data.?[dst_offset .. dst_offset + nf],
                src.i_data.?[src_offset .. src_offset + nf],
            );
        } else {
            @memcpy(
                self.f_data.?[dst_offset .. dst_offset + nf],
                src.f_data.?[src_offset .. src_offset + nf],
            );
        }
    }

    /// Zero out all features at timestep t.
    pub fn zeroTimeStep(self: *NetworkIO, t: usize) void {
        std.debug.assert(t < self.width_);

        const offset = t * self.num_features_;

        if (self.int_mode_) {
            @memset(self.i_data.?[offset .. offset + self.num_features_], 0);
        } else {
            @memset(self.f_data.?[offset .. offset + self.num_features_], 0.0);
        }
    }
};

// ── Tests ────────────────────────────────────────────────────────────────────

test "NetworkIO create and read/write timestep (float)" {
    const allocator = std.testing.allocator;
    var nio = try NetworkIO.init(allocator, 10, 5, false);
    defer nio.deinit();

    // Write [1, 2, 3, 4, 5] at t=3.
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    nio.writeTimeStep(3, &input);

    // Read back and verify.
    var out: [5]f32 = undefined;
    nio.readTimeStep(3, &out);

    for (0..5) |j| {
        try std.testing.expectApproxEqAbs(input[j], out[j], 1e-6);
    }
}

test "NetworkIO create and read/write timestep (int8)" {
    const allocator = std.testing.allocator;
    var nio = try NetworkIO.init(allocator, 10, 5, true);
    defer nio.deinit();

    // Write float data that should survive quantization reasonably.
    const input = [_]f32{ 0.5, -0.5, 1.0, -1.0, 0.0 };
    nio.writeTimeStep(3, &input);

    // Read back and verify within quantization tolerance.
    var out: [5]f32 = undefined;
    nio.readTimeStep(3, &out);

    // Quantization tolerance: 1/127 ~= 0.008.
    const quant_tol: f32 = 0.01;
    for (0..5) |j| {
        try std.testing.expectApproxEqAbs(input[j], out[j], quant_tol);
    }
}

test "NetworkIO width and numFeatures" {
    const allocator = std.testing.allocator;
    var nio = try NetworkIO.init(allocator, 20, 48, false);
    defer nio.deinit();

    try std.testing.expectEqual(@as(usize, 20), nio.width());
    try std.testing.expectEqual(@as(usize, 48), nio.numFeatures());
}

test "NetworkIO f() returns correct slice" {
    const allocator = std.testing.allocator;
    var nio = try NetworkIO.init(allocator, 10, 4, false);
    defer nio.deinit();

    // Write data at t=2.
    const input = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    nio.writeTimeStep(2, &input);

    // Access via f() and verify same data.
    const slice = nio.f(2);
    try std.testing.expectEqual(@as(usize, 4), slice.len);
    for (0..4) |j| {
        try std.testing.expectApproxEqAbs(input[j], slice[j], 1e-6);
    }
}

test "NetworkIO copyTimeStepFrom" {
    const allocator = std.testing.allocator;

    var src = try NetworkIO.init(allocator, 10, 4, false);
    defer src.deinit();

    var dst = try NetworkIO.init(allocator, 10, 4, false);
    defer dst.deinit();

    // Write data to src at t=5.
    const input = [_]f32{ 100.0, 200.0, 300.0, 400.0 };
    src.writeTimeStep(5, &input);

    // Copy from src t=5 to dst t=2.
    dst.copyTimeStepFrom(2, &src, 5);

    // Verify the copied data.
    var out: [4]f32 = undefined;
    dst.readTimeStep(2, &out);

    for (0..4) |j| {
        try std.testing.expectApproxEqAbs(input[j], out[j], 1e-6);
    }
}

test "NetworkIO resize grows buffer" {
    const allocator = std.testing.allocator;
    var nio = try NetworkIO.init(allocator, 5, 3, false);
    defer nio.deinit();

    // Write some data at t=0.
    const input = [_]f32{ 1.0, 2.0, 3.0 };
    nio.writeTimeStep(0, &input);

    // Resize larger.
    try nio.resize(20, 10);

    try std.testing.expectEqual(@as(usize, 20), nio.width());
    try std.testing.expectEqual(@as(usize, 10), nio.numFeatures());

    // Old data should be cleared (buffer is zeroed).
    var out: [10]f32 = undefined;
    nio.readTimeStep(0, &out);
    for (0..10) |j| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[j], 1e-6);
    }
}

test "NetworkIO zeroTimeStep" {
    const allocator = std.testing.allocator;
    var nio = try NetworkIO.init(allocator, 10, 4, false);
    defer nio.deinit();

    // Write data at t=7.
    const input = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    nio.writeTimeStep(7, &input);

    // Zero that timestep.
    nio.zeroTimeStep(7);

    // Verify all zeros.
    var out: [4]f32 = undefined;
    nio.readTimeStep(7, &out);
    for (0..4) |j| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[j], 1e-6);
    }
}
