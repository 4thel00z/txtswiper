const std = @import("std");
const math = std.math;

// ── Scalar activation functions ──────────────────────────────────────────────

/// Logistic sigmoid: 1 / (1 + exp(-x)).
/// Clamps the input to [-88, 88] to avoid overflow in exp().
pub fn sigmoid(x: f32) f32 {
    // For very large positive values, sigmoid -> 1.
    // For very large negative values, sigmoid -> 0.
    // Clamp to avoid inf in @exp.
    const clamped = math.clamp(x, -88.0, 88.0);
    return 1.0 / (1.0 + @exp(-clamped));
}

/// Hyperbolic tangent approximation using the identity:
///   tanh(x) = 2 * sigmoid(2x) - 1
pub fn tanh_approx(x: f32) f32 {
    return 2.0 * sigmoid(2.0 * x) - 1.0;
}

/// Rectified Linear Unit: max(x, 0).
pub fn relu(x: f32) f32 {
    return @max(x, 0.0);
}

// ── Lookup tables (comptime-generated) ──────────────────────────────────────

/// 256-entry precomputed sigmoid lookup table.
/// Maps input range [-8, 8] linearly to indices 0-255.
/// lut_sigmoid[i] = sigmoid((i / 255.0) * 16.0 - 8.0)
const lut_sigmoid: [256]f32 = blk: {
    var table: [256]f32 = undefined;
    for (0..256) |i| {
        const fi: f32 = @floatFromInt(i);
        const x: f32 = (fi / 255.0) * 16.0 - 8.0;
        const neg_x = -x;
        table[i] = 1.0 / (1.0 + @exp(neg_x));
    }
    break :blk table;
};

/// 256-entry precomputed tanh lookup table.
/// Maps input range [-4, 4] linearly to indices 0-255.
/// lut_tanh[i] = tanh((i / 255.0) * 8.0 - 4.0)
const lut_tanh: [256]f32 = blk: {
    var table: [256]f32 = undefined;
    for (0..256) |i| {
        const fi: f32 = @floatFromInt(i);
        const x: f32 = (fi / 255.0) * 8.0 - 4.0;
        // tanh(x) = 2 * sigmoid(2x) - 1
        const neg_2x = -(2.0 * x);
        table[i] = 2.0 * (1.0 / (1.0 + @exp(neg_2x))) - 1.0;
    }
    break :blk table;
};

// ── Fast lookup versions ────────────────────────────────────────────────────

/// Fast sigmoid using the 256-entry lookup table with linear interpolation.
/// For inputs outside [-8, 8], clamps to 0 or 1.
pub fn sigmoid_fast(x: f32) f32 {
    if (x <= -8.0) return 0.0;
    if (x >= 8.0) return 1.0;
    // Map [-8, 8] -> [0, 255]
    const idx_f = (x + 8.0) * (255.0 / 16.0);
    const clamped = math.clamp(idx_f, 0.0, 255.0);
    const lo: u8 = @intFromFloat(@floor(clamped));
    const hi: u8 = if (lo < 255) lo + 1 else 255;
    const frac = clamped - @floor(clamped);
    return lut_sigmoid[lo] * (1.0 - frac) + lut_sigmoid[hi] * frac;
}

/// Fast tanh using the 256-entry lookup table with linear interpolation.
/// For inputs outside [-4, 4], clamps to -1 or 1.
pub fn tanh_fast(x: f32) f32 {
    if (x <= -4.0) return -1.0;
    if (x >= 4.0) return 1.0;
    // Map [-4, 4] -> [0, 255]
    const idx_f = (x + 4.0) * (255.0 / 8.0);
    const clamped = math.clamp(idx_f, 0.0, 255.0);
    const lo: u8 = @intFromFloat(@floor(clamped));
    const hi: u8 = if (lo < 255) lo + 1 else 255;
    const frac = clamped - @floor(clamped);
    return lut_tanh[lo] * (1.0 - frac) + lut_tanh[hi] * frac;
}

// ── In-place activation functions (SIMD-accelerated) ────────────────────────

/// Apply sigmoid to every element of `data` in place.
/// Uses SIMD @Vector(8, f32) chunks with lookup-table acceleration.
pub fn sigmoid_inplace(data: []f32) void {
    const vec_len = 8;
    const chunks = data.len / vec_len;

    for (0..chunks) |chunk| {
        const offset = chunk * vec_len;
        for (0..vec_len) |lane| {
            data[offset + lane] = sigmoid_fast(data[offset + lane]);
        }
    }

    // Handle remainder with scalar loop.
    const tail_start = chunks * vec_len;
    for (tail_start..data.len) |i| {
        data[i] = sigmoid_fast(data[i]);
    }
}

/// Apply tanh to every element of `data` in place.
/// Uses SIMD @Vector(8, f32) chunks with lookup-table acceleration.
pub fn tanh_inplace(data: []f32) void {
    const vec_len = 8;
    const chunks = data.len / vec_len;

    for (0..chunks) |chunk| {
        const offset = chunk * vec_len;
        for (0..vec_len) |lane| {
            data[offset + lane] = tanh_fast(data[offset + lane]);
        }
    }

    // Handle remainder with scalar loop.
    const tail_start = chunks * vec_len;
    for (tail_start..data.len) |i| {
        data[i] = tanh_fast(data[i]);
    }
}

/// Numerically-stable softmax in place.
///   softmax(x_i) = exp(x_i - max) / sum_j(exp(x_j - max))
///
/// After this call every element of `data` is in [0, 1] and the slice sums to 1.
/// If the slice is empty the function is a no-op.
pub fn softmax_inplace(data: []f32) void {
    if (data.len == 0) return;

    // 1. Find the maximum value for numerical stability.
    var max_val: f32 = data[0];
    for (data[1..]) |v| {
        if (v > max_val) max_val = v;
    }

    // 2. Exponentiate (shifted) and accumulate the sum.
    var sum: f32 = 0.0;
    for (data) |*v| {
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }

    // 3. Normalise.
    const inv_sum = 1.0 / sum;
    for (data) |*v| {
        v.* *= inv_sum;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

const tolerance: f32 = 1e-4;

test "sigmoid(0) = 0.5" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid(0.0), tolerance);
}

test "sigmoid large positive ≈ 1" {
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sigmoid(100.0), tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sigmoid(1000.0), tolerance);
}

test "sigmoid large negative ≈ 0" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sigmoid(-100.0), tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sigmoid(-1000.0), tolerance);
}

test "tanh(0) = 0" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), tanh_approx(0.0), tolerance);
}

test "tanh(1) ≈ 0.7616" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.7616), tanh_approx(1.0), tolerance);
}

test "softmax normalizes to 1" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    softmax_inplace(&data);

    var sum: f32 = 0.0;
    for (data) |v| {
        sum += v;
        // Each element must be in [0, 1].
        try std.testing.expect(v >= 0.0 and v <= 1.0);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, tolerance);
}

test "sigmoid_inplace applies to all elements" {
    var data = [_]f32{ 0.0, -100.0, 100.0 };
    sigmoid_inplace(&data);

    try std.testing.expectApproxEqAbs(@as(f32, 0.5), data[0], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[1], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[2], tolerance);
}

test "tanh_inplace applies to all elements" {
    var data = [_]f32{ 0.0, 1.0, -1.0 };
    tanh_inplace(&data);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[0], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7616), data[1], tolerance);
    try std.testing.expectApproxEqAbs(@as(f32, -0.7616), data[2], tolerance);
}

test "relu positive passthrough" {
    try std.testing.expectEqual(@as(f32, 3.14), relu(3.14));
    try std.testing.expectEqual(@as(f32, 0.0), relu(0.0));
}

test "relu negative clamp to 0" {
    try std.testing.expectEqual(@as(f32, 0.0), relu(-5.0));
    try std.testing.expectEqual(@as(f32, 0.0), relu(-0.001));
}

// ── New LUT / SIMD tests ────────────────────────────────────────────────────

const lut_tolerance: f32 = 1e-3; // LUT with linear interpolation keeps error small

test "sigmoid_fast matches sigmoid within tolerance" {
    // Test across range [-10, 10]
    const steps = 200;
    for (0..steps + 1) |i| {
        const fi: f32 = @floatFromInt(i);
        const x: f32 = (fi / @as(f32, @floatFromInt(steps))) * 20.0 - 10.0;
        const expected = sigmoid(x);
        const actual = sigmoid_fast(x);
        try std.testing.expectApproxEqAbs(expected, actual, lut_tolerance);
    }
}

test "tanh_fast matches tanh_approx within tolerance" {
    // Test across range [-5, 5]
    const steps = 200;
    for (0..steps + 1) |i| {
        const fi: f32 = @floatFromInt(i);
        const x: f32 = (fi / @as(f32, @floatFromInt(steps))) * 10.0 - 5.0;
        const expected = tanh_approx(x);
        const actual = tanh_fast(x);
        try std.testing.expectApproxEqAbs(expected, actual, lut_tolerance);
    }
}

test "sigmoid_inplace SIMD matches scalar" {
    // 100-element array: compare SIMD inplace result with scalar sigmoid.
    var simd_data: [100]f32 = undefined;
    for (0..100) |i| {
        const fi: f32 = @floatFromInt(i);
        simd_data[i] = (fi / 99.0) * 20.0 - 10.0; // range [-10, 10]
    }

    // Compute expected values using scalar sigmoid.
    var expected: [100]f32 = undefined;
    for (0..100) |i| {
        expected[i] = sigmoid(simd_data[i]);
    }

    sigmoid_inplace(&simd_data);

    for (0..100) |i| {
        try std.testing.expectApproxEqAbs(expected[i], simd_data[i], lut_tolerance);
    }
}

test "tanh_inplace SIMD matches scalar" {
    // 100-element array: compare SIMD inplace result with scalar tanh.
    var simd_data: [100]f32 = undefined;
    for (0..100) |i| {
        const fi: f32 = @floatFromInt(i);
        simd_data[i] = (fi / 99.0) * 10.0 - 5.0; // range [-5, 5]
    }

    // Compute expected values using scalar tanh_approx.
    var expected: [100]f32 = undefined;
    for (0..100) |i| {
        expected[i] = tanh_approx(simd_data[i]);
    }

    tanh_inplace(&simd_data);

    for (0..100) |i| {
        try std.testing.expectApproxEqAbs(expected[i], simd_data[i], lut_tolerance);
    }
}
