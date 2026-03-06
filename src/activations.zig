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

// ── In-place activation functions ────────────────────────────────────────────

/// Apply sigmoid to every element of `data` in place.
pub fn sigmoid_inplace(data: []f32) void {
    for (data) |*v| {
        v.* = sigmoid(v.*);
    }
}

/// Apply tanh (via tanh_approx) to every element of `data` in place.
pub fn tanh_inplace(data: []f32) void {
    for (data) |*v| {
        v.* = tanh_approx(v.*);
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
