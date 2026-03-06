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
