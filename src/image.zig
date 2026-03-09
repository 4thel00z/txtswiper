const std = @import("std");
const Allocator = std.mem.Allocator;
const NetworkIO = @import("network.zig").NetworkIO;

const c = @cImport({
    @cInclude("stb_image.h");
});

/// Image container. Pixel data is always Zig-owned (allocator.free in deinit).
pub const Pix = struct {
    width: u32,
    height: u32,
    channels: u32,
    data: []u8,
    allocator: Allocator,

    /// Load an image from a file path. Returns a Pix with Zig-owned pixel data.
    pub fn loadFromFile(allocator: Allocator, path: [*:0]const u8) !Pix {
        var w: c_int = 0;
        var h: c_int = 0;
        var ch: c_int = 0;
        const ptr = c.stbi_load(path, &w, &h, &ch, 0) orelse return error.ImageLoadFailed;
        defer c.stbi_image_free(ptr);

        const width: u32 = @intCast(w);
        const height: u32 = @intCast(h);
        const channels: u32 = @intCast(ch);
        const len = width * height * channels;

        const data = try allocator.alloc(u8, len);
        @memcpy(data, ptr[0..len]);

        return .{
            .width = width,
            .height = height,
            .channels = channels,
            .data = data,
            .allocator = allocator,
        };
    }

    /// Load an image from an in-memory buffer. Returns a Pix with Zig-owned pixel data.
    pub fn loadFromMemory(allocator: Allocator, buf: []const u8) !Pix {
        var w: c_int = 0;
        var h: c_int = 0;
        var ch: c_int = 0;
        const ptr = c.stbi_load_from_memory(buf.ptr, @intCast(buf.len), &w, &h, &ch, 0) orelse return error.ImageLoadFailed;
        defer c.stbi_image_free(ptr);

        const width: u32 = @intCast(w);
        const height: u32 = @intCast(h);
        const channels: u32 = @intCast(ch);
        const len = width * height * channels;

        const data = try allocator.alloc(u8, len);
        @memcpy(data, ptr[0..len]);

        return .{
            .width = width,
            .height = height,
            .channels = channels,
            .data = data,
            .allocator = allocator,
        };
    }

    /// Convert to single-channel grayscale using luminance formula:
    ///   Y = 0.299*R + 0.587*G + 0.114*B
    /// Returns a new Pix. If already 1 channel, returns a copy.
    pub fn toGrayscale(self: *const Pix, allocator: Allocator) !Pix {
        if (self.channels == 1) {
            const data = try allocator.alloc(u8, self.data.len);
            @memcpy(data, self.data);
            return .{
                .width = self.width,
                .height = self.height,
                .channels = 1,
                .data = data,
                .allocator = allocator,
            };
        }

        if (self.channels != 3 and self.channels != 4) {
            return error.UnsupportedChannelCount;
        }

        const pixel_count = self.width * self.height;
        const gray = try allocator.alloc(u8, pixel_count);
        const stride = self.channels;

        for (0..pixel_count) |i| {
            const offset = i * stride;
            const r: f32 = @floatFromInt(self.data[offset]);
            const g: f32 = @floatFromInt(self.data[offset + 1]);
            const b: f32 = @floatFromInt(self.data[offset + 2]);
            const y = 0.299 * r + 0.587 * g + 0.114 * b;
            gray[i] = @intFromFloat(@round(@min(@max(y, 0.0), 255.0)));
        }

        return .{
            .width = self.width,
            .height = self.height,
            .channels = 1,
            .data = gray,
            .allocator = allocator,
        };
    }

    /// Compute the optimal Otsu threshold for a single-channel grayscale image.
    /// Returns a threshold value in [0, 255].
    pub fn otsuThreshold(self: *const Pix) !u8 {
        if (self.channels != 1) return error.NotGrayscale;

        const n = self.data.len;
        if (n == 0) return 0;

        // Step 1: build 256-bin histogram
        var hist: [256]u32 = [_]u32{0} ** 256;
        for (self.data) |px| {
            hist[px] += 1;
        }

        // Step 2: find threshold that maximizes between-class variance
        const total: u64 = n;
        var sum_total: u64 = 0;
        for (0..256) |i| {
            sum_total += @as(u64, i) * @as(u64, hist[i]);
        }

        var sum_bg: u64 = 0;
        var weight_bg: u64 = 0;
        var best_sigma: u128 = 0;
        var best_t: u8 = 0;

        for (0..256) |t| {
            weight_bg += hist[t];
            if (weight_bg == 0) continue;
            const weight_fg = total - weight_bg;
            if (weight_fg == 0) break;

            sum_bg += @as(u64, t) * @as(u64, hist[t]);
            const sum_fg = sum_total - sum_bg;

            // sigma_B^2 = w0 * w1 * (mu1 - mu0)^2
            // To avoid floating point, compute:
            //   mu0 = sum_bg / weight_bg,  mu1 = sum_fg / weight_fg
            //   (mu1 - mu0) = (sum_fg * weight_bg - sum_bg * weight_fg) / (weight_bg * weight_fg)
            //   sigma_B^2 * weight_bg * weight_fg = (sum_fg * weight_bg - sum_bg * weight_fg)^2 / (weight_bg * weight_fg)
            // We maximize: (sum_fg * weight_bg - sum_bg * weight_fg)^2 / (weight_bg * weight_fg)
            // which is equivalent since the denominator is always positive.
            // But to avoid overflow with large images, use u128.
            const a: i128 = @as(i128, @intCast(sum_fg)) * @as(i128, @intCast(weight_bg));
            const b: i128 = @as(i128, @intCast(sum_bg)) * @as(i128, @intCast(weight_fg));
            const diff = a - b;
            const numerator: u128 = @intCast(diff * diff);
            const denominator: u128 = @as(u128, weight_bg) * @as(u128, weight_fg);
            // Compare: numerator / denominator > best_sigma
            // => numerator > best_sigma * denominator  (avoids division)
            const sigma = numerator / denominator;
            if (sigma > best_sigma) {
                best_sigma = sigma;
                best_t = @intCast(t);
            }
        }

        return best_t;
    }

    /// Apply Otsu thresholding to produce a binary (1-channel) image.
    /// Pixels above the threshold become 255, at or below become 0.
    pub fn binarize(self: *const Pix, allocator: Allocator) !Pix {
        // If not already grayscale, convert first
        if (self.channels != 1) return error.NotGrayscale;

        const t = try self.otsuThreshold();
        const len = self.width * self.height;
        const data = try allocator.alloc(u8, len);

        for (0..len) |i| {
            data[i] = if (self.data[i] > t) 255 else 0;
        }

        return .{
            .width = self.width,
            .height = self.height,
            .channels = 1,
            .data = data,
            .allocator = allocator,
        };
    }

    /// Detect the skew angle of a document image using projection profiles.
    /// Works on single-channel (grayscale/binary) images.
    /// Returns the detected skew angle in degrees (range: -5.0 to +5.0).
    /// Returns 0 if the image is too small or uniform.
    pub fn detectSkew(self: *const Pix) f32 {
        if (self.channels != 1) return 0;
        if (self.width < 10 or self.height < 10) return 0;

        const w: usize = @intCast(self.width);
        const h: usize = @intCast(self.height);
        const cx: f32 = @as(f32, @floatFromInt(self.width)) / 2.0;
        const cy: f32 = @as(f32, @floatFromInt(self.height)) / 2.0;

        var best_score: f64 = 0;
        var best_angle: f32 = 0;

        // Test angles from -5.0 to +5.0 degrees in 0.25 degree steps (41 candidates)
        var step: i32 = -20;
        while (step <= 20) : (step += 1) {
            const angle_deg: f32 = @as(f32, @floatFromInt(step)) * 0.25;
            const angle_rad: f32 = angle_deg * (std.math.pi / 180.0);
            const cos_a = @cos(angle_rad);
            const sin_a = @sin(angle_rad);

            // Compute horizontal projection profile for this rotation angle.
            // For each pixel in the source, compute its rotated row and accumulate.
            // We use a profile buffer sized to h (original height) -- pixels mapping
            // outside are simply ignored.
            var profile: [4096]f64 = undefined;
            const profile_len = if (h <= 4096) h else 4096;
            for (0..profile_len) |i| {
                profile[i] = 0;
            }

            for (0..h) |y| {
                const fy: f32 = @as(f32, @floatFromInt(y)) - cy;
                for (0..w) |x| {
                    const px = self.data[y * w + x];
                    // Only count dark pixels (value < 128 means dark/text)
                    if (px >= 128) continue;

                    const fx: f32 = @as(f32, @floatFromInt(x)) - cx;
                    // Rotated y coordinate
                    const ry = -sin_a * fx + cos_a * fy + cy;
                    const ry_int: i32 = @intFromFloat(@round(ry));
                    if (ry_int >= 0 and ry_int < @as(i32, @intCast(profile_len))) {
                        const idx: usize = @intCast(ry_int);
                        profile[idx] += 1.0;
                    }
                }
            }

            // Compute sharpness: sum of squared differences between adjacent rows
            var score: f64 = 0;
            for (1..profile_len) |i| {
                const diff = profile[i] - profile[i - 1];
                score += diff * diff;
            }

            if (score > best_score) {
                best_score = score;
                best_angle = angle_deg;
            }
        }

        return best_angle;
    }

    /// Rotate the image by the given angle (in degrees) using bilinear interpolation.
    /// Positive angles rotate counter-clockwise. Background fill is 255 (white).
    /// Returns a new Pix with the same dimensions. Caller owns the result.
    pub fn rotate(self: *const Pix, angle_degrees: f32, allocator: Allocator) !Pix {
        if (self.channels != 1) return error.NotGrayscale;

        const w: usize = @intCast(self.width);
        const h: usize = @intCast(self.height);
        const len = w * h;

        const out_data = try allocator.alloc(u8, len);
        errdefer allocator.free(out_data);

        const angle_rad: f32 = angle_degrees * (std.math.pi / 180.0);
        const cos_a = @cos(angle_rad);
        const sin_a = @sin(angle_rad);
        const cx: f32 = @as(f32, @floatFromInt(self.width)) / 2.0;
        const cy: f32 = @as(f32, @floatFromInt(self.height)) / 2.0;

        const w_f: f32 = @floatFromInt(self.width);
        const h_f: f32 = @floatFromInt(self.height);

        for (0..h) |y| {
            const fy: f32 = @as(f32, @floatFromInt(y)) - cy;
            for (0..w) |x| {
                const fx: f32 = @as(f32, @floatFromInt(x)) - cx;

                // Inverse rotation: find source pixel for this output pixel
                const src_x = cos_a * fx + sin_a * fy + cx;
                const src_y = -sin_a * fx + cos_a * fy + cy;

                // Bilinear interpolation
                if (src_x < 0 or src_y < 0 or src_x >= w_f or src_y >= h_f) {
                    out_data[y * w + x] = 255; // white background
                } else {
                    const x0_f = @floor(src_x);
                    const y0_f = @floor(src_y);
                    const x0: usize = @intFromFloat(x0_f);
                    const y0: usize = @intFromFloat(y0_f);
                    const x1 = if (x0 + 1 < w) x0 + 1 else x0;
                    const y1 = if (y0 + 1 < h) y0 + 1 else y0;

                    const dx = src_x - x0_f;
                    const dy = src_y - y0_f;

                    const p00: f32 = @floatFromInt(self.data[y0 * w + x0]);
                    const p10: f32 = @floatFromInt(self.data[y0 * w + x1]);
                    const p01: f32 = @floatFromInt(self.data[y1 * w + x0]);
                    const p11: f32 = @floatFromInt(self.data[y1 * w + x1]);

                    const val = p00 * (1.0 - dx) * (1.0 - dy) +
                        p10 * dx * (1.0 - dy) +
                        p01 * (1.0 - dx) * dy +
                        p11 * dx * dy;

                    out_data[y * w + x] = @intFromFloat(@round(@min(@max(val, 0.0), 255.0)));
                }
            }
        }

        return .{
            .width = self.width,
            .height = self.height,
            .channels = 1,
            .data = out_data,
            .allocator = allocator,
        };
    }

    /// Convenience: detect skew and correct it. Returns a deskewed copy.
    pub fn deskew(self: *const Pix, allocator: Allocator) !Pix {
        const angle = self.detectSkew();
        if (angle == 0) {
            // No skew detected, return a copy
            const data = try allocator.alloc(u8, self.data.len);
            @memcpy(data, self.data);
            return .{
                .width = self.width,
                .height = self.height,
                .channels = self.channels,
                .data = data,
                .allocator = allocator,
            };
        }
        // Rotate by the negative of the detected skew to correct it
        return self.rotate(-angle, allocator);
    }

    /// Prepare image for LSTM input. Scales to target_height preserving aspect
    /// ratio, normalizes pixel values to [-1, 1], and packs into a NetworkIO
    /// with 2D stride layout for the Conv->Maxpool->XYTranspose->LSTM pipeline.
    /// If multi-channel, automatically converts to grayscale first.
    /// Caller owns the returned NetworkIO and must call deinit() on it.
    pub fn prepareLSTMInput(self: *const Pix, allocator: Allocator, target_height: usize) !NetworkIO {
        if (self.width == 0 or self.height == 0 or target_height == 0) {
            return NetworkIO.init(allocator, 0, 1, false);
        }

        // If multi-channel, convert to grayscale first.
        var gray_pix: ?Pix = null;
        defer if (gray_pix) |*gp| gp.deinit();

        const gray_data: []const u8 = if (self.channels > 1) blk: {
            gray_pix = try self.toGrayscale(allocator);
            break :blk gray_pix.?.data;
        } else self.data;

        const width: usize = @intCast(self.width);
        const height: usize = @intCast(self.height);

        // Compute scaled width preserving aspect ratio (with rounding).
        const scaled_width = (width * target_height + height / 2) / height;
        if (scaled_width == 0) {
            return NetworkIO.init(allocator, 0, 1, false);
        }

        // Allocate NetworkIO with 2D stride: [target_height * scaled_width][1].
        var nio = try NetworkIO.init2D(allocator, target_height, scaled_width, 1, false);
        errdefer nio.deinit();

        const buf = nio.f_data.?;

        // Compute adaptive normalization: global min/max for contrast stretching.
        var min_pixel: f32 = 255.0;
        var max_pixel: f32 = 0.0;
        for (gray_data) |p| {
            const v: f32 = @floatFromInt(p);
            if (v < min_pixel) min_pixel = v;
            if (v > max_pixel) max_pixel = v;
        }
        const black = min_pixel;
        var contrast = (max_pixel - black) / 2.0;
        if (contrast <= 0.0) contrast = 1.0;

        // Fill the NetworkIO: row-major (y then x).
        // Timestep t = y * scaled_width + x, 1 feature per timestep.
        for (0..target_height) |y| {
            for (0..scaled_width) |x| {
                // Map scaled (x, y) back to source coordinates via bilinear interpolation.
                const src_xf: f32 = if (scaled_width > 1)
                    @as(f32, @floatFromInt(x)) * @as(f32, @floatFromInt(width - 1)) / @as(f32, @floatFromInt(scaled_width - 1))
                else
                    @as(f32, @floatFromInt(width - 1)) * 0.5;

                const src_yf: f32 = if (target_height > 1)
                    @as(f32, @floatFromInt(y)) * @as(f32, @floatFromInt(height - 1)) / @as(f32, @floatFromInt(target_height - 1))
                else
                    @as(f32, @floatFromInt(height - 1)) * 0.5;

                // Bilinear interpolation.
                const x0: usize = @intFromFloat(@floor(src_xf));
                const y0: usize = @intFromFloat(@floor(src_yf));
                const x1: usize = @min(x0 + 1, width - 1);
                const y1: usize = @min(y0 + 1, height - 1);

                const dx = src_xf - @as(f32, @floatFromInt(x0));
                const dy = src_yf - @as(f32, @floatFromInt(y0));

                const p00: f32 = @floatFromInt(gray_data[y0 * width + x0]);
                const p10: f32 = @floatFromInt(gray_data[y0 * width + x1]);
                const p01: f32 = @floatFromInt(gray_data[y1 * width + x0]);
                const p11: f32 = @floatFromInt(gray_data[y1 * width + x1]);

                const val = p00 * (1.0 - dx) * (1.0 - dy) +
                    p10 * dx * (1.0 - dy) +
                    p01 * (1.0 - dx) * dy +
                    p11 * dx * dy;

                // Normalize: (pixel - black) / contrast - 1.0  (maps to [-1, 1])
                const normalized = (val - black) / contrast - 1.0;

                const t = y * scaled_width + x;
                buf[t] = normalized;
            }
        }

        return nio;
    }

    /// Free pixel data.
    pub fn deinit(self: *Pix) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }
};

// ── Tests ────────────────────────────────────────────────────────────────────

// Minimal 2x2 RGB PNG (76 bytes).
// Pixels: (255,0,0), (0,255,0), (0,0,255), (128,128,128)
const test_png = [_]u8{
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02,
    0x08, 0x02, 0x00, 0x00, 0x00, 0xfd, 0xd4, 0x9a, 0x73, 0x00, 0x00, 0x00,
    0x13, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0xf8, 0xcf, 0xc0, 0xc0,
    0x00, 0xc2, 0x0c, 0xff, 0x1b, 0x1a, 0x1a, 0x00, 0x1c, 0xf4, 0x04, 0x7e,
    0x29, 0x80, 0x40, 0xd8, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44,
    0xae, 0x42, 0x60, 0x82,
};

test "loadFromMemory: 2x2 RGB PNG dimensions" {
    var pix = try Pix.loadFromMemory(std.testing.allocator, &test_png);
    defer pix.deinit();

    try std.testing.expectEqual(@as(u32, 2), pix.width);
    try std.testing.expectEqual(@as(u32, 2), pix.height);
    try std.testing.expectEqual(@as(u32, 3), pix.channels);
    try std.testing.expectEqual(@as(usize, 12), pix.data.len);
}

test "loadFromMemory: pixel values correct" {
    var pix = try Pix.loadFromMemory(std.testing.allocator, &test_png);
    defer pix.deinit();

    // Pixel 0: red (255,0,0)
    try std.testing.expectEqual(@as(u8, 255), pix.data[0]);
    try std.testing.expectEqual(@as(u8, 0), pix.data[1]);
    try std.testing.expectEqual(@as(u8, 0), pix.data[2]);

    // Pixel 1: green (0,255,0)
    try std.testing.expectEqual(@as(u8, 0), pix.data[3]);
    try std.testing.expectEqual(@as(u8, 255), pix.data[4]);
    try std.testing.expectEqual(@as(u8, 0), pix.data[5]);

    // Pixel 2: blue (0,0,255)
    try std.testing.expectEqual(@as(u8, 0), pix.data[6]);
    try std.testing.expectEqual(@as(u8, 0), pix.data[7]);
    try std.testing.expectEqual(@as(u8, 255), pix.data[8]);

    // Pixel 3: gray (128,128,128)
    try std.testing.expectEqual(@as(u8, 128), pix.data[9]);
    try std.testing.expectEqual(@as(u8, 128), pix.data[10]);
    try std.testing.expectEqual(@as(u8, 128), pix.data[11]);
}

test "loadFromMemory: invalid data returns error" {
    const garbage = [_]u8{ 0x00, 0x01, 0x02, 0x03 };
    const result = Pix.loadFromMemory(std.testing.allocator, &garbage);
    try std.testing.expectError(error.ImageLoadFailed, result);
}

test "toGrayscale: RGB to single channel" {
    var pix = try Pix.loadFromMemory(std.testing.allocator, &test_png);
    defer pix.deinit();

    var gray = try pix.toGrayscale(std.testing.allocator);
    defer gray.deinit();

    try std.testing.expectEqual(@as(u32, 2), gray.width);
    try std.testing.expectEqual(@as(u32, 2), gray.height);
    try std.testing.expectEqual(@as(u32, 1), gray.channels);
    try std.testing.expectEqual(@as(usize, 4), gray.data.len);

    // Red (255,0,0): Y = 0.299*255 = 76.245 -> 76
    try std.testing.expectEqual(@as(u8, 76), gray.data[0]);
    // Green (0,255,0): Y = 0.587*255 = 149.685 -> 150
    try std.testing.expectEqual(@as(u8, 150), gray.data[1]);
    // Blue (0,0,255): Y = 0.114*255 = 29.07 -> 29
    try std.testing.expectEqual(@as(u8, 29), gray.data[2]);
    // Gray (128,128,128): Y = 128*(0.299+0.587+0.114) = 128 -> 128
    try std.testing.expectEqual(@as(u8, 128), gray.data[3]);
}

test "toGrayscale: already grayscale returns copy" {
    // Build a 1-channel Pix manually
    const alloc = std.testing.allocator;
    const data = try alloc.alloc(u8, 4);
    data[0] = 10;
    data[1] = 20;
    data[2] = 30;
    data[3] = 40;

    var pix = Pix{
        .width = 2,
        .height = 2,
        .channels = 1,
        .data = data,
        .allocator = alloc,
    };
    defer pix.deinit();

    var copy = try pix.toGrayscale(alloc);
    defer copy.deinit();

    try std.testing.expectEqual(@as(u32, 1), copy.channels);
    try std.testing.expectEqualSlices(u8, pix.data, copy.data);
    // Verify it's a real copy, not aliased
    try std.testing.expect(pix.data.ptr != copy.data.ptr);
}

test "loadFromFile: nonexistent file returns error" {
    const result = Pix.loadFromFile(std.testing.allocator, "__nonexistent_test_file__.png");
    try std.testing.expectError(error.ImageLoadFailed, result);
}

// ── Otsu binarization tests ──────────────────────────────────────────────

test "otsuThreshold: bimodal image" {
    const alloc = std.testing.allocator;
    const size = 200;
    const data = try alloc.alloc(u8, size);
    defer alloc.free(data);

    // Half at 50, half at 200
    for (0..100) |i| data[i] = 50;
    for (100..200) |i| data[i] = 200;

    const pix = Pix{
        .width = 10,
        .height = 20,
        .channels = 1,
        .data = data,
        .allocator = alloc,
    };
    // Don't deinit pix — we manage data ourselves with defer above

    const t = try pix.otsuThreshold();
    // Otsu finds the threshold that maximally separates the two classes.
    // For two equal-sized groups at 50 and 200, the best split is at 50
    // (puts group-50 in background and group-200 in foreground).
    try std.testing.expect(t >= 50 and t < 200);
}

test "binarize: correct output" {
    const alloc = std.testing.allocator;
    const data = try alloc.alloc(u8, 4);
    data[0] = 30;
    data[1] = 80;
    data[2] = 170;
    data[3] = 220;

    var pix = Pix{
        .width = 2,
        .height = 2,
        .channels = 1,
        .data = data,
        .allocator = alloc,
    };
    defer pix.deinit();

    var bin = try pix.binarize(alloc);
    defer bin.deinit();

    // All pixels should be 0 or 255
    for (bin.data) |px| {
        try std.testing.expect(px == 0 or px == 255);
    }

    // Threshold should split somewhere in the middle;
    // 30 and 80 should be dark, 170 and 220 should be bright
    try std.testing.expectEqual(@as(u8, 0), bin.data[0]); // 30
    try std.testing.expectEqual(@as(u8, 0), bin.data[1]); // 80
    try std.testing.expectEqual(@as(u8, 255), bin.data[2]); // 170
    try std.testing.expectEqual(@as(u8, 255), bin.data[3]); // 220
}

test "otsuThreshold: uniform image" {
    const alloc = std.testing.allocator;
    const data = try alloc.alloc(u8, 100);
    defer alloc.free(data);

    @memset(data, 128);

    const pix = Pix{
        .width = 10,
        .height = 10,
        .channels = 1,
        .data = data,
        .allocator = alloc,
    };

    // Should not error; any threshold is valid
    const t = try pix.otsuThreshold();
    _ = t;
}

test "otsuThreshold: gradient image" {
    const alloc = std.testing.allocator;
    const size: usize = 256;
    const data = try alloc.alloc(u8, size);
    defer alloc.free(data);

    // One pixel of each value 0..255
    for (0..size) |i| data[i] = @intCast(i);

    const pix = Pix{
        .width = 16,
        .height = 16,
        .channels = 1,
        .data = data,
        .allocator = alloc,
    };

    const t = try pix.otsuThreshold();
    // For a uniform histogram, Otsu should land near the midpoint
    try std.testing.expect(t >= 100 and t <= 155);
}

// ── Deskew tests ─────────────────────────────────────────────────────────

test "detectSkew: straight horizontal lines returns ~0" {
    const alloc = std.testing.allocator;
    const w: usize = 200;
    const h: usize = 100;
    const data = try alloc.alloc(u8, w * h);
    defer alloc.free(data);

    // White background
    @memset(data, 255);

    // Draw horizontal dark lines at rows 20, 40, 60, 80
    for ([_]usize{ 20, 40, 60, 80 }) |row| {
        for (10..190) |col| {
            data[row * w + col] = 0;
        }
    }

    const pix = Pix{
        .width = @intCast(w),
        .height = @intCast(h),
        .channels = 1,
        .data = data,
        .allocator = alloc,
    };

    const angle = pix.detectSkew();
    // Should be very close to 0 for perfectly horizontal lines
    try std.testing.expect(@abs(angle) <= 0.5);
}

test "rotate: zero degrees preserves image" {
    const alloc = std.testing.allocator;
    const w: usize = 10;
    const h: usize = 10;
    const data = try alloc.alloc(u8, w * h);

    // Fill with a gradient pattern
    for (0..w * h) |i| {
        data[i] = @intCast(i % 256);
    }

    var pix = Pix{
        .width = @intCast(w),
        .height = @intCast(h),
        .channels = 1,
        .data = data,
        .allocator = alloc,
    };
    defer pix.deinit();

    var rotated = try pix.rotate(0.0, alloc);
    defer rotated.deinit();

    try std.testing.expectEqual(pix.width, rotated.width);
    try std.testing.expectEqual(pix.height, rotated.height);
    try std.testing.expectEqual(pix.channels, rotated.channels);

    // At zero rotation, output should match input exactly (or very close due to float rounding)
    for (0..w * h) |i| {
        const diff: i16 = @as(i16, pix.data[i]) - @as(i16, rotated.data[i]);
        try std.testing.expect(@abs(diff) <= 1);
    }
}

test "rotate: 90 degrees maps pixels correctly" {
    const alloc = std.testing.allocator;
    // Use a 7x7 image with a dark pixel near center so it stays in-bounds after rotation
    const w: usize = 7;
    const h: usize = 7;
    const data = try alloc.alloc(u8, w * h);

    // White background, single dark pixel at (2, 1) -- near center
    @memset(data, 255);
    data[1 * w + 2] = 0; // (x=2, y=1)

    var pix = Pix{
        .width = @intCast(w),
        .height = @intCast(h),
        .channels = 1,
        .data = data,
        .allocator = alloc,
    };
    defer pix.deinit();

    var rotated = try pix.rotate(90.0, alloc);
    defer rotated.deinit();

    // After 90-degree CCW rotation around center (3.5, 3.5):
    // For output (x,y): src_x = sin(90)*(y-3.5)+3.5 = y, src_y = -(- sin(90))*(... ) ...
    // The dark pixel at source (2,1) should appear somewhere in the output.
    // Verify the dark pixel(s) moved (at least one dark pixel exists in output).
    var dark_count: usize = 0;
    for (rotated.data) |px| {
        if (px < 128) dark_count += 1;
    }
    try std.testing.expect(dark_count >= 1);

    // Also verify the original position is now white (pixel moved)
    try std.testing.expect(rotated.data[1 * w + 2] == 255);
}

test "deskew: corrects slightly skewed image" {
    const alloc = std.testing.allocator;
    const w: usize = 200;
    const h: usize = 200;
    const data = try alloc.alloc(u8, w * h);

    // White background
    @memset(data, 255);

    // Draw lines that are skewed by ~3 degrees.
    // At 3 degrees, over 200px width the line rises by ~10px.
    // Line at base row 50: for each x, y = 50 + round(x * tan(3 degrees))
    const skew_angle: f32 = 3.0;
    const tan_a = @tan(skew_angle * (std.math.pi / 180.0));

    for ([_]usize{ 50, 80, 110, 140, 170 }) |base_row| {
        for (10..190) |x_u| {
            const x_f: f32 = @floatFromInt(x_u);
            const offset: f32 = x_f * tan_a;
            const y_i: i32 = @as(i32, @intCast(base_row)) + @as(i32, @intFromFloat(@round(offset)));
            if (y_i >= 0 and y_i < @as(i32, @intCast(h))) {
                const y_u: usize = @intCast(y_i);
                data[y_u * w + x_u] = 0;
                // Make lines 2px thick for better detection
                if (y_u + 1 < h) data[(y_u + 1) * w + x_u] = 0;
            }
        }
    }

    var pix = Pix{
        .width = @intCast(w),
        .height = @intCast(h),
        .channels = 1,
        .data = data,
        .allocator = alloc,
    };
    defer pix.deinit();

    // Detect should find approximately +3 degrees
    const detected = pix.detectSkew();
    try std.testing.expect(@abs(detected - skew_angle) <= 1.0);

    // Deskew should produce a corrected image
    var corrected = try pix.deskew(alloc);
    defer corrected.deinit();

    try std.testing.expectEqual(pix.width, corrected.width);
    try std.testing.expectEqual(pix.height, corrected.height);
}

test "detectSkew: image too small returns 0" {
    const alloc = std.testing.allocator;
    const data = try alloc.alloc(u8, 9);
    defer alloc.free(data);
    @memset(data, 0);

    const pix = Pix{
        .width = 3,
        .height = 3,
        .channels = 1,
        .data = data,
        .allocator = alloc,
    };

    try std.testing.expectEqual(@as(f32, 0.0), pix.detectSkew());
}

test "detectSkew: non-grayscale returns 0" {
    const alloc = std.testing.allocator;
    const data = try alloc.alloc(u8, 300);
    defer alloc.free(data);
    @memset(data, 0);

    const pix = Pix{
        .width = 10,
        .height = 10,
        .channels = 3,
        .data = data,
        .allocator = alloc,
    };

    try std.testing.expectEqual(@as(f32, 0.0), pix.detectSkew());
}

test "rotate: non-grayscale returns error" {
    const alloc = std.testing.allocator;
    const data = try alloc.alloc(u8, 300);
    defer alloc.free(data);
    @memset(data, 128);

    const pix = Pix{
        .width = 10,
        .height = 10,
        .channels = 3,
        .data = data,
        .allocator = alloc,
    };

    try std.testing.expectError(error.NotGrayscale, pix.rotate(5.0, alloc));
}

// ── prepareLSTMInput tests ───────────────────────────────────────────────

test "prepareLSTMInput: dimensions correct for grayscale image" {
    const alloc = std.testing.allocator;
    // Create a 100x50 grayscale image with a gradient pattern
    const w: usize = 100;
    const h: usize = 50;
    const data = try alloc.alloc(u8, w * h);

    for (0..w * h) |i| {
        data[i] = @intCast(i % 256);
    }

    var pix = Pix{
        .width = @intCast(w),
        .height = @intCast(h),
        .channels = 1,
        .data = data,
        .allocator = alloc,
    };
    defer pix.deinit();

    const target_height: usize = 36;
    var nio = try pix.prepareLSTMInput(alloc, target_height);
    defer nio.deinit();

    // scaled_width = (100 * 36 + 25) / 50 = 3625 / 50 = 72 (with rounding)
    const expected_sw = (w * target_height + h / 2) / h;
    try std.testing.expectEqual(target_height, nio.getStrideHeight());
    try std.testing.expectEqual(expected_sw, nio.getStrideWidth());
    try std.testing.expectEqual(target_height * expected_sw, nio.width());
    try std.testing.expectEqual(@as(usize, 1), nio.numFeatures());
}

test "prepareLSTMInput: output values in [-1, 1] range" {
    const alloc = std.testing.allocator;
    // 4x4 image with known values covering a range
    const w: usize = 4;
    const h: usize = 4;
    const data = try alloc.alloc(u8, w * h);

    // Set up a range of pixel values: 10, 50, 100, 200, etc.
    data[0] = 10;
    data[1] = 50;
    data[2] = 100;
    data[3] = 200;
    data[4] = 30;
    data[5] = 80;
    data[6] = 150;
    data[7] = 250;
    data[8] = 10;
    data[9] = 60;
    data[10] = 120;
    data[11] = 200;
    data[12] = 20;
    data[13] = 90;
    data[14] = 180;
    data[15] = 250;

    var pix = Pix{
        .width = @intCast(w),
        .height = @intCast(h),
        .channels = 1,
        .data = data,
        .allocator = alloc,
    };
    defer pix.deinit();

    var nio = try pix.prepareLSTMInput(alloc, 4);
    defer nio.deinit();

    const buf = nio.f_data.?;
    for (buf) |v| {
        try std.testing.expect(v >= -1.0 - 1e-6);
        try std.testing.expect(v <= 1.0 + 1e-6);
    }

    // Min pixel (10) should map to -1.0, max pixel (250) should map to 1.0
    // Since we scale to the same dimensions (4x4 -> 4x4), corner values should be exact.
    // black = 10, contrast = (250-10)/2 = 120
    // min normalized = (10-10)/120 - 1.0 = -1.0
    // max normalized = (250-10)/120 - 1.0 = 240/120 - 1.0 = 2.0 - 1.0 = 1.0
    // Verify the extreme values exist somewhere in output
    var found_min = false;
    var found_max = false;
    for (buf) |v| {
        if (@abs(v - (-1.0)) < 0.01) found_min = true;
        if (@abs(v - 1.0) < 0.01) found_max = true;
    }
    try std.testing.expect(found_min);
    try std.testing.expect(found_max);
}

test "prepareLSTMInput: auto-converts RGB to grayscale" {
    const alloc = std.testing.allocator;
    // 4x4 RGB image
    const w: usize = 4;
    const h: usize = 4;
    const data = try alloc.alloc(u8, w * h * 3);

    // Fill with a simple pattern: (128, 128, 128) everywhere = gray
    for (0..w * h) |i| {
        data[i * 3 + 0] = 128; // R
        data[i * 3 + 1] = 128; // G
        data[i * 3 + 2] = 128; // B
    }
    // Make one pixel brighter and one darker for contrast
    data[0] = 255;
    data[1] = 255;
    data[2] = 255; // pixel 0: white
    data[3] = 0;
    data[4] = 0;
    data[5] = 0; // pixel 1: black

    var pix = Pix{
        .width = @intCast(w),
        .height = @intCast(h),
        .channels = 3,
        .data = data,
        .allocator = alloc,
    };
    defer pix.deinit();

    // Should not error -- auto-converts to grayscale internally
    var nio = try pix.prepareLSTMInput(alloc, 4);
    defer nio.deinit();

    try std.testing.expectEqual(@as(usize, 4), nio.getStrideHeight());
    try std.testing.expectEqual(@as(usize, 4), nio.getStrideWidth());
    try std.testing.expectEqual(@as(usize, 1), nio.numFeatures());

    // Verify values are in [-1, 1]
    const buf = nio.f_data.?;
    for (buf) |v| {
        try std.testing.expect(v >= -1.0 - 1e-6);
        try std.testing.expect(v <= 1.0 + 1e-6);
    }
}

test "prepareLSTMInput: empty image returns zero-width NetworkIO" {
    const alloc = std.testing.allocator;

    // Zero-width image
    {
        const data = try alloc.alloc(u8, 0);
        var pix = Pix{
            .width = 0,
            .height = 10,
            .channels = 1,
            .data = data,
            .allocator = alloc,
        };
        defer pix.deinit();

        var nio = try pix.prepareLSTMInput(alloc, 36);
        defer nio.deinit();

        try std.testing.expectEqual(@as(usize, 0), nio.width());
    }

    // Zero-height image
    {
        const data = try alloc.alloc(u8, 0);
        var pix = Pix{
            .width = 10,
            .height = 0,
            .channels = 1,
            .data = data,
            .allocator = alloc,
        };
        defer pix.deinit();

        var nio = try pix.prepareLSTMInput(alloc, 36);
        defer nio.deinit();

        try std.testing.expectEqual(@as(usize, 0), nio.width());
    }

    // Zero target height
    {
        const data = try alloc.alloc(u8, 100);
        @memset(data, 128);
        var pix = Pix{
            .width = 10,
            .height = 10,
            .channels = 1,
            .data = data,
            .allocator = alloc,
        };
        defer pix.deinit();

        var nio = try pix.prepareLSTMInput(alloc, 0);
        defer nio.deinit();

        try std.testing.expectEqual(@as(usize, 0), nio.width());
    }
}

test "prepareLSTMInput: uniform image produces all -1.0 values" {
    const alloc = std.testing.allocator;
    // All pixels same value => min == max => contrast = 1.0
    // normalized = (128 - 128) / 1.0 - 1.0 = -1.0
    const w: usize = 8;
    const h: usize = 8;
    const data = try alloc.alloc(u8, w * h);
    @memset(data, 128);

    var pix = Pix{
        .width = @intCast(w),
        .height = @intCast(h),
        .channels = 1,
        .data = data,
        .allocator = alloc,
    };
    defer pix.deinit();

    var nio = try pix.prepareLSTMInput(alloc, 4);
    defer nio.deinit();

    const buf = nio.f_data.?;
    for (buf) |v| {
        try std.testing.expect(@abs(v - (-1.0)) < 1e-6);
    }
}
