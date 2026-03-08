const std = @import("std");
const Allocator = std.mem.Allocator;

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
