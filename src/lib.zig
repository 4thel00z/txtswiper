const std = @import("std");
const c = @cImport({
    @cInclude("stb_image.h");
});

pub const activations = @import("activations.zig");
pub const weights = @import("weights.zig");
pub const network = @import("network.zig");
pub const tessdata = @import("tessdata.zig");
pub const recognizer = @import("recognizer.zig");
pub const image = @import("image.zig");
pub const layout = @import("layout.zig");
pub const types = @import("types.zig");

/// Load an image from a file path. Returns pixel data and image dimensions.
/// Caller owns the returned pixel slice and must free it with `stbiFree`.
pub fn loadImage(path: [*:0]const u8) !struct { data: [*]u8, width: c_int, height: c_int, channels: c_int } {
    var width: c_int = 0;
    var height: c_int = 0;
    var channels: c_int = 0;
    const data = c.stbi_load(path, &width, &height, &channels, 0);
    if (data == null) {
        return error.ImageLoadFailed;
    }
    return .{ .data = data, .width = width, .height = height, .channels = channels };
}

/// Free pixel data previously returned by `loadImage`.
pub fn stbiFree(data: [*]u8) void {
    c.stbi_image_free(data);
}

test "stb_image linked" {
    // Verify that stb_image symbols are available at link time.
    // We do not load an actual file here; a null path should return null.
    var w: c_int = 0;
    var h: c_int = 0;
    var ch: c_int = 0;
    const result = c.stbi_load("__nonexistent__.png", &w, &h, &ch, 0);
    try std.testing.expect(result == null);
}

test {
    _ = activations;
    _ = weights;
    _ = network;
    _ = tessdata;
    _ = recognizer;
    _ = image;
    _ = layout;
    _ = types;
}
