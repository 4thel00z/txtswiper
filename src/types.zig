const std = @import("std");
const Allocator = std.mem.Allocator;

/// Axis-aligned bounding box in pixel coordinates.
pub const BBox = struct {
    x: u32, // left edge
    y: u32, // top edge
    width: u32,
    height: u32,

    pub fn right(self: BBox) u32 {
        return self.x + self.width;
    }

    pub fn bottom(self: BBox) u32 {
        return self.y + self.height;
    }
};

/// A recognized word with text and confidence.
pub const OcrWord = struct {
    text: []const u8, // UTF-8 text (owned)
    confidence: f32, // 0.0 to 1.0
    bbox: BBox,
};

/// A text line containing words.
pub const OcrLine = struct {
    words: []OcrWord, // owned array of words
    bbox: BBox,
    baseline_slope: f32,
    baseline_intercept: f32,
};

/// A text block (column/paragraph).
pub const OcrBlock = struct {
    lines: []OcrLine, // owned array of lines
    bbox: BBox,
};

/// Full page OCR result.
pub const PageResult = struct {
    blocks: []OcrBlock, // owned array of blocks
    page_width: u32,
    page_height: u32,
    allocator: Allocator,

    /// Free all owned memory recursively.
    pub fn deinit(self: *PageResult) void {
        for (self.blocks) |*block| {
            for (block.lines) |*line| {
                for (line.words) |*word| {
                    self.allocator.free(word.text);
                }
                self.allocator.free(line.words);
            }
            self.allocator.free(block.lines);
        }
        self.allocator.free(self.blocks);
        self.* = undefined;
    }
};

// ── Tests ────────────────────────────────────────────────────────────────────

test "BBox: right and bottom" {
    const bb = BBox{ .x = 10, .y = 20, .width = 30, .height = 40 };
    try std.testing.expectEqual(@as(u32, 40), bb.right());
    try std.testing.expectEqual(@as(u32, 60), bb.bottom());
}

test "BBox: zero-sized box" {
    const bb = BBox{ .x = 5, .y = 10, .width = 0, .height = 0 };
    try std.testing.expectEqual(@as(u32, 5), bb.right());
    try std.testing.expectEqual(@as(u32, 10), bb.bottom());
}

test "PageResult deinit: frees all owned memory" {
    const alloc = std.testing.allocator;

    // Allocate text for words
    const text1 = try alloc.dupe(u8, "hello");
    const text2 = try alloc.dupe(u8, "world");
    const text3 = try alloc.dupe(u8, "foo");

    // Allocate words for line 1 (2 words)
    const words1 = try alloc.alloc(OcrWord, 2);
    words1[0] = .{
        .text = text1,
        .confidence = 0.95,
        .bbox = .{ .x = 10, .y = 10, .width = 50, .height = 20 },
    };
    words1[1] = .{
        .text = text2,
        .confidence = 0.88,
        .bbox = .{ .x = 70, .y = 10, .width = 50, .height = 20 },
    };

    // Allocate words for line 2 (1 word)
    const words2 = try alloc.alloc(OcrWord, 1);
    words2[0] = .{
        .text = text3,
        .confidence = 0.72,
        .bbox = .{ .x = 10, .y = 40, .width = 30, .height = 20 },
    };

    // Allocate lines for block (2 lines)
    const lines = try alloc.alloc(OcrLine, 2);
    lines[0] = .{
        .words = words1,
        .bbox = .{ .x = 10, .y = 10, .width = 110, .height = 20 },
        .baseline_slope = 0.0,
        .baseline_intercept = 30.0,
    };
    lines[1] = .{
        .words = words2,
        .bbox = .{ .x = 10, .y = 40, .width = 30, .height = 20 },
        .baseline_slope = 0.0,
        .baseline_intercept = 60.0,
    };

    // Allocate blocks (1 block)
    const blocks = try alloc.alloc(OcrBlock, 1);
    blocks[0] = .{
        .lines = lines,
        .bbox = .{ .x = 10, .y = 10, .width = 110, .height = 50 },
    };

    var result = PageResult{
        .blocks = blocks,
        .page_width = 200,
        .page_height = 100,
        .allocator = alloc,
    };

    // If deinit leaks, the testing allocator will catch it.
    result.deinit();
}

test "PageResult deinit: empty (0 blocks)" {
    const alloc = std.testing.allocator;

    const blocks = try alloc.alloc(OcrBlock, 0);

    var result = PageResult{
        .blocks = blocks,
        .page_width = 0,
        .page_height = 0,
        .allocator = alloc,
    };

    result.deinit();
}
