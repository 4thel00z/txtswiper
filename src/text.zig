const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("types.zig");
const PageResult = types.PageResult;

/// Render a PageResult as plain UTF-8 text.
/// Words separated by spaces, lines by newlines, blocks by double newlines.
/// Caller owns the returned string.
pub fn renderText(allocator: Allocator, page: *const PageResult) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();

    for (page.blocks, 0..) |block, bi| {
        if (bi > 0) try buf.appendSlice("\n\n");

        for (block.lines, 0..) |line, li| {
            if (li > 0) try buf.append('\n');

            for (line.words, 0..) |word, wi| {
                if (wi > 0) try buf.append(' ');
                try buf.appendSlice(word.text);
            }
        }
    }

    return buf.toOwnedSlice();
}

// ── Tests ────────────────────────────────────────────────────────────────────

const testing = std.testing;
const OcrWord = types.OcrWord;
const OcrLine = types.OcrLine;
const OcrBlock = types.OcrBlock;
const BBox = types.BBox;

const zero_box = BBox{ .x = 0, .y = 0, .width = 0, .height = 0 };

fn makeWord(text: []const u8) OcrWord {
    return .{ .text = text, .confidence = 1.0, .bbox = zero_box };
}

fn makeLine(words: []OcrWord) OcrLine {
    return .{ .words = words, .bbox = zero_box, .baseline_slope = 0.0, .baseline_intercept = 0.0 };
}

fn makeBlock(lines: []OcrLine) OcrBlock {
    return .{ .lines = lines, .bbox = zero_box };
}

test "renderText: single word" {
    var words = [_]OcrWord{makeWord("Hello")};
    var lines = [_]OcrLine{makeLine(&words)};
    var blocks = [_]OcrBlock{makeBlock(&lines)};
    const page = PageResult{
        .blocks = &blocks,
        .page_width = 100,
        .page_height = 100,
        .allocator = testing.allocator,
    };
    const result = try renderText(testing.allocator, &page);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings("Hello", result);
}

test "renderText: single line with three words" {
    var words = [_]OcrWord{ makeWord("Hello"), makeWord("world"), makeWord("foo") };
    var lines = [_]OcrLine{makeLine(&words)};
    var blocks = [_]OcrBlock{makeBlock(&lines)};
    const page = PageResult{
        .blocks = &blocks,
        .page_width = 100,
        .page_height = 100,
        .allocator = testing.allocator,
    };
    const result = try renderText(testing.allocator, &page);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings("Hello world foo", result);
}

test "renderText: two lines in one block" {
    var words1 = [_]OcrWord{makeWord("line1")};
    var words2 = [_]OcrWord{makeWord("line2")};
    var lines = [_]OcrLine{ makeLine(&words1), makeLine(&words2) };
    var blocks = [_]OcrBlock{makeBlock(&lines)};
    const page = PageResult{
        .blocks = &blocks,
        .page_width = 100,
        .page_height = 100,
        .allocator = testing.allocator,
    };
    const result = try renderText(testing.allocator, &page);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings("line1\nline2", result);
}

test "renderText: two blocks" {
    var words1 = [_]OcrWord{makeWord("block1")};
    var lines1 = [_]OcrLine{makeLine(&words1)};
    var words2 = [_]OcrWord{makeWord("block2")};
    var lines2 = [_]OcrLine{makeLine(&words2)};
    var blocks = [_]OcrBlock{ makeBlock(&lines1), makeBlock(&lines2) };
    const page = PageResult{
        .blocks = &blocks,
        .page_width = 100,
        .page_height = 100,
        .allocator = testing.allocator,
    };
    const result = try renderText(testing.allocator, &page);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings("block1\n\nblock2", result);
}

test "renderText: full document (2 blocks, 2 lines each)" {
    var w1a = [_]OcrWord{ makeWord("Hello"), makeWord("world") };
    var w1b = [_]OcrWord{makeWord("second line")};
    var lines1 = [_]OcrLine{ makeLine(&w1a), makeLine(&w1b) };

    var w2a = [_]OcrWord{makeWord("Another")};
    var w2b = [_]OcrWord{ makeWord("last"), makeWord("line") };
    var lines2 = [_]OcrLine{ makeLine(&w2a), makeLine(&w2b) };

    var blocks = [_]OcrBlock{ makeBlock(&lines1), makeBlock(&lines2) };
    const page = PageResult{
        .blocks = &blocks,
        .page_width = 200,
        .page_height = 200,
        .allocator = testing.allocator,
    };
    const result = try renderText(testing.allocator, &page);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings("Hello world\nsecond line\n\nAnother\nlast line", result);
}

test "renderText: empty page" {
    var blocks = [_]OcrBlock{};
    const page = PageResult{
        .blocks = &blocks,
        .page_width = 0,
        .page_height = 0,
        .allocator = testing.allocator,
    };
    const result = try renderText(testing.allocator, &page);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings("", result);
}
