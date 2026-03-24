const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("types.zig");
const PageResult = types.PageResult;
const BBox = types.BBox;

/// Render a PageResult as structured JSON.
/// Caller owns the returned string.
pub fn renderJson(allocator: Allocator, page: *const PageResult) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();

    try buf.appendSlice("{\"page\":{\"width\":");
    try appendInt(&buf, page.page_width);
    try buf.appendSlice(",\"height\":");
    try appendInt(&buf, page.page_height);
    try buf.appendSlice(",\"blocks\":[");

    for (page.blocks, 0..) |block, bi| {
        if (bi > 0) try buf.append(',');
        try renderBlock(&buf, block);
    }

    try buf.appendSlice("]}}");
    return buf.toOwnedSlice();
}

fn renderBlock(buf: *std.ArrayList(u8), block: types.OcrBlock) !void {
    try buf.appendSlice("{\"bbox\":");
    try renderBBox(buf, block.bbox);
    try buf.appendSlice(",\"lines\":[");

    for (block.lines, 0..) |line, li| {
        if (li > 0) try buf.append(',');
        try renderLine(buf, line);
    }

    try buf.appendSlice("]}");
}

fn renderLine(buf: *std.ArrayList(u8), line: types.OcrLine) !void {
    try buf.appendSlice("{\"bbox\":");
    try renderBBox(buf, line.bbox);
    try buf.appendSlice(",\"baseline\":{\"slope\":");
    try appendFloat(buf, line.baseline_slope);
    try buf.appendSlice(",\"intercept\":");
    try appendFloat(buf, line.baseline_intercept);
    try buf.appendSlice("},\"words\":[");

    for (line.words, 0..) |word, wi| {
        if (wi > 0) try buf.append(',');
        try renderWord(buf, word);
    }

    try buf.appendSlice("]}");
}

fn renderWord(buf: *std.ArrayList(u8), word: types.OcrWord) !void {
    try buf.appendSlice("{\"text\":\"");
    try appendJsonEscaped(buf, word.text);
    try buf.appendSlice("\",\"confidence\":");
    try appendFloat(buf, word.confidence);
    try buf.appendSlice(",\"bbox\":");
    try renderBBox(buf, word.bbox);
    try buf.append('}');
}

fn renderBBox(buf: *std.ArrayList(u8), bbox: BBox) !void {
    try buf.appendSlice("{\"x\":");
    try appendInt(buf, bbox.x);
    try buf.appendSlice(",\"y\":");
    try appendInt(buf, bbox.y);
    try buf.appendSlice(",\"width\":");
    try appendInt(buf, bbox.width);
    try buf.appendSlice(",\"height\":");
    try appendInt(buf, bbox.height);
    try buf.append('}');
}

fn appendInt(buf: *std.ArrayList(u8), value: u32) !void {
    var tmp: [20]u8 = undefined;
    const slice = std.fmt.bufPrint(&tmp, "{d}", .{value}) catch unreachable;
    try buf.appendSlice(slice);
}

fn appendFloat(buf: *std.ArrayList(u8), value: f32) !void {
    // Format with enough precision; strip unnecessary trailing zeros for clean output.
    var tmp: [64]u8 = undefined;
    const slice = std.fmt.bufPrint(&tmp, "{d:.6}", .{value}) catch unreachable;

    // Find the end of meaningful digits (keep at least one decimal place).
    var end: usize = slice.len;
    if (std.mem.indexOfScalar(u8, slice, '.')) |dot| {
        // Strip trailing zeros, but keep at least X.0
        while (end > dot + 2 and slice[end - 1] == '0') {
            end -= 1;
        }
    }
    try buf.appendSlice(slice[0..end]);
}

/// Escape a string for JSON output.
fn appendJsonEscaped(buf: *std.ArrayList(u8), s: []const u8) !void {
    for (s) |ch| {
        switch (ch) {
            '"' => try buf.appendSlice("\\\""),
            '\\' => try buf.appendSlice("\\\\"),
            '\n' => try buf.appendSlice("\\n"),
            '\t' => try buf.appendSlice("\\t"),
            '\r' => try buf.appendSlice("\\r"),
            0x08 => try buf.appendSlice("\\b"),
            0x0C => try buf.appendSlice("\\f"),
            else => {
                if (ch < 0x20) {
                    // Control character: emit \u00XX
                    var tmp: [6]u8 = undefined;
                    _ = std.fmt.bufPrint(&tmp, "\\u{X:0>4}", .{ch}) catch unreachable;
                    try buf.appendSlice(&tmp);
                } else {
                    try buf.append(ch);
                }
            },
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

const testing = std.testing;
const OcrWord = types.OcrWord;
const OcrLine = types.OcrLine;
const OcrBlock = types.OcrBlock;

const zero_box = BBox{ .x = 0, .y = 0, .width = 0, .height = 0 };

fn makeWord(text: []const u8) OcrWord {
    return .{ .text = text, .confidence = 1.0, .bbox = zero_box };
}

fn makeWordFull(text: []const u8, confidence: f32, bbox: BBox) OcrWord {
    return .{ .text = text, .confidence = confidence, .bbox = bbox };
}

fn makeLine(words: []OcrWord) OcrLine {
    return .{ .words = words, .bbox = zero_box, .baseline_slope = 0.0, .baseline_intercept = 0.0 };
}

fn makeLineFull(words: []OcrWord, bbox: BBox, slope: f32, intercept: f32) OcrLine {
    return .{ .words = words, .bbox = bbox, .baseline_slope = slope, .baseline_intercept = intercept };
}

fn makeBlock(lines: []OcrLine) OcrBlock {
    return .{ .lines = lines, .bbox = zero_box };
}

fn makeBlockFull(lines: []OcrLine, bbox: BBox) OcrBlock {
    return .{ .lines = lines, .bbox = bbox };
}

test "renderJson: single word" {
    const bbox = BBox{ .x = 10, .y = 20, .width = 50, .height = 30 };
    var words = [_]OcrWord{makeWordFull("Hello", 0.95, bbox)};
    var lines = [_]OcrLine{makeLineFull(&words, bbox, 0.0, 45.0)};
    const block_bbox = BBox{ .x = 10, .y = 20, .width = 400, .height = 100 };
    var blocks = [_]OcrBlock{makeBlockFull(&lines, block_bbox)};
    const page = PageResult{
        .blocks = &blocks,
        .page_width = 800,
        .page_height = 600,
        .allocator = testing.allocator,
    };
    const result = try renderJson(testing.allocator, &page);
    defer testing.allocator.free(result);

    // Parse with std.json to verify it's valid JSON.
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, result, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    const pg = root.get("page").?.object;
    try testing.expectEqual(@as(i64, 800), pg.get("width").?.integer);
    try testing.expectEqual(@as(i64, 600), pg.get("height").?.integer);

    const blocks_arr = pg.get("blocks").?.array;
    try testing.expectEqual(@as(usize, 1), blocks_arr.items.len);

    const b0 = blocks_arr.items[0].object;
    const b0_lines = b0.get("lines").?.array;
    try testing.expectEqual(@as(usize, 1), b0_lines.items.len);

    const l0 = b0_lines.items[0].object;
    const l0_words = l0.get("words").?.array;
    try testing.expectEqual(@as(usize, 1), l0_words.items.len);

    const w0 = l0_words.items[0].object;
    try testing.expectEqualStrings("Hello", w0.get("text").?.string);
}

test "renderJson: JSON string escaping" {
    var words = [_]OcrWord{makeWord("He said \"hello\\world\"\nnewline\ttab")};
    var lines = [_]OcrLine{makeLine(&words)};
    var blocks = [_]OcrBlock{makeBlock(&lines)};
    const page = PageResult{
        .blocks = &blocks,
        .page_width = 100,
        .page_height = 100,
        .allocator = testing.allocator,
    };
    const result = try renderJson(testing.allocator, &page);
    defer testing.allocator.free(result);

    // Should be valid JSON.
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, result, .{});
    defer parsed.deinit();

    const pg = parsed.value.object.get("page").?.object;
    const w0 = pg.get("blocks").?.array.items[0].object
        .get("lines").?.array.items[0].object
        .get("words").?.array.items[0].object;

    try testing.expectEqualStrings("He said \"hello\\world\"\nnewline\ttab", w0.get("text").?.string);
}

test "renderJson: empty page" {
    var blocks = [_]OcrBlock{};
    const page = PageResult{
        .blocks = &blocks,
        .page_width = 0,
        .page_height = 0,
        .allocator = testing.allocator,
    };
    const result = try renderJson(testing.allocator, &page);
    defer testing.allocator.free(result);

    try testing.expectEqualStrings("{\"page\":{\"width\":0,\"height\":0,\"blocks\":[]}}", result);
}

test "renderJson: parse roundtrip" {
    const bbox1 = BBox{ .x = 10, .y = 20, .width = 50, .height = 30 };
    const bbox2 = BBox{ .x = 70, .y = 20, .width = 60, .height = 30 };
    var words1 = [_]OcrWord{ makeWordFull("Hello", 0.95, bbox1), makeWordFull("world", 0.88, bbox2) };
    const line_bbox = BBox{ .x = 10, .y = 20, .width = 120, .height = 30 };
    var lines = [_]OcrLine{makeLineFull(&words1, line_bbox, 0.01, 45.5)};
    const block_bbox = BBox{ .x = 10, .y = 20, .width = 400, .height = 100 };
    var blocks = [_]OcrBlock{makeBlockFull(&lines, block_bbox)};
    const page = PageResult{
        .blocks = &blocks,
        .page_width = 800,
        .page_height = 600,
        .allocator = testing.allocator,
    };
    const result = try renderJson(testing.allocator, &page);
    defer testing.allocator.free(result);

    // Full roundtrip: parse and verify every field.
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, result, .{});
    defer parsed.deinit();

    const pg = parsed.value.object.get("page").?.object;
    try testing.expectEqual(@as(i64, 800), pg.get("width").?.integer);
    try testing.expectEqual(@as(i64, 600), pg.get("height").?.integer);

    const b0 = pg.get("blocks").?.array.items[0].object;
    const b0_bbox = b0.get("bbox").?.object;
    try testing.expectEqual(@as(i64, 10), b0_bbox.get("x").?.integer);
    try testing.expectEqual(@as(i64, 400), b0_bbox.get("width").?.integer);

    const l0 = b0.get("lines").?.array.items[0].object;
    const baseline = l0.get("baseline").?.object;
    // slope = 0.01, intercept = 45.5
    try testing.expectApproxEqAbs(@as(f64, 0.01), baseline.get("slope").?.float, 0.001);
    try testing.expectApproxEqAbs(@as(f64, 45.5), baseline.get("intercept").?.float, 0.001);

    const l0_words = l0.get("words").?.array;
    try testing.expectEqual(@as(usize, 2), l0_words.items.len);
    try testing.expectEqualStrings("Hello", l0_words.items[0].object.get("text").?.string);
    try testing.expectEqualStrings("world", l0_words.items[1].object.get("text").?.string);

    // confidence
    try testing.expectApproxEqAbs(@as(f64, 0.95), l0_words.items[0].object.get("confidence").?.float, 0.001);
    try testing.expectApproxEqAbs(@as(f64, 0.88), l0_words.items[1].object.get("confidence").?.float, 0.001);
}

test "renderJson: control character escaping" {
    // Test a word containing a control character (0x01).
    var words = [_]OcrWord{makeWord("a\x01b")};
    var lines = [_]OcrLine{makeLine(&words)};
    var blocks = [_]OcrBlock{makeBlock(&lines)};
    const page = PageResult{
        .blocks = &blocks,
        .page_width = 10,
        .page_height = 10,
        .allocator = testing.allocator,
    };
    const result = try renderJson(testing.allocator, &page);
    defer testing.allocator.free(result);

    // Verify the control char is escaped as \u0001.
    try testing.expect(std.mem.indexOf(u8, result, "\\u0001") != null);

    // Must still be valid JSON.
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, result, .{});
    defer parsed.deinit();

    const w0 = parsed.value.object.get("page").?.object
        .get("blocks").?.array.items[0].object
        .get("lines").?.array.items[0].object
        .get("words").?.array.items[0].object;
    try testing.expectEqualStrings("a\x01b", w0.get("text").?.string);
}
