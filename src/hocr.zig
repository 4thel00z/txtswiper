const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("types.zig");
const PageResult = types.PageResult;
const BBox = types.BBox;

/// Render a PageResult as hOCR HTML.
/// Caller owns the returned string.
pub fn renderHocr(allocator: Allocator, page: *const PageResult) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();
    const w = buf.writer();

    // XML prologue and HTML boilerplate
    try w.writeAll(
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
        \\<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
        \\<head>
        \\<meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>
        \\<title>OCR Output</title>
        \\</head>
        \\<body>
        \\
    );

    const page_id: u32 = 1;
    try w.print("<div class=\"ocr_page\" id=\"page_{d}\" title=\"bbox 0 0 {d} {d}\">\n", .{
        page_id,
        page.page_width,
        page.page_height,
    });

    for (page.blocks, 1..) |block, block_idx| {
        try w.print("  <div class=\"ocr_carea\" id=\"block_{d}_{d}\" title=\"bbox {d} {d} {d} {d}\">\n", .{
            page_id,
            block_idx,
            block.bbox.x,
            block.bbox.y,
            block.bbox.right(),
            block.bbox.bottom(),
        });

        for (block.lines, 1..) |line, line_idx| {
            try w.print("    <span class=\"ocr_line\" id=\"line_{d}_{d}_{d}\" title=\"bbox {d} {d} {d} {d}; baseline {d:.1} {d:.1}\">\n", .{
                page_id,
                block_idx,
                line_idx,
                line.bbox.x,
                line.bbox.y,
                line.bbox.right(),
                line.bbox.bottom(),
                line.baseline_slope,
                line.baseline_intercept,
            });

            for (line.words, 1..) |word, word_idx| {
                const conf_pct = @as(u32, @intFromFloat(@round(word.confidence * 100.0)));
                try w.print("      <span class=\"ocrx_word\" id=\"word_{d}_{d}_{d}_{d}\" title=\"bbox {d} {d} {d} {d}; x_wconf {d}\">", .{
                    page_id,
                    block_idx,
                    line_idx,
                    word_idx,
                    word.bbox.x,
                    word.bbox.y,
                    word.bbox.right(),
                    word.bbox.bottom(),
                    conf_pct,
                });

                try writeEscaped(w, word.text);

                try w.writeAll("</span>\n");
            }

            try w.writeAll("    </span>\n");
        }

        try w.writeAll("  </div>\n");
    }

    try w.writeAll("</div>\n");
    try w.writeAll("</body>\n</html>\n");

    return buf.toOwnedSlice();
}

/// Write HTML-escaped text to the writer.
fn writeEscaped(writer: anytype, text: []const u8) !void {
    for (text) |ch| {
        switch (ch) {
            '&' => try writer.writeAll("&amp;"),
            '<' => try writer.writeAll("&lt;"),
            '>' => try writer.writeAll("&gt;"),
            '"' => try writer.writeAll("&quot;"),
            else => try writer.writeByte(ch),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

const testing = std.testing;

/// Helper: build a PageResult without owning text (for tests that manage their own memory).
fn makeTestPage(
    blocks: []types.OcrBlock,
    width: u32,
    height: u32,
) PageResult {
    return .{
        .blocks = blocks,
        .page_width = width,
        .page_height = height,
        .allocator = testing.allocator,
    };
}

test "hocr: single word" {
    var words = [_]types.OcrWord{
        .{
            .text = "hello",
            .confidence = 0.95,
            .bbox = .{ .x = 10, .y = 20, .width = 50, .height = 15 },
        },
    };
    var lines = [_]types.OcrLine{
        .{
            .words = &words,
            .bbox = .{ .x = 10, .y = 20, .width = 50, .height = 15 },
            .baseline_slope = 0.0,
            .baseline_intercept = 35.0,
        },
    };
    var blocks = [_]types.OcrBlock{
        .{
            .lines = &lines,
            .bbox = .{ .x = 10, .y = 20, .width = 50, .height = 15 },
        },
    };

    const page = makeTestPage(&blocks, 200, 100);
    const html = try renderHocr(testing.allocator, &page);
    defer testing.allocator.free(html);

    // Verify XML declaration
    try testing.expect(std.mem.startsWith(u8, html, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));

    // Verify page bbox
    try testing.expect(std.mem.indexOf(u8, html, "bbox 0 0 200 100") != null);

    // Verify block bbox: right = 10+50 = 60, bottom = 20+15 = 35
    try testing.expect(std.mem.indexOf(u8, html, "bbox 10 20 60 35") != null);

    // Verify word text and confidence (0.95 * 100 = 95)
    try testing.expect(std.mem.indexOf(u8, html, "x_wconf 95") != null);
    try testing.expect(std.mem.indexOf(u8, html, ">hello</span>") != null);

    // Verify IDs
    try testing.expect(std.mem.indexOf(u8, html, "id=\"page_1\"") != null);
    try testing.expect(std.mem.indexOf(u8, html, "id=\"block_1_1\"") != null);
    try testing.expect(std.mem.indexOf(u8, html, "id=\"line_1_1_1\"") != null);
    try testing.expect(std.mem.indexOf(u8, html, "id=\"word_1_1_1_1\"") != null);

    // Verify baseline
    try testing.expect(std.mem.indexOf(u8, html, "baseline 0.0 35.0") != null);
}

test "hocr: html escaping" {
    var words = [_]types.OcrWord{
        .{
            .text = "<b>&\"test\"</b>",
            .confidence = 0.80,
            .bbox = .{ .x = 0, .y = 0, .width = 40, .height = 10 },
        },
    };
    var lines = [_]types.OcrLine{
        .{
            .words = &words,
            .bbox = .{ .x = 0, .y = 0, .width = 40, .height = 10 },
            .baseline_slope = 0.0,
            .baseline_intercept = 10.0,
        },
    };
    var blocks = [_]types.OcrBlock{
        .{
            .lines = &lines,
            .bbox = .{ .x = 0, .y = 0, .width = 40, .height = 10 },
        },
    };

    const page = makeTestPage(&blocks, 100, 50);
    const html = try renderHocr(testing.allocator, &page);
    defer testing.allocator.free(html);

    // The word text should be escaped
    try testing.expect(std.mem.indexOf(u8, html, "&lt;b&gt;&amp;&quot;test&quot;&lt;/b&gt;") != null);

    // Confidence: 0.80 * 100 = 80
    try testing.expect(std.mem.indexOf(u8, html, "x_wconf 80") != null);
}

test "hocr: empty page" {
    var blocks = [_]types.OcrBlock{};

    const page = makeTestPage(&blocks, 640, 480);
    const html = try renderHocr(testing.allocator, &page);
    defer testing.allocator.free(html);

    // Should still be valid HTML
    try testing.expect(std.mem.startsWith(u8, html, "<?xml"));
    try testing.expect(std.mem.indexOf(u8, html, "bbox 0 0 640 480") != null);
    try testing.expect(std.mem.indexOf(u8, html, "</body>") != null);
    try testing.expect(std.mem.indexOf(u8, html, "</html>") != null);

    // No blocks or words
    try testing.expect(std.mem.indexOf(u8, html, "ocr_carea") == null);
    try testing.expect(std.mem.indexOf(u8, html, "ocrx_word") == null);
}

test "hocr: multiple blocks with unique ids" {
    // Block 1: 1 line, 2 words
    var words1 = [_]types.OcrWord{
        .{
            .text = "foo",
            .confidence = 0.90,
            .bbox = .{ .x = 0, .y = 0, .width = 30, .height = 10 },
        },
        .{
            .text = "bar",
            .confidence = 0.85,
            .bbox = .{ .x = 35, .y = 0, .width = 30, .height = 10 },
        },
    };
    var lines1 = [_]types.OcrLine{
        .{
            .words = &words1,
            .bbox = .{ .x = 0, .y = 0, .width = 65, .height = 10 },
            .baseline_slope = 0.1,
            .baseline_intercept = 10.0,
        },
    };

    // Block 2: 1 line, 1 word
    var words2 = [_]types.OcrWord{
        .{
            .text = "baz",
            .confidence = 0.70,
            .bbox = .{ .x = 0, .y = 50, .width = 30, .height = 10 },
        },
    };
    var lines2 = [_]types.OcrLine{
        .{
            .words = &words2,
            .bbox = .{ .x = 0, .y = 50, .width = 30, .height = 10 },
            .baseline_slope = -0.2,
            .baseline_intercept = 60.0,
        },
    };

    var blocks = [_]types.OcrBlock{
        .{
            .lines = &lines1,
            .bbox = .{ .x = 0, .y = 0, .width = 65, .height = 10 },
        },
        .{
            .lines = &lines2,
            .bbox = .{ .x = 0, .y = 50, .width = 30, .height = 10 },
        },
    };

    const page = makeTestPage(&blocks, 300, 200);
    const html = try renderHocr(testing.allocator, &page);
    defer testing.allocator.free(html);

    // Verify both blocks present with unique IDs
    try testing.expect(std.mem.indexOf(u8, html, "id=\"block_1_1\"") != null);
    try testing.expect(std.mem.indexOf(u8, html, "id=\"block_1_2\"") != null);

    // Lines have correct hierarchical IDs
    try testing.expect(std.mem.indexOf(u8, html, "id=\"line_1_1_1\"") != null);
    try testing.expect(std.mem.indexOf(u8, html, "id=\"line_1_2_1\"") != null);

    // Words have correct hierarchical IDs
    try testing.expect(std.mem.indexOf(u8, html, "id=\"word_1_1_1_1\"") != null);
    try testing.expect(std.mem.indexOf(u8, html, "id=\"word_1_1_1_2\"") != null);
    try testing.expect(std.mem.indexOf(u8, html, "id=\"word_1_2_1_1\"") != null);

    // Verify all word texts present
    try testing.expect(std.mem.indexOf(u8, html, ">foo</span>") != null);
    try testing.expect(std.mem.indexOf(u8, html, ">bar</span>") != null);
    try testing.expect(std.mem.indexOf(u8, html, ">baz</span>") != null);

    // Verify baselines
    try testing.expect(std.mem.indexOf(u8, html, "baseline 0.1 10.0") != null);
    try testing.expect(std.mem.indexOf(u8, html, "baseline -0.2 60.0") != null);
}
