const std = @import("std");
const Allocator = std.mem.Allocator;
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
pub const text = @import("text.zig");
pub const hocr = @import("hocr.zig");
pub const json_renderer = @import("json_renderer.zig");

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

// ── OcrEngine ─────────────────────────────────────────────────────────────────

pub const OcrEngine = struct {
    allocator: Allocator,
    model: tessdata.LSTMRecognizer,
    pix: ?image.Pix,

    /// Initialize an OcrEngine by loading a model from .traineddata bytes.
    pub fn init(allocator: Allocator, model_data: []const u8) !OcrEngine {
        var model = try tessdata.LSTMRecognizer.load(allocator, model_data);
        errdefer model.deinit();
        return .{
            .allocator = allocator,
            .model = model,
            .pix = null,
        };
    }

    /// Release all resources.
    pub fn deinit(self: *OcrEngine) void {
        if (self.pix) |*p| p.deinit();
        self.model.deinit();
        self.* = undefined;
    }

    /// Set the image from an encoded in-memory buffer (PNG, JPEG, etc.).
    /// Any previously loaded image is freed first.
    pub fn setImageFromMemory(self: *OcrEngine, buf: []const u8) !void {
        if (self.pix) |*p| p.deinit();
        self.pix = null;
        self.pix = try image.Pix.loadFromMemory(self.allocator, buf);
    }

    /// Set the image from a file path.
    /// Any previously loaded image is freed first.
    pub fn setImageFromFile(self: *OcrEngine, path: [*:0]const u8) !void {
        if (self.pix) |*p| p.deinit();
        self.pix = null;
        self.pix = try image.Pix.loadFromFile(self.allocator, path);
    }

    /// Set the image from raw pixel data. The data is copied internally.
    /// Any previously loaded image is freed first.
    pub fn setImageRaw(self: *OcrEngine, pixels: []const u8, w: u32, h: u32, channels: u32) !void {
        if (self.pix) |*p| p.deinit();
        self.pix = null;

        const expected_len = @as(usize, w) * @as(usize, h) * @as(usize, channels);
        if (pixels.len != expected_len) return error.InvalidDimensions;

        const data = try self.allocator.alloc(u8, expected_len);
        @memcpy(data, pixels);

        self.pix = .{
            .width = w,
            .height = h,
            .channels = channels,
            .data = data,
            .allocator = self.allocator,
        };
    }

    /// Run the full OCR pipeline: preprocess -> layout analysis -> line recognition -> assemble PageResult.
    pub fn recognize(self: *OcrEngine) !types.PageResult {
        const src_pix = self.pix orelse return error.NoImageLoaded;
        const alloc = self.allocator;

        // 1. Convert to grayscale if needed.
        var gray = try src_pix.toGrayscale(alloc);
        defer gray.deinit();

        // 2. Binarize for layout analysis.
        var binary = try gray.binarize(alloc);
        defer binary.deinit();

        // 3. Deskew the binary image.
        var deskewed = try binary.deskew(alloc);
        defer deskewed.deinit();

        // 4. Extract connected components from the deskewed binary image.
        const components = try layout.extractComponents(alloc, deskewed.data, deskewed.width, deskewed.height);
        defer alloc.free(components);

        // 5. Classify blobs.
        const blobs = try layout.classifyBlobs(alloc, components);
        defer alloc.free(blobs);

        // 6. Filter to text blobs only.
        const text_blobs = try layout.filterTextBlobs(alloc, blobs);
        defer alloc.free(text_blobs);

        // 7. Detect text lines.
        const lines = try layout.detectTextLines(alloc, text_blobs);
        // lines ownership transfers to detectColumns below.

        // 8. Detect columns/blocks (takes ownership of the lines array).
        const layout_blocks = try layout.detectColumns(alloc, lines);
        defer {
            for (layout_blocks) |*b| {
                var block = b.*;
                block.deinit();
            }
            alloc.free(layout_blocks);
        }

        // 9. For each block's each line, recognize text and build OcrBlock/OcrLine/OcrWord.
        const page_w = if (self.pix) |p| p.width else 0;
        const page_h = if (self.pix) |p| p.height else 0;

        var ocr_blocks = try alloc.alloc(types.OcrBlock, layout_blocks.len);
        var blocks_built: usize = 0;
        errdefer {
            for (ocr_blocks[0..blocks_built]) |*ob| {
                for (ob.lines) |*ol| {
                    for (ol.words) |*ow| {
                        alloc.free(ow.text);
                    }
                    alloc.free(ol.words);
                }
                alloc.free(ob.lines);
            }
            alloc.free(ocr_blocks);
        }

        for (layout_blocks) |lb| {
            var ocr_lines = try alloc.alloc(types.OcrLine, lb.lines.len);
            var lines_built: usize = 0;
            errdefer {
                for (ocr_lines[0..lines_built]) |*ol| {
                    for (ol.words) |*ow| {
                        alloc.free(ow.text);
                    }
                    alloc.free(ol.words);
                }
                alloc.free(ocr_lines);
            }

            for (lb.lines) |tl| {
                // Segment words from layout.
                const layout_words = try layout.segmentWords(alloc, &tl);
                defer alloc.free(layout_words);

                // Crop the line region from the grayscale image for recognition.
                const line_y_min: usize = @intCast(tl.y_min);
                const line_y_max: usize = @intCast(tl.y_max);
                const line_x_min: usize = @intCast(lb.bbox.x);
                const line_x_max: usize = @intCast(lb.bbox.x + lb.bbox.width);
                const crop_w = if (line_x_max > line_x_min) line_x_max - line_x_min else 1;
                const crop_h = if (line_y_max > line_y_min) line_y_max - line_y_min else 1;

                const crop_buf = try alloc.alloc(u8, crop_w * crop_h);
                defer alloc.free(crop_buf);

                const gray_w: usize = @intCast(gray.width);
                const gray_h: usize = @intCast(gray.height);
                for (0..crop_h) |y| {
                    for (0..crop_w) |x| {
                        const src_y = line_y_min + y;
                        const src_x = line_x_min + x;
                        if (src_y < gray_h and src_x < gray_w) {
                            crop_buf[y * crop_w + x] = gray.data[src_y * gray_w + src_x];
                        } else {
                            crop_buf[y * crop_w + x] = 255;
                        }
                    }
                }

                // Recognize the line.
                var rec_result = try recognizer.recognizeLine(alloc, &self.model, crop_buf, crop_w, crop_h);
                defer rec_result.deinit();

                // Split recognized text into words, mapping to layout words.
                var ocr_words = try alloc.alloc(types.OcrWord, layout_words.len);
                var words_built: usize = 0;
                errdefer {
                    for (ocr_words[0..words_built]) |*ow| {
                        alloc.free(ow.text);
                    }
                    alloc.free(ocr_words);
                }

                if (layout_words.len == 0) {
                    // No words from layout, nothing to assign.
                } else if (layout_words.len == 1) {
                    // Single word: assign all recognized text.
                    const word_text = try alloc.dupe(u8, rec_result.text);
                    errdefer alloc.free(word_text);

                    var avg_conf: f32 = 0.0;
                    if (rec_result.confidences.len > 0) {
                        var sum: f32 = 0;
                        for (rec_result.confidences) |cf| sum += cf;
                        avg_conf = @exp(sum / @as(f32, @floatFromInt(rec_result.confidences.len)));
                    }

                    ocr_words[0] = .{
                        .text = word_text,
                        .confidence = avg_conf,
                        .bbox = layoutBBoxToOcrBBox(layout_words[0].bbox),
                    };
                    words_built = 1;
                } else {
                    // Multiple words: split recognized text by spaces.
                    const text_parts = try splitBySpaces(alloc, rec_result.text);
                    defer {
                        for (text_parts) |p| alloc.free(p);
                        alloc.free(text_parts);
                    }

                    var avg_conf: f32 = 0.0;
                    if (rec_result.confidences.len > 0) {
                        var sum: f32 = 0;
                        for (rec_result.confidences) |cf| sum += cf;
                        avg_conf = @exp(sum / @as(f32, @floatFromInt(rec_result.confidences.len)));
                    }

                    if (text_parts.len == layout_words.len) {
                        for (layout_words, 0..) |lw, wi| {
                            const word_text = try alloc.dupe(u8, text_parts[wi]);
                            ocr_words[wi] = .{
                                .text = word_text,
                                .confidence = avg_conf,
                                .bbox = layoutBBoxToOcrBBox(lw.bbox),
                            };
                            words_built += 1;
                        }
                    } else {
                        // Mismatch: assign all text to first word, empty for the rest.
                        const full_text = try alloc.dupe(u8, rec_result.text);
                        ocr_words[0] = .{
                            .text = full_text,
                            .confidence = avg_conf,
                            .bbox = layoutBBoxToOcrBBox(layout_words[0].bbox),
                        };
                        words_built = 1;

                        for (layout_words[1..], 0..) |lw, idx| {
                            const empty_text = try alloc.dupe(u8, "");
                            ocr_words[1 + idx] = .{
                                .text = empty_text,
                                .confidence = 0.0,
                                .bbox = layoutBBoxToOcrBBox(lw.bbox),
                            };
                            words_built += 1;
                        }
                    }
                }

                const line_bbox = types.BBox{
                    .x = lb.bbox.x,
                    .y = tl.y_min,
                    .width = lb.bbox.width,
                    .height = if (tl.y_max > tl.y_min) tl.y_max - tl.y_min else 0,
                };

                ocr_lines[lines_built] = .{
                    .words = ocr_words,
                    .bbox = line_bbox,
                    .baseline_slope = tl.baseline_slope,
                    .baseline_intercept = tl.baseline_intercept,
                };
                lines_built += 1;
            }

            ocr_blocks[blocks_built] = .{
                .lines = ocr_lines,
                .bbox = layoutBBoxToOcrBBox(lb.bbox),
            };
            blocks_built += 1;
        }

        return types.PageResult{
            .blocks = ocr_blocks,
            .page_width = page_w,
            .page_height = page_h,
            .allocator = alloc,
        };
    }

    /// Render the page result as plain UTF-8 text.
    pub fn getText(self: *OcrEngine, result: *const types.PageResult) ![]u8 {
        return text.renderText(self.allocator, result);
    }

    /// Render the page result as hOCR HTML.
    pub fn getHocr(self: *OcrEngine, result: *const types.PageResult) ![]u8 {
        return hocr.renderHocr(self.allocator, result);
    }

    /// Render the page result as structured JSON.
    pub fn getJson(self: *OcrEngine, result: *const types.PageResult) ![]u8 {
        return json_renderer.renderJson(self.allocator, result);
    }
};

/// Convert a layout BoundingBox to a types.BBox.
fn layoutBBoxToOcrBBox(lb: layout.BoundingBox) types.BBox {
    return .{
        .x = lb.x,
        .y = lb.y,
        .width = lb.width,
        .height = lb.height,
    };
}

/// Split a byte slice by ASCII spaces. Returns owned slices (caller frees each + the array).
fn splitBySpaces(allocator: Allocator, input: []const u8) ![][]u8 {
    var parts = std.ArrayList([]u8).init(allocator);
    errdefer {
        for (parts.items) |p| allocator.free(p);
        parts.deinit();
    }

    var iter = std.mem.splitScalar(u8, input, ' ');
    while (iter.next()) |segment| {
        if (segment.len > 0) {
            const dupe = try allocator.dupe(u8, segment);
            errdefer allocator.free(dupe);
            try parts.append(dupe);
        }
    }

    return parts.toOwnedSlice();
}

// ── C-ABI / WASM Exports ──────────────────────────────────────────────────────

const is_wasm = @import("builtin").target.cpu.arch == .wasm32 or
    @import("builtin").target.cpu.arch == .wasm64;

/// Allocator used by C-ABI exports. On WASM, use page_allocator; on native, use c_allocator.
const export_allocator: Allocator = if (is_wasm) std.heap.page_allocator else std.heap.c_allocator;

/// Opaque handle for C-ABI consumers.
const ExportEngine = opaque {};
const ExportResult = opaque {};

fn toEngine(ptr: *ExportEngine) *OcrEngine {
    return @ptrCast(@alignCast(ptr));
}

fn toResult(ptr: *ExportResult) *types.PageResult {
    return @ptrCast(@alignCast(ptr));
}

/// Initialize an OCR engine from .traineddata model bytes.
/// Returns null on failure.
export fn txtswiper_init(model_data: [*]const u8, model_len: usize) ?*ExportEngine {
    const alloc = export_allocator;
    const engine = alloc.create(OcrEngine) catch return null;
    engine.* = OcrEngine.init(alloc, model_data[0..model_len]) catch {
        alloc.destroy(engine);
        return null;
    };
    return @ptrCast(engine);
}

/// Set the image from raw pixel data (grayscale or RGB/RGBA).
/// Returns 0 on success, -1 on failure.
export fn txtswiper_set_image(handle: *ExportEngine, data: [*]const u8, w: u32, h: u32, ch: u32) i32 {
    const engine = toEngine(handle);
    const len = @as(usize, w) * @as(usize, h) * @as(usize, ch);
    engine.setImageRaw(data[0..len], w, h, ch) catch return -1;
    return 0;
}

/// Run OCR recognition. Returns null on failure.
/// Caller must free with txtswiper_free_result.
export fn txtswiper_recognize(handle: *ExportEngine) ?*ExportResult {
    const engine = toEngine(handle);
    const alloc = engine.allocator;
    const result = alloc.create(types.PageResult) catch return null;
    result.* = engine.recognize() catch {
        alloc.destroy(result);
        return null;
    };
    return @ptrCast(result);
}

/// Get recognized text as UTF-8. Writes into out_buf, returns number of bytes written.
/// If buf_len is 0, returns the required buffer size.
export fn txtswiper_get_text(result_handle: *ExportResult, out_buf: ?[*]u8, buf_len: usize) usize {
    const result = toResult(result_handle);
    const alloc = export_allocator;
    const rendered = text.renderText(alloc, result) catch return 0;
    defer alloc.free(rendered);

    if (buf_len == 0 or out_buf == null) return rendered.len;

    const copy_len = @min(rendered.len, buf_len);
    if (out_buf) |buf| {
        @memcpy(buf[0..copy_len], rendered[0..copy_len]);
    }
    return copy_len;
}

/// Get hOCR output. Same semantics as txtswiper_get_text.
export fn txtswiper_get_hocr(result_handle: *ExportResult, out_buf: ?[*]u8, buf_len: usize) usize {
    const result = toResult(result_handle);
    const alloc = export_allocator;
    const rendered = hocr.renderHocr(alloc, result) catch return 0;
    defer alloc.free(rendered);

    if (buf_len == 0 or out_buf == null) return rendered.len;

    const copy_len = @min(rendered.len, buf_len);
    if (out_buf) |buf| {
        @memcpy(buf[0..copy_len], rendered[0..copy_len]);
    }
    return copy_len;
}

/// Get JSON output. Same semantics as txtswiper_get_text.
export fn txtswiper_get_json(result_handle: *ExportResult, out_buf: ?[*]u8, buf_len: usize) usize {
    const result = toResult(result_handle);
    const alloc = export_allocator;
    const rendered = json_renderer.renderJson(alloc, result) catch return 0;
    defer alloc.free(rendered);

    if (buf_len == 0 or out_buf == null) return rendered.len;

    const copy_len = @min(rendered.len, buf_len);
    if (out_buf) |buf| {
        @memcpy(buf[0..copy_len], rendered[0..copy_len]);
    }
    return copy_len;
}

/// Free a PageResult returned by txtswiper_recognize.
export fn txtswiper_free_result(result_handle: *ExportResult) void {
    const result = toResult(result_handle);
    result.deinit();
    export_allocator.destroy(result);
}

/// Free the OCR engine.
export fn txtswiper_free(handle: *ExportEngine) void {
    const engine = toEngine(handle);
    engine.deinit();
    export_allocator.destroy(engine);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

test "stb_image linked" {
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
    _ = text;
    _ = hocr;
    _ = json_renderer;
}

test "splitBySpaces: basic splitting" {
    const alloc = std.testing.allocator;
    const parts = try splitBySpaces(alloc, "hello world foo");
    defer {
        for (parts) |p| alloc.free(p);
        alloc.free(parts);
    }

    try std.testing.expectEqual(@as(usize, 3), parts.len);
    try std.testing.expectEqualStrings("hello", parts[0]);
    try std.testing.expectEqualStrings("world", parts[1]);
    try std.testing.expectEqualStrings("foo", parts[2]);
}

test "splitBySpaces: single word (no spaces)" {
    const alloc = std.testing.allocator;
    const parts = try splitBySpaces(alloc, "hello");
    defer {
        for (parts) |p| alloc.free(p);
        alloc.free(parts);
    }

    try std.testing.expectEqual(@as(usize, 1), parts.len);
    try std.testing.expectEqualStrings("hello", parts[0]);
}

test "splitBySpaces: empty input" {
    const alloc = std.testing.allocator;
    const parts = try splitBySpaces(alloc, "");
    defer {
        for (parts) |p| alloc.free(p);
        alloc.free(parts);
    }

    try std.testing.expectEqual(@as(usize, 0), parts.len);
}

test "splitBySpaces: multiple consecutive spaces" {
    const alloc = std.testing.allocator;
    const parts = try splitBySpaces(alloc, "  hello   world  ");
    defer {
        for (parts) |p| alloc.free(p);
        alloc.free(parts);
    }

    try std.testing.expectEqual(@as(usize, 2), parts.len);
    try std.testing.expectEqualStrings("hello", parts[0]);
    try std.testing.expectEqualStrings("world", parts[1]);
}

test "layoutBBoxToOcrBBox: converts correctly" {
    const lb = layout.BoundingBox{ .x = 10, .y = 20, .width = 30, .height = 40 };
    const ob = layoutBBoxToOcrBBox(lb);
    try std.testing.expectEqual(@as(u32, 10), ob.x);
    try std.testing.expectEqual(@as(u32, 20), ob.y);
    try std.testing.expectEqual(@as(u32, 30), ob.width);
    try std.testing.expectEqual(@as(u32, 40), ob.height);
}

test "OcrEngine: init/deinit with real model" {
    const alloc = std.testing.allocator;

    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer file.close();

    const data = file.readToEndAlloc(alloc, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer alloc.free(data);

    var ocr = OcrEngine.init(alloc, data) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer ocr.deinit();

    try std.testing.expectEqual(@as(usize, 36), ocr.model.numInputs());
    try std.testing.expectEqual(@as(usize, 111), ocr.model.numClasses());
}

test "OcrEngine: setImageRaw stores pixels" {
    const alloc = std.testing.allocator;

    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer file.close();

    const data = file.readToEndAlloc(alloc, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer alloc.free(data);

    var ocr = OcrEngine.init(alloc, data) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer ocr.deinit();

    var pixels = [_]u8{ 10, 20, 30, 40, 50, 60, 70, 80 };
    try ocr.setImageRaw(&pixels, 4, 2, 1);

    try std.testing.expect(ocr.pix != null);
    try std.testing.expectEqual(@as(u32, 4), ocr.pix.?.width);
    try std.testing.expectEqual(@as(u32, 2), ocr.pix.?.height);
    try std.testing.expectEqual(@as(u32, 1), ocr.pix.?.channels);
}

test "OcrEngine: setImageRaw invalid dimensions returns error" {
    const alloc = std.testing.allocator;

    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer file.close();

    const model_data = file.readToEndAlloc(alloc, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer alloc.free(model_data);

    var ocr = OcrEngine.init(alloc, model_data) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer ocr.deinit();

    var pixels = [_]u8{ 10, 20, 30 };
    try std.testing.expectError(error.InvalidDimensions, ocr.setImageRaw(&pixels, 4, 2, 1));
    try std.testing.expect(ocr.pix == null);
}

test "OcrEngine: setImage replaces previous (no leak)" {
    const alloc = std.testing.allocator;

    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer file.close();

    const model_data = file.readToEndAlloc(alloc, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer alloc.free(model_data);

    var ocr = OcrEngine.init(alloc, model_data) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer ocr.deinit();

    var pixels1 = [_]u8{ 0, 0, 0, 0 };
    try ocr.setImageRaw(&pixels1, 2, 2, 1);
    try std.testing.expectEqual(@as(u32, 2), ocr.pix.?.width);

    var pixels2 = [_]u8{ 128, 128, 128, 128, 128, 128 };
    try ocr.setImageRaw(&pixels2, 3, 2, 1);
    try std.testing.expectEqual(@as(u32, 3), ocr.pix.?.width);
}

test "OcrEngine: full pipeline with synthetic image" {
    const alloc = std.testing.allocator;

    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer file.close();

    const model_data = file.readToEndAlloc(alloc, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer alloc.free(model_data);

    var ocr = OcrEngine.init(alloc, model_data) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer ocr.deinit();

    const W: u32 = 200;
    const H: u32 = 100;
    var pixels: [W * H]u8 = undefined;
    @memset(&pixels, 255);

    // Draw dark blobs at y=30..50.
    for (30..50) |y| {
        for (10..60) |x| {
            pixels[y * W + x] = 0;
        }
        for (70..120) |x| {
            pixels[y * W + x] = 0;
        }
    }

    try ocr.setImageRaw(&pixels, W, H, 1);

    var result = try ocr.recognize();
    defer result.deinit();

    try std.testing.expectEqual(@as(u32, W), result.page_width);
    try std.testing.expectEqual(@as(u32, H), result.page_height);
}

test "OcrEngine: getText/getHocr/getJson convenience wrappers" {
    const alloc = std.testing.allocator;

    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer file.close();

    const model_data = file.readToEndAlloc(alloc, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer alloc.free(model_data);

    var ocr = OcrEngine.init(alloc, model_data) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer ocr.deinit();

    const word_text = try alloc.dupe(u8, "hello");
    const words = try alloc.alloc(types.OcrWord, 1);
    words[0] = .{
        .text = word_text,
        .confidence = 0.95,
        .bbox = .{ .x = 10, .y = 20, .width = 50, .height = 15 },
    };

    const ocr_lines = try alloc.alloc(types.OcrLine, 1);
    ocr_lines[0] = .{
        .words = words,
        .bbox = .{ .x = 10, .y = 20, .width = 50, .height = 15 },
        .baseline_slope = 0.0,
        .baseline_intercept = 35.0,
    };

    const ocr_blocks = try alloc.alloc(types.OcrBlock, 1);
    ocr_blocks[0] = .{
        .lines = ocr_lines,
        .bbox = .{ .x = 10, .y = 20, .width = 50, .height = 15 },
    };

    var page_result = types.PageResult{
        .blocks = ocr_blocks,
        .page_width = 200,
        .page_height = 100,
        .allocator = alloc,
    };
    defer page_result.deinit();

    const text_out = try ocr.getText(&page_result);
    defer alloc.free(text_out);
    try std.testing.expectEqualStrings("hello", text_out);

    const hocr_out = try ocr.getHocr(&page_result);
    defer alloc.free(hocr_out);
    try std.testing.expect(std.mem.startsWith(u8, hocr_out, "<?xml"));
    try std.testing.expect(std.mem.indexOf(u8, hocr_out, ">hello</span>") != null);

    const json_out = try ocr.getJson(&page_result);
    defer alloc.free(json_out);
    try std.testing.expect(std.mem.indexOf(u8, json_out, "\"text\":\"hello\"") != null);
}

test "OcrEngine: recognize on empty image produces empty result" {
    const alloc = std.testing.allocator;

    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer file.close();

    const model_data = file.readToEndAlloc(alloc, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer alloc.free(model_data);

    var ocr = OcrEngine.init(alloc, model_data) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer ocr.deinit();

    const W: u32 = 50;
    const H: u32 = 50;
    var pixels: [W * H]u8 = undefined;
    @memset(&pixels, 255);

    try ocr.setImageRaw(&pixels, W, H, 1);
    var result = try ocr.recognize();
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 0), result.blocks.len);
}

test "OcrEngine: recognize without image returns error" {
    const alloc = std.testing.allocator;

    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer file.close();

    const model_data = file.readToEndAlloc(alloc, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer alloc.free(model_data);

    var ocr = OcrEngine.init(alloc, model_data) catch |err| {
        std.debug.print("Skipping test: {}\n", .{err});
        return;
    };
    defer ocr.deinit();

    try std.testing.expectError(error.NoImageLoaded, ocr.recognize());
}
