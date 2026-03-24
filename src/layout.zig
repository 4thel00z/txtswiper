const std = @import("std");
const Allocator = std.mem.Allocator;

// ── Union-Find ───────────────────────────────────────────────────────────────

const UnionFind = struct {
    parent: []u32,
    allocator: Allocator,

    fn init(allocator: Allocator, size: usize) !UnionFind {
        const parent = try allocator.alloc(u32, size);
        for (0..size) |i| {
            parent[i] = @intCast(i);
        }
        return .{ .parent = parent, .allocator = allocator };
    }

    fn deinit(self: *UnionFind) void {
        self.allocator.free(self.parent);
        self.* = undefined;
    }

    fn find(self: *UnionFind, x: u32) u32 {
        var current = x;
        while (self.parent[current] != current) {
            // Path compression: point to grandparent
            self.parent[current] = self.parent[self.parent[current]];
            current = self.parent[current];
        }
        return current;
    }

    fn unite(self: *UnionFind, a: u32, b: u32) void {
        const root_a = self.find(a);
        const root_b = self.find(b);
        if (root_a != root_b) {
            // Always make the smaller label the root for determinism
            if (root_a < root_b) {
                self.parent[root_b] = root_a;
            } else {
                self.parent[root_a] = root_b;
            }
        }
    }
};

// ── BoundingBox ──────────────────────────────────────────────────────────────

pub const BoundingBox = struct {
    x: u32, // left
    y: u32, // top
    width: u32,
    height: u32,

    pub fn right(self: BoundingBox) u32 {
        return self.x + self.width;
    }

    pub fn bottom(self: BoundingBox) u32 {
        return self.y + self.height;
    }

    pub fn area(self: BoundingBox) u32 {
        return self.width * self.height;
    }

    pub fn contains(self: BoundingBox, px: u32, py: u32) bool {
        return px >= self.x and px < self.right() and
            py >= self.y and py < self.bottom();
    }

    pub fn overlapsVertically(self: BoundingBox, other: BoundingBox) bool {
        return self.y < other.bottom() and other.y < self.bottom();
    }
};

// ── Component ────────────────────────────────────────────────────────────────

pub const Component = struct {
    label: u32,
    bbox: BoundingBox,
    pixel_count: u32,
};

// ── Connected Component Extraction ───────────────────────────────────────────

/// Extract connected components from a binary image.
/// Binary image: foreground pixels = 0 (dark/text), background = 255 (white).
/// Uses 8-connectivity (includes diagonal neighbors).
/// Returns array of Components sorted by (y, x) of bounding box top-left.
pub fn extractComponents(allocator: Allocator, pixels: []const u8, width: u32, height: u32) ![]Component {
    if (width == 0 or height == 0) return try allocator.alloc(Component, 0);

    const w: usize = @intCast(width);
    const h: usize = @intCast(height);
    const total = w * h;

    // Labels buffer, 0 = unlabeled / background
    const labels = try allocator.alloc(u32, total);
    defer allocator.free(labels);
    @memset(labels, 0);

    // Worst case: every other pixel is a separate component (~total/2 labels + 1 for 0).
    // We start label IDs at 1, so allocate total+1 for the union-find.
    // In practice the number of labels is much smaller, but we size for the worst case.
    const max_labels = total + 1;
    var uf = try UnionFind.init(allocator, max_labels);
    defer uf.deinit();

    var next_label: u32 = 1;

    // ── Pass 1: Label assignment ─────────────────────────────────────────
    for (0..h) |y| {
        for (0..w) |x| {
            const idx = y * w + x;

            // Skip background pixels (255 = white = background)
            if (pixels[idx] != 0) continue;

            // Collect labels of 8-connected neighbors already visited:
            //   NW  N  NE
            //    W  *
            // (only the 4 neighbors that come before the current pixel in raster order)
            var min_label: u32 = 0; // 0 = none found yet
            var neighbor_labels: [4]u32 = undefined;
            var n_count: usize = 0;

            // West (x-1, y)
            if (x > 0 and labels[idx - 1] != 0) {
                const l = uf.find(labels[idx - 1]);
                neighbor_labels[n_count] = l;
                n_count += 1;
                if (min_label == 0 or l < min_label) min_label = l;
            }

            // North-West (x-1, y-1)
            if (x > 0 and y > 0 and labels[(y - 1) * w + (x - 1)] != 0) {
                const l = uf.find(labels[(y - 1) * w + (x - 1)]);
                neighbor_labels[n_count] = l;
                n_count += 1;
                if (min_label == 0 or l < min_label) min_label = l;
            }

            // North (x, y-1)
            if (y > 0 and labels[(y - 1) * w + x] != 0) {
                const l = uf.find(labels[(y - 1) * w + x]);
                neighbor_labels[n_count] = l;
                n_count += 1;
                if (min_label == 0 or l < min_label) min_label = l;
            }

            // North-East (x+1, y-1)
            if (x + 1 < w and y > 0 and labels[(y - 1) * w + (x + 1)] != 0) {
                const l = uf.find(labels[(y - 1) * w + (x + 1)]);
                neighbor_labels[n_count] = l;
                n_count += 1;
                if (min_label == 0 or l < min_label) min_label = l;
            }

            if (n_count == 0) {
                // No labeled neighbors: assign new label
                labels[idx] = next_label;
                next_label += 1;
            } else {
                // Assign minimum label and union all neighbors
                labels[idx] = min_label;
                for (0..n_count) |i| {
                    uf.unite(min_label, neighbor_labels[i]);
                }
            }
        }
    }

    // If no foreground pixels found at all, return empty
    if (next_label == 1) return try allocator.alloc(Component, 0);

    // ── Pass 2: Resolve labels and compute bounding boxes ────────────────

    // Map: root_label -> index in component list (built on the fly)
    var root_to_idx = std.AutoHashMap(u32, usize).init(allocator);
    defer root_to_idx.deinit();

    var components = std.ArrayList(Component).init(allocator);
    defer components.deinit();

    for (0..h) |y| {
        for (0..w) |x| {
            const idx = y * w + x;
            if (labels[idx] == 0) continue;

            const root = uf.find(labels[idx]);
            labels[idx] = root;

            const px: u32 = @intCast(x);
            const py: u32 = @intCast(y);

            const gop = try root_to_idx.getOrPut(root);
            if (!gop.found_existing) {
                // First pixel of this component
                gop.value_ptr.* = components.items.len;
                try components.append(.{
                    .label = root,
                    .bbox = .{
                        .x = px,
                        .y = py,
                        .width = 1,
                        .height = 1,
                    },
                    .pixel_count = 1,
                });
            } else {
                // Expand bounding box
                const ci = gop.value_ptr.*;
                var comp = &components.items[ci];
                comp.pixel_count += 1;

                const cur_right = comp.bbox.right();
                const cur_bottom = comp.bbox.bottom();

                if (px < comp.bbox.x) comp.bbox.x = px;
                if (py < comp.bbox.y) comp.bbox.y = py;

                const new_right = if (px + 1 > cur_right) px + 1 else cur_right;
                const new_bottom = if (py + 1 > cur_bottom) py + 1 else cur_bottom;

                comp.bbox.width = new_right - comp.bbox.x;
                comp.bbox.height = new_bottom - comp.bbox.y;
            }
        }
    }

    // ── Sort by (y, x) of top-left corner ────────────────────────────────
    const result = try components.toOwnedSlice();

    std.mem.sort(Component, result, {}, struct {
        fn lessThan(_: void, a: Component, b: Component) bool {
            if (a.bbox.y != b.bbox.y) return a.bbox.y < b.bbox.y;
            return a.bbox.x < b.bbox.x;
        }
    }.lessThan);

    return result;
}

// ── Blob Classification ─────────────────────────────────────────────────────

pub const BlobType = enum {
    text,
    noise,
    small,
    large,
};

pub const Blob = struct {
    component: Component,
    blob_type: BlobType,
};

/// Classify components into text, noise, small, and large categories
/// based on size statistics relative to the estimated median height.
/// Returns a new array of Blobs with classification tags.
/// If fewer than 3 components, all are classified as text.
pub fn classifyBlobs(allocator: Allocator, components: []const Component) ![]Blob {
    if (components.len == 0) return try allocator.alloc(Blob, 0);

    const blobs = try allocator.alloc(Blob, components.len);
    errdefer allocator.free(blobs);

    // Fewer than 3 components: not enough data for statistics, classify all as text
    if (components.len < 3) {
        for (components, 0..) |comp, i| {
            blobs[i] = .{ .component = comp, .blob_type = .text };
        }
        return blobs;
    }

    // Collect heights and sort to find the median
    const heights = try allocator.alloc(u32, components.len);
    defer allocator.free(heights);
    for (components, 0..) |comp, i| {
        heights[i] = comp.bbox.height;
    }
    std.mem.sort(u32, heights, {}, std.sort.asc(u32));

    const median_h: f64 = blk: {
        const mid = heights.len / 2;
        if (heights.len % 2 == 1) {
            break :blk @as(f64, @floatFromInt(heights[mid]));
        } else {
            const a: f64 = @floatFromInt(heights[mid - 1]);
            const b: f64 = @floatFromInt(heights[mid]);
            break :blk (a + b) / 2.0;
        }
    };

    // Classification thresholds
    const noise_h = median_h * 0.25;
    const small_h = median_h * 0.5;
    const text_max_h = median_h * 3.0;
    const text_max_w = median_h * 10.0;
    const min_fill_ratio: f64 = 0.1;

    for (components, 0..) |comp, i| {
        const h: f64 = @floatFromInt(comp.bbox.height);
        const w: f64 = @floatFromInt(comp.bbox.width);
        const bbox_area = comp.bbox.area();
        const fill_ratio: f64 = if (bbox_area > 0)
            @as(f64, @floatFromInt(comp.pixel_count)) / @as(f64, @floatFromInt(bbox_area))
        else
            0.0;

        // Area ratio filter: very low fill → noise regardless of size
        if (fill_ratio < min_fill_ratio) {
            blobs[i] = .{ .component = comp, .blob_type = .noise };
            continue;
        }

        if (h < noise_h) {
            blobs[i] = .{ .component = comp, .blob_type = .noise };
        } else if (h < small_h) {
            blobs[i] = .{ .component = comp, .blob_type = .small };
        } else if (h <= text_max_h and w <= text_max_w) {
            blobs[i] = .{ .component = comp, .blob_type = .text };
        } else {
            blobs[i] = .{ .component = comp, .blob_type = .large };
        }
    }

    return blobs;
}

/// Filter blobs, returning only those classified as text.
pub fn filterTextBlobs(allocator: Allocator, blobs: []const Blob) ![]Blob {
    var count: usize = 0;
    for (blobs) |b| {
        if (b.blob_type == .text) count += 1;
    }

    const result = try allocator.alloc(Blob, count);
    var idx: usize = 0;
    for (blobs) |b| {
        if (b.blob_type == .text) {
            result[idx] = b;
            idx += 1;
        }
    }
    return result;
}

// ── Text Line Detection ─────────────────────────────────────────────────────

pub const TextLine = struct {
    blobs: []Blob, // blobs in this line, sorted left-to-right
    y_min: u32, // top of line
    y_max: u32, // bottom of line
    baseline_slope: f32, // slope of fitted baseline
    baseline_intercept: f32, // intercept of fitted baseline
    allocator: Allocator,

    pub fn deinit(self: *TextLine) void {
        self.allocator.free(self.blobs);
        self.* = undefined;
    }
};

/// Fit a baseline (y = slope * x + intercept) through blob bottom edges
/// using linear least squares.
fn fitBaseline(blobs: []const Blob) struct { slope: f32, intercept: f32 } {
    if (blobs.len == 0) return .{ .slope = 0, .intercept = 0 };

    if (blobs.len == 1) {
        const bottom: f32 = @floatFromInt(blobs[0].component.bbox.bottom());
        return .{ .slope = 0, .intercept = bottom };
    }

    const n: f32 = @floatFromInt(blobs.len);
    var sum_x: f64 = 0;
    var sum_y: f64 = 0;
    var sum_xy: f64 = 0;
    var sum_xx: f64 = 0;

    for (blobs) |blob| {
        const bbox = blob.component.bbox;
        // x = horizontal center, y = bottom edge
        const x: f64 = @as(f64, @floatFromInt(bbox.x)) + @as(f64, @floatFromInt(bbox.width)) / 2.0;
        const y: f64 = @floatFromInt(bbox.bottom());

        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    const nf: f64 = @floatCast(n);
    const denom = nf * sum_xx - sum_x * sum_x;

    if (@abs(denom) < 1e-9) {
        // All blobs at same x: slope=0, intercept=mean(y)
        return .{ .slope = 0, .intercept = @floatCast(sum_y / nf) };
    }

    const slope = (nf * sum_xy - sum_x * sum_y) / denom;
    const intercept = (sum_y - slope * sum_x) / nf;

    return .{ .slope = @floatCast(slope), .intercept = @floatCast(intercept) };
}

/// Temporary row accumulator used during line detection.
const RowAccum = struct {
    blob_indices: std.ArrayList(usize),
    y_min: u32,
    y_max: u32,

    fn init(allocator: Allocator) RowAccum {
        return .{
            .blob_indices = std.ArrayList(usize).init(allocator),
            .y_min = std.math.maxInt(u32),
            .y_max = 0,
        };
    }

    fn deinit(self: *RowAccum) void {
        self.blob_indices.deinit();
    }

    /// Compute vertical overlap between this row and a blob's y-range.
    fn verticalOverlap(self: RowAccum, blob_y: u32, blob_bottom: u32) u32 {
        const overlap_top = @max(self.y_min, blob_y);
        const overlap_bot = @min(self.y_max, blob_bottom);
        if (overlap_bot > overlap_top) {
            return overlap_bot - overlap_top;
        }
        return 0;
    }
};

/// Group text blobs into lines by vertical proximity.
/// Blobs should be pre-filtered to text type.
/// Returns lines sorted top-to-bottom.
pub fn detectTextLines(allocator: Allocator, blobs: []const Blob) ![]TextLine {
    if (blobs.len == 0) return try allocator.alloc(TextLine, 0);

    // Step 1: Sort blobs by y-center, then x for ties.
    // We work on indices to avoid mutating the input.
    const indices = try allocator.alloc(usize, blobs.len);
    defer allocator.free(indices);
    for (0..blobs.len) |i| {
        indices[i] = i;
    }

    const SortCtx = struct {
        blob_slice: []const Blob,

        fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            const ba = ctx.blob_slice[a].component.bbox;
            const bb = ctx.blob_slice[b].component.bbox;
            const ya_center = ba.y + ba.height / 2;
            const yb_center = bb.y + bb.height / 2;
            if (ya_center != yb_center) return ya_center < yb_center;
            return ba.x < bb.x;
        }
    };

    std.mem.sort(usize, indices, SortCtx{ .blob_slice = blobs }, SortCtx.lessThan);

    // Step 2: Greedy row grouping by vertical overlap.
    var rows = std.ArrayList(RowAccum).init(allocator);
    defer {
        for (rows.items) |*row| {
            row.deinit();
        }
        rows.deinit();
    }

    for (indices) |blob_idx| {
        const bbox = blobs[blob_idx].component.bbox;
        const blob_y = bbox.y;
        const blob_bottom = bbox.bottom();

        // Find best overlapping row
        var best_row: ?usize = null;
        var best_overlap: u32 = 0;

        for (rows.items, 0..) |row, ri| {
            const overlap = row.verticalOverlap(blob_y, blob_bottom);
            if (overlap > 0 and overlap > best_overlap) {
                best_overlap = overlap;
                best_row = ri;
            }
        }

        if (best_row) |ri| {
            // Add to existing row, update bounds
            try rows.items[ri].blob_indices.append(blob_idx);
            if (blob_y < rows.items[ri].y_min) rows.items[ri].y_min = blob_y;
            if (blob_bottom > rows.items[ri].y_max) rows.items[ri].y_max = blob_bottom;
        } else {
            // Create new row
            var new_row = RowAccum.init(allocator);
            try new_row.blob_indices.append(blob_idx);
            new_row.y_min = blob_y;
            new_row.y_max = blob_bottom;
            try rows.append(new_row);
        }
    }

    // Step 3: Build TextLine output for each row.
    const lines = try allocator.alloc(TextLine, rows.items.len);
    errdefer {
        for (lines) |*line| {
            if (line.blobs.len > 0) line.deinit();
        }
        allocator.free(lines);
    }

    for (rows.items, 0..) |row, li| {
        // Collect blobs for this row
        const row_blobs = try allocator.alloc(Blob, row.blob_indices.items.len);

        for (row.blob_indices.items, 0..) |blob_idx, bi| {
            row_blobs[bi] = blobs[blob_idx];
        }

        // Sort blobs within row by x (left to right)
        std.mem.sort(Blob, row_blobs, {}, struct {
            fn lessThan(_: void, a: Blob, b: Blob) bool {
                return a.component.bbox.x < b.component.bbox.x;
            }
        }.lessThan);

        // Fit baseline
        const baseline = fitBaseline(row_blobs);

        lines[li] = .{
            .blobs = row_blobs,
            .y_min = row.y_min,
            .y_max = row.y_max,
            .baseline_slope = baseline.slope,
            .baseline_intercept = baseline.intercept,
            .allocator = allocator,
        };
    }

    // Sort lines top-to-bottom by y_min
    std.mem.sort(TextLine, lines, {}, struct {
        fn lessThan(_: void, a: TextLine, b: TextLine) bool {
            return a.y_min < b.y_min;
        }
    }.lessThan);

    return lines;
}

// ── Word Segmentation ────────────────────────────────────────────────────────

pub const Word = struct {
    blobs: []const Blob, // slice into the line's blob array (not owned)
    bbox: BoundingBox, // bounding box of all blobs in the word
};

/// Segment a text line into words based on inter-blob gap analysis.
/// The blobs in the text line must be sorted left-to-right.
/// Returns an array of Words (caller owns the array, not the blob data).
pub fn segmentWords(allocator: Allocator, line: *const TextLine) ![]Word {
    const blobs = line.blobs;

    // 0 blobs → 0 words
    if (blobs.len == 0) return try allocator.alloc(Word, 0);

    // 1 blob → 1 word
    if (blobs.len == 1) {
        const words = try allocator.alloc(Word, 1);
        words[0] = .{
            .blobs = blobs[0..1],
            .bbox = blobs[0].component.bbox,
        };
        return words;
    }

    // Step 1: Compute inter-blob gaps
    const n_gaps = blobs.len - 1;
    const gaps = try allocator.alloc(i32, n_gaps);
    defer allocator.free(gaps);

    for (0..n_gaps) |i| {
        const right_edge: i32 = @intCast(blobs[i].component.bbox.right());
        const left_edge: i32 = @intCast(blobs[i + 1].component.bbox.x);
        gaps[i] = left_edge - right_edge;
    }

    // Compute median blob width for fallback threshold
    const widths = try allocator.alloc(u32, blobs.len);
    defer allocator.free(widths);
    for (blobs, 0..) |b, i| {
        widths[i] = b.component.bbox.width;
    }
    std.mem.sort(u32, widths, {}, std.sort.asc(u32));
    const median_blob_width: f64 = blk: {
        const mid = widths.len / 2;
        if (widths.len % 2 == 1) {
            break :blk @as(f64, @floatFromInt(widths[mid]));
        } else {
            const a: f64 = @floatFromInt(widths[mid - 1]);
            const b: f64 = @floatFromInt(widths[mid]);
            break :blk (a + b) / 2.0;
        }
    };

    // Step 2: Determine space threshold
    const threshold: f64 = thr: {
        // Sort a copy of gaps to find the max jump
        const sorted_gaps = try allocator.alloc(i32, n_gaps);
        defer allocator.free(sorted_gaps);
        @memcpy(sorted_gaps, gaps);
        std.mem.sort(i32, sorted_gaps, {}, std.sort.asc(i32));

        // Fallback for 1-2 gaps: use median_blob_width * 0.5
        if (n_gaps <= 2) {
            break :thr median_blob_width * 0.5;
        }

        // Find the largest jump between consecutive sorted gaps
        var max_jump: f64 = 0;
        var max_jump_idx: usize = 0;
        for (0..n_gaps - 1) |i| {
            const jump: f64 = @as(f64, @floatFromInt(sorted_gaps[i + 1])) - @as(f64, @floatFromInt(sorted_gaps[i]));
            if (jump > max_jump) {
                max_jump = jump;
                max_jump_idx = i;
            }
        }

        // If no clear jump, treat as single word
        if (max_jump < median_blob_width * 0.3) {
            break :thr @as(f64, @floatFromInt(sorted_gaps[n_gaps - 1])) + 1.0;
        }

        // Threshold = midpoint of the jump
        const lo: f64 = @floatFromInt(sorted_gaps[max_jump_idx]);
        const hi: f64 = @floatFromInt(sorted_gaps[max_jump_idx + 1]);
        break :thr (lo + hi) / 2.0;
    };

    // Step 3: Segment into words by walking gaps
    var word_list = std.ArrayList(Word).init(allocator);
    defer word_list.deinit();

    var word_start: usize = 0;

    for (0..n_gaps) |i| {
        const gap_f: f64 = @floatFromInt(gaps[i]);
        if (gap_f >= threshold) {
            // End current word at blob i, start new word at blob i+1
            try word_list.append(buildWord(blobs[word_start .. i + 1]));
            word_start = i + 1;
        }
    }

    // Final word
    try word_list.append(buildWord(blobs[word_start..blobs.len]));

    return try word_list.toOwnedSlice();
}

/// Build a Word from a slice of blobs, computing the union bounding box.
fn buildWord(blobs: []const Blob) Word {
    std.debug.assert(blobs.len > 0);

    var x_min: u32 = blobs[0].component.bbox.x;
    var y_min: u32 = blobs[0].component.bbox.y;
    var x_max: u32 = blobs[0].component.bbox.right();
    var y_max: u32 = blobs[0].component.bbox.bottom();

    for (blobs[1..]) |b| {
        const bbox = b.component.bbox;
        if (bbox.x < x_min) x_min = bbox.x;
        if (bbox.y < y_min) y_min = bbox.y;
        if (bbox.right() > x_max) x_max = bbox.right();
        if (bbox.bottom() > y_max) y_max = bbox.bottom();
    }

    return .{
        .blobs = blobs,
        .bbox = .{
            .x = x_min,
            .y = y_min,
            .width = x_max - x_min,
            .height = y_max - y_min,
        },
    };
}

// ── Column / Block Detection ─────────────────────────────────────────────────

pub const TextBlock = struct {
    lines: []TextLine, // owned lines in this block, sorted top-to-bottom
    bbox: BoundingBox, // bounding box of entire block
    allocator: Allocator,

    pub fn deinit(self: *TextBlock) void {
        for (self.lines) |*line| {
            line.deinit();
        }
        self.allocator.free(self.lines);
        self.* = undefined;
    }
};

/// Temporary column accumulator used during column detection.
const ColumnAccum = struct {
    line_indices: std.ArrayList(usize),
    x_min: u32,
    x_max: u32,

    fn init(allocator: Allocator) ColumnAccum {
        return .{
            .line_indices = std.ArrayList(usize).init(allocator),
            .x_min = std.math.maxInt(u32),
            .x_max = 0,
        };
    }

    fn deinit(self: *ColumnAccum) void {
        self.line_indices.deinit();
    }

    fn width(self: ColumnAccum) u32 {
        if (self.x_max <= self.x_min) return 0;
        return self.x_max - self.x_min;
    }

    fn avgX(self: ColumnAccum) u32 {
        return self.x_min + self.width() / 2;
    }
};

/// Compute the x-range of a TextLine from its leftmost to rightmost blob edge.
fn lineXRange(line: *const TextLine) struct { x_min: u32, x_max: u32 } {
    if (line.blobs.len == 0) return .{ .x_min = 0, .x_max = 0 };

    var x_min: u32 = std.math.maxInt(u32);
    var x_max: u32 = 0;

    for (line.blobs) |blob| {
        const bbox = blob.component.bbox;
        if (bbox.x < x_min) x_min = bbox.x;
        const r = bbox.right();
        if (r > x_max) x_max = r;
    }

    return .{ .x_min = x_min, .x_max = x_max };
}

/// Detect text columns/blocks from text lines.
/// Lines are grouped by x-range overlap into blocks.
/// Returns blocks sorted left-to-right, lines within each block sorted top-to-bottom.
/// Takes ownership of the TextLine array and its contents.
/// Caller must free the returned slice and call deinit on each TextBlock.
pub fn detectColumns(allocator: Allocator, lines: []TextLine) ![]TextBlock {
    if (lines.len == 0) {
        allocator.free(lines);
        return try allocator.alloc(TextBlock, 0);
    }

    // Sort lines top-to-bottom by y_min
    std.mem.sort(TextLine, lines, {}, struct {
        fn lessThan(_: void, a: TextLine, b: TextLine) bool {
            return a.y_min < b.y_min;
        }
    }.lessThan);

    // Group lines into columns by x-range overlap
    var columns = std.ArrayList(ColumnAccum).init(allocator);
    defer {
        for (columns.items) |*col| {
            col.deinit();
        }
        columns.deinit();
    }

    for (lines, 0..) |*line, li| {
        const xr = lineXRange(line);
        const line_width = if (xr.x_max > xr.x_min) xr.x_max - xr.x_min else @as(u32, 1);

        var best_col: ?usize = null;
        var best_ratio: f64 = 0;

        for (columns.items, 0..) |col, ci| {
            const col_w = if (col.width() > 0) col.width() else @as(u32, 1);

            // Compute overlap
            const overlap_left = @max(xr.x_min, col.x_min);
            const overlap_right = @min(xr.x_max, col.x_max);

            if (overlap_right > overlap_left) {
                const overlap: f64 = @floatFromInt(overlap_right - overlap_left);
                const narrower: f64 = @floatFromInt(@min(line_width, col_w));
                const ratio = overlap / narrower;

                if (ratio > 0.5 and ratio > best_ratio) {
                    best_ratio = ratio;
                    best_col = ci;
                }
            }
        }

        if (best_col) |ci| {
            try columns.items[ci].line_indices.append(li);
            // Update column x-range to union
            if (xr.x_min < columns.items[ci].x_min) columns.items[ci].x_min = xr.x_min;
            if (xr.x_max > columns.items[ci].x_max) columns.items[ci].x_max = xr.x_max;
        } else {
            var new_col = ColumnAccum.init(allocator);
            try new_col.line_indices.append(li);
            new_col.x_min = xr.x_min;
            new_col.x_max = xr.x_max;
            try columns.append(new_col);
        }
    }

    // Sort columns left-to-right by average x position
    std.mem.sort(ColumnAccum, columns.items, {}, struct {
        fn lessThan(_: void, a: ColumnAccum, b: ColumnAccum) bool {
            return a.avgX() < b.avgX();
        }
    }.lessThan);

    // Build TextBlock output for each column
    const blocks = try allocator.alloc(TextBlock, columns.items.len);
    var blocks_built: usize = 0;
    errdefer {
        for (blocks[0..blocks_built]) |*block| {
            block.deinit();
        }
        allocator.free(blocks);
    }

    for (columns.items) |col| {
        const block_lines = try allocator.alloc(TextLine, col.line_indices.items.len);

        for (col.line_indices.items, 0..) |line_idx, bi| {
            block_lines[bi] = lines[line_idx];
        }

        // Sort lines within block top-to-bottom (should already be, but ensure)
        std.mem.sort(TextLine, block_lines, {}, struct {
            fn lessThan(_: void, a: TextLine, b: TextLine) bool {
                return a.y_min < b.y_min;
            }
        }.lessThan);

        // Compute block bounding box
        var bb_x_min: u32 = std.math.maxInt(u32);
        var bb_y_min: u32 = std.math.maxInt(u32);
        var bb_x_max: u32 = 0;
        var bb_y_max: u32 = 0;

        for (block_lines) |*bl| {
            const xr = lineXRange(bl);
            if (xr.x_min < bb_x_min) bb_x_min = xr.x_min;
            if (xr.x_max > bb_x_max) bb_x_max = xr.x_max;
            if (bl.y_min < bb_y_min) bb_y_min = bl.y_min;
            if (bl.y_max > bb_y_max) bb_y_max = bl.y_max;
        }

        blocks[blocks_built] = .{
            .lines = block_lines,
            .bbox = .{
                .x = bb_x_min,
                .y = bb_y_min,
                .width = if (bb_x_max > bb_x_min) bb_x_max - bb_x_min else 0,
                .height = if (bb_y_max > bb_y_min) bb_y_max - bb_y_min else 0,
            },
            .allocator = allocator,
        };
        blocks_built += 1;
    }

    // Free the original lines array (contents have been moved into blocks)
    allocator.free(lines);

    return blocks;
}

// ── Tests ────────────────────────────────────────────────────────────────────

test "single blob: 3x3 square in 5x5 image" {
    const alloc = std.testing.allocator;

    // 5x5, all white except 3x3 dark square in center (rows 1..3, cols 1..3)
    var pixels: [25]u8 = .{255} ** 25;
    for (1..4) |y| {
        for (1..4) |x| {
            pixels[y * 5 + x] = 0;
        }
    }

    const comps = try extractComponents(alloc, &pixels, 5, 5);
    defer alloc.free(comps);

    try std.testing.expectEqual(@as(usize, 1), comps.len);
    try std.testing.expectEqual(@as(u32, 1), comps[0].bbox.x);
    try std.testing.expectEqual(@as(u32, 1), comps[0].bbox.y);
    try std.testing.expectEqual(@as(u32, 3), comps[0].bbox.width);
    try std.testing.expectEqual(@as(u32, 3), comps[0].bbox.height);
    try std.testing.expectEqual(@as(u32, 9), comps[0].pixel_count);
}

test "two separate blobs" {
    const alloc = std.testing.allocator;

    // 10x5 image, two 2x2 blobs separated by gap
    var pixels: [50]u8 = .{255} ** 50;

    // Blob A at top-left: rows 0..1, cols 0..1
    pixels[0 * 10 + 0] = 0;
    pixels[0 * 10 + 1] = 0;
    pixels[1 * 10 + 0] = 0;
    pixels[1 * 10 + 1] = 0;

    // Blob B at bottom-right: rows 3..4, cols 7..8
    pixels[3 * 10 + 7] = 0;
    pixels[3 * 10 + 8] = 0;
    pixels[4 * 10 + 7] = 0;
    pixels[4 * 10 + 8] = 0;

    const comps = try extractComponents(alloc, &pixels, 10, 5);
    defer alloc.free(comps);

    try std.testing.expectEqual(@as(usize, 2), comps.len);

    // Sorted by (y, x), so blob A comes first
    try std.testing.expectEqual(@as(u32, 0), comps[0].bbox.x);
    try std.testing.expectEqual(@as(u32, 0), comps[0].bbox.y);
    try std.testing.expectEqual(@as(u32, 2), comps[0].bbox.width);
    try std.testing.expectEqual(@as(u32, 2), comps[0].bbox.height);
    try std.testing.expectEqual(@as(u32, 4), comps[0].pixel_count);

    try std.testing.expectEqual(@as(u32, 7), comps[1].bbox.x);
    try std.testing.expectEqual(@as(u32, 3), comps[1].bbox.y);
    try std.testing.expectEqual(@as(u32, 2), comps[1].bbox.width);
    try std.testing.expectEqual(@as(u32, 2), comps[1].bbox.height);
    try std.testing.expectEqual(@as(u32, 4), comps[1].pixel_count);
}

test "L-shaped blob: 8-connectivity makes single component" {
    const alloc = std.testing.allocator;

    // 5x5 image with L-shape:
    // row 0: X X .  .  .
    // row 1: X X .  .  .
    // row 2: X .  .  .  .
    // row 3: X .  .  .  .
    // row 4: X X X .  .
    var pixels: [25]u8 = .{255} ** 25;
    // Top part
    pixels[0 * 5 + 0] = 0;
    pixels[0 * 5 + 1] = 0;
    pixels[1 * 5 + 0] = 0;
    pixels[1 * 5 + 1] = 0;
    // Vertical stem
    pixels[2 * 5 + 0] = 0;
    pixels[3 * 5 + 0] = 0;
    // Bottom bar
    pixels[4 * 5 + 0] = 0;
    pixels[4 * 5 + 1] = 0;
    pixels[4 * 5 + 2] = 0;

    const comps = try extractComponents(alloc, &pixels, 5, 5);
    defer alloc.free(comps);

    try std.testing.expectEqual(@as(usize, 1), comps.len);
    try std.testing.expectEqual(@as(u32, 0), comps[0].bbox.x);
    try std.testing.expectEqual(@as(u32, 0), comps[0].bbox.y);
    try std.testing.expectEqual(@as(u32, 3), comps[0].bbox.width);
    try std.testing.expectEqual(@as(u32, 5), comps[0].bbox.height);
    try std.testing.expectEqual(@as(u32, 9), comps[0].pixel_count);
}

test "empty image: all white yields 0 components" {
    const alloc = std.testing.allocator;
    var pixels: [25]u8 = .{255} ** 25;

    const comps = try extractComponents(alloc, &pixels, 5, 5);
    defer alloc.free(comps);

    try std.testing.expectEqual(@as(usize, 0), comps.len);
}

test "full image: all dark yields 1 component" {
    const alloc = std.testing.allocator;
    var pixels: [25]u8 = .{0} ** 25;

    const comps = try extractComponents(alloc, &pixels, 5, 5);
    defer alloc.free(comps);

    try std.testing.expectEqual(@as(usize, 1), comps.len);
    try std.testing.expectEqual(@as(u32, 0), comps[0].bbox.x);
    try std.testing.expectEqual(@as(u32, 0), comps[0].bbox.y);
    try std.testing.expectEqual(@as(u32, 5), comps[0].bbox.width);
    try std.testing.expectEqual(@as(u32, 5), comps[0].bbox.height);
    try std.testing.expectEqual(@as(u32, 25), comps[0].pixel_count);
}

test "diagonal connectivity: two diagonal pixels are 1 component with 8-conn" {
    const alloc = std.testing.allocator;

    // 3x3 image with two pixels connected only diagonally
    var pixels: [9]u8 = .{255} ** 9;
    pixels[0 * 3 + 0] = 0; // (0, 0)
    pixels[1 * 3 + 1] = 0; // (1, 1)

    const comps = try extractComponents(alloc, &pixels, 3, 3);
    defer alloc.free(comps);

    try std.testing.expectEqual(@as(usize, 1), comps.len);
    try std.testing.expectEqual(@as(u32, 0), comps[0].bbox.x);
    try std.testing.expectEqual(@as(u32, 0), comps[0].bbox.y);
    try std.testing.expectEqual(@as(u32, 2), comps[0].bbox.width);
    try std.testing.expectEqual(@as(u32, 2), comps[0].bbox.height);
    try std.testing.expectEqual(@as(u32, 2), comps[0].pixel_count);
}

test "zero-size image yields 0 components" {
    const alloc = std.testing.allocator;
    const comps = try extractComponents(alloc, &.{}, 0, 0);
    defer alloc.free(comps);
    try std.testing.expectEqual(@as(usize, 0), comps.len);
}

test "BoundingBox: methods" {
    const bb = BoundingBox{ .x = 10, .y = 20, .width = 30, .height = 40 };
    try std.testing.expectEqual(@as(u32, 40), bb.right());
    try std.testing.expectEqual(@as(u32, 60), bb.bottom());
    try std.testing.expectEqual(@as(u32, 1200), bb.area());

    // contains
    try std.testing.expect(bb.contains(10, 20));
    try std.testing.expect(bb.contains(25, 40));
    try std.testing.expect(!bb.contains(9, 20));
    try std.testing.expect(!bb.contains(40, 20)); // right edge is exclusive
    try std.testing.expect(!bb.contains(10, 60)); // bottom edge is exclusive

    // overlapsVertically
    const bb2 = BoundingBox{ .x = 50, .y = 30, .width = 10, .height = 10 };
    try std.testing.expect(bb.overlapsVertically(bb2));
    try std.testing.expect(bb2.overlapsVertically(bb));

    const bb3 = BoundingBox{ .x = 50, .y = 60, .width = 10, .height = 10 };
    try std.testing.expect(!bb.overlapsVertically(bb3));
}

test "UnionFind: basic operations" {
    var uf = try UnionFind.init(std.testing.allocator, 10);
    defer uf.deinit();

    // Initially every element is its own root
    try std.testing.expectEqual(@as(u32, 3), uf.find(3));
    try std.testing.expectEqual(@as(u32, 7), uf.find(7));

    // Unite 3 and 7 (3 < 7, so 3 becomes root)
    uf.unite(3, 7);
    try std.testing.expectEqual(uf.find(3), uf.find(7));
    try std.testing.expectEqual(@as(u32, 3), uf.find(7));

    // Unite 5 and 7 (transitively merges with 3's set)
    uf.unite(5, 7);
    try std.testing.expectEqual(uf.find(3), uf.find(5));
    try std.testing.expectEqual(@as(u32, 3), uf.find(5));

    // 9 is still its own root
    try std.testing.expectEqual(@as(u32, 9), uf.find(9));
}

// ── Blob Classification Tests ───────────────────────────────────────────────

test "classifyBlobs: mixed sizes" {
    const alloc = std.testing.allocator;

    // Create components with varying heights:
    // - tiny noise (height=2), normal text (height=20, 22, 18), huge image (height=200)
    // Sorted heights: 2, 18, 20, 22, 200 → median = 20
    // noise_h = 5, small_h = 10, text_max_h = 60, text_max_w = 200
    const components = [_]Component{
        // tiny noise: height 2 < 5 → noise
        .{ .label = 1, .bbox = .{ .x = 0, .y = 0, .width = 3, .height = 2 }, .pixel_count = 4 },
        // normal text: height 20
        .{ .label = 2, .bbox = .{ .x = 10, .y = 0, .width = 15, .height = 20 }, .pixel_count = 200 },
        // normal text: height 22
        .{ .label = 3, .bbox = .{ .x = 30, .y = 0, .width = 14, .height = 22 }, .pixel_count = 200 },
        // normal text: height 18
        .{ .label = 4, .bbox = .{ .x = 50, .y = 0, .width = 16, .height = 18 }, .pixel_count = 200 },
        // huge image: height 200 > 60 → large
        .{ .label = 5, .bbox = .{ .x = 0, .y = 30, .width = 100, .height = 200 }, .pixel_count = 15000 },
    };

    const blobs = try classifyBlobs(alloc, &components);
    defer alloc.free(blobs);

    try std.testing.expectEqual(@as(usize, 5), blobs.len);
    try std.testing.expectEqual(BlobType.noise, blobs[0].blob_type); // height 2
    try std.testing.expectEqual(BlobType.text, blobs[1].blob_type); // height 20
    try std.testing.expectEqual(BlobType.text, blobs[2].blob_type); // height 22
    try std.testing.expectEqual(BlobType.text, blobs[3].blob_type); // height 18
    try std.testing.expectEqual(BlobType.large, blobs[4].blob_type); // height 200
}

test "classifyBlobs: all same size → all text" {
    const alloc = std.testing.allocator;

    const components = [_]Component{
        .{ .label = 1, .bbox = .{ .x = 0, .y = 0, .width = 10, .height = 20 }, .pixel_count = 150 },
        .{ .label = 2, .bbox = .{ .x = 15, .y = 0, .width = 10, .height = 20 }, .pixel_count = 150 },
        .{ .label = 3, .bbox = .{ .x = 30, .y = 0, .width = 10, .height = 20 }, .pixel_count = 150 },
        .{ .label = 4, .bbox = .{ .x = 45, .y = 0, .width = 10, .height = 20 }, .pixel_count = 150 },
    };

    const blobs = try classifyBlobs(alloc, &components);
    defer alloc.free(blobs);

    for (blobs) |b| {
        try std.testing.expectEqual(BlobType.text, b.blob_type);
    }
}

test "classifyBlobs: sparse component (low fill ratio) → noise" {
    const alloc = std.testing.allocator;

    // 5 components so we have enough for statistics. Median height = 20.
    // The sparse one has a normal-sized bbox but very few pixels.
    const components = [_]Component{
        .{ .label = 1, .bbox = .{ .x = 0, .y = 0, .width = 10, .height = 20 }, .pixel_count = 150 },
        .{ .label = 2, .bbox = .{ .x = 15, .y = 0, .width = 10, .height = 20 }, .pixel_count = 150 },
        .{ .label = 3, .bbox = .{ .x = 30, .y = 0, .width = 10, .height = 20 }, .pixel_count = 150 },
        // Sparse: 100x20 bbox = 2000 area, but only 5 pixels → fill = 0.0025 < 0.1 → noise
        .{ .label = 4, .bbox = .{ .x = 45, .y = 0, .width = 100, .height = 20 }, .pixel_count = 5 },
        .{ .label = 5, .bbox = .{ .x = 150, .y = 0, .width = 10, .height = 20 }, .pixel_count = 150 },
    };

    const blobs = try classifyBlobs(alloc, &components);
    defer alloc.free(blobs);

    try std.testing.expectEqual(BlobType.text, blobs[0].blob_type);
    try std.testing.expectEqual(BlobType.text, blobs[1].blob_type);
    try std.testing.expectEqual(BlobType.text, blobs[2].blob_type);
    try std.testing.expectEqual(BlobType.noise, blobs[3].blob_type); // sparse
    try std.testing.expectEqual(BlobType.text, blobs[4].blob_type);
}

test "classifyBlobs: empty input → 0 blobs" {
    const alloc = std.testing.allocator;

    const blobs = try classifyBlobs(alloc, &[_]Component{});
    defer alloc.free(blobs);

    try std.testing.expectEqual(@as(usize, 0), blobs.len);
}

test "classifyBlobs: fewer than 3 components → all text" {
    const alloc = std.testing.allocator;

    const components = [_]Component{
        .{ .label = 1, .bbox = .{ .x = 0, .y = 0, .width = 5, .height = 3 }, .pixel_count = 10 },
        .{ .label = 2, .bbox = .{ .x = 10, .y = 0, .width = 200, .height = 300 }, .pixel_count = 50000 },
    };

    const blobs = try classifyBlobs(alloc, &components);
    defer alloc.free(blobs);

    try std.testing.expectEqual(@as(usize, 2), blobs.len);
    try std.testing.expectEqual(BlobType.text, blobs[0].blob_type);
    try std.testing.expectEqual(BlobType.text, blobs[1].blob_type);
}

test "filterTextBlobs: returns only text blobs" {
    const alloc = std.testing.allocator;

    const blobs = [_]Blob{
        .{ .component = .{ .label = 1, .bbox = .{ .x = 0, .y = 0, .width = 3, .height = 2 }, .pixel_count = 4 }, .blob_type = .noise },
        .{ .component = .{ .label = 2, .bbox = .{ .x = 10, .y = 0, .width = 15, .height = 20 }, .pixel_count = 200 }, .blob_type = .text },
        .{ .component = .{ .label = 3, .bbox = .{ .x = 30, .y = 0, .width = 14, .height = 22 }, .pixel_count = 200 }, .blob_type = .text },
        .{ .component = .{ .label = 4, .bbox = .{ .x = 0, .y = 30, .width = 100, .height = 200 }, .pixel_count = 15000 }, .blob_type = .large },
        .{ .component = .{ .label = 5, .bbox = .{ .x = 50, .y = 0, .width = 8, .height = 6 }, .pixel_count = 30 }, .blob_type = .small },
    };

    const text_blobs = try filterTextBlobs(alloc, &blobs);
    defer alloc.free(text_blobs);

    try std.testing.expectEqual(@as(usize, 2), text_blobs.len);
    try std.testing.expectEqual(@as(u32, 2), text_blobs[0].component.label);
    try std.testing.expectEqual(@as(u32, 3), text_blobs[1].component.label);
}

// ── Text Line Detection Tests ───────────────────────────────────────────────

fn makeTextBlob(label: u32, x: u32, y: u32, width: u32, height: u32) Blob {
    return .{
        .component = .{
            .label = label,
            .bbox = .{ .x = x, .y = y, .width = width, .height = height },
            .pixel_count = width * height,
        },
        .blob_type = .text,
    };
}

test "detectTextLines: single line of 5 blobs" {
    const alloc = std.testing.allocator;

    // 5 blobs at roughly the same y, spread horizontally
    const blobs = [_]Blob{
        makeTextBlob(1, 10, 100, 12, 20),
        makeTextBlob(2, 30, 102, 12, 20),
        makeTextBlob(3, 50, 98, 12, 20),
        makeTextBlob(4, 70, 101, 12, 20),
        makeTextBlob(5, 90, 99, 12, 20),
    };

    const lines = try detectTextLines(alloc, &blobs);
    defer {
        for (lines) |*line| {
            var l = line;
            l.deinit();
        }
        alloc.free(lines);
    }

    try std.testing.expectEqual(@as(usize, 1), lines.len);
    try std.testing.expectEqual(@as(usize, 5), lines[0].blobs.len);

    // Blobs should be sorted by x within the line
    for (1..lines[0].blobs.len) |i| {
        try std.testing.expect(lines[0].blobs[i].component.bbox.x > lines[0].blobs[i - 1].component.bbox.x);
    }

    // Slope should be near zero for a horizontal line
    try std.testing.expect(@abs(lines[0].baseline_slope) < 0.1);
}

test "detectTextLines: two separate lines" {
    const alloc = std.testing.allocator;

    // Line 1: y around 50
    // Line 2: y around 200 (well separated)
    const blobs = [_]Blob{
        makeTextBlob(1, 10, 50, 12, 20),
        makeTextBlob(2, 30, 52, 12, 20),
        makeTextBlob(3, 50, 48, 12, 20),
        makeTextBlob(4, 10, 200, 12, 20),
        makeTextBlob(5, 30, 202, 12, 20),
        makeTextBlob(6, 50, 198, 12, 20),
    };

    const lines = try detectTextLines(alloc, &blobs);
    defer {
        for (lines) |*line| {
            var l = line;
            l.deinit();
        }
        alloc.free(lines);
    }

    try std.testing.expectEqual(@as(usize, 2), lines.len);

    // Lines should be sorted top-to-bottom
    try std.testing.expect(lines[0].y_min < lines[1].y_min);

    // First line should have blobs around y=50
    try std.testing.expectEqual(@as(usize, 3), lines[0].blobs.len);
    try std.testing.expect(lines[0].y_min < 55);

    // Second line should have blobs around y=200
    try std.testing.expectEqual(@as(usize, 3), lines[1].blobs.len);
    try std.testing.expect(lines[1].y_min > 150);
}

test "detectTextLines: slightly sloped line" {
    const alloc = std.testing.allocator;

    // Blobs with a gradual y increase (slope ~ 0.5 px per px)
    // Each blob is 20px apart in x, and y increases by 10 each time
    const blobs = [_]Blob{
        makeTextBlob(1, 10, 100, 12, 20),
        makeTextBlob(2, 30, 110, 12, 20),
        makeTextBlob(3, 50, 120, 12, 20),
        makeTextBlob(4, 70, 130, 12, 20),
    };

    const lines = try detectTextLines(alloc, &blobs);
    defer {
        for (lines) |*line| {
            var l = line;
            l.deinit();
        }
        alloc.free(lines);
    }

    try std.testing.expectEqual(@as(usize, 1), lines.len);
    try std.testing.expectEqual(@as(usize, 4), lines[0].blobs.len);

    // Slope should be positive (y increases left to right)
    try std.testing.expect(lines[0].baseline_slope > 0.0);
    // Slope should be around 0.5 (10px rise over 20px run)
    try std.testing.expect(lines[0].baseline_slope > 0.3);
    try std.testing.expect(lines[0].baseline_slope < 0.7);
}

test "detectTextLines: single blob" {
    const alloc = std.testing.allocator;

    const blobs = [_]Blob{
        makeTextBlob(1, 50, 100, 12, 20),
    };

    const lines = try detectTextLines(alloc, &blobs);
    defer {
        for (lines) |*line| {
            var l = line;
            l.deinit();
        }
        alloc.free(lines);
    }

    try std.testing.expectEqual(@as(usize, 1), lines.len);
    try std.testing.expectEqual(@as(usize, 1), lines[0].blobs.len);
    try std.testing.expectEqual(@as(f32, 0), lines[0].baseline_slope);
    // Intercept should be the blob bottom (100 + 20 = 120)
    try std.testing.expectEqual(@as(f32, 120.0), lines[0].baseline_intercept);
}

test "detectTextLines: empty input" {
    const alloc = std.testing.allocator;

    const lines = try detectTextLines(alloc, &[_]Blob{});
    defer alloc.free(lines);

    try std.testing.expectEqual(@as(usize, 0), lines.len);
}

// ── Word Segmentation Tests ─────────────────────────────────────────────────

fn makeWordTestLine(allocator: Allocator, blobs: []const Blob) TextLine {
    return .{
        .blobs = @constCast(blobs),
        .y_min = 0,
        .y_max = 20,
        .baseline_slope = 0,
        .baseline_intercept = 20,
        .allocator = allocator,
    };
}

test "segmentWords: single word (3 blobs, small uniform gaps)" {
    const alloc = std.testing.allocator;

    // 3 blobs with small gaps of 2px between them
    const blobs = [_]Blob{
        makeTextBlob(1, 10, 0, 12, 20), // right = 22
        makeTextBlob(2, 24, 0, 12, 20), // gap = 2, right = 36
        makeTextBlob(3, 38, 0, 12, 20), // gap = 2, right = 50
    };

    var line = makeWordTestLine(alloc, &blobs);
    const words = try segmentWords(alloc, &line);
    defer alloc.free(words);

    try std.testing.expectEqual(@as(usize, 1), words.len);
    try std.testing.expectEqual(@as(usize, 3), words[0].blobs.len);
    try std.testing.expectEqual(@as(u32, 10), words[0].bbox.x);
    try std.testing.expectEqual(@as(u32, 40), words[0].bbox.width); // 50 - 10
}

test "segmentWords: two words (5 blobs, large gap in middle)" {
    const alloc = std.testing.allocator;

    // Word 1: blobs at 10, 24, 38 (gaps of 2)
    // Word 2: blobs at 80, 94 (gap of 2)
    // Gap between words: 80 - 50 = 30 (large)
    const blobs = [_]Blob{
        makeTextBlob(1, 10, 0, 12, 20), // right = 22
        makeTextBlob(2, 24, 0, 12, 20), // gap = 2, right = 36
        makeTextBlob(3, 38, 0, 12, 20), // gap = 2, right = 50
        makeTextBlob(4, 80, 0, 12, 20), // gap = 30, right = 92
        makeTextBlob(5, 94, 0, 12, 20), // gap = 2, right = 106
    };

    var line = makeWordTestLine(alloc, &blobs);
    const words = try segmentWords(alloc, &line);
    defer alloc.free(words);

    try std.testing.expectEqual(@as(usize, 2), words.len);
    try std.testing.expectEqual(@as(usize, 3), words[0].blobs.len);
    try std.testing.expectEqual(@as(usize, 2), words[1].blobs.len);

    // First word bbox
    try std.testing.expectEqual(@as(u32, 10), words[0].bbox.x);
    try std.testing.expectEqual(@as(u32, 40), words[0].bbox.width);

    // Second word bbox
    try std.testing.expectEqual(@as(u32, 80), words[1].bbox.x);
    try std.testing.expectEqual(@as(u32, 26), words[1].bbox.width); // 106 - 80
}

test "segmentWords: uniform large gaps → single word (no bimodal split)" {
    const alloc = std.testing.allocator;

    // 4 blobs with identical large gaps (100px each).
    // sorted gaps = [100, 100, 100], max_jump = 0 < median_width*0.3
    // No bimodal separation → algorithm correctly treats as single word.
    const blobs = [_]Blob{
        makeTextBlob(1, 0, 0, 10, 20), // right = 10
        makeTextBlob(2, 110, 0, 10, 20), // gap = 100
        makeTextBlob(3, 220, 0, 10, 20), // gap = 100
        makeTextBlob(4, 330, 0, 10, 20), // gap = 100
    };

    var line = makeWordTestLine(alloc, &blobs);
    const words = try segmentWords(alloc, &line);
    defer alloc.free(words);

    try std.testing.expectEqual(@as(usize, 1), words.len);
    try std.testing.expectEqual(@as(usize, 4), words[0].blobs.len);
}

test "segmentWords: fallback threshold splits two groups" {
    const alloc = std.testing.allocator;

    // 3 blobs, 2 gaps → fallback path (n_gaps <= 2).
    // threshold = median_blob_width * 0.5 = 5 * 0.5 = 2.5
    // gap0 = 2 < 2.5 → same word; gap1 = 50 >= 2.5 → new word.
    const blobs = [_]Blob{
        makeTextBlob(1, 0, 0, 5, 20), // right = 5
        makeTextBlob(2, 7, 0, 5, 20), // gap = 2, right = 12
        makeTextBlob(3, 62, 0, 5, 20), // gap = 50, right = 67
    };

    var line = makeWordTestLine(alloc, &blobs);
    const words = try segmentWords(alloc, &line);
    defer alloc.free(words);

    try std.testing.expectEqual(@as(usize, 2), words.len);
    try std.testing.expectEqual(@as(usize, 2), words[0].blobs.len);
    try std.testing.expectEqual(@as(usize, 1), words[1].blobs.len);
}

test "segmentWords: single blob → single word" {
    const alloc = std.testing.allocator;

    const blobs = [_]Blob{
        makeTextBlob(1, 42, 5, 15, 20),
    };

    var line = makeWordTestLine(alloc, &blobs);
    const words = try segmentWords(alloc, &line);
    defer alloc.free(words);

    try std.testing.expectEqual(@as(usize, 1), words.len);
    try std.testing.expectEqual(@as(usize, 1), words[0].blobs.len);
    try std.testing.expectEqual(@as(u32, 42), words[0].bbox.x);
    try std.testing.expectEqual(@as(u32, 5), words[0].bbox.y);
    try std.testing.expectEqual(@as(u32, 15), words[0].bbox.width);
    try std.testing.expectEqual(@as(u32, 20), words[0].bbox.height);
}

test "segmentWords: empty line → 0 words" {
    const alloc = std.testing.allocator;

    var line = makeWordTestLine(alloc, &[_]Blob{});
    const words = try segmentWords(alloc, &line);
    defer alloc.free(words);

    try std.testing.expectEqual(@as(usize, 0), words.len);
}

test "segmentWords: three words via clear gap jumps" {
    const alloc = std.testing.allocator;

    // Word 1: blobs at 0..12, 14..26 (gap=2)
    // Word 2: blobs at 76..88, 90..102 (gap=2)
    // Word 3: blobs at 152..164, 166..178 (gap=2)
    // Inter-word gaps: 76-26=50 and 152-102=50
    // Gaps array: [2, 50, 2, 50, 2]
    // Sorted: [2, 2, 2, 50, 50]
    // Max jump at index 2→3: 50-2=48, median_blob_width=12, 0.3*12=3.6, 48>3.6 → clear
    // Threshold = (2+50)/2 = 26
    // Walk: gap0=2<26 no break, gap1=50>=26 break, gap2=2<26 no break, gap3=50>=26 break, gap4=2<26 no break
    // Words: [blob0,blob1], [blob2,blob3], [blob4,blob5]

    const blobs = [_]Blob{
        makeTextBlob(1, 0, 0, 12, 20), // right=12
        makeTextBlob(2, 14, 0, 12, 20), // gap=2, right=26
        makeTextBlob(3, 76, 0, 12, 20), // gap=50, right=88
        makeTextBlob(4, 90, 0, 12, 20), // gap=2, right=102
        makeTextBlob(5, 152, 0, 12, 20), // gap=50, right=164
        makeTextBlob(6, 166, 0, 12, 20), // gap=2, right=178
    };

    var line = makeWordTestLine(alloc, &blobs);
    const words = try segmentWords(alloc, &line);
    defer alloc.free(words);

    try std.testing.expectEqual(@as(usize, 3), words.len);
    try std.testing.expectEqual(@as(usize, 2), words[0].blobs.len);
    try std.testing.expectEqual(@as(usize, 2), words[1].blobs.len);
    try std.testing.expectEqual(@as(usize, 2), words[2].blobs.len);

    // Check word bboxes
    try std.testing.expectEqual(@as(u32, 0), words[0].bbox.x);
    try std.testing.expectEqual(@as(u32, 76), words[1].bbox.x);
    try std.testing.expectEqual(@as(u32, 152), words[2].bbox.x);
}

// ── Column Detection Tests ──────────────────────────────────────────────────

/// Helper: create a TextLine with allocated blobs for column detection tests.
/// The blobs are allocated with the given allocator so TextBlock.deinit() can free them.
fn makeTestTextLine(allocator: Allocator, blobs_data: []const Blob, y_min: u32, y_max: u32) !TextLine {
    const blobs = try allocator.alloc(Blob, blobs_data.len);
    @memcpy(blobs, blobs_data);
    return .{
        .blobs = blobs,
        .y_min = y_min,
        .y_max = y_max,
        .baseline_slope = 0,
        .baseline_intercept = @floatFromInt(y_max),
        .allocator = allocator,
    };
}

test "detectColumns: empty input → 0 blocks" {
    const alloc = std.testing.allocator;

    const lines = try alloc.alloc(TextLine, 0);
    const blocks = try detectColumns(alloc, lines);
    defer alloc.free(blocks);

    try std.testing.expectEqual(@as(usize, 0), blocks.len);
}

test "detectColumns: single line → 1 block" {
    const alloc = std.testing.allocator;

    const blob_data = [_]Blob{
        makeTextBlob(1, 10, 100, 50, 20),
        makeTextBlob(2, 70, 100, 50, 20),
    };

    const lines = try alloc.alloc(TextLine, 1);
    lines[0] = try makeTestTextLine(alloc, &blob_data, 100, 120);

    const blocks = try detectColumns(alloc, lines);
    defer {
        for (blocks) |*b| {
            var block = b;
            block.deinit();
        }
        alloc.free(blocks);
    }

    try std.testing.expectEqual(@as(usize, 1), blocks.len);
    try std.testing.expectEqual(@as(usize, 1), blocks[0].lines.len);
}

test "detectColumns: single column (3 lines with similar x-range)" {
    const alloc = std.testing.allocator;

    // 3 lines all spanning roughly x=10..120
    const blob_data_1 = [_]Blob{
        makeTextBlob(1, 10, 50, 50, 20),
        makeTextBlob(2, 70, 50, 50, 20), // right = 120
    };
    const blob_data_2 = [_]Blob{
        makeTextBlob(3, 15, 80, 45, 20),
        makeTextBlob(4, 65, 80, 50, 20), // right = 115
    };
    const blob_data_3 = [_]Blob{
        makeTextBlob(5, 12, 110, 48, 20),
        makeTextBlob(6, 68, 110, 52, 20), // right = 120
    };

    const lines = try alloc.alloc(TextLine, 3);
    lines[0] = try makeTestTextLine(alloc, &blob_data_1, 50, 70);
    lines[1] = try makeTestTextLine(alloc, &blob_data_2, 80, 100);
    lines[2] = try makeTestTextLine(alloc, &blob_data_3, 110, 130);

    const blocks = try detectColumns(alloc, lines);
    defer {
        for (blocks) |*b| {
            var block = b;
            block.deinit();
        }
        alloc.free(blocks);
    }

    try std.testing.expectEqual(@as(usize, 1), blocks.len);
    try std.testing.expectEqual(@as(usize, 3), blocks[0].lines.len);

    // Lines should be sorted top-to-bottom
    try std.testing.expect(blocks[0].lines[0].y_min <= blocks[0].lines[1].y_min);
    try std.testing.expect(blocks[0].lines[1].y_min <= blocks[0].lines[2].y_min);
}

test "detectColumns: two columns (left and right halves)" {
    const alloc = std.testing.allocator;

    // Left column: x range ~10..100
    const left_blob_1 = [_]Blob{
        makeTextBlob(1, 10, 50, 40, 20),
        makeTextBlob(2, 55, 50, 45, 20), // right = 100
    };
    const left_blob_2 = [_]Blob{
        makeTextBlob(3, 12, 80, 38, 20),
        makeTextBlob(4, 58, 80, 42, 20), // right = 100
    };

    // Right column: x range ~200..300
    const right_blob_1 = [_]Blob{
        makeTextBlob(5, 200, 50, 45, 20),
        makeTextBlob(6, 255, 50, 45, 20), // right = 300
    };
    const right_blob_2 = [_]Blob{
        makeTextBlob(7, 205, 80, 40, 20),
        makeTextBlob(8, 260, 80, 40, 20), // right = 300
    };

    const lines = try alloc.alloc(TextLine, 4);
    lines[0] = try makeTestTextLine(alloc, &left_blob_1, 50, 70);
    lines[1] = try makeTestTextLine(alloc, &right_blob_1, 50, 70);
    lines[2] = try makeTestTextLine(alloc, &left_blob_2, 80, 100);
    lines[3] = try makeTestTextLine(alloc, &right_blob_2, 80, 100);

    const blocks = try detectColumns(alloc, lines);
    defer {
        for (blocks) |*b| {
            var block = b;
            block.deinit();
        }
        alloc.free(blocks);
    }

    try std.testing.expectEqual(@as(usize, 2), blocks.len);

    // Blocks should be sorted left-to-right
    try std.testing.expect(blocks[0].bbox.x < blocks[1].bbox.x);

    // Left block
    try std.testing.expectEqual(@as(usize, 2), blocks[0].lines.len);
    try std.testing.expect(blocks[0].bbox.x <= 12);
    try std.testing.expect(blocks[0].bbox.right() >= 100);

    // Right block
    try std.testing.expectEqual(@as(usize, 2), blocks[1].lines.len);
    try std.testing.expect(blocks[1].bbox.x >= 200);
    try std.testing.expect(blocks[1].bbox.right() >= 300);
}

test "detectColumns: mixed width lines that still overlap → 1 block" {
    const alloc = std.testing.allocator;

    // Line 1: x=10..200 (wide)
    const blob_data_1 = [_]Blob{
        makeTextBlob(1, 10, 50, 90, 20),
        makeTextBlob(2, 110, 50, 90, 20), // right = 200
    };
    // Line 2: x=50..150 (narrower, but >50% overlap with line 1)
    const blob_data_2 = [_]Blob{
        makeTextBlob(3, 50, 80, 50, 20),
        makeTextBlob(4, 105, 80, 45, 20), // right = 150
    };
    // Line 3: x=30..180 (medium width)
    const blob_data_3 = [_]Blob{
        makeTextBlob(5, 30, 110, 70, 20),
        makeTextBlob(6, 110, 110, 70, 20), // right = 180
    };

    const lines = try alloc.alloc(TextLine, 3);
    lines[0] = try makeTestTextLine(alloc, &blob_data_1, 50, 70);
    lines[1] = try makeTestTextLine(alloc, &blob_data_2, 80, 100);
    lines[2] = try makeTestTextLine(alloc, &blob_data_3, 110, 130);

    const blocks = try detectColumns(alloc, lines);
    defer {
        for (blocks) |*b| {
            var block = b;
            block.deinit();
        }
        alloc.free(blocks);
    }

    try std.testing.expectEqual(@as(usize, 1), blocks.len);
    try std.testing.expectEqual(@as(usize, 3), blocks[0].lines.len);
}
