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
