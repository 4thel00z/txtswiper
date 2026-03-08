const std = @import("std");
const math = std.math;
const tessdata = @import("tessdata.zig");
const network_mod = @import("network.zig");

const Allocator = std.mem.Allocator;
const Unicharset = tessdata.Unicharset;
const UnicharCompress = tessdata.UnicharCompress;
const NetworkIO = network_mod.NetworkIO;
const LSTMRecognizer = tessdata.LSTMRecognizer;

// ── Constants ────────────────────────────────────────────────────────────────

/// Sentinel value meaning "no valid unichar mapping for this code".
pub const INVALID_UNICHAR_ID: u32 = std.math.maxInt(u32);

/// Minimum probability value to avoid log(0).
const MIN_PROB: f32 = 1e-30;

// ── CtcResult ────────────────────────────────────────────────────────────────

/// Holds the output of CTC decoding: a sequence of unichar IDs with per-character
/// confidence scores and timestep positions.
pub const CtcResult = struct {
    /// Decoded character IDs (unicharset indices).
    unichar_ids: []u32,
    /// Confidence per character (natural log probability).
    confidences: []f32,
    /// Timestep position of each character (for bounding box reconstruction).
    timesteps: []u32,
    allocator: Allocator,

    /// Release all owned memory.
    pub fn deinit(self: *CtcResult) void {
        self.allocator.free(self.unichar_ids);
        self.allocator.free(self.confidences);
        self.allocator.free(self.timesteps);
    }

    /// Number of decoded characters.
    pub fn len(self: CtcResult) usize {
        return self.unichar_ids.len;
    }

    /// Convert decoded unichar IDs to a UTF-8 string using the unicharset.
    /// Caller owns the returned slice.
    pub fn toText(self: CtcResult, unicharset: *const Unicharset, allocator: Allocator) ![]u8 {
        // First pass: compute total length.
        var total_len: usize = 0;
        for (self.unichar_ids) |uid| {
            const s = unicharset.unicharToStr(uid);
            total_len += s.len;
        }

        // Allocate and fill.
        const buf = try allocator.alloc(u8, total_len);
        var pos: usize = 0;
        for (self.unichar_ids) |uid| {
            const s = unicharset.unicharToStr(uid);
            @memcpy(buf[pos .. pos + s.len], s);
            pos += s.len;
        }

        return buf;
    }
};

// ── Recoder Reverse Lookup ───────────────────────────────────────────────────

/// Build a reverse lookup table: output code -> unichar_id.
///
/// For single-code characters (the common case in English), this is a direct
/// 1:1 mapping. Multi-code characters are skipped (they would need a more
/// complex state machine to decode, which is a future enhancement).
///
/// Returns a slice indexed by output code. The value at each index is the
/// unichar_id that maps to that code, or INVALID_UNICHAR_ID if no single
/// character maps to it.
pub fn buildCodeToUnichar(
    allocator: Allocator,
    recoder: *const UnicharCompress,
) ![]u32 {
    // Find the maximum code value to size the table.
    var max_code: u32 = 0;
    for (recoder.entries) |entry| {
        for (entry.codes) |code| {
            if (code >= 0) {
                const c: u32 = @intCast(code);
                if (c > max_code) max_code = c;
            }
        }
    }

    const table_size = max_code + 1;
    const table = try allocator.alloc(u32, table_size);
    @memset(table, INVALID_UNICHAR_ID);

    // Populate: for each unichar_id with a single code, map code -> unichar_id.
    for (recoder.entries, 0..) |entry, uid| {
        if (entry.codes.len == 1 and entry.codes[0] >= 0) {
            const code: u32 = @intCast(entry.codes[0]);
            if (code < table_size) {
                table[code] = @intCast(uid);
            }
        }
    }

    return table;
}

// ── Greedy CTC Decode ────────────────────────────────────────────────────────

/// Greedy CTC decode: at each timestep pick the code with highest probability,
/// fold consecutive duplicates, and remove null (blank) characters.
///
/// Parameters:
///   - allocator: Memory allocator for the result arrays.
///   - softmax_output: Flat row-major array of shape [num_timesteps * num_classes].
///   - num_timesteps: Number of timesteps in the softmax output.
///   - num_classes: Number of output classes (codes) per timestep.
///   - null_char: Index of the CTC blank character.
///   - recoder: Optional recoder for mapping output codes to unichar IDs.
///              If null, codes are used directly as unichar IDs.
///   - code_to_unichar: Optional prebuilt reverse lookup table. If provided,
///                      used instead of recoder for faster single-code lookup.
///
/// Returns a CtcResult with the decoded sequence.
pub fn ctcGreedyDecode(
    allocator: Allocator,
    softmax_output: []const f32,
    num_timesteps: usize,
    num_classes: usize,
    null_char: u32,
    recoder: ?*const UnicharCompress,
    code_to_unichar: ?[]const u32,
) !CtcResult {
    if (num_timesteps == 0 or num_classes == 0) {
        return CtcResult{
            .unichar_ids = try allocator.alloc(u32, 0),
            .confidences = try allocator.alloc(f32, 0),
            .timesteps = try allocator.alloc(u32, 0),
            .allocator = allocator,
        };
    }

    // Temporary buffers sized to max possible output length.
    var ids = try allocator.alloc(u32, num_timesteps);
    defer allocator.free(ids);
    var confs = try allocator.alloc(f32, num_timesteps);
    defer allocator.free(confs);
    var ts = try allocator.alloc(u32, num_timesteps);
    defer allocator.free(ts);

    var out_len: usize = 0;
    var prev_code: i64 = -1; // Use i64 so -1 is a valid "no previous" sentinel.

    for (0..num_timesteps) |t| {
        const row = softmax_output[t * num_classes .. (t + 1) * num_classes];

        // Argmax over the row.
        var best_code: u32 = 0;
        var best_prob: f32 = row[0];
        for (1..num_classes) |c| {
            if (row[c] > best_prob) {
                best_prob = row[c];
                best_code = @intCast(c);
            }
        }

        // Skip CTC blank.
        if (best_code == null_char) {
            prev_code = -1; // Null breaks the duplicate chain.
            continue;
        }

        // CTC folding: skip consecutive duplicates.
        if (@as(i64, @intCast(best_code)) == prev_code) {
            continue;
        }

        // Map code to unichar_id.
        const unichar_id = codeToUnicharId(best_code, recoder, code_to_unichar);

        ids[out_len] = unichar_id;
        confs[out_len] = @log(@max(best_prob, MIN_PROB));
        ts[out_len] = @intCast(t);
        out_len += 1;

        prev_code = @intCast(best_code);
    }

    // Copy to right-sized result arrays.
    const result_ids = try allocator.alloc(u32, out_len);
    @memcpy(result_ids, ids[0..out_len]);
    errdefer allocator.free(result_ids);

    const result_confs = try allocator.alloc(f32, out_len);
    @memcpy(result_confs, confs[0..out_len]);
    errdefer allocator.free(result_confs);

    const result_ts = try allocator.alloc(u32, out_len);
    @memcpy(result_ts, ts[0..out_len]);

    return CtcResult{
        .unichar_ids = result_ids,
        .confidences = result_confs,
        .timesteps = result_ts,
        .allocator = allocator,
    };
}

// ── Beam Search CTC Decode ───────────────────────────────────────────────────

/// Configuration for CTC beam search decoding.
pub const BeamSearchConfig = struct {
    /// Number of beam hypotheses to maintain at each timestep.
    beam_width: usize = 10,
    /// Number of top codes to consider per timestep (pruning).
    top_n: usize = 5,
};

/// A single beam hypothesis in the search.
const Beam = struct {
    /// Sequence of output codes accumulated so far.
    codes: std.ArrayList(u32),
    /// Cumulative log probability score.
    score: f64,
    /// Last emitted code (-1 means no code emitted yet, or last was null).
    last_code: i64,

    fn init(allocator: Allocator) Beam {
        return Beam{
            .codes = std.ArrayList(u32).init(allocator),
            .score = 0.0,
            .last_code = -1,
        };
    }

    fn deinit(self: *Beam) void {
        self.codes.deinit();
    }

    fn clone(self: *const Beam) !Beam {
        const new_codes = try self.codes.clone();
        return Beam{
            .codes = new_codes,
            .score = self.score,
            .last_code = self.last_code,
        };
    }
};

/// CTC beam search decode: maintains multiple hypotheses and prunes by score.
///
/// This produces better results than greedy decoding by exploring multiple
/// candidate sequences and keeping the top beam_width hypotheses at each step.
///
/// Parameters are the same as ctcGreedyDecode, plus a BeamSearchConfig.
pub fn ctcBeamDecode(
    allocator: Allocator,
    softmax_output: []const f32,
    num_timesteps: usize,
    num_classes: usize,
    null_char: u32,
    recoder: ?*const UnicharCompress,
    code_to_unichar: ?[]const u32,
    config: BeamSearchConfig,
) !CtcResult {
    if (num_timesteps == 0 or num_classes == 0) {
        return CtcResult{
            .unichar_ids = try allocator.alloc(u32, 0),
            .confidences = try allocator.alloc(f32, 0),
            .timesteps = try allocator.alloc(u32, 0),
            .allocator = allocator,
        };
    }

    const effective_top_n = @min(config.top_n, num_classes);

    // Initialize with a single empty beam.
    var beams = std.ArrayList(Beam).init(allocator);
    defer {
        for (beams.items) |*b| b.deinit();
        beams.deinit();
    }

    var initial_beam = Beam.init(allocator);
    errdefer initial_beam.deinit();
    try beams.append(initial_beam);

    // Temporary storage for top-N code indices at each timestep.
    const top_indices = try allocator.alloc(u32, effective_top_n);
    defer allocator.free(top_indices);

    for (0..num_timesteps) |t| {
        const row = softmax_output[t * num_classes .. (t + 1) * num_classes];

        // Find top-N codes by probability.
        findTopN(row, num_classes, top_indices, effective_top_n);

        // New beams collected for this timestep. We use a hashmap keyed on
        // the code sequence to merge beams that produce the same decoded output.
        var new_beam_map = std.AutoHashMap(u64, Beam).init(allocator);
        defer {
            // Any beams remaining in the map that are NOT moved to the output
            // list need to be freed. But we move everything out below, so
            // just deinit the map structure itself.
            var it = new_beam_map.valueIterator();
            while (it.next()) |b| b.deinit();
            new_beam_map.deinit();
        }

        for (beams.items) |*beam| {
            for (top_indices) |code| {
                const log_prob: f64 = @floatCast(@log(@as(f32, @max(row[code], MIN_PROB))));
                const new_score = beam.score + log_prob;

                if (code == null_char) {
                    // Null extension: same decoded sequence, reset last_code.
                    const key = hashCodeSequence(beam.codes.items);
                    const entry = try new_beam_map.getOrPut(key);
                    if (entry.found_existing) {
                        if (new_score > entry.value_ptr.score) {
                            entry.value_ptr.score = new_score;
                            entry.value_ptr.last_code = -1;
                        }
                    } else {
                        var new_beam = try beam.clone();
                        new_beam.score = new_score;
                        new_beam.last_code = -1;
                        entry.value_ptr.* = new_beam;
                    }
                } else if (@as(i64, @intCast(code)) == beam.last_code) {
                    // Duplicate: CTC folding, don't add new character.
                    const key = hashCodeSequence(beam.codes.items);
                    const entry = try new_beam_map.getOrPut(key);
                    if (entry.found_existing) {
                        if (new_score > entry.value_ptr.score) {
                            entry.value_ptr.score = new_score;
                            entry.value_ptr.last_code = beam.last_code;
                        }
                    } else {
                        var new_beam = try beam.clone();
                        new_beam.score = new_score;
                        // last_code stays the same (still the duplicate)
                        entry.value_ptr.* = new_beam;
                    }
                } else {
                    // New character.
                    var new_beam = try beam.clone();
                    errdefer new_beam.deinit();
                    try new_beam.codes.append(code);
                    new_beam.score = new_score;
                    new_beam.last_code = @intCast(code);

                    const key = hashCodeSequence(new_beam.codes.items);
                    const entry = try new_beam_map.getOrPut(key);
                    if (entry.found_existing) {
                        if (new_score > entry.value_ptr.score) {
                            entry.value_ptr.deinit();
                            entry.value_ptr.* = new_beam;
                        } else {
                            new_beam.deinit();
                        }
                    } else {
                        entry.value_ptr.* = new_beam;
                    }
                }
            }
        }

        // Collect all candidate beams from the map.
        var candidates = std.ArrayList(Beam).init(allocator);
        defer candidates.deinit();

        var map_iter = new_beam_map.iterator();
        while (map_iter.next()) |entry| {
            try candidates.append(entry.value_ptr.*);
        }
        // Clear the map without deiniting the beams (they are now owned by candidates).
        new_beam_map.clearRetainingCapacity();

        // Sort by score descending.
        std.mem.sort(Beam, candidates.items, {}, struct {
            fn cmp(_: void, a: Beam, b: Beam) bool {
                return a.score > b.score;
            }
        }.cmp);

        // Prune to beam_width.
        const keep = @min(candidates.items.len, config.beam_width);

        // Free pruned beams.
        for (candidates.items[keep..]) |*b| b.deinit();

        // Replace the beam list.
        for (beams.items) |*b| b.deinit();
        beams.clearRetainingCapacity();
        for (candidates.items[0..keep]) |beam| {
            try beams.append(beam);
        }
    }

    // Find the best beam.
    if (beams.items.len == 0) {
        return CtcResult{
            .unichar_ids = try allocator.alloc(u32, 0),
            .confidences = try allocator.alloc(f32, 0),
            .timesteps = try allocator.alloc(u32, 0),
            .allocator = allocator,
        };
    }

    var best_idx: usize = 0;
    for (beams.items, 0..) |beam, idx| {
        if (beam.score > beams.items[best_idx].score) {
            best_idx = idx;
        }
    }

    const best = &beams.items[best_idx];
    const n = best.codes.items.len;

    // Convert codes to unichar_ids.
    const result_ids = try allocator.alloc(u32, n);
    errdefer allocator.free(result_ids);
    const result_confs = try allocator.alloc(f32, n);
    errdefer allocator.free(result_confs);
    const result_ts = try allocator.alloc(u32, n);

    for (0..n) |idx| {
        const code = best.codes.items[idx];
        result_ids[idx] = codeToUnicharId(code, recoder, code_to_unichar);
        // For beam search, we report the total normalized score as confidence
        // for each character, since individual per-character scores are not
        // tracked in this implementation.
        const score_f32: f32 = @floatCast(best.score);
        result_confs[idx] = if (n > 0) score_f32 / @as(f32, @floatFromInt(n)) else 0.0;
        result_ts[idx] = 0; // Timestep tracking not available in beam search.
    }

    return CtcResult{
        .unichar_ids = result_ids,
        .confidences = result_confs,
        .timesteps = result_ts,
        .allocator = allocator,
    };
}

// ── Internal Helpers ─────────────────────────────────────────────────────────

/// Map an output code to a unichar_id using either a prebuilt lookup table
/// or the recoder. Falls back to using the code directly as the unichar_id.
fn codeToUnicharId(
    code: u32,
    recoder: ?*const UnicharCompress,
    code_to_unichar: ?[]const u32,
) u32 {
    // Prefer the prebuilt table if available.
    if (code_to_unichar) |table| {
        if (code < table.len and table[code] != INVALID_UNICHAR_ID) {
            return table[code];
        }
    }

    // Fall back to scanning the recoder entries.
    if (recoder) |rec| {
        for (rec.entries, 0..) |entry, uid| {
            if (entry.codes.len == 1 and entry.codes[0] >= 0) {
                if (@as(u32, @intCast(entry.codes[0])) == code) {
                    return @intCast(uid);
                }
            }
        }
        return INVALID_UNICHAR_ID;
    }

    // No recoder: code IS the unichar_id.
    return code;
}

/// Find the indices of the top-N elements in `values` by value (descending).
/// Results are written to `out_indices` which must have length >= top_n.
fn findTopN(values: []const f32, num_values: usize, out_indices: []u32, top_n: usize) void {
    const n = @min(top_n, num_values);
    if (n == 0) return;

    // Simple selection: maintain a sorted list of top-N.
    // For small top_n (5-20) this is efficient enough.
    var top_vals: [64]f32 = undefined;
    var top_idx: [64]u32 = undefined;
    var count: usize = 0;

    for (0..num_values) |i| {
        const val = values[i];
        const ci: u32 = @intCast(i);

        if (count < n) {
            // Insert into the list (sorted descending).
            var pos: usize = count;
            while (pos > 0 and val > top_vals[pos - 1]) {
                top_vals[pos] = top_vals[pos - 1];
                top_idx[pos] = top_idx[pos - 1];
                pos -= 1;
            }
            top_vals[pos] = val;
            top_idx[pos] = ci;
            count += 1;
        } else if (val > top_vals[count - 1]) {
            // Replace the smallest in our top-N.
            var pos: usize = count - 1;
            while (pos > 0 and val > top_vals[pos - 1]) {
                top_vals[pos] = top_vals[pos - 1];
                top_idx[pos] = top_idx[pos - 1];
                pos -= 1;
            }
            top_vals[pos] = val;
            top_idx[pos] = ci;
        }
    }

    for (0..n) |i| {
        out_indices[i] = top_idx[i];
    }
}

/// Hash a code sequence for beam deduplication. Uses FNV-1a.
fn hashCodeSequence(codes: []const u32) u64 {
    var h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for (codes) |code| {
        const bytes = std.mem.asBytes(&code);
        for (bytes) |b| {
            h ^= @intCast(b);
            h *%= 0x100000001b3; // FNV prime
        }
    }
    return h;
}

// ── Image Input Preparation ─────────────────────────────────────────────────

/// Prepare a grayscale image as network input.
///
/// Scales the image to `target_height` while preserving aspect ratio using
/// bilinear interpolation. Normalizes pixel values from [0, 255] to [-1.0, 1.0].
///
/// Matches Tesseract's Copy2DImage layout: the image is flattened row-by-row into
/// a 1D sequence of timesteps with 1 feature each (the grayscale pixel value).
/// Total timesteps = target_height * scaled_width. The network's 2D convolutions
/// and XYTranspose layers work on this flattened representation.
///
/// Parameters:
///   - allocator: Memory allocator for the result.
///   - pixels: Grayscale pixels, row-major [height][width], values 0-255.
///   - width: Width of the input image in pixels.
///   - height: Height of the input image in pixels.
///   - target_height: Target height to scale to (from network input shape, e.g. 36).
///
/// Returns a NetworkIO of shape [target_height * scaled_width][1].
pub fn prepareInput(
    allocator: Allocator,
    pixels: []const u8,
    width: usize,
    height: usize,
    target_height: usize,
) !NetworkIO {
    if (width == 0 or height == 0 or target_height == 0) {
        return NetworkIO.init(allocator, 0, 1, false);
    }

    // Compute scaled width preserving aspect ratio.
    const scaled_width = (width * target_height + height / 2) / height;
    if (scaled_width == 0) {
        return NetworkIO.init(allocator, 0, 1, false);
    }

    // Total timesteps = height * width in the scaled image (row-major 2D layout).
    var nio = try NetworkIO.init2D(allocator, target_height, scaled_width, 1, false);
    errdefer nio.deinit();

    const buf = nio.f_data.?;

    // Compute adaptive normalization (matching Tesseract's ComputeBlackWhite).
    // Find min/max pixel values for contrast stretching.
    var min_pixel: f32 = 255.0;
    var max_pixel: f32 = 0.0;
    for (pixels) |p| {
        const v: f32 = @floatFromInt(p);
        if (v < min_pixel) min_pixel = v;
        if (v > max_pixel) max_pixel = v;
    }
    const black = min_pixel;
    var contrast = (max_pixel - black) / 2.0;
    if (contrast <= 0.0) contrast = 1.0;

    // Fill the NetworkIO: row-major (for each row y, for each column x).
    // Timestep t = y * scaled_width + x, 1 feature per timestep.
    for (0..target_height) |y| {
        for (0..scaled_width) |x| {
            // Map scaled (x, y) back to source image coordinates via bilinear interpolation.
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

            const p00: f32 = @floatFromInt(pixels[y0 * width + x0]);
            const p10: f32 = @floatFromInt(pixels[y0 * width + x1]);
            const p01: f32 = @floatFromInt(pixels[y1 * width + x0]);
            const p11: f32 = @floatFromInt(pixels[y1 * width + x1]);

            const val = p00 * (1.0 - dx) * (1.0 - dy) +
                p10 * dx * (1.0 - dy) +
                p01 * (1.0 - dx) * dy +
                p11 * dx * dy;

            // Normalize: match Tesseract's SetPixel: (pixel - black) / contrast - 1.0
            const normalized = (val - black) / contrast - 1.0;

            const t = y * scaled_width + x;
            buf[t] = normalized; // 1 feature per timestep
        }
    }

    return nio;
}

// ── Recognition Result ──────────────────────────────────────────────────────

/// Holds the output of line recognition: UTF-8 text with per-character
/// confidence scores and timestep positions.
pub const RecognitionResult = struct {
    /// UTF-8 text (owned).
    text: []u8,
    /// Per-character confidence (log probability).
    confidences: []f32,
    /// Per-character timestep positions.
    timesteps: []u32,
    allocator: Allocator,

    /// Release all owned memory.
    pub fn deinit(self: *RecognitionResult) void {
        self.allocator.free(self.text);
        self.allocator.free(self.confidences);
        self.allocator.free(self.timesteps);
    }
};

// ── Line Recognition Pipeline ───────────────────────────────────────────────

/// Recognize text from a grayscale line image using a loaded LSTM model.
///
/// Orchestrates the complete pipeline:
///   1. Prepare image input (scale, normalize, pack into NetworkIO)
///   2. Network forward pass (LSTM inference)
///   3. CTC beam search decode (softmax output -> character sequence)
///   4. Convert to UTF-8 text with confidences
///
/// Parameters:
///   - allocator: Memory allocator for all intermediate and result buffers.
///   - model: Loaded LSTMRecognizer with network, unicharset, and recoder.
///   - pixels: Grayscale pixels, row-major [height][width], values 0-255.
///   - width: Width of the input image in pixels.
///   - height: Height of the input image in pixels.
///
/// Returns a RecognitionResult with UTF-8 text and per-character metadata.
/// Caller must call deinit() to free the result.
pub fn recognizeLine(
    allocator: Allocator,
    model: *LSTMRecognizer,
    pixels: []const u8,
    width: usize,
    height: usize,
) !RecognitionResult {
    // 1. Determine target height from the network input shape.
    const target_height = model.numInputs();

    // 2. Prepare image as network input.
    var input = try prepareInput(allocator, pixels, width, height, target_height);
    defer input.deinit();

    // Handle degenerate case: empty input produces empty result.
    if (input.width() == 0) {
        const empty_text = try allocator.alloc(u8, 0);
        errdefer allocator.free(empty_text);
        const empty_confs = try allocator.alloc(f32, 0);
        errdefer allocator.free(empty_confs);
        const empty_ts = try allocator.alloc(u32, 0);
        return RecognitionResult{
            .text = empty_text,
            .confidences = empty_confs,
            .timesteps = empty_ts,
            .allocator = allocator,
        };
    }

    // 3. Run network forward pass.
    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try model.network.forward(&input, &output, allocator);

    // 4. Extract softmax probabilities.
    const num_timesteps = output.width();
    const num_classes = output.numFeatures();
    const softmax_data = output.f_data.?[0 .. num_timesteps * num_classes];

    // 5. Build reverse lookup table for the recoder.
    var code_to_unichar: ?[]u32 = null;
    defer {
        if (code_to_unichar) |tbl| allocator.free(tbl);
    }
    var recoder_ptr: ?*const UnicharCompress = null;
    if (model.recoder) |*r| {
        code_to_unichar = try buildCodeToUnichar(allocator, r);
        recoder_ptr = r;
    }

    // 6. CTC beam search decode.
    const null_char: u32 = @intCast(model.null_char);
    var ctc_result = try ctcBeamDecode(
        allocator,
        softmax_data,
        num_timesteps,
        num_classes,
        null_char,
        recoder_ptr,
        code_to_unichar,
        .{ .beam_width = 10, .top_n = 5 },
    );
    defer ctc_result.deinit();

    // 7. Convert unichar IDs to UTF-8 text.
    const text = try ctc_result.toText(&model.unicharset, allocator);
    errdefer allocator.free(text);

    // 8. Copy confidence and timestep arrays to owned result.
    const result_confs = try allocator.alloc(f32, ctc_result.len());
    errdefer allocator.free(result_confs);
    @memcpy(result_confs, ctc_result.confidences);

    const result_ts = try allocator.alloc(u32, ctc_result.len());
    @memcpy(result_ts, ctc_result.timesteps);

    return RecognitionResult{
        .text = text,
        .confidences = result_confs,
        .timesteps = result_ts,
        .allocator = allocator,
    };
}

// ── Tests ────────────────────────────────────────────────────────────────────

const testing = std.testing;
const test_allocator = testing.allocator;

test "greedy decode: basic CTC folding" {
    // 3 timesteps, 4 classes (codes 0-3), null_char=0:
    //   t=0: [0.1, 0.7, 0.1, 0.1]  -> code 1
    //   t=1: [0.1, 0.7, 0.1, 0.1]  -> code 1 (duplicate, folded)
    //   t=2: [0.1, 0.1, 0.8, 0.0]  -> code 2
    // Result: [1, 2] with confidence [log(0.7), log(0.8)]
    const softmax = [_]f32{
        0.1, 0.7, 0.1, 0.1,
        0.1, 0.7, 0.1, 0.1,
        0.1, 0.1, 0.8, 0.0,
    };

    var result = try ctcGreedyDecode(test_allocator, &softmax, 3, 4, 0, null, null);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.len());
    try testing.expectEqual(@as(u32, 1), result.unichar_ids[0]);
    try testing.expectEqual(@as(u32, 2), result.unichar_ids[1]);
    try testing.expectApproxEqAbs(@log(@as(f32, 0.7)), result.confidences[0], 1e-5);
    try testing.expectApproxEqAbs(@log(@as(f32, 0.8)), result.confidences[1], 1e-5);
}

test "greedy decode: null removal" {
    // 3 timesteps, 3 classes, null_char=0:
    //   t=0: [0.8, 0.1, 0.1]  -> code 0 (null, removed)
    //   t=1: [0.1, 0.8, 0.1]  -> code 1
    //   t=2: [0.8, 0.1, 0.1]  -> code 0 (null, removed)
    // Result: [1]
    const softmax = [_]f32{
        0.8, 0.1, 0.1,
        0.1, 0.8, 0.1,
        0.8, 0.1, 0.1,
    };

    var result = try ctcGreedyDecode(test_allocator, &softmax, 3, 3, 0, null, null);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.len());
    try testing.expectEqual(@as(u32, 1), result.unichar_ids[0]);
}

test "greedy decode: CTC folding with null separator" {
    // 4 timesteps, 3 classes, null_char=0:
    //   t=0: [0.1, 0.8, 0.1]  -> code 1
    //   t=1: [0.8, 0.1, 0.1]  -> code 0 (null)
    //   t=2: [0.1, 0.8, 0.1]  -> code 1 (NOT folded - null separated)
    //   t=3: [0.1, 0.1, 0.8]  -> code 2
    // Result: [1, 1, 2]
    const softmax = [_]f32{
        0.1, 0.8, 0.1,
        0.8, 0.1, 0.1,
        0.1, 0.8, 0.1,
        0.1, 0.1, 0.8,
    };

    var result = try ctcGreedyDecode(test_allocator, &softmax, 4, 3, 0, null, null);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.len());
    try testing.expectEqual(@as(u32, 1), result.unichar_ids[0]);
    try testing.expectEqual(@as(u32, 1), result.unichar_ids[1]);
    try testing.expectEqual(@as(u32, 2), result.unichar_ids[2]);
}

test "beam search: matches greedy on simple cases" {
    // Same test case as "basic CTC folding" above.
    const softmax = [_]f32{
        0.1, 0.7, 0.1, 0.1,
        0.1, 0.7, 0.1, 0.1,
        0.1, 0.1, 0.8, 0.0,
    };

    var greedy = try ctcGreedyDecode(test_allocator, &softmax, 3, 4, 0, null, null);
    defer greedy.deinit();

    var beam = try ctcBeamDecode(test_allocator, &softmax, 3, 4, 0, null, null, .{
        .beam_width = 10,
        .top_n = 4,
    });
    defer beam.deinit();

    // Beam search should produce the same decoded sequence.
    try testing.expectEqual(greedy.len(), beam.len());
    for (0..greedy.len()) |idx| {
        try testing.expectEqual(greedy.unichar_ids[idx], beam.unichar_ids[idx]);
    }
}

test "beam search: produces valid output on null-separated input" {
    // Same test as "CTC folding with null separator".
    const softmax = [_]f32{
        0.1, 0.8, 0.1,
        0.8, 0.1, 0.1,
        0.1, 0.8, 0.1,
        0.1, 0.1, 0.8,
    };

    var result = try ctcBeamDecode(test_allocator, &softmax, 4, 3, 0, null, null, .{
        .beam_width = 10,
        .top_n = 3,
    });
    defer result.deinit();

    // Should decode to [1, 1, 2] just like greedy.
    try testing.expectEqual(@as(usize, 3), result.len());
    try testing.expectEqual(@as(u32, 1), result.unichar_ids[0]);
    try testing.expectEqual(@as(u32, 1), result.unichar_ids[1]);
    try testing.expectEqual(@as(u32, 2), result.unichar_ids[2]);
}

test "toText: unichar ID to string conversion" {
    // Build a minimal unicharset with known characters.
    // IDs: 0=NULL, 1=' ', 2='H', 3='i', 4='!'
    const unichar_data =
        \\5
        \\NULL 00
        \\NULL 00
        \\H 01
        \\i 02
        \\! 10
    ;

    var unicharset = try Unicharset.load(test_allocator, unichar_data);
    defer unicharset.deinit();

    // Create a CtcResult with unichar_ids = [2, 3, 4] -> "Hi!"
    var ids = [_]u32{ 2, 3, 4 };
    var confs = [_]f32{ -0.1, -0.2, -0.3 };
    var ts = [_]u32{ 0, 1, 2 };

    const result = CtcResult{
        .unichar_ids = &ids,
        .confidences = &confs,
        .timesteps = &ts,
        .allocator = test_allocator,
    };

    const text = try result.toText(&unicharset, test_allocator);
    defer test_allocator.free(text);

    try testing.expectEqualStrings("Hi!", text);
}

test "greedy decode: empty input" {
    var result = try ctcGreedyDecode(test_allocator, &[_]f32{}, 0, 0, 0, null, null);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.len());
}

test "beam search: empty input" {
    var result = try ctcBeamDecode(test_allocator, &[_]f32{}, 0, 0, 0, null, null, .{});
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.len());
}

test "greedy decode: all nulls" {
    // Every timestep is a null character -> empty output.
    const softmax = [_]f32{
        0.9, 0.05, 0.05,
        0.9, 0.05, 0.05,
        0.9, 0.05, 0.05,
    };

    var result = try ctcGreedyDecode(test_allocator, &softmax, 3, 3, 0, null, null);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.len());
}

test "greedy decode: single timestep" {
    const softmax = [_]f32{ 0.1, 0.2, 0.7 };

    var result = try ctcGreedyDecode(test_allocator, &softmax, 1, 3, 0, null, null);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.len());
    try testing.expectEqual(@as(u32, 2), result.unichar_ids[0]);
}

test "buildCodeToUnichar: basic reverse lookup" {
    // We can't easily construct a UnicharCompress in tests because its load
    // requires a BinaryReader. Instead we verify the algorithm logic via
    // the decode path with code_to_unichar parameter.
    //
    // Simulate a reverse table where code 1 -> unichar 5, code 2 -> unichar 3.
    var table = [_]u32{ INVALID_UNICHAR_ID, 5, 3 };

    const softmax = [_]f32{
        0.1, 0.8, 0.1,
        0.1, 0.1, 0.8,
    };

    var result = try ctcGreedyDecode(test_allocator, &softmax, 2, 3, 0, null, &table);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.len());
    try testing.expectEqual(@as(u32, 5), result.unichar_ids[0]); // code 1 -> unichar 5
    try testing.expectEqual(@as(u32, 3), result.unichar_ids[1]); // code 2 -> unichar 3
}

test "findTopN: correctness" {
    const values = [_]f32{ 0.1, 0.5, 0.3, 0.8, 0.2 };
    var indices: [3]u32 = undefined;

    findTopN(&values, 5, &indices, 3);

    // Top 3 should be indices 3 (0.8), 1 (0.5), 2 (0.3)
    try testing.expectEqual(@as(u32, 3), indices[0]);
    try testing.expectEqual(@as(u32, 1), indices[1]);
    try testing.expectEqual(@as(u32, 2), indices[2]);
}

test "hashCodeSequence: deterministic" {
    const a = [_]u32{ 1, 2, 3 };
    const b = [_]u32{ 1, 2, 3 };
    const c = [_]u32{ 1, 2, 4 };

    try testing.expectEqual(hashCodeSequence(&a), hashCodeSequence(&b));
    try testing.expect(hashCodeSequence(&a) != hashCodeSequence(&c));
}

test "beam search: null removal only" {
    // All timesteps are null except one.
    const softmax = [_]f32{
        0.8, 0.1, 0.1,
        0.1, 0.8, 0.1,
        0.8, 0.1, 0.1,
    };

    var result = try ctcBeamDecode(test_allocator, &softmax, 3, 3, 0, null, null, .{
        .beam_width = 10,
        .top_n = 3,
    });
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.len());
    try testing.expectEqual(@as(u32, 1), result.unichar_ids[0]);
}

// ── prepareInput Tests ──────────────────────────────────────────────────────

test "prepareInput: basic no scaling" {
    // 10x5 image, target_height=5 -> no vertical scaling needed.
    // Output: timesteps = 5 * 10 = 50, features = 1 (Tesseract 2D layout).
    const width: usize = 10;
    const height: usize = 5;
    var pixels: [width * height]u8 = undefined;

    // Fill with known values: gradient along x, constant along y.
    for (0..height) |y| {
        for (0..width) |x| {
            pixels[y * width + x] = @intCast(x * 25); // 0, 25, 50, ..., 225
        }
    }

    var nio = try prepareInput(test_allocator, &pixels, width, height, 5);
    defer nio.deinit();

    // Dimensions: total_timesteps = 5 * 10 = 50, num_features = 1.
    try testing.expectEqual(@as(usize, 50), nio.width());
    try testing.expectEqual(@as(usize, 1), nio.numFeatures());

    // With adaptive normalization: pixel 0 (black) maps to -1.0,
    // pixel 225 (max) maps to +1.0. First timestep (row 0, col 0) has pixel 0.
    const first_val = nio.f(0)[0];
    try testing.expectApproxEqAbs(@as(f32, -1.0), first_val, 0.02);
}

test "prepareInput: normalization values" {
    // 3x1 image with specific values to verify normalization.
    // With adaptive normalization: black=0, contrast=(255-0)/2=127.5
    // pixel 0 -> (0 - 0) / 127.5 - 1.0 = -1.0
    // pixel 128 -> (128 - 0) / 127.5 - 1.0 ≈ 0.004
    // pixel 255 -> (255 - 0) / 127.5 - 1.0 = 1.0
    const pixels = [_]u8{ 0, 128, 255 };

    var nio = try prepareInput(test_allocator, &pixels, 3, 1, 1);
    defer nio.deinit();

    // 1x3 scaled -> total timesteps = 1 * 3 = 3, features = 1.
    try testing.expectEqual(@as(usize, 3), nio.width());
    try testing.expectEqual(@as(usize, 1), nio.numFeatures());

    try testing.expectApproxEqAbs(@as(f32, -1.0), nio.f(0)[0], 0.01);
    try testing.expectApproxEqAbs(@as(f32, 0.004), nio.f(1)[0], 0.02);
    try testing.expectApproxEqAbs(@as(f32, 1.0), nio.f(2)[0], 0.01);
}

test "prepareInput: with scaling" {
    // 20x10 image, target_height=5 -> scale to height 5.
    // Scaled width = 20 * 5 / 10 = 10. Total timesteps = 5 * 10 = 50.
    const width: usize = 20;
    const height: usize = 10;
    var pixels: [width * height]u8 = undefined;
    @memset(&pixels, 128); // uniform gray

    var nio = try prepareInput(test_allocator, &pixels, width, height, 5);
    defer nio.deinit();

    // Dimensions: total_timesteps = 5 * 10 = 50, features = 1.
    try testing.expectEqual(@as(usize, 50), nio.width());
    try testing.expectEqual(@as(usize, 1), nio.numFeatures());

    // All pixels are 128, so min=max=128, contrast=1.0.
    // Normalized: (128 - 128) / 1.0 - 1.0 = -1.0 (uniform value).
    const ts = nio.f(0);
    for (ts) |v| {
        try testing.expectApproxEqAbs(@as(f32, -1.0), v, 0.02);
    }
}

test "prepareInput: empty image" {
    var nio = try prepareInput(test_allocator, &[_]u8{}, 0, 0, 36);
    defer nio.deinit();

    try testing.expectEqual(@as(usize, 0), nio.width());
}

// ── recognizeLine Integration Tests ─────────────────────────────────────────

test "recognizeLine: integration with real model" {
    // Load the eng.traineddata model.
    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: cannot open test file: {}\n", .{err});
        return;
    };
    defer file.close();

    const data = file.readToEndAlloc(test_allocator, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: cannot read test file: {}\n", .{err});
        return;
    };
    defer test_allocator.free(data);

    var model = LSTMRecognizer.load(test_allocator, data) catch |err| {
        std.debug.print("Skipping test: cannot load model: {}\n", .{err});
        return;
    };
    defer model.deinit();

    // Verify model basics.
    try testing.expectEqual(@as(usize, 36), model.numInputs());
    try testing.expectEqual(@as(usize, 111), model.numClasses());

    // Create a synthetic white image (255 = white, should produce minimal/no text).
    // 100 pixels wide, 36 pixels tall (matching model input height).
    const img_w: usize = 100;
    const img_h: usize = 36;
    var pixels: [img_w * img_h]u8 = undefined;
    @memset(&pixels, 255);

    // Run the pipeline. This is the key integration test - it should not crash
    // and should produce valid output even if the text is meaningless.
    var result = try recognizeLine(test_allocator, &model, &pixels, img_w, img_h);
    defer result.deinit();

    // Verify the result is structurally valid.
    // Text should be valid UTF-8 (may be empty or whitespace for a blank image).
    try testing.expect(std.unicode.utf8ValidateSlice(result.text));

    // Confidences and timesteps arrays should match in length.
    try testing.expectEqual(result.confidences.len, result.timesteps.len);
}

test "recognizeLine: solid black image" {
    // Load the eng.traineddata model.
    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: cannot open test file: {}\n", .{err});
        return;
    };
    defer file.close();

    const data = file.readToEndAlloc(test_allocator, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: cannot read test file: {}\n", .{err});
        return;
    };
    defer test_allocator.free(data);

    var model = LSTMRecognizer.load(test_allocator, data) catch |err| {
        std.debug.print("Skipping test: cannot load model: {}\n", .{err});
        return;
    };
    defer model.deinit();

    // Solid black image (0 = black).
    const img_w: usize = 80;
    const img_h: usize = 36;
    var pixels: [img_w * img_h]u8 = undefined;
    @memset(&pixels, 0);

    var result = try recognizeLine(test_allocator, &model, &pixels, img_w, img_h);
    defer result.deinit();

    // Should produce valid UTF-8 and not crash.
    try testing.expect(std.unicode.utf8ValidateSlice(result.text));
    try testing.expectEqual(result.confidences.len, result.timesteps.len);
}

test "RecognitionResult: deinit frees all memory" {
    // Allocate a result manually and verify deinit cleans up without leaks.
    // The testing allocator will catch any leaks.
    const text = try test_allocator.alloc(u8, 5);
    @memcpy(text, "hello");
    const confs = try test_allocator.alloc(f32, 3);
    const ts = try test_allocator.alloc(u32, 3);

    var result = RecognitionResult{
        .text = text,
        .confidences = confs,
        .timesteps = ts,
        .allocator = test_allocator,
    };
    result.deinit();
    // If we get here without the testing allocator complaining, all memory was freed.
}
