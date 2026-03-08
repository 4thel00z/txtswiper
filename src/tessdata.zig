const std = @import("std");
const network_mod = @import("network.zig");
const weights_mod = @import("weights.zig");

// ── TessdataType ─────────────────────────────────────────────────────────────

/// Component types stored in a .traineddata container.
/// Indices match the Tesseract 5.x offset table layout (24 entries).
pub const TessdataType = enum(u8) {
    lang_config = 0,
    unicharset = 1,
    ambigs = 2,
    inttemp = 3,
    pffmtable = 4,
    normproto = 5,
    punc_dawg = 6,
    system_dawg = 7,
    number_dawg = 8,
    freq_dawg = 9,
    fixed_length_dawgs = 10, // deprecated
    cube_unicharset = 11, // deprecated
    cube_system_dawg = 12, // deprecated
    shape_table = 13,
    bigram_dawg = 14,
    unambig_dawg = 15,
    params_model = 16,
    lstm = 17,
    lstm_punc_dawg = 18,
    lstm_system_dawg = 19,
    lstm_number_dawg = 20,
    lstm_unicharset = 21,
    lstm_recoder = 22,
    version = 23,

    pub const count = 24;
};

// ── Errors ───────────────────────────────────────────────────────────────────

pub const TessdataError = error{
    /// File is too small to contain a valid header.
    FileTooSmall,
    /// num_entries is not 24 (unsupported format).
    BadEntryCount,
    /// An offset points past the end of the file.
    OffsetOutOfRange,
};

pub const UnicharsetError = error{
    /// Data is empty or contains no lines.
    EmptyData,
    /// First line is not a valid integer count.
    BadCharCount,
    /// More data lines than the declared count.
    TooManyEntries,
    /// Fewer data lines than the declared count.
    TooFewEntries,
    /// A line has no fields at all.
    EmptyLine,
    /// Properties field is not a valid hex integer.
    BadProperties,
    /// other_case / direction / mirror field is not a valid integer.
    BadIntField,
    /// Memory allocation failed.
    OutOfMemory,
};

// ── TessdataManager ──────────────────────────────────────────────────────────

/// Maximum num_entries value before we assume the file is big-endian.
const kMaxNumTessdataEntries: u32 = 1000;

/// Size of the header: 4 bytes (num_entries) + 24 * 8 bytes (offset table).
const header_size: usize = 4 + TessdataType.count * @sizeOf(i64);

/// Parses the .traineddata binary container format used by Tesseract OCR.
///
/// The format is:
///   - 4 bytes: u32 num_entries (always 24 for Tesseract 5.x)
///   - 192 bytes: i64[24] offset table (-1 means entry absent)
///   - remainder: concatenated entry data
///
/// TessdataManager operates on a borrowed byte slice; it does not allocate.
pub const TessdataManager = struct {
    /// Byte offsets for each component, or -1 if absent.
    offsets: [TessdataType.count]i64,
    /// Computed sizes for each component, or 0 if absent.
    sizes: [TessdataType.count]usize,
    /// The underlying file data (borrowed, not owned).
    data: []const u8,
    /// Whether the file was stored in big-endian format.
    is_big_endian: bool,

    /// Parse a .traineddata container from a byte buffer.
    /// The buffer must remain valid for the lifetime of this manager.
    pub fn init(data: []const u8) TessdataError!TessdataManager {
        if (data.len < header_size) return TessdataError.FileTooSmall;

        // Read num_entries as little-endian first.
        var num_entries = std.mem.readInt(u32, data[0..4], .little);
        var is_big_endian = false;

        // Endianness detection: if num_entries > kMaxNumTessdataEntries,
        // the file was written big-endian.
        if (num_entries > kMaxNumTessdataEntries) {
            num_entries = @byteSwap(num_entries);
            is_big_endian = true;
        }

        if (num_entries != TessdataType.count) return TessdataError.BadEntryCount;

        // Read offset table.
        var offsets: [TessdataType.count]i64 = undefined;
        for (0..TessdataType.count) |i| {
            const start = 4 + i * 8;
            const raw = std.mem.readInt(i64, data[start..][0..8], .little);
            offsets[i] = if (is_big_endian) @bitCast(@byteSwap(@as(u64, @bitCast(raw)))) else raw;
        }

        // Validate offsets and compute sizes.
        var sizes: [TessdataType.count]usize = .{0} ** TessdataType.count;
        for (0..TessdataType.count) |i| {
            if (offsets[i] == -1) continue;
            if (offsets[i] < 0) return TessdataError.OffsetOutOfRange;

            const off: u64 = @intCast(offsets[i]);
            if (off >= data.len) return TessdataError.OffsetOutOfRange;

            // Find the next present entry to determine this entry's size.
            var next_offset: u64 = data.len;
            for ((i + 1)..TessdataType.count) |j| {
                if (offsets[j] != -1) {
                    if (offsets[j] < 0) return TessdataError.OffsetOutOfRange;
                    next_offset = @intCast(offsets[j]);
                    if (next_offset > data.len) return TessdataError.OffsetOutOfRange;
                    break;
                }
            }

            if (next_offset < off) return TessdataError.OffsetOutOfRange;
            sizes[i] = @intCast(next_offset - off);
        }

        return TessdataManager{
            .offsets = offsets,
            .sizes = sizes,
            .data = data,
            .is_big_endian = is_big_endian,
        };
    }

    /// Returns a slice to the data for the requested component, or null if absent.
    pub fn getComponent(self: TessdataManager, comp_type: TessdataType) ?[]const u8 {
        const idx = @intFromEnum(comp_type);
        if (self.offsets[idx] == -1) return null;

        const off: usize = @intCast(self.offsets[idx]);
        const size = self.sizes[idx];
        return self.data[off..][0..size];
    }

    /// Number of entry types supported by this container (always 24).
    pub fn numEntries(self: TessdataManager) usize {
        _ = self;
        return TessdataType.count;
    }

    /// Returns the number of components actually present in this container.
    pub fn numPresent(self: TessdataManager) usize {
        var n: usize = 0;
        for (self.offsets) |off| {
            if (off != -1) n += 1;
        }
        return n;
    }

    /// Returns true if the given component is present.
    pub fn hasComponent(self: TessdataManager, comp_type: TessdataType) bool {
        return self.offsets[@intFromEnum(comp_type)] != -1;
    }
};

// ── Unicharset ──────────────────────────────────────────────────────────────

/// Property bit flags for UNICHAR entries (hex-encoded in the text format).
/// bit0 = alpha, bit1 = lower, bit2 = upper, bit3 = digit, bit4 = punct
const CHAR_PROP_ALPHA: u8 = 0x01;
const CHAR_PROP_LOWER: u8 = 0x02;
const CHAR_PROP_UPPER: u8 = 0x04;
const CHAR_PROP_DIGIT: u8 = 0x08;
const CHAR_PROP_PUNCT: u8 = 0x10;

/// Parses the UNICHARSET text format used by Tesseract OCR.
///
/// The UNICHARSET maps integer IDs to Unicode characters and their properties.
/// The LSTM recognizer uses it to decode network output (class IDs) back to
/// Unicode strings.
///
/// Format:
///   - First line: integer count of characters
///   - Following lines: one per character with space-separated fields
///   - "NULL" on the first entry means the space character (ASCII 32)
///
/// Unicharset owns all string data; call deinit() to release memory.
pub const Unicharset = struct {
    chars: []UnicharEntry,
    allocator: std.mem.Allocator,

    pub const UnicharEntry = struct {
        unichar: []const u8, // UTF-8 string (owned copy)
        properties: u8, // hex property flags
        script: []const u8, // script name (owned copy)
        other_case: u32, // UNICHAR_ID of case pair
        direction: u8, // 0-22
        mirror: u32, // UNICHAR_ID of mirror
        normed: []const u8, // normalized form (owned copy)
    };

    /// Parse a UNICHARSET from raw text data (as returned by TessdataManager.getComponent).
    pub fn load(allocator: std.mem.Allocator, data: []const u8) (UnicharsetError || std.mem.Allocator.Error)!Unicharset {
        if (data.len == 0) return UnicharsetError.EmptyData;

        var line_iter = std.mem.splitScalar(u8, data, '\n');

        // First line: character count.
        const count_line = line_iter.next() orelse return UnicharsetError.EmptyData;
        const char_count = std.fmt.parseInt(u32, std.mem.trim(u8, count_line, &std.ascii.whitespace), 10) catch {
            return UnicharsetError.BadCharCount;
        };

        // Allocate entry array.
        const chars = try allocator.alloc(UnicharEntry, char_count);

        // Track how many entries have been fully populated so errdefer can
        // release their owned strings.
        var populated: u32 = 0;
        errdefer {
            for (chars[0..populated]) |*entry| {
                if (entry.unichar.len > 0) allocator.free(entry.unichar);
                if (entry.script.len > 0) allocator.free(entry.script);
                if (entry.normed.len > 0) allocator.free(entry.normed);
            }
            allocator.free(chars);
        }

        var idx: u32 = 0;
        while (line_iter.next()) |raw_line| {
            // Skip trailing empty lines (common after last entry).
            if (raw_line.len == 0) continue;

            if (idx >= char_count) return UnicharsetError.TooManyEntries;

            // Strip debug comment: everything after \t# is ignored.
            const line = if (std.mem.indexOf(u8, raw_line, "\t#")) |tab_pos|
                raw_line[0..tab_pos]
            else
                raw_line;

            // Parse fields (space-separated).
            var fields = std.mem.splitScalar(u8, std.mem.trimRight(u8, line, &std.ascii.whitespace), ' ');

            // Field 0: unichar string.
            const unichar_raw = fields.next() orelse return UnicharsetError.EmptyLine;

            // "NULL" is the space character.
            const unichar_str = if (std.mem.eql(u8, unichar_raw, "NULL")) " " else unichar_raw;

            // Field 1: properties (hex integer).
            const props_str = fields.next() orelse {
                // Minimal line: just the unichar. Use defaults.
                chars[idx] = UnicharEntry{
                    .unichar = try allocator.dupe(u8, unichar_str),
                    .properties = 0,
                    .script = &.{},
                    .other_case = 0,
                    .direction = 0,
                    .mirror = 0,
                    .normed = &.{},
                };
                idx += 1;
                populated = idx;
                continue;
            };
            const properties: u8 = std.fmt.parseInt(u8, props_str, 16) catch {
                return UnicharsetError.BadProperties;
            };

            // Field 2: either metrics (contains commas) or script name.
            const field2 = fields.next();

            var script_str: []const u8 = "Common";
            var other_case: u32 = 0;
            var direction: u8 = 0;
            var mirror: u32 = 0;
            var normed_str: []const u8 = unichar_str;

            if (field2) |f2| {
                if (std.mem.indexOfScalar(u8, f2, ',') != null) {
                    // f2 is the metrics field (comma-separated values) - skip it.
                    // Field 3: script
                    if (fields.next()) |s| {
                        script_str = s;
                    }
                    // Field 4: other_case
                    if (fields.next()) |oc| {
                        other_case = std.fmt.parseInt(u32, oc, 10) catch return UnicharsetError.BadIntField;
                    }
                    // Field 5: direction
                    if (fields.next()) |d| {
                        direction = std.fmt.parseInt(u8, d, 10) catch return UnicharsetError.BadIntField;
                    }
                    // Field 6: mirror
                    if (fields.next()) |m| {
                        mirror = std.fmt.parseInt(u32, m, 10) catch return UnicharsetError.BadIntField;
                    }
                    // Field 7: normed
                    if (fields.next()) |n| {
                        normed_str = if (std.mem.eql(u8, n, "NULL")) " " else n;
                    }
                } else {
                    // f2 is the script name (short format, no metrics).
                    script_str = f2;
                    // Field 3: other_case
                    if (fields.next()) |oc| {
                        other_case = std.fmt.parseInt(u32, oc, 10) catch return UnicharsetError.BadIntField;
                    }
                    // Remaining fields in short format are optional.
                    if (fields.next()) |d| {
                        direction = std.fmt.parseInt(u8, d, 10) catch return UnicharsetError.BadIntField;
                    }
                    if (fields.next()) |m| {
                        mirror = std.fmt.parseInt(u32, m, 10) catch return UnicharsetError.BadIntField;
                    }
                    if (fields.next()) |n| {
                        normed_str = if (std.mem.eql(u8, n, "NULL")) " " else n;
                    }
                }
            }

            // Allocate owned copies of strings.
            const owned_unichar = try allocator.dupe(u8, unichar_str);
            errdefer allocator.free(owned_unichar);
            const owned_script = try allocator.dupe(u8, script_str);
            errdefer allocator.free(owned_script);
            const owned_normed = try allocator.dupe(u8, normed_str);
            errdefer allocator.free(owned_normed);

            chars[idx] = UnicharEntry{
                .unichar = owned_unichar,
                .properties = properties,
                .script = owned_script,
                .other_case = other_case,
                .direction = direction,
                .mirror = mirror,
                .normed = owned_normed,
            };

            idx += 1;
            populated = idx;
        }

        if (idx < char_count) return UnicharsetError.TooFewEntries;

        return Unicharset{
            .chars = chars,
            .allocator = allocator,
        };
    }

    /// Release all owned memory.
    pub fn deinit(self: *Unicharset) void {
        for (self.chars) |*entry| {
            if (entry.unichar.len > 0) self.allocator.free(entry.unichar);
            if (entry.script.len > 0) self.allocator.free(entry.script);
            if (entry.normed.len > 0) self.allocator.free(entry.normed);
        }
        self.allocator.free(self.chars);
        self.chars = &.{};
    }

    /// Number of characters in this UNICHARSET.
    pub fn size(self: Unicharset) usize {
        return self.chars.len;
    }

    /// Map a UNICHAR_ID to its UTF-8 string representation.
    /// Returns an empty string if the ID is out of range.
    pub fn unicharToStr(self: Unicharset, id: u32) []const u8 {
        if (id >= self.chars.len) return "";
        return self.chars[id].unichar;
    }

    /// Returns true if the character at `id` has the alpha property.
    pub fn isAlpha(self: Unicharset, id: u32) bool {
        if (id >= self.chars.len) return false;
        return self.chars[id].properties & CHAR_PROP_ALPHA != 0;
    }

    /// Returns true if the character at `id` has the digit property.
    pub fn isDigit(self: Unicharset, id: u32) bool {
        if (id >= self.chars.len) return false;
        return self.chars[id].properties & CHAR_PROP_DIGIT != 0;
    }

    /// Returns true if the character at `id` has the upper-case property.
    pub fn isUpper(self: Unicharset, id: u32) bool {
        if (id >= self.chars.len) return false;
        return self.chars[id].properties & CHAR_PROP_UPPER != 0;
    }

    /// Returns true if the character at `id` has the lower-case property.
    pub fn isLower(self: Unicharset, id: u32) bool {
        if (id >= self.chars.len) return false;
        return self.chars[id].properties & CHAR_PROP_LOWER != 0;
    }

    /// Returns true if the character at `id` has the punctuation property.
    pub fn isPunct(self: Unicharset, id: u32) bool {
        if (id >= self.chars.len) return false;
        return self.chars[id].properties & CHAR_PROP_PUNCT != 0;
    }
};

// ── SquishedDawg ────────────────────────────────────────────────────────────

/// Errors specific to DAWG loading.
pub const DawgError = error{
    /// Magic number is not 42 (kDawgMagicNumber).
    BadMagicNumber,
    /// Data is too small to contain a valid DAWG header.
    DataTooSmall,
    /// The unicharset_size value is invalid (<= 0).
    BadUnicharsetSize,
    /// The num_edges value is invalid (<= 0).
    BadEdgeCount,
    /// The data is too small to hold all declared edges.
    TruncatedEdges,
    /// Memory allocation failed.
    OutOfMemory,
};

/// Magic number written at the start of a serialised DAWG.
const kDawgMagicNumber: i16 = 42;

/// Sentinel value meaning "no edge found".
const NO_EDGE: u64 = 0xFFFFFFFFFFFFFFFF;

/// Flag bit offsets within the 3-bit flags field.
const MARKER_FLAG: u64 = 1; // bit 0
const DIRECTION_FLAG: u64 = 2; // bit 1
const WERD_END_FLAG: u64 = 4; // bit 2

/// Number of flag bits packed into each edge record.
const NUM_FLAG_BITS: u6 = 3;

/// Parses and queries a Squished DAWG (Directed Acyclic Word Graph) used by
/// Tesseract OCR for dictionary-based word validation during beam search.
///
/// The DAWG stores words as paths through a graph of edges. Each edge is
/// packed into a u64 record containing a unichar ID, flags, and a pointer
/// to the next node. Node 0 (the root) has its edges sorted by unichar ID
/// for binary search; all other nodes use linear search.
///
/// Three DAWG types are used for LSTM inference:
///   - `lstm_punc_dawg` (punctuation patterns)
///   - `lstm_system_dawg` (system dictionary / word list)
///   - `lstm_number_dawg` (number patterns)
pub const SquishedDawg = struct {
    /// Flat array of edge records. Nodes are implicit: a node starts at a
    /// given index and its edges are consecutive until MARKER_FLAG is set.
    edges: []u64,
    unicharset_size: u32,

    // Computed bit masks derived from unicharset_size.
    letter_mask: u64,
    flags_mask: u64,
    next_node_mask: u64,
    flag_start_bit: u6,
    next_node_start_bit: u6,

    /// Number of forward edges in node 0 (for binary search bounds).
    num_forward_edges_in_node0: u64,

    allocator: std.mem.Allocator,

    /// Parse a SquishedDawg from raw binary data (as returned by
    /// TessdataManager.getComponent).
    ///
    /// Binary layout:
    ///   i16: magic number (42)
    ///   i32: unicharset_size
    ///   i32: num_edges
    ///   u64[num_edges]: edge records
    pub fn load(allocator: std.mem.Allocator, data: []const u8) DawgError!SquishedDawg {
        // Minimum size: 2 (magic) + 4 (unicharset_size) + 4 (num_edges) = 10
        if (data.len < 10) return DawgError.DataTooSmall;

        // Read magic number (i16, little-endian).
        const magic = std.mem.readInt(i16, data[0..2], .little);
        if (magic != kDawgMagicNumber) return DawgError.BadMagicNumber;

        // Read unicharset_size (i32, little-endian).
        const uc_size_i32 = std.mem.readInt(i32, data[2..6], .little);
        if (uc_size_i32 <= 0) return DawgError.BadUnicharsetSize;
        const uc_size: u32 = @intCast(uc_size_i32);

        // Read num_edges (i32, little-endian).
        const num_edges_i32 = std.mem.readInt(i32, data[6..10], .little);
        if (num_edges_i32 <= 0) return DawgError.BadEdgeCount;
        const num_edges: u32 = @intCast(num_edges_i32);

        // Check there is enough data for all edges.
        const edge_data_size: usize = @as(usize, num_edges) * 8;
        if (data.len < 10 + edge_data_size) return DawgError.TruncatedEdges;

        // Compute bit layout from unicharset_size.
        // flag_start_bit = ceil(log2(unicharset_size + 1))
        const flag_start_bit = computeFlagStartBit(uc_size);
        const next_node_start_bit: u6 = flag_start_bit + NUM_FLAG_BITS;

        const letter_mask: u64 = if (flag_start_bit >= 64) ~@as(u64, 0) else (@as(u64, 1) << flag_start_bit) -% 1;
        const next_node_mask: u64 = if (next_node_start_bit >= 64) 0 else ~@as(u64, 0) << next_node_start_bit;
        const flags_mask: u64 = ~(letter_mask | next_node_mask);

        // Allocate and copy edge records.
        const edges = allocator.alloc(u64, num_edges) catch return DawgError.OutOfMemory;
        errdefer allocator.free(edges);

        const edge_bytes = data[10 .. 10 + edge_data_size];
        for (0..num_edges) |i| {
            const start = i * 8;
            edges[i] = std.mem.readInt(u64, edge_bytes[start..][0..8], .little);
        }

        // Count forward edges in node 0.
        var dawg = SquishedDawg{
            .edges = edges,
            .unicharset_size = uc_size,
            .letter_mask = letter_mask,
            .flags_mask = flags_mask,
            .next_node_mask = next_node_mask,
            .flag_start_bit = flag_start_bit,
            .next_node_start_bit = next_node_start_bit,
            .num_forward_edges_in_node0 = 0,
            .allocator = allocator,
        };

        dawg.num_forward_edges_in_node0 = dawg.countForwardEdges(0);

        return dawg;
    }

    /// Release all owned memory.
    pub fn deinit(self: *SquishedDawg) void {
        self.allocator.free(self.edges);
        self.edges = &.{};
    }

    // ── Edge accessors ──────────────────────────────────────────────────

    /// Extract the UNICHAR_ID from an edge record.
    pub fn unicharId(self: SquishedDawg, edge: u64) u32 {
        return @intCast(edge & self.letter_mask);
    }

    /// Extract the next-node index from an edge record.
    pub fn nextNode(self: SquishedDawg, edge: u64) u64 {
        return (edge & self.next_node_mask) >> self.next_node_start_bit;
    }

    /// Returns true if this edge has the MARKER_FLAG set (last edge in node).
    pub fn isMarkerFlag(self: SquishedDawg, edge: u64) bool {
        return (edge & (MARKER_FLAG << self.flag_start_bit)) != 0;
    }

    /// Returns true if this edge has the DIRECTION_FLAG set (backward edge).
    pub fn isBackwardEdge(self: SquishedDawg, edge: u64) bool {
        return (edge & (DIRECTION_FLAG << self.flag_start_bit)) != 0;
    }

    /// Returns true if this edge has the WERD_END_FLAG set (word boundary).
    pub fn isWordEnd(self: SquishedDawg, edge: u64) bool {
        return (edge & (WERD_END_FLAG << self.flag_start_bit)) != 0;
    }

    /// Returns true if the edge slot is occupied (not an empty edge).
    /// An empty edge has the value next_node_mask (all next-node bits set,
    /// everything else zero).
    fn isEdgeOccupied(self: SquishedDawg, edge_ref: u64) bool {
        if (edge_ref >= self.edges.len) return false;
        return self.edges[@intCast(edge_ref)] != self.next_node_mask;
    }

    /// Returns true if this is a forward edge (occupied and direction = 0).
    fn isForwardEdge(self: SquishedDawg, edge_ref: u64) bool {
        if (!self.isEdgeOccupied(edge_ref)) return false;
        return !self.isBackwardEdge(self.edges[@intCast(edge_ref)]);
    }

    // ── Lookup ──────────────────────────────────────────────────────────

    /// Find an edge out of `node` with matching `unichar_id`.
    /// If `require_word_end` is true, the edge must also have WERD_END_FLAG.
    /// Returns the edge index, or null if not found.
    ///
    /// For node 0: uses binary search (edges sorted by unichar_id).
    /// For other nodes: linear scan until MARKER_FLAG.
    pub fn edgeCharOf(self: SquishedDawg, node: u64, unichar_id: u32, require_word_end: bool) ?u64 {
        if (node == 0) {
            // Binary search within the forward edges of node 0.
            if (self.num_forward_edges_in_node0 == 0) return null;
            var start: i64 = 0;
            var end: i64 = @as(i64, @intCast(self.num_forward_edges_in_node0)) - 1;

            while (start <= end) {
                const mid_idx: u64 = @intCast(@divTrunc(start + end, 2));
                const rec = self.edges[@intCast(mid_idx)];
                const edge_uid = self.unicharId(rec);

                if (unichar_id == edge_uid) {
                    // Unichar matches. Check word_end requirement.
                    if (!require_word_end or self.isWordEnd(rec)) {
                        return mid_idx;
                    }
                    // In Tesseract's binary search, if unichar matches but
                    // word_end doesn't, the comparison logic treats the query
                    // as "greater" so it searches right. But for simplicity,
                    // since there's typically only one edge per unichar_id in
                    // node 0, we do a linear scan around mid.
                    // Search left from mid for a match.
                    var probe: i64 = @as(i64, @intCast(mid_idx)) - 1;
                    while (probe >= 0) {
                        const prec = self.edges[@intCast(@as(u64, @intCast(probe)))];
                        if (self.unicharId(prec) != unichar_id) break;
                        if (!require_word_end or self.isWordEnd(prec)) {
                            return @intCast(@as(u64, @intCast(probe)));
                        }
                        probe -= 1;
                    }
                    // Search right from mid.
                    var probe_r: u64 = mid_idx + 1;
                    while (probe_r < self.num_forward_edges_in_node0) {
                        const prec = self.edges[@intCast(probe_r)];
                        if (self.unicharId(prec) != unichar_id) break;
                        if (!require_word_end or self.isWordEnd(prec)) {
                            return probe_r;
                        }
                        probe_r += 1;
                    }
                    return null;
                } else if (unichar_id > edge_uid) {
                    start = @as(i64, @intCast(mid_idx)) + 1;
                } else {
                    end = @as(i64, @intCast(mid_idx)) - 1;
                }
            }
            return null;
        } else {
            // Linear search for non-root nodes.
            var edge_ref = node;
            if (edge_ref == NO_EDGE or edge_ref >= self.edges.len) return null;
            if (!self.isEdgeOccupied(edge_ref)) return null;

            while (true) {
                const idx: usize = @intCast(edge_ref);
                const rec = self.edges[idx];
                if (self.unicharId(rec) == unichar_id) {
                    if (!require_word_end or self.isWordEnd(rec)) {
                        return edge_ref;
                    }
                }
                if (self.isMarkerFlag(rec)) break; // last edge in this node
                edge_ref += 1;
                if (edge_ref >= self.edges.len) break;
            }
            return null;
        }
    }

    /// Returns true if the given word (as a slice of unichar IDs) is in the DAWG.
    pub fn wordInDawg(self: SquishedDawg, word: []const u32) bool {
        if (word.len == 0) return false;

        var node: u64 = 0;
        for (word[0 .. word.len - 1]) |uid| {
            const edge_idx = self.edgeCharOf(node, uid, false) orelse return false;
            node = self.nextNode(self.edges[@intCast(edge_idx)]);
            if (node == 0) return false; // dead end: all words terminated
        }

        // Last character: require word_end flag.
        const last_uid = word[word.len - 1];
        return self.edgeCharOf(node, last_uid, true) != null;
    }

    /// Returns true if the given prefix is a valid start of some word in the DAWG.
    /// If the prefix has length 0, returns true (empty prefix matches anything).
    pub fn prefixInDawg(self: SquishedDawg, word: []const u32) bool {
        if (word.len == 0) return true;

        var node: u64 = 0;
        for (word[0 .. word.len - 1]) |uid| {
            const edge_idx = self.edgeCharOf(node, uid, false) orelse return false;
            node = self.nextNode(self.edges[@intCast(edge_idx)]);
            if (node == 0) return false;
        }

        // Last character: don't require word_end — just need the edge to exist.
        const last_uid = word[word.len - 1];
        return self.edgeCharOf(node, last_uid, false) != null;
    }

    /// Number of edges in this DAWG.
    pub fn numEdges(self: SquishedDawg) usize {
        return self.edges.len;
    }

    // ── Internal helpers ────────────────────────────────────────────────

    /// Count the number of consecutive forward edges starting at `node`.
    fn countForwardEdges(self: SquishedDawg, node: u64) u64 {
        var edge_ref = node;
        var count: u64 = 0;

        if (!self.isForwardEdge(edge_ref)) return 0;

        while (edge_ref < self.edges.len) {
            count += 1;
            if (self.isMarkerFlag(self.edges[@intCast(edge_ref)])) break;
            edge_ref += 1;
        }

        return count;
    }

    /// Compute flag_start_bit = ceil(log2(unicharset_size + 1)).
    /// Uses integer bit counting to avoid floating-point precision issues.
    fn computeFlagStartBit(uc_size: u32) u6 {
        // We need ceil(log2(uc_size + 1)).
        // If uc_size + 1 is a power of two, log2 is exact.
        // Otherwise, floor(log2) + 1 gives ceiling.
        const val: u64 = @as(u64, uc_size) + 1;
        if (val <= 1) return 1; // edge case: uc_size == 0

        // Number of bits needed to represent (val - 1), which equals floor(log2(val-1)) + 1.
        // For a power of two val, we need exactly log2(val) bits to represent val itself,
        // but ceil(log2(val)) = log2(val) when val is a power of two.
        //
        // std.math.log2_int gives floor(log2(x)) for x > 0.
        // ceil(log2(x)) = floor(log2(x-1)) + 1 when x > 1, or = floor(log2(x)) when x is pow2.
        const bits = 64 - @clz(val - 1);
        return @intCast(bits);
    }
};

// ── UnicharCompress ──────────────────────────────────────────────────────────

/// Training flag: model uses int8 quantized weights.
const TF_INT_MODE: i32 = 0x01;
/// Training flag: model has a compressed unicharset recoder.
const TF_COMPRESS_UNICHARSET: i32 = 0x40;

/// Recoder: maps UNICHARSET character IDs to network output codes.
/// In Tesseract, the recoder compresses the unicharset so that the network
/// output layer can be smaller than the full unicharset. Each unichar_id maps
/// to a short sequence of code values (a RecodedCharID).
pub const UnicharCompress = struct {
    entries: []RecodedCharID,
    allocator: std.mem.Allocator,

    pub const RecodedCharID = struct {
        self_normalized: bool,
        codes: []i32, // encoded character code sequence
    };

    /// Deserialize the recoder from the TESSDATA_LSTM component stream.
    /// Format: u32 entry_count, then for each entry:
    ///   i8 self_normalized, i32 length (0-9), i32[length] code values
    pub fn load(allocator: std.mem.Allocator, reader: *weights_mod.BinaryReader) !UnicharCompress {
        const entry_count = reader.readU32() catch return error.UnexpectedEof;

        const entries = allocator.alloc(RecodedCharID, entry_count) catch return error.OutOfMemory;
        var populated: u32 = 0;
        errdefer {
            for (entries[0..populated]) |*e| {
                if (e.codes.len > 0) allocator.free(e.codes);
            }
            allocator.free(entries);
        }

        for (0..entry_count) |i| {
            const self_norm_i8 = reader.readI8() catch return error.UnexpectedEof;
            const length = reader.readI32() catch return error.UnexpectedEof;
            if (length < 0 or length > 9) return error.InvalidRecoderEntry;

            const len: usize = @intCast(length);
            const codes = allocator.alloc(i32, len) catch return error.OutOfMemory;
            errdefer allocator.free(codes);

            for (0..len) |c| {
                codes[c] = reader.readI32() catch return error.UnexpectedEof;
            }

            entries[i] = RecodedCharID{
                .self_normalized = self_norm_i8 != 0,
                .codes = codes,
            };
            populated += 1;
        }

        return UnicharCompress{
            .entries = entries,
            .allocator = allocator,
        };
    }

    /// Release all owned memory.
    pub fn deinit(self: *UnicharCompress) void {
        for (self.entries) |*e| {
            if (e.codes.len > 0) self.allocator.free(e.codes);
        }
        self.allocator.free(self.entries);
        self.entries = &.{};
    }

    /// Encode a unichar_id to its recoded form.
    /// Returns null if the id is out of range.
    pub fn encode(self: UnicharCompress, unichar_id: u32) ?[]const i32 {
        if (unichar_id >= self.entries.len) return null;
        return self.entries[unichar_id].codes;
    }

    /// Number of entries.
    pub fn size(self: UnicharCompress) usize {
        return self.entries.len;
    }
};

// ── LSTMRecognizer ──────────────────────────────────────────────────────────

/// Errors specific to LSTMRecognizer loading.
pub const RecognizerError = error{
    /// The TESSDATA_LSTM component is missing from the .traineddata file.
    MissingLSTMComponent,
    /// The TESSDATA_LSTM_UNICHARSET component is missing.
    MissingUnicharset,
    /// A recoder entry has an invalid length (not 0-9).
    InvalidRecoderEntry,
    /// Network deserialization failed.
    NetworkDeserializeFailed,
    /// Unexpected end of data in the LSTM component.
    UnexpectedEof,
    /// Memory allocation failed.
    OutOfMemory,
};

/// Complete LSTM model ready for inference.
/// Ties together: network graph, unicharset, recoder, dictionaries, and params.
pub const LSTMRecognizer = struct {
    // Core components
    network: network_mod.Layer, // The neural network graph
    unicharset: Unicharset, // Character set

    // Recoder (maps unicharset IDs -> network output codes)
    recoder: ?UnicharCompress, // null if not compressed

    // Training metadata (useful for inference config)
    network_str: []const u8, // Network spec string
    training_flags: i32,
    null_char: i32, // Blank/CTC null character index

    // Optional dictionaries
    punc_dawg: ?SquishedDawg,
    system_dawg: ?SquishedDawg,
    number_dawg: ?SquishedDawg,

    allocator: std.mem.Allocator,

    /// Load complete model from .traineddata file data.
    pub fn load(allocator: std.mem.Allocator, traineddata: []const u8) !LSTMRecognizer {
        // 1. Parse the container
        const mgr = TessdataManager.init(traineddata) catch
            return error.MissingLSTMComponent;

        // 2. Get the LSTM component
        const lstm_data = mgr.getComponent(.lstm) orelse
            return error.MissingLSTMComponent;

        // 3. Create a reader over the LSTM component
        var reader = weights_mod.BinaryReader.init(lstm_data);

        // 4. Deserialize the network graph
        var net = network_mod.deserializeNetwork(allocator, &reader) catch
            return error.NetworkDeserializeFailed;
        errdefer network_mod.deinitLayer(allocator, &net);

        // 5. Read network_str (spec string)
        const net_str = reader.readString(allocator) catch
            return error.UnexpectedEof;
        errdefer allocator.free(net_str);

        // 6. Read training_flags
        const training_flags = reader.readI32() catch return error.UnexpectedEof;

        // 7. Read training_iteration (consumed, not stored)
        _ = reader.readI32() catch return error.UnexpectedEof;

        // 8. Read sample_iteration (consumed, not stored)
        _ = reader.readI32() catch return error.UnexpectedEof;

        // 9. Read null_char
        const null_char = reader.readI32() catch return error.UnexpectedEof;

        // 10. Read adam_beta (consumed, not stored)
        _ = reader.readF32() catch return error.UnexpectedEof;

        // 11. Read learning_rate (consumed, not stored)
        _ = reader.readF32() catch return error.UnexpectedEof;

        // 12. Read momentum (consumed, not stored)
        _ = reader.readF32() catch return error.UnexpectedEof;

        // 13-14. Load unicharset and recoder.
        // Modern .traineddata files have separate TESSDATA_LSTM_UNICHARSET and
        // TESSDATA_LSTM_RECODER components. Older files embed them inline in
        // the LSTM component stream. We detect which format by checking if the
        // separate components exist (matching Tesseract's include_charsets logic).
        const has_separate_charsets = mgr.hasComponent(.lstm_unicharset) and
            mgr.hasComponent(.lstm_recoder);

        var unicharset: Unicharset = undefined;
        var recoder: ?UnicharCompress = null;
        errdefer {
            if (recoder) |*r| r.deinit();
        }

        if (has_separate_charsets) {
            // Modern format: charsets are in separate components.
            // The LSTM stream ends after momentum (no inline unicharset/recoder).
            const unicharset_data = mgr.getComponent(.lstm_unicharset).?;
            unicharset = Unicharset.load(allocator, unicharset_data) catch
                return error.OutOfMemory;

            if ((training_flags & TF_COMPRESS_UNICHARSET) != 0) {
                const recoder_data = mgr.getComponent(.lstm_recoder).?;
                var recoder_reader = weights_mod.BinaryReader.init(recoder_data);
                recoder = UnicharCompress.load(allocator, &recoder_reader) catch
                    return error.UnexpectedEof;
            }
        } else {
            // Legacy format: unicharset was inline (already consumed by network
            // deserializer or needs to be read here). For now we only support
            // models with separate components.
            if (mgr.getComponent(.lstm_unicharset)) |unicharset_data| {
                unicharset = Unicharset.load(allocator, unicharset_data) catch
                    return error.OutOfMemory;
            } else {
                return error.MissingUnicharset;
            }

            // Inline recoder follows momentum in the LSTM stream.
            if ((training_flags & TF_COMPRESS_UNICHARSET) != 0) {
                recoder = UnicharCompress.load(allocator, &reader) catch
                    return error.UnexpectedEof;
            }
        }
        errdefer unicharset.deinit();

        // 15-17. Load optional DAWG dictionaries
        var punc_dawg: ?SquishedDawg = null;
        errdefer {
            if (punc_dawg) |*d| d.deinit();
        }
        if (mgr.getComponent(.lstm_punc_dawg)) |dawg_data| {
            punc_dawg = SquishedDawg.load(allocator, dawg_data) catch null;
        }

        var system_dawg: ?SquishedDawg = null;
        errdefer {
            if (system_dawg) |*d| d.deinit();
        }
        if (mgr.getComponent(.lstm_system_dawg)) |dawg_data| {
            system_dawg = SquishedDawg.load(allocator, dawg_data) catch null;
        }

        var number_dawg: ?SquishedDawg = null;
        errdefer {
            if (number_dawg) |*d| d.deinit();
        }
        if (mgr.getComponent(.lstm_number_dawg)) |dawg_data| {
            number_dawg = SquishedDawg.load(allocator, dawg_data) catch null;
        }

        return LSTMRecognizer{
            .network = net,
            .unicharset = unicharset,
            .recoder = recoder,
            .network_str = net_str,
            .training_flags = training_flags,
            .null_char = null_char,
            .punc_dawg = punc_dawg,
            .system_dawg = system_dawg,
            .number_dawg = number_dawg,
            .allocator = allocator,
        };
    }

    /// Free all resources.
    pub fn deinit(self: *LSTMRecognizer) void {
        network_mod.deinitLayer(self.allocator, &self.network);
        self.unicharset.deinit();
        if (self.recoder) |*r| r.deinit();
        self.allocator.free(self.network_str);
        if (self.punc_dawg) |*d| d.deinit();
        if (self.system_dawg) |*d| d.deinit();
        if (self.number_dawg) |*d| d.deinit();
    }

    /// Number of output classes (from network's output layer).
    pub fn numClasses(self: LSTMRecognizer) usize {
        return self.network.numOutputs();
    }

    /// Number of input features (from network's input layer).
    pub fn numInputs(self: LSTMRecognizer) usize {
        return self.network.numInputs();
    }

    /// Whether the model uses int8 quantized weights.
    pub fn isIntMode(self: LSTMRecognizer) bool {
        return (self.training_flags & TF_INT_MODE) != 0;
    }
};

// ── Tests ────────────────────────────────────────────────────────────────────

test "parse eng.traineddata from file" {
    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: cannot open test file: {}\n", .{err});
        return;
    };
    defer file.close();

    const data = file.readToEndAlloc(std.testing.allocator, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: cannot read test file: {}\n", .{err});
        return;
    };
    defer std.testing.allocator.free(data);

    const mgr = try TessdataManager.init(data);

    // Verify entry count.
    try std.testing.expectEqual(@as(usize, 24), mgr.numEntries());

    // Verify not big-endian.
    try std.testing.expect(!mgr.is_big_endian);

    // LSTM component should be present and have size > 0.
    const lstm = mgr.getComponent(.lstm);
    try std.testing.expect(lstm != null);
    try std.testing.expect(lstm.?.len > 0);
    // Exact size: next offset (401832) - this offset (196) = 401636
    try std.testing.expectEqual(@as(usize, 401636), lstm.?.len);

    // LSTM_UNICHARSET should be present.
    const lstm_uni = mgr.getComponent(.lstm_unicharset);
    try std.testing.expect(lstm_uni != null);
    try std.testing.expect(lstm_uni.?.len > 0);
    // Exact size: 4112046 - 4105686 = 6360
    try std.testing.expectEqual(@as(usize, 6360), lstm_uni.?.len);

    // LSTM_RECODER should be present.
    const lstm_rec = mgr.getComponent(.lstm_recoder);
    try std.testing.expect(lstm_rec != null);
    try std.testing.expectEqual(@as(usize, 1012), lstm_rec.?.len);

    // VERSION component should be present with known size.
    const ver = mgr.getComponent(.version);
    try std.testing.expect(ver != null);
    try std.testing.expectEqual(@as(usize, 30), ver.?.len);

    // Legacy unicharset (index 1) should be absent.
    try std.testing.expect(mgr.getComponent(.unicharset) == null);

    // Legacy inttemp should be absent.
    try std.testing.expect(mgr.getComponent(.inttemp) == null);

    // All legacy entries (0-16) should be absent.
    try std.testing.expect(!mgr.hasComponent(.lang_config));
    try std.testing.expect(!mgr.hasComponent(.ambigs));
    try std.testing.expect(!mgr.hasComponent(.normproto));
    try std.testing.expect(!mgr.hasComponent(.shape_table));
    try std.testing.expect(!mgr.hasComponent(.params_model));

    // All LSTM entries (17-23) should be present.
    try std.testing.expect(mgr.hasComponent(.lstm));
    try std.testing.expect(mgr.hasComponent(.lstm_punc_dawg));
    try std.testing.expect(mgr.hasComponent(.lstm_system_dawg));
    try std.testing.expect(mgr.hasComponent(.lstm_number_dawg));
    try std.testing.expect(mgr.hasComponent(.lstm_unicharset));
    try std.testing.expect(mgr.hasComponent(.lstm_recoder));
    try std.testing.expect(mgr.hasComponent(.version));

    // 7 present components total.
    try std.testing.expectEqual(@as(usize, 7), mgr.numPresent());
}

test "parse minimal hand-crafted traineddata" {
    // Build a minimal container with 2 entries present: lstm (17) and version (23).
    var buf: [header_size + 16]u8 = undefined;

    // num_entries = 24, little-endian.
    std.mem.writeInt(u32, buf[0..4], 24, .little);

    // Offset table: all -1 except entries 17 and 23.
    for (0..TessdataType.count) |i| {
        const start = 4 + i * 8;
        std.mem.writeInt(i64, buf[start..][0..8], -1, .little);
    }

    // Entry 17 (lstm) at offset header_size, 8 bytes of data.
    const lstm_offset: i64 = @intCast(header_size);
    std.mem.writeInt(i64, buf[4 + 17 * 8 ..][0..8], lstm_offset, .little);

    // Entry 23 (version) at offset header_size + 8, 8 bytes of data.
    const ver_offset: i64 = @intCast(header_size + 8);
    std.mem.writeInt(i64, buf[4 + 23 * 8 ..][0..8], ver_offset, .little);

    // Fill entry data with recognizable bytes.
    @memset(buf[header_size .. header_size + 8], 0xAA);
    @memset(buf[header_size + 8 .. header_size + 16], 0xBB);

    const mgr = try TessdataManager.init(&buf);

    try std.testing.expectEqual(@as(usize, 24), mgr.numEntries());
    try std.testing.expectEqual(@as(usize, 2), mgr.numPresent());
    try std.testing.expect(!mgr.is_big_endian);

    // LSTM present with correct data.
    const lstm = mgr.getComponent(.lstm).?;
    try std.testing.expectEqual(@as(usize, 8), lstm.len);
    try std.testing.expectEqual(@as(u8, 0xAA), lstm[0]);

    // VERSION present with correct data.
    const ver = mgr.getComponent(.version).?;
    try std.testing.expectEqual(@as(usize, 8), ver.len);
    try std.testing.expectEqual(@as(u8, 0xBB), ver[0]);

    // Others absent.
    try std.testing.expect(mgr.getComponent(.unicharset) == null);
    try std.testing.expect(mgr.getComponent(.lang_config) == null);
}

test "big-endian byte-swap path" {
    // Build a container where num_entries is stored big-endian (= byte-swapped 24).
    var buf: [header_size + 10]u8 = undefined;

    // num_entries = 24 in big-endian = 0x18000000 -> stored as 00 00 00 18.
    std.mem.writeInt(u32, buf[0..4], 24, .big);

    // Offset table: all -1 in big-endian.
    for (0..TessdataType.count) |i| {
        const start = 4 + i * 8;
        std.mem.writeInt(i64, buf[start..][0..8], -1, .big);
    }

    // Entry 17 (lstm) at offset header_size, 10 bytes (rest of file).
    const lstm_offset: i64 = @intCast(header_size);
    std.mem.writeInt(i64, buf[4 + 17 * 8 ..][0..8], lstm_offset, .big);

    // Fill data region.
    @memset(buf[header_size..], 0xCC);

    const mgr = try TessdataManager.init(&buf);

    try std.testing.expect(mgr.is_big_endian);
    try std.testing.expectEqual(@as(usize, 24), mgr.numEntries());

    const lstm = mgr.getComponent(.lstm).?;
    try std.testing.expectEqual(@as(usize, 10), lstm.len);
    try std.testing.expectEqual(@as(u8, 0xCC), lstm[0]);
}

test "file too small returns error" {
    const tiny = [_]u8{ 0x18, 0x00, 0x00 }; // only 3 bytes
    try std.testing.expectError(TessdataError.FileTooSmall, TessdataManager.init(&tiny));
}

test "bad entry count returns error" {
    // num_entries = 25 (not 24).
    var buf: [header_size]u8 = undefined;
    @memset(&buf, 0xFF);
    std.mem.writeInt(u32, buf[0..4], 25, .little);
    try std.testing.expectError(TessdataError.BadEntryCount, TessdataManager.init(&buf));
}

test "offset past end of file returns error" {
    var buf: [header_size]u8 = undefined;
    std.mem.writeInt(u32, buf[0..4], 24, .little);

    // All entries absent.
    for (0..TessdataType.count) |i| {
        const start = 4 + i * 8;
        std.mem.writeInt(i64, buf[start..][0..8], -1, .little);
    }

    // Set entry 17 to an offset past EOF.
    std.mem.writeInt(i64, buf[4 + 17 * 8 ..][0..8], 99999, .little);

    try std.testing.expectError(TessdataError.OffsetOutOfRange, TessdataManager.init(&buf));
}

test "TessdataType enum values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(TessdataType.lang_config));
    try std.testing.expectEqual(@as(u8, 17), @intFromEnum(TessdataType.lstm));
    try std.testing.expectEqual(@as(u8, 21), @intFromEnum(TessdataType.lstm_unicharset));
    try std.testing.expectEqual(@as(u8, 23), @intFromEnum(TessdataType.version));
}

// ── Unicharset Tests ────────────────────────────────────────────────────────

test "load unicharset from eng.traineddata" {
    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: cannot open test file: {}\n", .{err});
        return;
    };
    defer file.close();

    const data = file.readToEndAlloc(std.testing.allocator, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: cannot read test file: {}\n", .{err});
        return;
    };
    defer std.testing.allocator.free(data);

    const mgr = try TessdataManager.init(data);
    const unicharset_data = mgr.getComponent(.lstm_unicharset).?;

    var uc = try Unicharset.load(std.testing.allocator, unicharset_data);
    defer uc.deinit();

    // 112 characters in eng.traineddata LSTM unicharset.
    try std.testing.expectEqual(@as(usize, 112), uc.size());

    // ID 0: NULL → space character.
    try std.testing.expectEqualStrings(" ", uc.unicharToStr(0));

    // ID 3: "C" (uppercase letter).
    try std.testing.expectEqualStrings("C", uc.unicharToStr(3));
    try std.testing.expect(uc.isAlpha(3));
    try std.testing.expect(uc.isUpper(3));
    try std.testing.expect(!uc.isLower(3));
    try std.testing.expect(!uc.isDigit(3));
    try std.testing.expect(!uc.isPunct(3));

    // ID 3: script should be "Latin".
    try std.testing.expectEqualStrings("Latin", uc.chars[3].script);

    // ID 3: other_case should be 87 (lowercase 'c' equivalent).
    try std.testing.expectEqual(@as(u32, 87), uc.chars[3].other_case);

    // ID 7 ("-"): punctuation.
    // Properties hex 10 = 0x10 = bit4 = punct.
    try std.testing.expectEqualStrings("-", uc.unicharToStr(7));
    try std.testing.expect(uc.isPunct(7));
    try std.testing.expect(!uc.isAlpha(7));
    try std.testing.expect(!uc.isDigit(7));

    // ID 14 ("8"): digit.
    // Properties hex 8 = 0x08 = bit3 = digit.
    try std.testing.expectEqualStrings("8", uc.unicharToStr(14));
    try std.testing.expect(uc.isDigit(14));
    try std.testing.expect(!uc.isAlpha(14));

    // ID 1 ("Joined"): special entry, properties 0x07 = alpha+lower+upper.
    try std.testing.expectEqualStrings("Joined", uc.unicharToStr(1));
    try std.testing.expect(uc.isAlpha(1));

    // ID 111 ("é"): lowercase alpha.
    try std.testing.expectEqualStrings("\xc3\xa9", uc.unicharToStr(111)); // é in UTF-8
    try std.testing.expect(uc.isAlpha(111));
    try std.testing.expect(uc.isLower(111));

    // Out-of-range ID returns empty string.
    try std.testing.expectEqualStrings("", uc.unicharToStr(999));
    try std.testing.expect(!uc.isAlpha(999));
}

test "unicharset property bit flags" {
    // Synthetic unicharset to verify each property bit independently.
    const data =
        "5\n" ++
        "NULL 0 Common 0\n" ++ // ID 0: space, no properties
        "a 3 0,255,0,255,0,0,0,0,0,0 Latin 0 0 0 a\n" ++ // ID 1: alpha+lower
        "A 5 0,255,0,255,0,0,0,0,0,0 Latin 1 0 1 A\n" ++ // ID 2: alpha+upper
        "7 8 0,255,0,255,0,0,0,0,0,0 Common 3 2 3 7\n" ++ // ID 3: digit
        ". 10 0,255,0,255,0,0,0,0,0,0 Common 4 6 4 .\n"; // ID 4: punct

    var uc = try Unicharset.load(std.testing.allocator, data);
    defer uc.deinit();

    try std.testing.expectEqual(@as(usize, 5), uc.size());

    // ID 0: space - no properties.
    try std.testing.expectEqualStrings(" ", uc.unicharToStr(0));
    try std.testing.expect(!uc.isAlpha(0));
    try std.testing.expect(!uc.isLower(0));
    try std.testing.expect(!uc.isUpper(0));
    try std.testing.expect(!uc.isDigit(0));
    try std.testing.expect(!uc.isPunct(0));

    // ID 1: "a" - alpha + lower.
    try std.testing.expect(uc.isAlpha(1));
    try std.testing.expect(uc.isLower(1));
    try std.testing.expect(!uc.isUpper(1));

    // ID 2: "A" - alpha + upper.
    try std.testing.expect(uc.isAlpha(2));
    try std.testing.expect(uc.isUpper(2));
    try std.testing.expect(!uc.isLower(2));

    // ID 3: "7" - digit.
    try std.testing.expect(uc.isDigit(3));
    try std.testing.expect(!uc.isAlpha(3));

    // ID 4: "." - punct.
    try std.testing.expect(uc.isPunct(4));
    try std.testing.expect(!uc.isAlpha(4));

    // Verify other_case linkage.
    try std.testing.expectEqual(@as(u32, 0), uc.chars[1].other_case); // 'a' -> 0
    try std.testing.expectEqual(@as(u32, 1), uc.chars[2].other_case); // 'A' -> 1
}

test "unicharset empty set" {
    const data = "0\n";
    var uc = try Unicharset.load(std.testing.allocator, data);
    defer uc.deinit();

    try std.testing.expectEqual(@as(usize, 0), uc.size());
    try std.testing.expectEqualStrings("", uc.unicharToStr(0));
    try std.testing.expect(!uc.isAlpha(0));
}

test "unicharset single entry" {
    const data = "1\nNULL 0 Common 0\n";
    var uc = try Unicharset.load(std.testing.allocator, data);
    defer uc.deinit();

    try std.testing.expectEqual(@as(usize, 1), uc.size());
    try std.testing.expectEqualStrings(" ", uc.unicharToStr(0));
}

test "unicharset short format lines" {
    // Lines with just unichar + properties + script + other_case (no metrics).
    const data =
        "2\n" ++
        "NULL 0 Common 0\n" ++
        "X 5 Latin 0\n";

    var uc = try Unicharset.load(std.testing.allocator, data);
    defer uc.deinit();

    try std.testing.expectEqual(@as(usize, 2), uc.size());
    try std.testing.expectEqualStrings(" ", uc.unicharToStr(0));
    try std.testing.expectEqualStrings("X", uc.unicharToStr(1));
    try std.testing.expect(uc.isAlpha(1));
    try std.testing.expect(uc.isUpper(1));
    try std.testing.expectEqualStrings("Latin", uc.chars[1].script);
}

test "unicharset error on empty data" {
    try std.testing.expectError(UnicharsetError.EmptyData, Unicharset.load(std.testing.allocator, ""));
}

test "unicharset error on bad count" {
    try std.testing.expectError(UnicharsetError.BadCharCount, Unicharset.load(std.testing.allocator, "abc\n"));
}

test "unicharset error on count mismatch" {
    // Declared 3 but only 1 line of data.
    try std.testing.expectError(UnicharsetError.TooFewEntries, Unicharset.load(std.testing.allocator, "3\nNULL 0 Common 0\n"));
}

test "unicharset normed field and comment stripping" {
    const data =
        "1\n" ++
        "C 5 0,255,0,255,0,0,0,0,0,0 Latin 87 0 3 C\t# C [43 ]A\n";

    var uc = try Unicharset.load(std.testing.allocator, data);
    defer uc.deinit();

    try std.testing.expectEqualStrings("C", uc.chars[0].unichar);
    try std.testing.expectEqualStrings("C", uc.chars[0].normed);
    try std.testing.expectEqualStrings("Latin", uc.chars[0].script);
    try std.testing.expectEqual(@as(u32, 87), uc.chars[0].other_case);
    try std.testing.expectEqual(@as(u8, 0), uc.chars[0].direction);
    try std.testing.expectEqual(@as(u32, 3), uc.chars[0].mirror);
}

// ── SquishedDawg Tests ──────────────────────────────────────────────────────

test "SquishedDawg computeFlagStartBit" {
    // ceil(log2(1)) = 0 ... but we need at least 1 bit for unichar 0
    // Actually: computeFlagStartBit(0) -> ceil(log2(0+1)) = ceil(log2(1)) = 0
    // But the code returns 1 for the edge case of val <= 1.
    // For uc_size=0, val=1: clz-based formula gives 0, but we clamp to 1.
    try std.testing.expectEqual(@as(u6, 1), SquishedDawg.computeFlagStartBit(0));

    // ceil(log2(2)) = 1
    try std.testing.expectEqual(@as(u6, 1), SquishedDawg.computeFlagStartBit(1));

    // ceil(log2(3)) = 2
    try std.testing.expectEqual(@as(u6, 2), SquishedDawg.computeFlagStartBit(2));

    // ceil(log2(4)) = 2
    try std.testing.expectEqual(@as(u6, 2), SquishedDawg.computeFlagStartBit(3));

    // ceil(log2(5)) = 3
    try std.testing.expectEqual(@as(u6, 3), SquishedDawg.computeFlagStartBit(4));

    // ceil(log2(9)) = 4
    try std.testing.expectEqual(@as(u6, 4), SquishedDawg.computeFlagStartBit(8));

    // ceil(log2(113)) = 7 (typical for eng.traineddata with 112 chars)
    try std.testing.expectEqual(@as(u6, 7), SquishedDawg.computeFlagStartBit(112));

    // ceil(log2(129)) = 8 (unicharset_size = 128)
    try std.testing.expectEqual(@as(u6, 8), SquishedDawg.computeFlagStartBit(128));

    // Cross-check with Tesseract's floating-point formula.
    // Tesseract: ceil(log(113.0) / log(2.0)) = ceil(6.820...) = 7
    try std.testing.expectEqual(@as(u6, 7), SquishedDawg.computeFlagStartBit(112));
}

test "SquishedDawg edge accessors on synthetic record" {
    // Create a minimal DAWG with unicharset_size = 7 (need 3 bits for IDs 0-7).
    // flag_start_bit = ceil(log2(8)) = 3
    // Bits: [0..2] = unichar_id, [3..5] = flags, [6..63] = next_node
    //
    // Build edge: unichar_id=5, flags=MARKER|WERD_END (0b101=5), next_node=42
    // edge = 5 | (5 << 3) | (42 << 6) = 5 | 40 | 2688 = 2733
    const flag_start: u6 = 3;
    const next_start: u6 = 6;
    const letter_mask: u64 = (1 << flag_start) - 1; // 0b111 = 7
    const next_node_mask: u64 = ~@as(u64, 0) << next_start;
    const flags_mask: u64 = ~(letter_mask | next_node_mask);

    const edge_val: u64 = 5 | (@as(u64, 5) << flag_start) | (@as(u64, 42) << next_start);

    // Build a 1-edge array for the struct.
    var edge_arr = [_]u64{edge_val};

    const dawg = SquishedDawg{
        .edges = &edge_arr,
        .unicharset_size = 7,
        .letter_mask = letter_mask,
        .flags_mask = flags_mask,
        .next_node_mask = next_node_mask,
        .flag_start_bit = flag_start,
        .next_node_start_bit = next_start,
        .num_forward_edges_in_node0 = 1,
        .allocator = std.testing.allocator,
    };

    try std.testing.expectEqual(@as(u32, 5), dawg.unicharId(edge_val));
    try std.testing.expectEqual(@as(u64, 42), dawg.nextNode(edge_val));
    try std.testing.expect(dawg.isMarkerFlag(edge_val)); // MARKER_FLAG (bit 0 of flags)
    try std.testing.expect(!dawg.isBackwardEdge(edge_val)); // DIRECTION_FLAG (bit 1) is 0
    try std.testing.expect(dawg.isWordEnd(edge_val)); // WERD_END_FLAG (bit 2 of flags)
}

test "SquishedDawg synthetic word lookup" {
    // Build a tiny DAWG that contains the word [1, 2, 3] (unichar IDs).
    // unicharset_size = 7 -> flag_start_bit = 3, next_node_start_bit = 6
    //
    // Graph structure:
    //   Node 0 (edge 0): uid=1, next_node=1, marker=1, word_end=0
    //   Node 1 (edge 1): uid=2, next_node=2, marker=1, word_end=0
    //   Node 2 (edge 2): uid=3, next_node=0, marker=1, word_end=1
    //
    const flag_start: u6 = 3;
    const next_start: u6 = 6;
    const letter_mask: u64 = (1 << flag_start) - 1;
    const next_node_mask: u64 = ~@as(u64, 0) << next_start;
    const flags_mask: u64 = ~(letter_mask | next_node_mask);

    // Helper to build an edge record.
    const mkEdge = struct {
        fn f(uid: u64, next: u64, marker: bool, word_end: bool) u64 {
            var flags: u64 = 0;
            if (marker) flags |= MARKER_FLAG;
            if (word_end) flags |= WERD_END_FLAG;
            return uid | (flags << flag_start) | (next << next_start);
        }
    }.f;

    var edges = [_]u64{
        mkEdge(1, 1, true, false), // node 0, edge 0: uid=1 -> node 1
        mkEdge(2, 2, true, false), // node 1, edge 1: uid=2 -> node 2
        mkEdge(3, 0, true, true), //  node 2, edge 2: uid=3, word_end
    };

    var dawg = SquishedDawg{
        .edges = &edges,
        .unicharset_size = 7,
        .letter_mask = letter_mask,
        .flags_mask = flags_mask,
        .next_node_mask = next_node_mask,
        .flag_start_bit = flag_start,
        .next_node_start_bit = next_start,
        .num_forward_edges_in_node0 = 1,
        .allocator = std.testing.allocator,
    };

    // The word [1, 2, 3] should be found.
    const word_good = [_]u32{ 1, 2, 3 };
    try std.testing.expect(dawg.wordInDawg(&word_good));

    // The prefix [1, 2] should be in the DAWG.
    const prefix = [_]u32{ 1, 2 };
    try std.testing.expect(dawg.prefixInDawg(&prefix));

    // The word [1, 2] is NOT a complete word (no word_end on uid=2).
    try std.testing.expect(!dawg.wordInDawg(&prefix));

    // A word [1, 2, 4] should not be found (uid=4 not at node 2).
    const word_bad = [_]u32{ 1, 2, 4 };
    try std.testing.expect(!dawg.wordInDawg(&word_bad));

    // A word [5] should not be found (uid=5 not at node 0).
    const word_missing = [_]u32{5};
    try std.testing.expect(!dawg.wordInDawg(&word_missing));

    // Empty word should not be found.
    const word_empty: []const u32 = &.{};
    try std.testing.expect(!dawg.wordInDawg(word_empty));
}

test "SquishedDawg synthetic multi-edge node" {
    // Build a DAWG with two words: [1, 3] and [2, 4]
    // Both start from node 0, which has 2 edges.
    // unicharset_size = 7 -> flag_start_bit = 3, next_node_start_bit = 6
    //
    // Node 0 (edges 0-1, sorted by uid for binary search):
    //   edge 0: uid=1, next_node=2, marker=0
    //   edge 1: uid=2, next_node=3, marker=1
    // Node at index 2 (edge 2):
    //   edge 2: uid=3, next_node=0, marker=1, word_end=1
    // Node at index 3 (edge 3):
    //   edge 3: uid=4, next_node=0, marker=1, word_end=1

    const flag_start: u6 = 3;
    const next_start: u6 = 6;
    const letter_mask: u64 = (1 << flag_start) - 1;
    const next_node_mask: u64 = ~@as(u64, 0) << next_start;
    const flags_mask: u64 = ~(letter_mask | next_node_mask);

    const mkEdge = struct {
        fn f(uid: u64, next: u64, marker: bool, word_end: bool) u64 {
            var flags: u64 = 0;
            if (marker) flags |= MARKER_FLAG;
            if (word_end) flags |= WERD_END_FLAG;
            return uid | (flags << flag_start) | (next << next_start);
        }
    }.f;

    var edges = [_]u64{
        mkEdge(1, 2, false, false), // node 0, edge 0
        mkEdge(2, 3, true, false), //  node 0, edge 1 (last in node)
        mkEdge(3, 0, true, true), //   node at 2
        mkEdge(4, 0, true, true), //   node at 3
    };

    var dawg = SquishedDawg{
        .edges = &edges,
        .unicharset_size = 7,
        .letter_mask = letter_mask,
        .flags_mask = flags_mask,
        .next_node_mask = next_node_mask,
        .flag_start_bit = flag_start,
        .next_node_start_bit = next_start,
        .num_forward_edges_in_node0 = 2,
        .allocator = std.testing.allocator,
    };

    // Word [1, 3] should be found.
    const word1 = [_]u32{ 1, 3 };
    try std.testing.expect(dawg.wordInDawg(&word1));

    // Word [2, 4] should be found.
    const word2 = [_]u32{ 2, 4 };
    try std.testing.expect(dawg.wordInDawg(&word2));

    // Word [1, 4] should NOT be found (uid=4 not at node 2).
    const bad = [_]u32{ 1, 4 };
    try std.testing.expect(!dawg.wordInDawg(&bad));

    // Word [2, 3] should NOT be found (uid=3 not at node 3).
    const bad2 = [_]u32{ 2, 3 };
    try std.testing.expect(!dawg.wordInDawg(&bad2));
}

test "SquishedDawg load error cases" {
    // Too small.
    const tiny = [_]u8{ 0x2A, 0x00 };
    try std.testing.expectError(DawgError.DataTooSmall, SquishedDawg.load(std.testing.allocator, &tiny));

    // Bad magic.
    var bad_magic: [18]u8 = undefined;
    std.mem.writeInt(i16, bad_magic[0..2], 99, .little);
    std.mem.writeInt(i32, bad_magic[2..6], 10, .little);
    std.mem.writeInt(i32, bad_magic[6..10], 1, .little);
    @memset(bad_magic[10..18], 0);
    try std.testing.expectError(DawgError.BadMagicNumber, SquishedDawg.load(std.testing.allocator, &bad_magic));

    // Bad unicharset_size (0).
    var bad_uc: [18]u8 = undefined;
    std.mem.writeInt(i16, bad_uc[0..2], 42, .little);
    std.mem.writeInt(i32, bad_uc[2..6], 0, .little);
    std.mem.writeInt(i32, bad_uc[6..10], 1, .little);
    @memset(bad_uc[10..18], 0);
    try std.testing.expectError(DawgError.BadUnicharsetSize, SquishedDawg.load(std.testing.allocator, &bad_uc));

    // Bad num_edges (0).
    var bad_ne: [10]u8 = undefined;
    std.mem.writeInt(i16, bad_ne[0..2], 42, .little);
    std.mem.writeInt(i32, bad_ne[2..6], 10, .little);
    std.mem.writeInt(i32, bad_ne[6..10], 0, .little);
    try std.testing.expectError(DawgError.BadEdgeCount, SquishedDawg.load(std.testing.allocator, &bad_ne));

    // Truncated edges: declares 2 edges but only has room for 1.
    var truncated: [18]u8 = undefined;
    std.mem.writeInt(i16, truncated[0..2], 42, .little);
    std.mem.writeInt(i32, truncated[2..6], 10, .little);
    std.mem.writeInt(i32, truncated[6..10], 2, .little); // 2 edges = 16 bytes needed
    @memset(truncated[10..18], 0); // only 8 bytes of edge data
    try std.testing.expectError(DawgError.TruncatedEdges, SquishedDawg.load(std.testing.allocator, &truncated));
}

test "SquishedDawg load synthetic binary" {
    // Build a binary DAWG blob manually and verify load + lookup.
    // Word: [1, 2] with unicharset_size=4 -> flag_start_bit = ceil(log2(5)) = 3
    //
    // Node 0 (edge 0): uid=1, next_node=1, marker=1, word_end=0
    // Node 1 (edge 1): uid=2, next_node=0, marker=1, word_end=1

    const flag_start: u6 = 3;
    const next_start: u6 = 6;

    const edge0: u64 = 1 | (@as(u64, MARKER_FLAG) << flag_start) | (@as(u64, 1) << next_start);
    const edge1: u64 = 2 | (@as(u64, MARKER_FLAG | WERD_END_FLAG) << flag_start) | (@as(u64, 0) << next_start);

    // Header: i16 magic(42), i32 uc_size(4), i32 num_edges(2)
    // Then 2 x u64 edges.
    var buf: [10 + 16]u8 = undefined;
    std.mem.writeInt(i16, buf[0..2], 42, .little);
    std.mem.writeInt(i32, buf[2..6], 4, .little);
    std.mem.writeInt(i32, buf[6..10], 2, .little);
    std.mem.writeInt(u64, buf[10..18], edge0, .little);
    std.mem.writeInt(u64, buf[18..26], edge1, .little);

    var dawg = try SquishedDawg.load(std.testing.allocator, &buf);
    defer dawg.deinit();

    try std.testing.expectEqual(@as(usize, 2), dawg.numEdges());
    try std.testing.expectEqual(@as(u32, 4), dawg.unicharset_size);
    try std.testing.expectEqual(@as(u6, 3), dawg.flag_start_bit);
    try std.testing.expectEqual(@as(u6, 6), dawg.next_node_start_bit);

    // Verify edge accessors on loaded data.
    try std.testing.expectEqual(@as(u32, 1), dawg.unicharId(dawg.edges[0]));
    try std.testing.expectEqual(@as(u64, 1), dawg.nextNode(dawg.edges[0]));
    try std.testing.expect(dawg.isMarkerFlag(dawg.edges[0]));
    try std.testing.expect(!dawg.isWordEnd(dawg.edges[0]));

    try std.testing.expectEqual(@as(u32, 2), dawg.unicharId(dawg.edges[1]));
    try std.testing.expect(dawg.isWordEnd(dawg.edges[1]));

    // Word lookup.
    const word = [_]u32{ 1, 2 };
    try std.testing.expect(dawg.wordInDawg(&word));

    const bad_word = [_]u32{ 1, 3 };
    try std.testing.expect(!dawg.wordInDawg(&bad_word));
}

test "SquishedDawg load from eng.traineddata" {
    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: cannot open test file: {}\n", .{err});
        return;
    };
    defer file.close();

    const data = file.readToEndAlloc(std.testing.allocator, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: cannot read test file: {}\n", .{err});
        return;
    };
    defer std.testing.allocator.free(data);

    const mgr = try TessdataManager.init(data);

    // Load all three LSTM DAWG types.
    const dawg_types = [_]TessdataType{
        .lstm_punc_dawg,
        .lstm_system_dawg,
        .lstm_number_dawg,
    };
    const dawg_names = [_][]const u8{
        "lstm_punc_dawg",
        "lstm_system_dawg",
        "lstm_number_dawg",
    };

    for (dawg_types, dawg_names) |dt, name| {
        const comp = mgr.getComponent(dt) orelse {
            std.debug.print("Skipping {s}: component not present\n", .{name});
            continue;
        };

        var dawg = try SquishedDawg.load(std.testing.allocator, comp);
        defer dawg.deinit();

        // All DAWGs should have > 0 edges.
        try std.testing.expect(dawg.numEdges() > 0);
        // The unicharset_size should match expected value for eng (112).
        try std.testing.expectEqual(@as(u32, 112), dawg.unicharset_size);
        // flag_start_bit for 112 chars = ceil(log2(113)) = 7
        try std.testing.expectEqual(@as(u6, 7), dawg.flag_start_bit);
        try std.testing.expectEqual(@as(u6, 10), dawg.next_node_start_bit);
    }

    // Test the system DAWG more thoroughly: it should be the largest.
    if (mgr.getComponent(.lstm_system_dawg)) |sys_data| {
        var sys_dawg = try SquishedDawg.load(std.testing.allocator, sys_data);
        defer sys_dawg.deinit();

        // The system DAWG for eng should have many edges (a real dictionary).
        try std.testing.expect(sys_dawg.numEdges() > 100);

        // Verify node 0 has forward edges for binary search.
        try std.testing.expect(sys_dawg.num_forward_edges_in_node0 > 0);
    }
}

// ── LSTMRecognizer Tests ────────────────────────────────────────────────────

test "LSTMRecognizer load complete eng.traineddata model" {
    const allocator = std.testing.allocator;

    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: cannot open test file: {}\n", .{err});
        return;
    };
    defer file.close();

    const data = file.readToEndAlloc(allocator, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: cannot read test file: {}\n", .{err});
        return;
    };
    defer allocator.free(data);

    var model = try LSTMRecognizer.load(allocator, data);
    defer model.deinit();

    // 1. Verify network loaded (it's a Series)
    try std.testing.expect(model.network == .series);

    // 2. Verify unicharset has 112 chars
    try std.testing.expectEqual(@as(usize, 112), model.unicharset.size());

    // 3. Verify null_char is a valid index (within unicharset range)
    try std.testing.expect(model.null_char >= 0);
    try std.testing.expect(@as(usize, @intCast(model.null_char)) < model.unicharset.size());

    // 4. Verify recoder loaded (eng model has TF_COMPRESS_UNICHARSET)
    try std.testing.expect(model.recoder != null);
    if (model.recoder) |recoder| {
        try std.testing.expect(recoder.size() > 0);
        // The recoder should have entries for all unicharset chars
        try std.testing.expectEqual(@as(usize, 112), recoder.size());
    }

    // 5. Verify at least one DAWG loaded
    const has_dawg = model.punc_dawg != null or model.system_dawg != null or model.number_dawg != null;
    try std.testing.expect(has_dawg);

    // All three should be present for eng
    try std.testing.expect(model.punc_dawg != null);
    try std.testing.expect(model.system_dawg != null);
    try std.testing.expect(model.number_dawg != null);

    // 6. Verify numClasses() returns reasonable number (~111 for eng)
    const num_classes = model.numClasses();
    try std.testing.expect(num_classes > 50);
    try std.testing.expect(num_classes < 500);

    // 7. Verify numInputs() is reasonable
    const num_inputs = model.numInputs();
    try std.testing.expect(num_inputs > 0);
}

test "LSTMRecognizer network_str starts with '['" {
    const allocator = std.testing.allocator;

    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: cannot open test file: {}\n", .{err});
        return;
    };
    defer file.close();

    const data = file.readToEndAlloc(allocator, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: cannot read test file: {}\n", .{err});
        return;
    };
    defer allocator.free(data);

    var model = try LSTMRecognizer.load(allocator, data);
    defer model.deinit();

    // The network_str is a spec string like "[1,1,0,1 Ct5,5,16 Mp3,3..."
    try std.testing.expect(model.network_str.len > 0);
    try std.testing.expectEqual(@as(u8, '['), model.network_str[0]);
}

test "LSTMRecognizer training_flags has TF_INT_MODE set" {
    const allocator = std.testing.allocator;

    const file = std.fs.cwd().openFile("test/testdata/eng.traineddata", .{}) catch |err| {
        std.debug.print("Skipping test: cannot open test file: {}\n", .{err});
        return;
    };
    defer file.close();

    const data = file.readToEndAlloc(allocator, 16 * 1024 * 1024) catch |err| {
        std.debug.print("Skipping test: cannot read test file: {}\n", .{err});
        return;
    };
    defer allocator.free(data);

    var model = try LSTMRecognizer.load(allocator, data);
    defer model.deinit();

    // eng model uses int8 quantized weights
    try std.testing.expect(model.isIntMode());
    try std.testing.expect((model.training_flags & TF_INT_MODE) != 0);

    // eng model should also have TF_COMPRESS_UNICHARSET
    try std.testing.expect((model.training_flags & TF_COMPRESS_UNICHARSET) != 0);
}
