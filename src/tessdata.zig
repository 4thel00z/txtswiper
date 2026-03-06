const std = @import("std");

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
