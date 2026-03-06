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
