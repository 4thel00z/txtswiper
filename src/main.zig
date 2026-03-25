const std = @import("std");
const lib = @import("lib.zig");
const OcrEngine = lib.OcrEngine;

const version = "0.1.0";

const Format = enum {
    text_fmt,
    hocr_fmt,
    json_fmt,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stderr = std.io.getStdErr().writer();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.next(); // skip program name

    var image_path: ?[]const u8 = null;
    var model_path: ?[]const u8 = null;
    var format: Format = .text_fmt;
    var output_path: ?[]const u8 = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            model_path = args.next() orelse {
                try stderr.writeAll("error: -m/--model requires a path argument\n");
                printUsage(stderr);
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--format")) {
            const fmt_str = args.next() orelse {
                try stderr.writeAll("error: --format requires an argument (text, hocr, json)\n");
                printUsage(stderr);
                std.process.exit(1);
            };
            if (std.mem.eql(u8, fmt_str, "hocr")) {
                format = .hocr_fmt;
            } else if (std.mem.eql(u8, fmt_str, "json")) {
                format = .json_fmt;
            } else if (std.mem.eql(u8, fmt_str, "text")) {
                format = .text_fmt;
            } else {
                try stderr.print("error: unknown format '{s}' (expected text, hocr, or json)\n", .{fmt_str});
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
            output_path = args.next() orelse {
                try stderr.writeAll("error: -o/--output requires a path argument\n");
                printUsage(stderr);
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            const stdout = std.io.getStdOut().writer();
            printUsage(stdout);
            return;
        } else if (std.mem.eql(u8, arg, "--version")) {
            const stdout = std.io.getStdOut().writer();
            try stdout.print("txtswiper v{s}\n", .{version});
            return;
        } else if (std.mem.startsWith(u8, arg, "-")) {
            try stderr.print("error: unknown option '{s}'\n", .{arg});
            printUsage(stderr);
            std.process.exit(1);
        } else {
            image_path = arg;
        }
    }

    const img_path = image_path orelse {
        try stderr.writeAll("error: no image file specified\n");
        printUsage(stderr);
        std.process.exit(1);
    };

    // Find and load model
    const model_data = findModel(allocator, model_path) catch |err| {
        switch (err) {
            error.ModelNotFound => {
                try stderr.writeAll("error: could not find eng.traineddata model file\n");
                try stderr.writeAll("  searched: ./eng.traineddata, TESSDATA_PREFIX, /usr/share/tessdata/\n");
                try stderr.writeAll("  use -m <path> to specify the model path explicitly\n");
            },
            else => {
                try stderr.print("error: failed to load model: {}\n", .{err});
            },
        }
        std.process.exit(1);
    };
    defer allocator.free(model_data);

    // Initialize OCR engine
    var engine = OcrEngine.init(allocator, model_data) catch |err| {
        try stderr.print("error: failed to initialize OCR engine: {}\n", .{err});
        std.process.exit(1);
    };
    defer engine.deinit();

    // Load image
    const img_path_z = try allocator.dupeZ(u8, img_path);
    defer allocator.free(img_path_z);

    engine.setImageFromFile(img_path_z) catch |err| {
        try stderr.print("error: failed to load image '{s}': {}\n", .{ img_path, err });
        std.process.exit(1);
    };

    // Run recognition
    var result = engine.recognize() catch |err| {
        try stderr.print("error: recognition failed: {}\n", .{err});
        std.process.exit(1);
    };
    defer result.deinit();

    // Render output
    const rendered = switch (format) {
        .text_fmt => engine.getText(&result) catch |err| {
            try stderr.print("error: text rendering failed: {}\n", .{err});
            std.process.exit(1);
        },
        .hocr_fmt => engine.getHocr(&result) catch |err| {
            try stderr.print("error: hOCR rendering failed: {}\n", .{err});
            std.process.exit(1);
        },
        .json_fmt => engine.getJson(&result) catch |err| {
            try stderr.print("error: JSON rendering failed: {}\n", .{err});
            std.process.exit(1);
        },
    };
    defer allocator.free(rendered);

    // Write output
    if (output_path) |out_path| {
        const file = std.fs.cwd().createFile(out_path, .{}) catch |err| {
            try stderr.print("error: could not create output file '{s}': {}\n", .{ out_path, err });
            std.process.exit(1);
        };
        defer file.close();
        file.writeAll(rendered) catch |err| {
            try stderr.print("error: failed to write output file: {}\n", .{err});
            std.process.exit(1);
        };
    } else {
        const stdout = std.io.getStdOut().writer();
        try stdout.writeAll(rendered);
        if (format == .text_fmt and rendered.len > 0 and rendered[rendered.len - 1] != '\n') {
            try stdout.writeByte('\n');
        }
    }
}

fn findModel(allocator: std.mem.Allocator, explicit_path: ?[]const u8) ![]const u8 {
    if (explicit_path) |p| {
        return readModelFile(allocator, p);
    }

    if (readModelFile(allocator, "eng.traineddata")) |data| {
        return data;
    } else |_| {}

    if (std.posix.getenv("TESSDATA_PREFIX")) |prefix| {
        const path = try std.fs.path.join(allocator, &.{ prefix, "eng.traineddata" });
        defer allocator.free(path);
        if (readModelFile(allocator, path)) |data| {
            return data;
        } else |_| {}
    }

    if (readModelFile(allocator, "/usr/share/tessdata/eng.traineddata")) |data| {
        return data;
    } else |_| {}

    return error.ModelNotFound;
}

fn readModelFile(allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    return try file.readToEndAlloc(allocator, 64 * 1024 * 1024);
}

fn printUsage(writer: anytype) void {
    writer.writeAll(
        \\Usage: txtswiper <image> [-m model_path] [--format text|hocr|json] [-o output_file]
        \\
        \\Recognize text in an image using LSTM-based OCR.
        \\
        \\Arguments:
        \\  <image>                  Path to the input image (PNG, JPEG, etc.)
        \\
        \\Options:
        \\  -m, --model <path>       Path to .traineddata model file
        \\  --format <fmt>           Output format: text (default), hocr, json
        \\  -o, --output <path>      Write output to file instead of stdout
        \\  -h, --help               Show this help message
        \\  --version                Show version information
        \\
        \\Model search order (when -m is not specified):
        \\  1. ./eng.traineddata
        \\  2. $TESSDATA_PREFIX/eng.traineddata
        \\  3. /usr/share/tessdata/eng.traineddata
        \\
    ) catch {};
}
