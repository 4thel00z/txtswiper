const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── stb_image C dependency ──────────────────────────────────────
    const stb_image = b.addStaticLibrary(.{
        .name = "stb_image",
        .target = target,
        .optimize = optimize,
    });
    stb_image.addCSourceFile(.{
        .file = b.path("deps/stb/stb_image_impl.c"),
        .flags = &.{"-std=c99"},
    });
    stb_image.addIncludePath(b.path("deps/stb"));
    stb_image.linkLibC();

    // ── Static library target ───────────────────────────────────────
    const lib = b.addStaticLibrary(.{
        .name = "txtswiper",
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib.addIncludePath(b.path("deps/stb"));
    lib.linkLibrary(stb_image);
    lib.linkLibC();
    b.installArtifact(lib);

    // ── Executable target ───────────────────────────────────────────
    const exe = b.addExecutable(.{
        .name = "txtswiper",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.addIncludePath(b.path("deps/stb"));
    exe.linkLibrary(stb_image);
    exe.linkLibC();
    b.installArtifact(exe);

    // ── Run step ────────────────────────────────────────────────────
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the txtswiper executable");
    run_step.dependOn(&run_cmd.step);

    // ── Tests ───────────────────────────────────────────────────────
    const lib_tests = b.addTest(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_tests.addIncludePath(b.path("deps/stb"));
    lib_tests.linkLibrary(stb_image);
    lib_tests.linkLibC();

    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);
}
