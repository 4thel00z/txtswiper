const std = @import("std");

pub fn main() void {
    const stdout = std.io.getStdOut().writer();
    stdout.print("txtswiper v0.1.0\n", .{}) catch {};
}
