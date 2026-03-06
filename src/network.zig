const std = @import("std");
const math = std.math;
const weights_mod = @import("weights.zig");
const activations = @import("activations.zig");

// ── NetworkIO: Activation Data Container ─────────────────────────────────────
//
// Stores a 2D array of shape [width][num_features] in either float32 or int8
// mode. Width corresponds to the number of timesteps (positions along the text
// line). num_features is the number of channels/features at each position.
//
// This is the core data container used by ALL neural network layers to pass
// activations between them.

pub const NetworkIO = struct {
    /// Float32 activation data, row-major: [width * num_features]
    f_data: ?[]f32,
    /// Int8 activation data for quantized mode: [width * num_features]
    i_data: ?[]i8,
    width_: usize,
    num_features_: usize,
    int_mode_: bool,
    allocator: std.mem.Allocator,

    /// Allocate a NetworkIO with the given dimensions and mode.
    /// The appropriate buffer (f_data or i_data) is allocated and zero-initialized.
    pub fn init(allocator: std.mem.Allocator, w: usize, num_features: usize, int_mode: bool) !NetworkIO {
        const total = w * num_features;
        if (int_mode) {
            const buf = try allocator.alloc(i8, total);
            @memset(buf, 0);
            return NetworkIO{
                .f_data = null,
                .i_data = buf,
                .width_ = w,
                .num_features_ = num_features,
                .int_mode_ = true,
                .allocator = allocator,
            };
        } else {
            const buf = try allocator.alloc(f32, total);
            @memset(buf, 0.0);
            return NetworkIO{
                .f_data = buf,
                .i_data = null,
                .width_ = w,
                .num_features_ = num_features,
                .int_mode_ = false,
                .allocator = allocator,
            };
        }
    }

    /// Free the allocated buffer.
    pub fn deinit(self: *NetworkIO) void {
        if (self.f_data) |buf| self.allocator.free(buf);
        if (self.i_data) |buf| self.allocator.free(buf);
        self.f_data = null;
        self.i_data = null;
    }

    /// Return the width (number of timesteps).
    pub fn width(self: NetworkIO) usize {
        return self.width_;
    }

    /// Return the number of features per timestep.
    pub fn numFeatures(self: NetworkIO) usize {
        return self.num_features_;
    }

    /// Write float data into timestep t.
    /// If int_mode is active, the data is quantized from f32 to i8 by scaling
    /// by 127 and clipping to [-127, 127].
    pub fn writeTimeStep(self: *NetworkIO, t: usize, data: []const f32) void {
        std.debug.assert(data.len == self.num_features_);
        std.debug.assert(t < self.width_);

        const offset = t * self.num_features_;

        if (self.int_mode_) {
            const buf = self.i_data.?;
            for (0..self.num_features_) |j| {
                const scaled = data[j] * 127.0;
                const clamped = math.clamp(scaled, -127.0, 127.0);
                buf[offset + j] = @intFromFloat(@round(clamped));
            }
        } else {
            const buf = self.f_data.?;
            @memcpy(buf[offset .. offset + self.num_features_], data);
        }
    }

    /// Read data from timestep t into out.
    /// If int_mode is active, the i8 data is converted back to f32 by dividing
    /// by 127.0.
    pub fn readTimeStep(self: *const NetworkIO, t: usize, out: []f32) void {
        std.debug.assert(out.len == self.num_features_);
        std.debug.assert(t < self.width_);

        const offset = t * self.num_features_;

        if (self.int_mode_) {
            const buf = self.i_data.?;
            for (0..self.num_features_) |j| {
                out[j] = @as(f32, @floatFromInt(buf[offset + j])) / 127.0;
            }
        } else {
            const buf = self.f_data.?;
            @memcpy(out, buf[offset .. offset + self.num_features_]);
        }
    }

    /// Return a slice to the float data at timestep t.
    /// Asserts that int_mode is not active.
    pub fn f(self: *NetworkIO, t: usize) []f32 {
        std.debug.assert(!self.int_mode_);
        std.debug.assert(t < self.width_);
        const offset = t * self.num_features_;
        return self.f_data.?[offset .. offset + self.num_features_];
    }

    /// Return a slice to the int8 data at timestep t.
    /// Asserts that int_mode is active.
    pub fn i(self: *NetworkIO, t: usize) []i8 {
        std.debug.assert(self.int_mode_);
        std.debug.assert(t < self.width_);
        const offset = t * self.num_features_;
        return self.i_data.?[offset .. offset + self.num_features_];
    }

    /// Resize the buffer to accommodate new_width * new_features.
    /// Only grows the allocation, never shrinks. Updates dimensions and zeros
    /// the entire buffer.
    pub fn resize(self: *NetworkIO, new_width: usize, new_features: usize) !void {
        const new_total = new_width * new_features;
        const old_total = self.width_ * self.num_features_;

        if (self.int_mode_) {
            if (new_total > old_total) {
                if (self.i_data) |buf| self.allocator.free(buf);
                self.i_data = try self.allocator.alloc(i8, new_total);
            }
            @memset(self.i_data.?, 0);
        } else {
            if (new_total > old_total) {
                if (self.f_data) |buf| self.allocator.free(buf);
                self.f_data = try self.allocator.alloc(f32, new_total);
            }
            @memset(self.f_data.?, 0.0);
        }

        self.width_ = new_width;
        self.num_features_ = new_features;
    }

    /// Copy one timestep from src into self at dest_t.
    /// Both must have the same num_features and the same mode.
    pub fn copyTimeStepFrom(self: *NetworkIO, dest_t: usize, src: *const NetworkIO, src_t: usize) void {
        std.debug.assert(self.num_features_ == src.num_features_);
        std.debug.assert(self.int_mode_ == src.int_mode_);
        std.debug.assert(dest_t < self.width_);
        std.debug.assert(src_t < src.width_);

        const nf = self.num_features_;
        const dst_offset = dest_t * nf;
        const src_offset = src_t * nf;

        if (self.int_mode_) {
            @memcpy(
                self.i_data.?[dst_offset .. dst_offset + nf],
                src.i_data.?[src_offset .. src_offset + nf],
            );
        } else {
            @memcpy(
                self.f_data.?[dst_offset .. dst_offset + nf],
                src.f_data.?[src_offset .. src_offset + nf],
            );
        }
    }

    /// Zero out all features at timestep t.
    pub fn zeroTimeStep(self: *NetworkIO, t: usize) void {
        std.debug.assert(t < self.width_);

        const offset = t * self.num_features_;

        if (self.int_mode_) {
            @memset(self.i_data.?[offset .. offset + self.num_features_], 0);
        } else {
            @memset(self.f_data.?[offset .. offset + self.num_features_], 0.0);
        }
    }
};

// ── Activation Type ──────────────────────────────────────────────────────────

pub const ActivationType = enum {
    tanh,
    sigmoid,
    relu,
    softmax,
    linear, // no activation
};

// ── FullyConnected Layer ─────────────────────────────────────────────────────
//
// Computes: output[t] = activation(W * input[t] + bias) for each timestep.
// The weight matrix has shape [no][ni+1] where the last column holds biases.

pub const FullyConnectedLayer = struct {
    weights: weights_mod.WeightMatrix,
    activation: ActivationType,
    ni: usize, // num inputs
    no: usize, // num outputs

    /// Create a FullyConnected layer with a float weight matrix of size [no][ni+1].
    pub fn init(allocator: std.mem.Allocator, ni: usize, no: usize, activation: ActivationType) !FullyConnectedLayer {
        const wm = try weights_mod.WeightMatrix.initFloat(allocator, no, ni);
        return FullyConnectedLayer{
            .weights = wm,
            .activation = activation,
            .ni = ni,
            .no = no,
        };
    }

    /// Free the weight matrix allocation.
    pub fn deinit(self: *FullyConnectedLayer, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.weights.deinit();
    }

    /// Forward pass: for each timestep, compute output = activation(W * input + bias).
    pub fn forward(self: *const FullyConnectedLayer, input: *const NetworkIO, output: *NetworkIO, allocator: std.mem.Allocator) !void {
        std.debug.assert(input.numFeatures() == self.ni);

        const w = input.width();
        try output.resize(w, self.no);

        // Allocate temporary buffers for one timestep.
        const temp_input = try allocator.alloc(f32, self.ni);
        defer allocator.free(temp_input);
        const temp_output = try allocator.alloc(f32, self.no);
        defer allocator.free(temp_output);

        for (0..w) |t| {
            // Read input timestep into temp buffer.
            input.readTimeStep(t, temp_input);

            // Matrix-vector multiply: temp_output = W * temp_input + bias.
            self.weights.matVecFloat(temp_input, temp_output);

            // Apply activation function.
            switch (self.activation) {
                .tanh => activations.tanh_inplace(temp_output),
                .sigmoid => activations.sigmoid_inplace(temp_output),
                .relu => {
                    for (temp_output) |*v| {
                        v.* = activations.relu(v.*);
                    }
                },
                .softmax => activations.softmax_inplace(temp_output),
                .linear => {}, // no activation
            }

            // Write result to output timestep.
            output.writeTimeStep(t, temp_output);
        }
    }
};

// ── LSTM Gate Definitions ────────────────────────────────────────────────────
//
// The LSTM uses a 5-gate architecture matching Tesseract:
//   CI  = Cell Input (tanh)
//   GI  = Input Gate (sigmoid)
//   GF1 = Forget Gate 1D (sigmoid)
//   GO  = Output Gate (sigmoid)
//   GFS = Forget Gate 2D (sigmoid) - only used in 2D mode
//
// For 1D mode we use only the first 4 gates.

pub const GateIndex = enum(u3) {
    CI = 0, // Cell Input
    GI = 1, // Input Gate
    GF1 = 2, // Forget Gate (1D)
    GO = 3, // Output Gate
    GFS = 4, // Forget Gate (2D) - unused in 1D mode
};

const NUM_GATES_1D: usize = 4; // CI, GI, GF1, GO

// ── LSTM Layer ───────────────────────────────────────────────────────────────
//
// Implements a 1D LSTM forward pass. At each timestep the layer concatenates
// the external input with the previous hidden output, passes this through four
// gated weight matrices (CI, GI, GF1, GO), updates the cell state, and
// produces the hidden output.

pub const LSTMLayer = struct {
    ni: usize, // num external inputs
    ns: usize, // num states (hidden units) = num outputs
    na: usize, // concatenated input size = ni + ns
    gate_weights: [NUM_GATES_1D]weights_mod.WeightMatrix,
    /// Cell state persisted across timesteps within a single forward call [ns].
    curr_state: []f32,
    /// Hidden output persisted across timesteps within a single forward call [ns].
    curr_output: []f32,
    allocator: std.mem.Allocator,

    /// Create an LSTM layer with `ni` external inputs and `ns` hidden states.
    /// Each of the 4 gate weight matrices has shape [ns][na] (na = ni + ns).
    pub fn init(allocator: std.mem.Allocator, ni: usize, ns: usize) !LSTMLayer {
        const na = ni + ns;

        var gate_weights: [NUM_GATES_1D]weights_mod.WeightMatrix = undefined;
        var gates_initialised: usize = 0;
        errdefer {
            for (0..gates_initialised) |g| {
                gate_weights[g].deinit();
            }
        }

        for (0..NUM_GATES_1D) |g| {
            gate_weights[g] = try weights_mod.WeightMatrix.initFloat(allocator, ns, na);
            gates_initialised += 1;
        }

        const curr_state = try allocator.alloc(f32, ns);
        errdefer allocator.free(curr_state);
        @memset(curr_state, 0.0);

        const curr_output = try allocator.alloc(f32, ns);
        errdefer allocator.free(curr_output);
        @memset(curr_output, 0.0);

        return LSTMLayer{
            .ni = ni,
            .ns = ns,
            .na = na,
            .gate_weights = gate_weights,
            .curr_state = curr_state,
            .curr_output = curr_output,
            .allocator = allocator,
        };
    }

    /// Free all weight matrices and state buffers.
    pub fn deinit(self: *LSTMLayer) void {
        for (0..NUM_GATES_1D) |g| {
            self.gate_weights[g].deinit();
        }
        self.allocator.free(self.curr_state);
        self.allocator.free(self.curr_output);
    }

    /// Zero the cell state and hidden output (called at the start of each forward pass).
    pub fn resetState(self: *LSTMLayer) void {
        @memset(self.curr_state, 0.0);
        @memset(self.curr_output, 0.0);
    }

    /// LSTM forward pass over all timesteps in `input`.
    ///
    /// For each timestep t:
    ///   1. Construct concatenated input: [fresh_input | prev_output]
    ///   2. Compute gate activations via weight matrices
    ///   3. Update cell state: state = state * GF1 + CI * GI
    ///   4. Compute output: output = tanh(state) * GO
    pub fn forward(self: *LSTMLayer, input: *const NetworkIO, output: *NetworkIO, allocator: std.mem.Allocator) !void {
        std.debug.assert(input.numFeatures() == self.ni);

        const w = input.width();
        try output.resize(w, self.ns);

        // Reset state for a fresh forward pass.
        self.resetState();

        // Allocate temporary buffers.
        const curr_input = try allocator.alloc(f32, self.na);
        defer allocator.free(curr_input);

        // Temp gate outputs: one buffer per gate, each of size ns.
        var temp_gates: [NUM_GATES_1D][]f32 = undefined;
        var temps_allocated: usize = 0;
        defer {
            for (0..temps_allocated) |g| {
                allocator.free(temp_gates[g]);
            }
        }
        for (0..NUM_GATES_1D) |g| {
            temp_gates[g] = try allocator.alloc(f32, self.ns);
            temps_allocated += 1;
        }

        for (0..w) |t| {
            // 1. CONSTRUCT INPUT: [fresh_input(0..ni) | prev_output(ni..na)]
            input.readTimeStep(t, curr_input[0..self.ni]);
            @memcpy(curr_input[self.ni..self.na], self.curr_output);

            // 2. GATE COMPUTATIONS
            const ci_idx = @intFromEnum(GateIndex.CI);
            const gi_idx = @intFromEnum(GateIndex.GI);
            const gf1_idx = @intFromEnum(GateIndex.GF1);
            const go_idx = @intFromEnum(GateIndex.GO);

            self.gate_weights[ci_idx].matVecFloat(curr_input, temp_gates[ci_idx]);
            self.gate_weights[gi_idx].matVecFloat(curr_input, temp_gates[gi_idx]);
            self.gate_weights[gf1_idx].matVecFloat(curr_input, temp_gates[gf1_idx]);
            self.gate_weights[go_idx].matVecFloat(curr_input, temp_gates[go_idx]);

            // Apply activations: CI=tanh, GI/GF1/GO=sigmoid
            activations.tanh_inplace(temp_gates[ci_idx]);
            activations.sigmoid_inplace(temp_gates[gi_idx]);
            activations.sigmoid_inplace(temp_gates[gf1_idx]);
            activations.sigmoid_inplace(temp_gates[go_idx]);

            // 3. STATE UPDATE
            for (0..self.ns) |i| {
                // Forget old state
                self.curr_state[i] *= temp_gates[gf1_idx][i];
                // Add new input
                self.curr_state[i] += temp_gates[ci_idx][i] * temp_gates[gi_idx][i];
                // Clamp for numerical stability
                self.curr_state[i] = math.clamp(self.curr_state[i], -100.0, 100.0);
            }

            // 4. OUTPUT: tanh(state) * GO
            for (0..self.ns) |i| {
                self.curr_output[i] = activations.tanh_fast(self.curr_state[i]) * temp_gates[go_idx][i];
            }

            // 5. WRITE OUTPUT
            output.writeTimeStep(t, self.curr_output);
        }
    }
};

// ── Tests ────────────────────────────────────────────────────────────────────

test "NetworkIO create and read/write timestep (float)" {
    const allocator = std.testing.allocator;
    var nio = try NetworkIO.init(allocator, 10, 5, false);
    defer nio.deinit();

    // Write [1, 2, 3, 4, 5] at t=3.
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    nio.writeTimeStep(3, &input);

    // Read back and verify.
    var out: [5]f32 = undefined;
    nio.readTimeStep(3, &out);

    for (0..5) |j| {
        try std.testing.expectApproxEqAbs(input[j], out[j], 1e-6);
    }
}

test "NetworkIO create and read/write timestep (int8)" {
    const allocator = std.testing.allocator;
    var nio = try NetworkIO.init(allocator, 10, 5, true);
    defer nio.deinit();

    // Write float data that should survive quantization reasonably.
    const input = [_]f32{ 0.5, -0.5, 1.0, -1.0, 0.0 };
    nio.writeTimeStep(3, &input);

    // Read back and verify within quantization tolerance.
    var out: [5]f32 = undefined;
    nio.readTimeStep(3, &out);

    // Quantization tolerance: 1/127 ~= 0.008.
    const quant_tol: f32 = 0.01;
    for (0..5) |j| {
        try std.testing.expectApproxEqAbs(input[j], out[j], quant_tol);
    }
}

test "NetworkIO width and numFeatures" {
    const allocator = std.testing.allocator;
    var nio = try NetworkIO.init(allocator, 20, 48, false);
    defer nio.deinit();

    try std.testing.expectEqual(@as(usize, 20), nio.width());
    try std.testing.expectEqual(@as(usize, 48), nio.numFeatures());
}

test "NetworkIO f() returns correct slice" {
    const allocator = std.testing.allocator;
    var nio = try NetworkIO.init(allocator, 10, 4, false);
    defer nio.deinit();

    // Write data at t=2.
    const input = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    nio.writeTimeStep(2, &input);

    // Access via f() and verify same data.
    const slice = nio.f(2);
    try std.testing.expectEqual(@as(usize, 4), slice.len);
    for (0..4) |j| {
        try std.testing.expectApproxEqAbs(input[j], slice[j], 1e-6);
    }
}

test "NetworkIO copyTimeStepFrom" {
    const allocator = std.testing.allocator;

    var src = try NetworkIO.init(allocator, 10, 4, false);
    defer src.deinit();

    var dst = try NetworkIO.init(allocator, 10, 4, false);
    defer dst.deinit();

    // Write data to src at t=5.
    const input = [_]f32{ 100.0, 200.0, 300.0, 400.0 };
    src.writeTimeStep(5, &input);

    // Copy from src t=5 to dst t=2.
    dst.copyTimeStepFrom(2, &src, 5);

    // Verify the copied data.
    var out: [4]f32 = undefined;
    dst.readTimeStep(2, &out);

    for (0..4) |j| {
        try std.testing.expectApproxEqAbs(input[j], out[j], 1e-6);
    }
}

test "NetworkIO resize grows buffer" {
    const allocator = std.testing.allocator;
    var nio = try NetworkIO.init(allocator, 5, 3, false);
    defer nio.deinit();

    // Write some data at t=0.
    const input = [_]f32{ 1.0, 2.0, 3.0 };
    nio.writeTimeStep(0, &input);

    // Resize larger.
    try nio.resize(20, 10);

    try std.testing.expectEqual(@as(usize, 20), nio.width());
    try std.testing.expectEqual(@as(usize, 10), nio.numFeatures());

    // Old data should be cleared (buffer is zeroed).
    var out: [10]f32 = undefined;
    nio.readTimeStep(0, &out);
    for (0..10) |j| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[j], 1e-6);
    }
}

test "NetworkIO zeroTimeStep" {
    const allocator = std.testing.allocator;
    var nio = try NetworkIO.init(allocator, 10, 4, false);
    defer nio.deinit();

    // Write data at t=7.
    const input = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    nio.writeTimeStep(7, &input);

    // Zero that timestep.
    nio.zeroTimeStep(7);

    // Verify all zeros.
    var out: [4]f32 = undefined;
    nio.readTimeStep(7, &out);
    for (0..4) |j| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[j], 1e-6);
    }
}

// ── FullyConnected Tests ─────────────────────────────────────────────────────

test "FullyConnected forward tanh" {
    const allocator = std.testing.allocator;

    // Create FC layer: 3 inputs, 2 outputs, tanh activation.
    var fc = try FullyConnectedLayer.init(allocator, 3, 2, .tanh);
    defer fc.deinit(allocator);

    // Set weights to known values:
    //   Row 0: [1, 0, 0, bias=0] -> output = tanh(1*x0 + 0*x1 + 0*x2 + 0)
    //   Row 1: [0, 1, 0, bias=0] -> output = tanh(0*x0 + 1*x1 + 0*x2 + 0)
    fc.weights.set(0, 0, 1.0);
    fc.weights.set(0, 1, 0.0);
    fc.weights.set(0, 2, 0.0);
    fc.weights.setBias(0, 0.0);

    fc.weights.set(1, 0, 0.0);
    fc.weights.set(1, 1, 1.0);
    fc.weights.set(1, 2, 0.0);
    fc.weights.setBias(1, 0.0);

    // Input: 2 timesteps: t0=[1.0, 2.0, 3.0], t1=[0.5, -0.5, 0.0]
    var input = try NetworkIO.init(allocator, 2, 3, false);
    defer input.deinit();
    const t0_in = [_]f32{ 1.0, 2.0, 3.0 };
    const t1_in = [_]f32{ 0.5, -0.5, 0.0 };
    input.writeTimeStep(0, &t0_in);
    input.writeTimeStep(1, &t1_in);

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try fc.forward(&input, &output, allocator);

    try std.testing.expectEqual(@as(usize, 2), output.width());
    try std.testing.expectEqual(@as(usize, 2), output.numFeatures());

    // Expected: t0=[tanh(1.0), tanh(2.0)], t1=[tanh(0.5), tanh(-0.5)]
    const tol: f32 = 1e-3;
    var out0: [2]f32 = undefined;
    var out1: [2]f32 = undefined;
    output.readTimeStep(0, &out0);
    output.readTimeStep(1, &out1);

    try std.testing.expectApproxEqAbs(activations.tanh_approx(1.0), out0[0], tol);
    try std.testing.expectApproxEqAbs(activations.tanh_approx(2.0), out0[1], tol);
    try std.testing.expectApproxEqAbs(activations.tanh_approx(0.5), out1[0], tol);
    try std.testing.expectApproxEqAbs(activations.tanh_approx(-0.5), out1[1], tol);
}

test "FullyConnected forward linear (no activation)" {
    const allocator = std.testing.allocator;

    // Create FC: 2 inputs, 2 outputs, linear activation.
    var fc = try FullyConnectedLayer.init(allocator, 2, 2, .linear);
    defer fc.deinit(allocator);

    // Set weights to identity + bias:
    //   Row 0: [1, 0, bias=0.5] -> output = 1*x0 + 0*x1 + 0.5
    //   Row 1: [0, 1, bias=-1.0] -> output = 0*x0 + 1*x1 - 1.0
    fc.weights.set(0, 0, 1.0);
    fc.weights.set(0, 1, 0.0);
    fc.weights.setBias(0, 0.5);

    fc.weights.set(1, 0, 0.0);
    fc.weights.set(1, 1, 1.0);
    fc.weights.setBias(1, -1.0);

    // Input: 2 timesteps.
    var input = try NetworkIO.init(allocator, 2, 2, false);
    defer input.deinit();
    const t0_in = [_]f32{ 3.0, 7.0 };
    const t1_in = [_]f32{ -2.0, 4.0 };
    input.writeTimeStep(0, &t0_in);
    input.writeTimeStep(1, &t1_in);

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try fc.forward(&input, &output, allocator);

    const tol: f32 = 1e-5;
    var out0: [2]f32 = undefined;
    var out1: [2]f32 = undefined;
    output.readTimeStep(0, &out0);
    output.readTimeStep(1, &out1);

    // t0: [1*3+0.5, 0*3+1*7-1.0] = [3.5, 6.0]
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), out0[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out0[1], tol);

    // t1: [1*(-2)+0.5, 0*(-2)+1*4-1.0] = [-1.5, 3.0]
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), out1[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out1[1], tol);
}

test "FullyConnected forward softmax" {
    const allocator = std.testing.allocator;

    // Create FC: 2 inputs, 3 outputs, softmax activation.
    var fc = try FullyConnectedLayer.init(allocator, 2, 3, .softmax);
    defer fc.deinit(allocator);

    // Set weights so pre-activation values differ per output.
    //   Row 0: [1, 0, bias=0]
    //   Row 1: [0, 1, bias=0]
    //   Row 2: [1, 1, bias=0]
    fc.weights.set(0, 0, 1.0);
    fc.weights.set(0, 1, 0.0);
    fc.weights.setBias(0, 0.0);

    fc.weights.set(1, 0, 0.0);
    fc.weights.set(1, 1, 1.0);
    fc.weights.setBias(1, 0.0);

    fc.weights.set(2, 0, 1.0);
    fc.weights.set(2, 1, 1.0);
    fc.weights.setBias(2, 0.0);

    // Input: 2 timesteps.
    var input = try NetworkIO.init(allocator, 2, 2, false);
    defer input.deinit();
    const t0_in = [_]f32{ 1.0, 2.0 };
    const t1_in = [_]f32{ -1.0, 0.5 };
    input.writeTimeStep(0, &t0_in);
    input.writeTimeStep(1, &t1_in);

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try fc.forward(&input, &output, allocator);

    const tol: f32 = 1e-4;

    // Verify output sums to 1.0 for each timestep and all values in [0, 1].
    for (0..2) |t| {
        var out: [3]f32 = undefined;
        output.readTimeStep(t, &out);

        var sum: f32 = 0.0;
        for (out) |v| {
            try std.testing.expect(v >= 0.0 and v <= 1.0);
            sum += v;
        }
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, tol);
    }
}

test "FullyConnected forward sigmoid" {
    const allocator = std.testing.allocator;

    // Create FC: 2 inputs, 2 outputs, sigmoid activation.
    var fc = try FullyConnectedLayer.init(allocator, 2, 2, .sigmoid);
    defer fc.deinit(allocator);

    // Set weights:
    //   Row 0: [1, 0, bias=0] -> sigmoid(x0)
    //   Row 1: [0, 1, bias=0] -> sigmoid(x1)
    fc.weights.set(0, 0, 1.0);
    fc.weights.set(0, 1, 0.0);
    fc.weights.setBias(0, 0.0);

    fc.weights.set(1, 0, 0.0);
    fc.weights.set(1, 1, 1.0);
    fc.weights.setBias(1, 0.0);

    // Input: 2 timesteps with values spanning positive, negative, and zero.
    var input = try NetworkIO.init(allocator, 2, 2, false);
    defer input.deinit();
    const t0_in = [_]f32{ 0.0, 5.0 };
    const t1_in = [_]f32{ -3.0, 100.0 };
    input.writeTimeStep(0, &t0_in);
    input.writeTimeStep(1, &t1_in);

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try fc.forward(&input, &output, allocator);

    const tol: f32 = 1e-3;

    // All outputs should be in [0, 1].
    for (0..2) |t| {
        var out: [2]f32 = undefined;
        output.readTimeStep(t, &out);
        for (out) |v| {
            try std.testing.expect(v >= 0.0 and v <= 1.0);
        }
    }

    // Check specific values.
    var out0: [2]f32 = undefined;
    var out1: [2]f32 = undefined;
    output.readTimeStep(0, &out0);
    output.readTimeStep(1, &out1);

    // t0: [sigmoid(0.0), sigmoid(5.0)] = [0.5, ~0.9933]
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out0[0], tol);
    try std.testing.expectApproxEqAbs(activations.sigmoid(5.0), out0[1], tol);

    // t1: [sigmoid(-3.0), sigmoid(100.0)] = [~0.0474, 1.0]
    try std.testing.expectApproxEqAbs(activations.sigmoid(-3.0), out1[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out1[1], tol);
}

// ── LSTM Tests ───────────────────────────────────────────────────────────────

test "LSTM forward single timestep" {
    const allocator = std.testing.allocator;

    // ni=2, ns=2  ->  na=4 (concatenated: [x0, x1, h0, h1])
    var lstm = try LSTMLayer.init(allocator, 2, 2);
    defer lstm.deinit();

    // CI weights: identity on fresh input -> CI = tanh([x0, x1, 0, 0] * W_CI)
    // Set CI to pass through: row0 takes x0, row1 takes x1
    const ci = @intFromEnum(GateIndex.CI);
    lstm.gate_weights[ci].set(0, 0, 1.0); // row0: weight on x0
    lstm.gate_weights[ci].set(1, 1, 1.0); // row1: weight on x1

    // GI weights: large positive bias so sigmoid -> ~1 (gate fully open)
    const gi = @intFromEnum(GateIndex.GI);
    lstm.gate_weights[gi].setBias(0, 10.0);
    lstm.gate_weights[gi].setBias(1, 10.0);

    // GF1 weights: large negative bias so sigmoid -> ~0 (forget everything)
    const gf1 = @intFromEnum(GateIndex.GF1);
    lstm.gate_weights[gf1].setBias(0, -10.0);
    lstm.gate_weights[gf1].setBias(1, -10.0);

    // GO weights: large positive bias so sigmoid -> ~1 (output gate open)
    const go = @intFromEnum(GateIndex.GO);
    lstm.gate_weights[go].setBias(0, 10.0);
    lstm.gate_weights[go].setBias(1, 10.0);

    // Input: single timestep with [1.0, 0.5]
    var input = try NetworkIO.init(allocator, 1, 2, false);
    defer input.deinit();
    const t0_in = [_]f32{ 1.0, 0.5 };
    input.writeTimeStep(0, &t0_in);

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try lstm.forward(&input, &output, allocator);

    // With GF1~0 and GI~1:
    //   state = 0 * ~0 + tanh(x) * ~1 = tanh(x)
    //   output = tanh(state) * ~1 = tanh(tanh(x))
    //
    // For x0=1.0: tanh(1.0) ~ 0.7616, tanh(0.7616) ~ 0.6411
    // For x1=0.5: tanh(0.5) ~ 0.4621, tanh(0.4621) ~ 0.4319
    var out: [2]f32 = undefined;
    output.readTimeStep(0, &out);

    const tol: f32 = 0.02; // LUT tolerance
    const expected0 = activations.tanh_fast(activations.tanh_fast(1.0));
    const expected1 = activations.tanh_fast(activations.tanh_fast(0.5));
    try std.testing.expectApproxEqAbs(expected0, out[0], tol);
    try std.testing.expectApproxEqAbs(expected1, out[1], tol);

    // Verify outputs are nonzero and reasonable (in [-1, 1])
    try std.testing.expect(out[0] != 0.0);
    try std.testing.expect(out[1] != 0.0);
    try std.testing.expect(@abs(out[0]) <= 1.0);
    try std.testing.expect(@abs(out[1]) <= 1.0);
}

test "LSTM forward multi timestep state carries" {
    const allocator = std.testing.allocator;

    // ni=1, ns=1  ->  na=2
    var lstm = try LSTMLayer.init(allocator, 1, 1);
    defer lstm.deinit();

    // CI: pass through fresh input (weight on x0 = 1.0)
    const ci = @intFromEnum(GateIndex.CI);
    lstm.gate_weights[ci].set(0, 0, 1.0);

    // GI: fully open (large positive bias)
    const gi = @intFromEnum(GateIndex.GI);
    lstm.gate_weights[gi].setBias(0, 10.0);

    // GF1: partially open (bias=2.0 -> sigmoid(2.0)~0.88), so state accumulates
    const gf1 = @intFromEnum(GateIndex.GF1);
    lstm.gate_weights[gf1].setBias(0, 2.0);

    // GO: fully open
    const go = @intFromEnum(GateIndex.GO);
    lstm.gate_weights[go].setBias(0, 10.0);

    // Input: 5 timesteps, each with [0.5]
    var input = try NetworkIO.init(allocator, 5, 1, false);
    defer input.deinit();
    for (0..5) |t| {
        const val = [_]f32{0.5};
        input.writeTimeStep(t, &val);
    }

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try lstm.forward(&input, &output, allocator);

    // Read outputs at t=0 and t=4
    var out0: [1]f32 = undefined;
    var out4: [1]f32 = undefined;
    output.readTimeStep(0, &out0);
    output.readTimeStep(4, &out4);

    // Verify outputs change over time (state accumulates so they should differ)
    try std.testing.expect(out0[0] != out4[0]);

    // Verify all outputs are nonzero
    for (0..5) |t| {
        var out: [1]f32 = undefined;
        output.readTimeStep(t, &out);
        try std.testing.expect(out[0] != 0.0);
    }
}

test "LSTM forward zero input produces zero output" {
    const allocator = std.testing.allocator;

    // ni=2, ns=2 -> na=4
    // All weights and biases are zero (default from initFloat).
    var lstm = try LSTMLayer.init(allocator, 2, 2);
    defer lstm.deinit();

    // Input: 3 timesteps, all zeros (default from NetworkIO.init).
    var input = try NetworkIO.init(allocator, 3, 2, false);
    defer input.deinit();

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try lstm.forward(&input, &output, allocator);

    // With all-zero weights and all-zero input:
    //   gate_output = W * 0 + 0 = 0 for all gates
    //   CI = tanh(0) = 0,  GI = sigmoid(0) = 0.5,  GF1 = sigmoid(0) = 0.5,  GO = sigmoid(0) = 0.5
    //   state = state * 0.5 + 0 * 0.5 = 0  (starts at 0, stays 0)
    //   output = tanh(0) * 0.5 = 0
    const tol: f32 = 1e-6;
    for (0..3) |t| {
        var out: [2]f32 = undefined;
        output.readTimeStep(t, &out);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], tol);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], tol);
    }
}

test "LSTM output dimensions correct" {
    const allocator = std.testing.allocator;

    // ni=4, ns=3 -> na=7
    var lstm = try LSTMLayer.init(allocator, 4, 3);
    defer lstm.deinit();

    // Verify internal dimensions.
    try std.testing.expectEqual(@as(usize, 4), lstm.ni);
    try std.testing.expectEqual(@as(usize, 3), lstm.ns);
    try std.testing.expectEqual(@as(usize, 7), lstm.na);

    // Input: width=10, features=4
    var input = try NetworkIO.init(allocator, 10, 4, false);
    defer input.deinit();

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try lstm.forward(&input, &output, allocator);

    // Verify output dimensions: width=10, features=3
    try std.testing.expectEqual(@as(usize, 10), output.width());
    try std.testing.expectEqual(@as(usize, 3), output.numFeatures());
}
