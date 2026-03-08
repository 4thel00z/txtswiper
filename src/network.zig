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

    // ── 2D Stride Info ──
    // These track the spatial dimensions of the underlying 2D image grid.
    // width_ (total timesteps) = stride_height * stride_width.
    // When stride_height == 1, the data is purely 1D.
    stride_height: usize = 1,
    stride_width: usize = 0, // 0 means "same as width_"

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
                .stride_height = 1,
                .stride_width = w,
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
                .stride_height = 1,
                .stride_width = w,
            };
        }
    }

    /// Allocate a NetworkIO with explicit 2D stride dimensions.
    /// Total timesteps = sh * sw. num_features is the depth per position.
    pub fn init2D(allocator: std.mem.Allocator, sh: usize, sw: usize, num_features: usize, int_mode: bool) !NetworkIO {
        var nio = try init(allocator, sh * sw, num_features, int_mode);
        nio.stride_height = sh;
        nio.stride_width = sw;
        return nio;
    }

    /// Return the effective stride width (image width in 2D).
    pub fn getStrideWidth(self: NetworkIO) usize {
        if (self.stride_width == 0) return self.width_;
        return self.stride_width;
    }

    /// Return the stride height (image height in 2D, 1 for pure 1D).
    pub fn getStrideHeight(self: NetworkIO) usize {
        return self.stride_height;
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
    /// the entire buffer. Resets stride to 1D (stride_height=1).
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
        self.stride_height = 1;
        self.stride_width = new_width;
    }

    /// Resize with explicit 2D stride dimensions. Total timesteps = sh * sw.
    pub fn resize2D(self: *NetworkIO, sh: usize, sw: usize, new_features: usize) !void {
        try self.resize(sh * sw, new_features);
        self.stride_height = sh;
        self.stride_width = sw;
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
    /// Preserves 2D stride from input.
    pub fn forward(self: *const FullyConnectedLayer, input: *const NetworkIO, output: *NetworkIO, allocator: std.mem.Allocator) !void {
        std.debug.assert(input.numFeatures() == self.ni);

        const w = input.width();
        try output.resize(w, self.no);
        // Preserve 2D stride info (spatial dims unchanged, only features change).
        output.stride_height = input.stride_height;
        output.stride_width = input.stride_width;

        // Allocate temporary buffers for one timestep.
        const temp_input = try allocator.alloc(f32, self.ni);
        defer allocator.free(temp_input);
        const temp_output = try allocator.alloc(f32, self.no);
        defer allocator.free(temp_output);

        for (0..w) |t| {
            // Read input timestep into temp buffer.
            input.readTimeStep(t, temp_input);

            // Matrix-vector multiply: temp_output = W * temp_input + bias.
            self.weights.matVec(temp_input, temp_output);

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
        // Preserve 2D stride info from input.
        output.stride_height = input.stride_height;
        output.stride_width = input.stride_width;

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

            self.gate_weights[ci_idx].matVec(curr_input, temp_gates[ci_idx]);
            self.gate_weights[gi_idx].matVec(curr_input, temp_gates[gi_idx]);
            self.gate_weights[gf1_idx].matVec(curr_input, temp_gates[gf1_idx]);
            self.gate_weights[go_idx].matVec(curr_input, temp_gates[go_idx]);

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

// ── ConvolveLayer ────────────────────────────────────────────────────────────
//
// A Convolve layer stacks spatial neighborhood features. In 1D mode (half_y=0),
// it creates a sliding window of size (2*half_x + 1) along the width axis.
// In 2D mode (half_y>0), it creates a (2*half_x+1) x (2*half_y+1) patch and
// stacks all features from the patch. This matches Tesseract's Convolve layer.
//
// Output features = ni * (2*half_x+1) * (2*half_y+1).

pub const ConvolveLayer = struct {
    ni: usize, // input features per position
    no: usize, // output features = ni * (2*half_x+1) * (2*half_y+1)
    half_x: usize, // half window in x (width) direction
    half_y: usize, // half window in y (height) direction (0 for 1D mode)

    /// Create a ConvolveLayer with `ni` input features and half-windows `half_x`, `half_y`.
    pub fn init(ni: usize, half_x: usize, half_y: usize) ConvolveLayer {
        return ConvolveLayer{
            .ni = ni,
            .no = ni * (2 * half_x + 1) * (2 * half_y + 1),
            .half_x = half_x,
            .half_y = half_y,
        };
    }

    /// Forward pass: for each position, stack features from the surrounding 2D window.
    /// Positions outside the image boundary are zero-padded.
    ///
    /// In 2D mode, the input's stride_height and stride_width determine the spatial
    /// layout. The output preserves the same 2D stride.
    pub fn forward(self: *const ConvolveLayer, input: *const NetworkIO, output: *NetworkIO, allocator: std.mem.Allocator) !void {
        _ = allocator;
        const sh = input.getStrideHeight();
        const sw = input.getStrideWidth();
        const ni = self.ni;
        const y_scale = 2 * self.half_y + 1;

        // Output has same spatial dimensions, different feature count.
        try output.resize2D(sh, sw, self.no);

        const in_buf = input.f_data.?;
        const out_buf = output.f_data.?;

        // Iterate over every 2D position.
        for (0..sh) |py| {
            for (0..sw) |px| {
                const t = py * sw + px;
                var out_ix: usize = 0;

                // x window: -half_x to +half_x
                var dxi: usize = 0;
                const x_window = 2 * self.half_x + 1;
                while (dxi < x_window) : (dxi += 1) {
                    const sx_shifted = px + dxi;
                    const sx_valid = sx_shifted >= self.half_x and (sx_shifted - self.half_x) < sw;

                    if (sx_valid) {
                        const sx = sx_shifted - self.half_x;

                        // y window: -half_y to +half_y
                        var dyi: usize = 0;
                        while (dyi < y_scale) : (dyi += 1) {
                            const sy_shifted = py + dyi;
                            const sy_valid = sy_shifted >= self.half_y and (sy_shifted - self.half_y) < sh;

                            const dst_off = t * self.no + out_ix;
                            if (sy_valid) {
                                const sy = sy_shifted - self.half_y;
                                const src_t = sy * sw + sx;
                                const src_off = src_t * input.num_features_;
                                @memcpy(out_buf[dst_off .. dst_off + ni], in_buf[src_off .. src_off + ni]);
                            } else {
                                @memset(out_buf[dst_off .. dst_off + ni], 0.0);
                            }
                            out_ix += ni;
                        }
                    } else {
                        // Entire x column is out of bounds.
                        const dst_off = t * self.no + out_ix;
                        @memset(out_buf[dst_off .. dst_off + y_scale * ni], 0.0);
                        out_ix += y_scale * ni;
                    }
                }
            }
        }
    }
};

// ── MaxpoolLayer ─────────────────────────────────────────────────────────────
//
// Downsamples along both width and height dimensions by taking the element-wise
// maximum over a 2D window. The feature count is preserved (this is Maxpool,
// not Reconfig which would stack features).
//
// For Tesseract's Reconfig/Maxpool (NT_MAXPOOL type): output spatial dimensions
// are [ceil(H/y_scale)][ceil(W/x_scale)], features stay the same.

pub const MaxpoolLayer = struct {
    ni: usize, // input features = output features
    no: usize, // same as ni (maxpool doesn't change feature count)
    x_scale: usize, // downsampling factor along width
    y_scale: usize, // downsampling factor along height (1 for 1D mode)

    /// Create a MaxpoolLayer with `ni` features and downsampling factors.
    pub fn init(ni: usize, x_scale: usize, y_scale: usize) MaxpoolLayer {
        return MaxpoolLayer{
            .ni = ni,
            .no = ni,
            .x_scale = x_scale,
            .y_scale = y_scale,
        };
    }

    /// Forward pass: downsample by taking element-wise max over 2D windows.
    ///
    /// In 2D mode, reads stride_height/stride_width from input to determine
    /// the spatial grid, and outputs a downsampled grid.
    pub fn forward(self: *const MaxpoolLayer, input: *const NetworkIO, output: *NetworkIO, allocator: std.mem.Allocator) !void {
        _ = allocator;
        const in_sh = input.getStrideHeight();
        const in_sw = input.getStrideWidth();
        const nf = self.ni;

        const out_sh = (in_sh + self.y_scale - 1) / self.y_scale;
        const out_sw = (in_sw + self.x_scale - 1) / self.x_scale;
        try output.resize2D(out_sh, out_sw, nf);

        const out_buf = output.f_data.?;
        const in_buf = input.f_data.?;

        for (0..out_sh) |oy| {
            for (0..out_sw) |ox| {
                const out_t = oy * out_sw + ox;
                const dst_offset = out_t * nf;

                // Initialize with very negative values for max operation.
                @memset(out_buf[dst_offset .. dst_offset + nf], -math.inf(f32));

                // Take element-wise max over the y_scale x x_scale window.
                const src_y_start = oy * self.y_scale;
                const src_x_start = ox * self.x_scale;

                var dy: usize = 0;
                while (dy < self.y_scale) : (dy += 1) {
                    const sy = src_y_start + dy;
                    if (sy >= in_sh) break;

                    var dx: usize = 0;
                    while (dx < self.x_scale) : (dx += 1) {
                        const sx = src_x_start + dx;
                        if (sx >= in_sw) break;

                        const src_t = sy * in_sw + sx;
                        const src_off = src_t * input.num_features_;

                        for (0..nf) |f_idx| {
                            out_buf[dst_offset + f_idx] = @max(
                                out_buf[dst_offset + f_idx],
                                in_buf[src_off + f_idx],
                            );
                        }
                    }
                }
            }
        }
    }
};

// ── InputLayer ───────────────────────────────────────────────────────────────
//
// The Input layer is the first layer in a Tesseract network. It defines the
// expected input dimensions (batch, height, width, depth) and simply passes
// the input through unchanged.

pub const InputLayer = struct {
    ni: usize, // num inputs (= height or depth depending on shape)
    no: usize, // num outputs (= depth)
    batch: i32,
    height: i32,
    width: i32,
    depth: i32,

    /// Forward pass: pass-through (copies input to output).
    /// Propagates the 2D stride information from input.
    pub fn forward(self: *const InputLayer, input: *const NetworkIO, output: *NetworkIO, allocator: std.mem.Allocator) !void {
        _ = allocator;
        _ = self;
        const w = input.width();
        const nf = input.numFeatures();
        try output.resize(w, nf);
        // Propagate 2D stride from input.
        output.stride_height = input.stride_height;
        output.stride_width = input.stride_width;
        for (0..w) |t| {
            output.copyTimeStepFrom(t, input, t);
        }
    }
};

// ── ReversedLayer ────────────────────────────────────────────────────────────
//
// Wraps a single child layer. In the forward pass it reverses the input
// timestep order, runs the child, then reverses the output back.
// Variants: RTLReversed (x-reversed), TTBReversed (y-reversed), XYTranspose.
// For inference, XYTranspose is a no-op on 1D data.

pub const ReversedType = enum {
    x_reversed, // RTLReversed: reverse along width (right-to-left)
    y_reversed, // TTBReversed: reverse along height (top-to-bottom)
    xy_transpose, // XYTranspose: swap x and y
};

pub const ReversedLayer = struct {
    child: Layer,
    reversed_type: ReversedType,
    ni: usize,
    no: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, child: Layer, reversed_type: ReversedType) !*ReversedLayer {
        const ni: usize = child.numInputs();
        const no: usize = child.numOutputs();
        const self = try allocator.create(ReversedLayer);
        self.* = .{
            .child = child,
            .reversed_type = reversed_type,
            .ni = ni,
            .no = no,
            .allocator = allocator,
        };
        return self;
    }

    /// Forward pass: reverse/transpose input, run child, reverse/transpose output.
    ///
    /// XYTranspose: transposes the 2D grid (swaps height and width), runs the
    /// child on the transposed data, then transposes back. This converts
    /// row-major scanning to column-major for the LSTM.
    ///
    /// X/Y Reversed: reverses timestep order, runs child, reverses back.
    pub fn forward(self: *ReversedLayer, input: *const NetworkIO, output: *NetworkIO, allocator: std.mem.Allocator) ForwardError!void {
        if (self.reversed_type == .xy_transpose) {
            // Transpose the 2D input: swap height and width.
            const sh = input.getStrideHeight();
            const sw = input.getStrideWidth();
            const nf = input.numFeatures();

            var transposed_input = try NetworkIO.init2D(allocator, sw, sh, nf, false);
            defer transposed_input.deinit();

            // src[y][x] -> dest[x][y]: src_t = y*sw+x, dest_t = x*sh+y
            for (0..sh) |y| {
                for (0..sw) |x| {
                    const src_t = y * sw + x;
                    const dest_t = x * sh + y;
                    transposed_input.copyTimeStepFrom(dest_t, input, src_t);
                }
            }

            // Run child on transposed input.
            var child_output = try NetworkIO.init(allocator, 1, 1, false);
            defer child_output.deinit();
            try self.child.forward(&transposed_input, &child_output, allocator);

            // Transpose the output back: child output has stride [sw_out][sh_out]
            // which we transpose to [sh_out][sw_out].
            const out_sh = child_output.getStrideHeight();
            const out_sw = child_output.getStrideWidth();
            const out_nf = child_output.numFeatures();

            try output.resize2D(out_sw, out_sh, out_nf);
            for (0..out_sh) |y| {
                for (0..out_sw) |x| {
                    const src_t = y * out_sw + x;
                    const dest_t = x * out_sh + y;
                    output.copyTimeStepFrom(dest_t, &child_output, src_t);
                }
            }
            return;
        }

        // X-reversed or Y-reversed: reverse timestep order.
        const w = input.width();
        const nf = input.numFeatures();
        var reversed_input = try NetworkIO.init(allocator, w, nf, false);
        defer reversed_input.deinit();
        // Propagate stride info.
        reversed_input.stride_height = input.stride_height;
        reversed_input.stride_width = input.stride_width;
        for (0..w) |t| {
            reversed_input.copyTimeStepFrom(w - 1 - t, input, t);
        }

        // Run child on reversed input
        var child_output = try NetworkIO.init(allocator, 1, 1, false);
        defer child_output.deinit();
        try self.child.forward(&reversed_input, &child_output, allocator);

        // Reverse the output back
        const out_w = child_output.width();
        const out_nf = child_output.numFeatures();
        try output.resize(out_w, out_nf);
        output.stride_height = child_output.stride_height;
        output.stride_width = child_output.stride_width;
        for (0..out_w) |t| {
            output.copyTimeStepFrom(out_w - 1 - t, &child_output, t);
        }
    }

    pub fn deinit(self: *ReversedLayer) void {
        deinitLayer(self.allocator, &self.child);
        self.allocator.destroy(self);
    }
};

// ── Layer Union ──────────────────────────────────────────────────────────────
//
// A tagged union that can hold any layer type. This allows SeriesLayer and
// ParallelLayer to compose arbitrary child layers. Series and Parallel use
// pointers because the union type is recursive (they contain slices of Layer).

/// Error set shared by all layer forward methods. Explicit because the
/// Layer -> Series/Parallel -> Layer recursion prevents Zig from inferring
/// the error set automatically.
pub const ForwardError = std.mem.Allocator.Error;

pub const Layer = union(enum) {
    fully_connected: FullyConnectedLayer,
    lstm: LSTMLayer,
    series: *SeriesLayer,
    parallel: *ParallelLayer,
    convolve: ConvolveLayer,
    maxpool: MaxpoolLayer,
    input: InputLayer,
    reversed: *ReversedLayer,

    /// Dispatch forward to the underlying layer.
    pub fn forward(self: *Layer, input_nio: *const NetworkIO, output: *NetworkIO, allocator: std.mem.Allocator) ForwardError!void {
        switch (self.*) {
            .fully_connected => |*fc| try fc.forward(input_nio, output, allocator),
            .lstm => |*l| try l.forward(input_nio, output, allocator),
            .series => |s| try s.forward(input_nio, output, allocator),
            .parallel => |p| try p.forward(input_nio, output, allocator),
            .convolve => |*c| try c.forward(input_nio, output, allocator),
            .maxpool => |*m| try m.forward(input_nio, output, allocator),
            .input => |*inp| try inp.forward(input_nio, output, allocator),
            .reversed => |r| try r.forward(input_nio, output, allocator),
        }
    }

    /// Return the number of output features for this layer.
    pub fn numOutputs(self: Layer) usize {
        return switch (self) {
            .fully_connected => |fc| fc.no,
            .lstm => |l| l.ns,
            .series => |s| s.no,
            .parallel => |p| p.no,
            .convolve => |c| c.no,
            .maxpool => |m| m.no,
            .input => |inp| inp.no,
            .reversed => |r| r.no,
        };
    }

    /// Return the number of input features for this layer.
    pub fn numInputs(self: Layer) usize {
        return switch (self) {
            .fully_connected => |fc| fc.ni,
            .lstm => |l| l.ni,
            .series => |s| s.ni,
            .parallel => |p| p.ni,
            .convolve => |c| c.ni,
            .maxpool => |m| m.ni,
            .input => |inp| inp.ni,
            .reversed => |r| r.ni,
        };
    }
};

// ── SeriesLayer ──────────────────────────────────────────────────────────────
//
// Chains a sequence of layers: the output of layer N becomes the input of
// layer N+1. The overall input dimension equals the first layer's input
// dimension, and the overall output dimension equals the last layer's output
// dimension.

pub const SeriesLayer = struct {
    layers: []Layer,
    ni: usize, // input features (= first layer's input count)
    no: usize, // output features (= last layer's output count)
    allocator: std.mem.Allocator,

    /// Allocate a SeriesLayer on the heap. Copies the provided layers slice.
    /// Sets ni from the first layer and no from the last layer.
    pub fn init(allocator: std.mem.Allocator, layers: []const Layer) !*SeriesLayer {
        const owned_layers = try allocator.alloc(Layer, layers.len);
        @memcpy(owned_layers, layers);

        const ni: usize = if (layers.len > 0) layers[0].numInputs() else 0;
        const no: usize = if (layers.len > 0) layers[layers.len - 1].numOutputs() else 0;

        const self = try allocator.create(SeriesLayer);
        self.* = SeriesLayer{
            .layers = owned_layers,
            .ni = ni,
            .no = no,
            .allocator = allocator,
        };
        return self;
    }

    /// Free the layers slice and the SeriesLayer itself.
    pub fn deinit(self: *SeriesLayer) void {
        self.allocator.free(self.layers);
        self.allocator.destroy(self);
    }

    /// Forward pass: chain layers sequentially.
    /// - 0 layers: copy input to output
    /// - 1 layer: forward through that single layer
    /// - 2+ layers: alternate between two temp buffers, final layer writes to output
    pub fn forward(self: *SeriesLayer, input: *const NetworkIO, output: *NetworkIO, allocator: std.mem.Allocator) ForwardError!void {
        const n = self.layers.len;

        if (n == 0) {
            // No-op: copy input to output, preserving stride.
            try output.resize(input.width(), input.numFeatures());
            output.stride_height = input.stride_height;
            output.stride_width = input.stride_width;
            for (0..input.width()) |t| {
                output.copyTimeStepFrom(t, input, t);
            }
            return;
        }

        if (n == 1) {
            try self.layers[0].forward(input, output, allocator);
            return;
        }

        // 2+ layers: use alternating temp buffers
        var buf_a = try NetworkIO.init(allocator, 1, 1, false);
        defer buf_a.deinit();
        var buf_b = try NetworkIO.init(allocator, 1, 1, false);
        defer buf_b.deinit();

        // First layer: input -> buf_a
        try self.layers[0].forward(input, &buf_a, allocator);

        // Middle layers: alternate between buf_a and buf_b
        var i: usize = 1;
        while (i < n - 1) : (i += 1) {
            if (i % 2 == 1) {
                // buf_a -> buf_b
                try self.layers[i].forward(&buf_a, &buf_b, allocator);
            } else {
                // buf_b -> buf_a
                try self.layers[i].forward(&buf_b, &buf_a, allocator);
            }
        }

        // Last layer: write to output
        const last_idx = n - 1;
        if (last_idx % 2 == 1) {
            // Previous wrote to buf_a (odd index reads from buf_a)
            try self.layers[last_idx].forward(&buf_a, output, allocator);
        } else {
            // Previous wrote to buf_b (even index reads from buf_b)
            try self.layers[last_idx].forward(&buf_b, output, allocator);
        }
    }
};

// ── ParallelLayer ────────────────────────────────────────────────────────────
//
// Runs all child layers on the SAME input in parallel and concatenates their
// outputs along the feature dimension. The output has width equal to the
// input width and num_features equal to the sum of all children's output
// feature counts.

pub const ParallelLayer = struct {
    layers: []Layer,
    ni: usize, // input features (same for all children)
    no: usize, // output features = sum of all children's outputs
    allocator: std.mem.Allocator,

    /// Allocate a ParallelLayer on the heap. Copies the provided layers slice.
    /// Computes no as the sum of all children's numOutputs().
    pub fn init(allocator: std.mem.Allocator, layers: []const Layer) !*ParallelLayer {
        const owned_layers = try allocator.alloc(Layer, layers.len);
        @memcpy(owned_layers, layers);

        const ni: usize = if (layers.len > 0) layers[0].numInputs() else 0;

        var no: usize = 0;
        for (layers) |layer| {
            no += layer.numOutputs();
        }

        const self = try allocator.create(ParallelLayer);
        self.* = ParallelLayer{
            .layers = owned_layers,
            .ni = ni,
            .no = no,
            .allocator = allocator,
        };
        return self;
    }

    /// Free the layers slice and the ParallelLayer itself.
    pub fn deinit(self: *ParallelLayer) void {
        self.allocator.free(self.layers);
        self.allocator.destroy(self);
    }

    /// Forward pass: run each child on the input, concatenate outputs.
    /// Output shape: [input.width()][self.no]
    /// Features are laid out as: child_0 features, then child_1 features, etc.
    pub fn forward(self: *ParallelLayer, input: *const NetworkIO, output: *NetworkIO, allocator: std.mem.Allocator) ForwardError!void {
        const w = input.width();
        try output.resize(w, self.no);
        // Preserve 2D stride from input.
        output.stride_height = input.stride_height;
        output.stride_width = input.stride_width;

        var temp = try NetworkIO.init(allocator, 1, 1, false);
        defer temp.deinit();

        // Temp buffer to read one timestep from the child output.
        const max_child_outputs = blk: {
            var m: usize = 0;
            for (self.layers) |layer| {
                const co = layer.numOutputs();
                if (co > m) m = co;
            }
            break :blk m;
        };
        const ts_buf = try allocator.alloc(f32, max_child_outputs);
        defer allocator.free(ts_buf);

        var offset: usize = 0;
        for (self.layers) |*layer| {
            try layer.forward(input, &temp, allocator);

            const child_no = layer.numOutputs();

            // Copy each timestep's features into the correct offset in output
            for (0..w) |t| {
                temp.readTimeStep(t, ts_buf[0..child_no]);

                // Write into the output at the correct feature offset
                const out_slice = output.f_data.?;
                const out_offset = t * self.no + offset;
                @memcpy(out_slice[out_offset .. out_offset + child_no], ts_buf[0..child_no]);
            }

            offset += child_no;
        }
    }
};

// ── Layer Lifecycle ──────────────────────────────────────────────────────────

/// Recursively free all resources owned by a deserialized Layer graph.
/// For layers created via deserialization, this properly cleans up heap-allocated
/// children, weight matrices, and state buffers.
pub fn deinitLayer(allocator: std.mem.Allocator, layer: *Layer) void {
    switch (layer.*) {
        .fully_connected => |*fc| fc.deinit(allocator),
        .lstm => |*l| l.deinit(),
        .series => |s| {
            for (s.layers) |*child| {
                deinitLayer(allocator, child);
            }
            s.deinit();
        },
        .parallel => |p| {
            for (p.layers) |*child| {
                deinitLayer(allocator, child);
            }
            p.deinit();
        },
        .reversed => |r| r.deinit(),
        .convolve => {},
        .maxpool => {},
        .input => {},
    }
}

// ── Network Deserialization ─────────────────────────────────────────────────
//
// Deserializes a complete Tesseract network graph from the TESSDATA_LSTM binary
// component. The format consists of a common header for every node followed by
// type-specific data that may recurse (containers like Series/Parallel read a
// child count and then recursively deserialize each child).

pub const DeserializeError = error{
    UnexpectedEof,
    InvalidTypeName,
    UnsupportedNetworkType,
    UnsupportedOldFormat,
    InvalidDimensions,
    Unsupported2DLSTM,
    OutOfMemory,
};

/// Mapping from Tesseract type name strings to our layer construction logic.
const TypeMapping = struct {
    name: []const u8,
    kind: TypeKind,
};

const TypeKind = enum {
    input,
    convolve,
    maxpool, // also Reconfig
    series,
    parallel,
    reversed_rtl,
    reversed_ttb,
    xy_transpose,
    lstm,
    lstm_softmax,
    lstm_binary_softmax,
    fc_softmax,
    fc_softmax_no_ctc,
    fc_sigmoid,
    fc_tanh,
    fc_relu,
    fc_linear,
    par_bidi_lstm,
    par_2d_lstm,
};

const type_mappings = [_]TypeMapping{
    .{ .name = "Input", .kind = .input },
    .{ .name = "Convolve", .kind = .convolve },
    .{ .name = "Maxpool", .kind = .maxpool },
    .{ .name = "Reconfig", .kind = .maxpool },
    .{ .name = "Series", .kind = .series },
    .{ .name = "Parallel", .kind = .parallel },
    .{ .name = "RTLReversed", .kind = .reversed_rtl },
    .{ .name = "TTBReversed", .kind = .reversed_ttb },
    .{ .name = "XYTranspose", .kind = .xy_transpose },
    .{ .name = "LSTM", .kind = .lstm },
    .{ .name = "SummLSTM", .kind = .lstm },
    .{ .name = "LSTMSoftmax", .kind = .lstm_softmax },
    .{ .name = "LSTMBinarySoftmax", .kind = .lstm_binary_softmax },
    .{ .name = "Softmax", .kind = .fc_softmax },
    .{ .name = "SoftmaxNoCTC", .kind = .fc_softmax_no_ctc },
    .{ .name = "Logistic", .kind = .fc_sigmoid },
    .{ .name = "LinLogistic", .kind = .fc_sigmoid },
    .{ .name = "Tanh", .kind = .fc_tanh },
    .{ .name = "LinTanh", .kind = .fc_tanh },
    .{ .name = "Relu", .kind = .fc_relu },
    .{ .name = "Linear", .kind = .fc_linear },
    .{ .name = "ParBidiLSTM", .kind = .par_bidi_lstm },
    .{ .name = "Par2dLSTM", .kind = .par_2d_lstm },
    .{ .name = "Replicated", .kind = .parallel },
    .{ .name = "DepParUDLSTM", .kind = .parallel },
};

fn lookupTypeKind(name: []const u8) ?TypeKind {
    for (type_mappings) |m| {
        if (std.mem.eql(u8, m.name, name)) return m.kind;
    }
    return null;
}

/// Network flags (matching Tesseract network.h)
const NF_LAYER_SPECIFIC_LR: i32 = 64;

/// Deserialize a complete network graph from Tesseract binary format.
/// Returns the root Layer. Caller must call deinitLayer() to free all resources.
pub fn deserializeNetwork(allocator: std.mem.Allocator, reader: *weights_mod.BinaryReader) DeserializeError!Layer {
    // ── Read common header ──
    // First byte: i8 = 0 (NT_NONE marker indicating string-based type lookup)
    const nt_marker = reader.readI8() catch return error.UnexpectedEof;
    if (nt_marker != 0) return error.InvalidTypeName;

    // Type name string
    const type_name = reader.readString(allocator) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.UnexpectedEof,
    };
    defer allocator.free(type_name);

    const kind = lookupTypeKind(type_name) orelse return error.UnsupportedNetworkType;

    // Training state (i8) - read and ignored for inference
    _ = reader.readI8() catch return error.UnexpectedEof;
    // Needs to backprop (i8) - read and ignored
    _ = reader.readI8() catch return error.UnexpectedEof;
    // Network flags (i32)
    const network_flags = reader.readI32() catch return error.UnexpectedEof;
    // ni (i32) - number of inputs
    const ni_i32 = reader.readI32() catch return error.UnexpectedEof;
    // no (i32) - number of outputs
    const no_i32 = reader.readI32() catch return error.UnexpectedEof;
    // num_weights (i32) - read and ignored
    _ = reader.readI32() catch return error.UnexpectedEof;
    // name string
    const name = reader.readString(allocator) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.UnexpectedEof,
    };
    defer allocator.free(name);

    const ni: usize = if (ni_i32 > 0) @intCast(ni_i32) else 0;
    const no: usize = if (no_i32 > 0) @intCast(no_i32) else 0;

    // ── Dispatch to type-specific deserialization ──
    switch (kind) {
        .input => {
            // StaticShape::DeSerialize: 5 x i32 (batch, height, width, depth, loss_type)
            const batch = reader.readI32() catch return error.UnexpectedEof;
            const height = reader.readI32() catch return error.UnexpectedEof;
            const width = reader.readI32() catch return error.UnexpectedEof;
            const depth = reader.readI32() catch return error.UnexpectedEof;
            _ = reader.readI32() catch return error.UnexpectedEof; // loss_type, ignored

            return Layer{ .input = InputLayer{
                .ni = ni,
                .no = no,
                .batch = batch,
                .height = height,
                .width = width,
                .depth = depth,
            } };
        },

        .convolve => {
            // Convolve::DeSerialize: i32 half_x, i32 half_y
            const half_x_i32 = reader.readI32() catch return error.UnexpectedEof;
            const half_y_i32 = reader.readI32() catch return error.UnexpectedEof;
            const half_x: usize = @intCast(half_x_i32);
            const half_y: usize = if (half_y_i32 > 0) @intCast(half_y_i32) else 0;

            return Layer{ .convolve = ConvolveLayer.init(ni, half_x, half_y) };
        },

        .maxpool => {
            // Reconfig::DeSerialize: i32 x_scale, i32 y_scale
            const x_scale_i32 = reader.readI32() catch return error.UnexpectedEof;
            const y_scale_i32 = reader.readI32() catch return error.UnexpectedEof;
            const x_scale: usize = @intCast(x_scale_i32);
            const y_scale: usize = if (y_scale_i32 > 0) @intCast(y_scale_i32) else 1;

            return Layer{ .maxpool = MaxpoolLayer.init(ni, x_scale, y_scale) };
        },

        .series, .parallel, .par_bidi_lstm, .par_2d_lstm => {
            return deserializePlumbing(allocator, reader, kind, ni, network_flags);
        },

        .reversed_rtl, .reversed_ttb, .xy_transpose => {
            return deserializePlumbing(allocator, reader, kind, ni, network_flags);
        },

        .lstm, .lstm_softmax, .lstm_binary_softmax => {
            return deserializeLSTM(allocator, reader, kind, ni, no);
        },

        .fc_softmax, .fc_softmax_no_ctc, .fc_sigmoid, .fc_tanh, .fc_relu, .fc_linear => {
            const activation: ActivationType = switch (kind) {
                .fc_softmax, .fc_softmax_no_ctc => .softmax,
                .fc_sigmoid => .sigmoid,
                .fc_tanh => .tanh,
                .fc_relu => .relu,
                .fc_linear => .linear,
                else => unreachable,
            };

            // FullyConnected::DeSerialize: WeightMatrix::DeSerialize(training=false)
            var wm = weights_mod.WeightMatrix.deserialize(allocator, reader) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                error.UnsupportedOldFormat => return error.UnsupportedOldFormat,
                error.InvalidDimensions => return error.InvalidDimensions,
                error.UnexpectedEof => return error.UnexpectedEof,
            };
            errdefer wm.deinit();

            return Layer{ .fully_connected = FullyConnectedLayer{
                .weights = wm,
                .activation = activation,
                .ni = wm.num_inputs,
                .no = wm.num_outputs,
            } };
        },
    }
}

/// Deserialize a Plumbing-derived type (Series, Parallel, Reversed).
/// Reads child_count + children, optionally followed by per-layer learning rates.
fn deserializePlumbing(
    allocator: std.mem.Allocator,
    reader: *weights_mod.BinaryReader,
    kind: TypeKind,
    ni: usize,
    network_flags: i32,
) DeserializeError!Layer {
    // Plumbing::DeSerialize: u32 child_count, then child_count recursive networks
    const child_count = reader.readU32() catch return error.UnexpectedEof;

    const children = allocator.alloc(Layer, child_count) catch return error.OutOfMemory;
    var children_initialized: usize = 0;
    errdefer {
        for (children[0..children_initialized]) |*child| {
            deinitLayer(allocator, child);
        }
        allocator.free(children);
    }

    for (0..child_count) |i| {
        children[i] = try deserializeNetwork(allocator, reader);
        children_initialized += 1;
    }

    // If NF_LAYER_SPECIFIC_LR flag is set, read (and skip) the learning rate vector.
    if ((network_flags & NF_LAYER_SPECIFIC_LR) != 0) {
        // GenericVector<float>::DeSerialize: u32 count, then count * sizeof(float) bytes.
        // Tesseract's Plumbing::learning_rates_ is std::vector<float>.
        const lr_count = reader.readU32() catch return error.UnexpectedEof;
        reader.skip(lr_count * 4) catch return error.UnexpectedEof;
    }

    switch (kind) {
        .series => {
            // Compute ni from first child, no from last child
            var series_no: usize = 0;
            if (children.len > 0) {
                series_no = children[children.len - 1].numOutputs();
            }
            const series_ni: usize = if (children.len > 0) children[0].numInputs() else 0;

            const series = allocator.create(SeriesLayer) catch return error.OutOfMemory;
            series.* = SeriesLayer{
                .layers = children,
                .ni = series_ni,
                .no = series_no,
                .allocator = allocator,
            };
            return Layer{ .series = series };
        },
        .parallel, .par_bidi_lstm, .par_2d_lstm => {
            var total_no: usize = 0;
            for (children) |child| {
                total_no += child.numOutputs();
            }

            const parallel = allocator.create(ParallelLayer) catch return error.OutOfMemory;
            parallel.* = ParallelLayer{
                .layers = children,
                .ni = ni,
                .no = total_no,
                .allocator = allocator,
            };
            return Layer{ .parallel = parallel };
        },
        .reversed_rtl, .reversed_ttb, .xy_transpose => {
            // Reversed has exactly 1 child
            if (children.len != 1) return error.InvalidDimensions;

            const rev_type: ReversedType = switch (kind) {
                .reversed_rtl => .x_reversed,
                .reversed_ttb => .y_reversed,
                .xy_transpose => .xy_transpose,
                else => unreachable,
            };

            const reversed = ReversedLayer.init(allocator, children[0], rev_type) catch return error.OutOfMemory;
            // Free the children slice (the single child is now owned by ReversedLayer)
            allocator.free(children);
            // Clear the errdefer tracking since we transferred ownership
            children_initialized = 0;

            return Layer{ .reversed = reversed };
        },
        else => unreachable,
    }
}

/// Deserialize an LSTM layer.
/// Reads na_, then 4 gate weight matrices (CI, GI, GF1, GO).
/// For LSTMSoftmax/LSTMBinarySoftmax types, also reads an embedded softmax network.
fn deserializeLSTM(
    allocator: std.mem.Allocator,
    reader: *weights_mod.BinaryReader,
    kind: TypeKind,
    ni: usize,
    no: usize,
) DeserializeError!Layer {
    // LSTM::DeSerialize: i32 na_
    const na_i32 = reader.readI32() catch return error.UnexpectedEof;
    if (na_i32 <= 0) return error.InvalidDimensions;
    const na: usize = @intCast(na_i32);

    // Determine ns (number of hidden states) from na and ni.
    // For 1D LSTM: na = ni + ns, so ns = na - ni
    // For 2D LSTM: na = ni + 2*ns, so ns = (na - ni) / 2
    // We check for 2D after reading the first gate weights.

    // Read 4 gate weight matrices: CI, GI, GF1, GO
    // We read CI first to determine ns.
    var gate_weights: [NUM_GATES_1D]weights_mod.WeightMatrix = undefined;
    var gates_initialized: usize = 0;
    errdefer {
        for (0..gates_initialized) |g| {
            gate_weights[g].deinit();
        }
    }

    // Read CI
    gate_weights[0] = weights_mod.WeightMatrix.deserialize(allocator, reader) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.UnsupportedOldFormat => return error.UnsupportedOldFormat,
        error.InvalidDimensions => return error.InvalidDimensions,
        error.UnexpectedEof => return error.UnexpectedEof,
    };
    gates_initialized = 1;

    const ns = gate_weights[0].num_outputs;

    // Check for 2D: if na - nf == ni + 2*ns, it's 2D
    // For standard LSTM type (not softmax), nf = 0
    const nf: usize = switch (kind) {
        .lstm => 0,
        .lstm_softmax => no,
        .lstm_binary_softmax => blk: {
            // ceil(log2(no))
            if (no <= 1) break :blk 0;
            var val: usize = no - 1;
            var bits: usize = 0;
            while (val > 0) {
                val >>= 1;
                bits += 1;
            }
            break :blk bits;
        },
        else => unreachable,
    };

    const is_2d = (na - nf) == ni + 2 * ns;
    if (is_2d) return error.Unsupported2DLSTM;

    // Read remaining gates: GI, GF1, GO
    for (1..NUM_GATES_1D) |g| {
        gate_weights[g] = weights_mod.WeightMatrix.deserialize(allocator, reader) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.UnsupportedOldFormat => return error.UnsupportedOldFormat,
            error.InvalidDimensions => return error.InvalidDimensions,
            error.UnexpectedEof => return error.UnexpectedEof,
        };
        gates_initialized += 1;
    }

    // For LSTMSoftmax/LSTMBinarySoftmax, read the embedded softmax network.
    // We skip this for now -- eng.traineddata uses plain LSTM type.
    if (kind == .lstm_softmax or kind == .lstm_binary_softmax) {
        return error.UnsupportedNetworkType;
    }

    // Allocate state buffers
    const curr_state = allocator.alloc(f32, ns) catch return error.OutOfMemory;
    errdefer allocator.free(curr_state);
    @memset(curr_state, 0.0);

    const curr_output = allocator.alloc(f32, ns) catch return error.OutOfMemory;
    errdefer allocator.free(curr_output);
    @memset(curr_output, 0.0);

    return Layer{ .lstm = LSTMLayer{
        .ni = ni,
        .ns = ns,
        .na = na,
        .gate_weights = gate_weights,
        .curr_state = curr_state,
        .curr_output = curr_output,
        .allocator = allocator,
    } };
}

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

// ── Series / Parallel / Layer Union Tests ────────────────────────────────────

test "SeriesLayer chains two FC layers" {
    const allocator = std.testing.allocator;

    // FC1: 3 inputs -> 2 outputs (linear)
    // Weights: each output = sum of all inputs
    //   Row 0: [1, 1, 1, bias=0] -> out0 = x0 + x1 + x2
    //   Row 1: [1, 1, 1, bias=0] -> out1 = x0 + x1 + x2
    var fc1 = try FullyConnectedLayer.init(allocator, 3, 2, .linear);
    defer fc1.deinit(allocator);
    fc1.weights.set(0, 0, 1.0);
    fc1.weights.set(0, 1, 1.0);
    fc1.weights.set(0, 2, 1.0);
    fc1.weights.setBias(0, 0.0);
    fc1.weights.set(1, 0, 1.0);
    fc1.weights.set(1, 1, 1.0);
    fc1.weights.set(1, 2, 1.0);
    fc1.weights.setBias(1, 0.0);

    // FC2: 2 inputs -> 1 output (linear)
    // Weight: out = sum of both inputs
    //   Row 0: [1, 1, bias=0] -> out0 = y0 + y1
    var fc2 = try FullyConnectedLayer.init(allocator, 2, 1, .linear);
    defer fc2.deinit(allocator);
    fc2.weights.set(0, 0, 1.0);
    fc2.weights.set(0, 1, 1.0);
    fc2.weights.setBias(0, 0.0);

    // Series(FC1, FC2): 3 inputs -> 1 output
    const child_layers = [_]Layer{
        Layer{ .fully_connected = fc1 },
        Layer{ .fully_connected = fc2 },
    };
    const series = try SeriesLayer.init(allocator, &child_layers);
    defer series.deinit();

    try std.testing.expectEqual(@as(usize, 3), series.ni);
    try std.testing.expectEqual(@as(usize, 1), series.no);

    // Input: 1 timestep, [1.0, 2.0, 3.0]
    var input = try NetworkIO.init(allocator, 1, 3, false);
    defer input.deinit();
    const t0_in = [_]f32{ 1.0, 2.0, 3.0 };
    input.writeTimeStep(0, &t0_in);

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try series.forward(&input, &output, allocator);

    // FC1: each output = 1+2+3 = 6 -> [6, 6]
    // FC2: output = 6+6 = 12 -> [12]
    var out: [1]f32 = undefined;
    output.readTimeStep(0, &out);

    const tol: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), out[0], tol);
}

test "ParallelLayer concatenates outputs" {
    const allocator = std.testing.allocator;

    // FC1: 2 inputs -> 1 output (linear), picks input[0]
    //   Row 0: [1, 0, bias=0]
    var fc1 = try FullyConnectedLayer.init(allocator, 2, 1, .linear);
    defer fc1.deinit(allocator);
    fc1.weights.set(0, 0, 1.0);
    fc1.weights.set(0, 1, 0.0);
    fc1.weights.setBias(0, 0.0);

    // FC2: 2 inputs -> 1 output (linear), picks input[1]
    //   Row 0: [0, 1, bias=0]
    var fc2 = try FullyConnectedLayer.init(allocator, 2, 1, .linear);
    defer fc2.deinit(allocator);
    fc2.weights.set(0, 0, 0.0);
    fc2.weights.set(0, 1, 1.0);
    fc2.weights.setBias(0, 0.0);

    // Parallel(FC1, FC2): 2 inputs -> 2 outputs (concatenated)
    const child_layers = [_]Layer{
        Layer{ .fully_connected = fc1 },
        Layer{ .fully_connected = fc2 },
    };
    const parallel = try ParallelLayer.init(allocator, &child_layers);
    defer parallel.deinit();

    try std.testing.expectEqual(@as(usize, 2), parallel.ni);
    try std.testing.expectEqual(@as(usize, 2), parallel.no);

    // Input: 1 timestep, [3.0, 7.0]
    var input = try NetworkIO.init(allocator, 1, 2, false);
    defer input.deinit();
    const t0_in = [_]f32{ 3.0, 7.0 };
    input.writeTimeStep(0, &t0_in);

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try parallel.forward(&input, &output, allocator);

    // FC1 picks input[0]=3.0, FC2 picks input[1]=7.0
    // Output: [3.0, 7.0] (concatenated)
    try std.testing.expectEqual(@as(usize, 1), output.width());
    try std.testing.expectEqual(@as(usize, 2), output.numFeatures());

    var out: [2]f32 = undefined;
    output.readTimeStep(0, &out);

    const tol: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), out[1], tol);
}

test "SeriesLayer single layer passthrough" {
    const allocator = std.testing.allocator;

    // FC: 2 inputs -> 2 outputs (linear), identity + bias
    //   Row 0: [1, 0, bias=0.5]
    //   Row 1: [0, 1, bias=-1.0]
    var fc = try FullyConnectedLayer.init(allocator, 2, 2, .linear);
    defer fc.deinit(allocator);
    fc.weights.set(0, 0, 1.0);
    fc.weights.set(0, 1, 0.0);
    fc.weights.setBias(0, 0.5);
    fc.weights.set(1, 0, 0.0);
    fc.weights.set(1, 1, 1.0);
    fc.weights.setBias(1, -1.0);

    // Series with 1 layer
    const child_layers = [_]Layer{Layer{ .fully_connected = fc }};
    const series = try SeriesLayer.init(allocator, &child_layers);
    defer series.deinit();

    try std.testing.expectEqual(@as(usize, 2), series.ni);
    try std.testing.expectEqual(@as(usize, 2), series.no);

    // Input: 1 timestep, [3.0, 7.0]
    var input = try NetworkIO.init(allocator, 1, 2, false);
    defer input.deinit();
    const t0_in = [_]f32{ 3.0, 7.0 };
    input.writeTimeStep(0, &t0_in);

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try series.forward(&input, &output, allocator);

    // Expected: [3.0+0.5, 7.0-1.0] = [3.5, 6.0]
    var out: [2]f32 = undefined;
    output.readTimeStep(0, &out);

    const tol: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), out[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out[1], tol);
}

test "Layer union dispatches correctly" {
    const allocator = std.testing.allocator;

    // Create a FullyConnected layer: 2 inputs -> 1 output, linear
    //   Row 0: [1, 1, bias=0] -> sum of inputs
    var fc = try FullyConnectedLayer.init(allocator, 2, 1, .linear);
    defer fc.deinit(allocator);
    fc.weights.set(0, 0, 1.0);
    fc.weights.set(0, 1, 1.0);
    fc.weights.setBias(0, 0.0);

    // Wrap in a Layer union
    var layer = Layer{ .fully_connected = fc };

    // Input: 1 timestep, [5.0, 3.0]
    var input = try NetworkIO.init(allocator, 1, 2, false);
    defer input.deinit();
    const t0_in = [_]f32{ 5.0, 3.0 };
    input.writeTimeStep(0, &t0_in);

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    // Call forward through the Layer union dispatch
    try layer.forward(&input, &output, allocator);

    // Expected: 5.0 + 3.0 = 8.0
    try std.testing.expectEqual(@as(usize, 1), output.width());
    try std.testing.expectEqual(@as(usize, 1), output.numFeatures());

    var out: [1]f32 = undefined;
    output.readTimeStep(0, &out);

    const tol: f32 = 1e-5;
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), out[0], tol);

    // Verify numOutputs dispatch
    try std.testing.expectEqual(@as(usize, 1), layer.numOutputs());
}

// ── ConvolveLayer / MaxpoolLayer Tests ───────────────────────────────────────

test "ConvolveLayer stacks neighbors" {
    const allocator = std.testing.allocator;

    // ni=2, half_x=1 (window of 3) -> no=6
    const conv = ConvolveLayer.init(2, 1, 0);
    try std.testing.expectEqual(@as(usize, 6), conv.no);

    // Input: width=4, features=2
    //   t=0: [1, 2], t=1: [3, 4], t=2: [5, 6], t=3: [7, 8]
    var input = try NetworkIO.init(allocator, 4, 2, false);
    defer input.deinit();
    input.writeTimeStep(0, &[_]f32{ 1.0, 2.0 });
    input.writeTimeStep(1, &[_]f32{ 3.0, 4.0 });
    input.writeTimeStep(2, &[_]f32{ 5.0, 6.0 });
    input.writeTimeStep(3, &[_]f32{ 7.0, 8.0 });

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try conv.forward(&input, &output, allocator);

    try std.testing.expectEqual(@as(usize, 4), output.width());
    try std.testing.expectEqual(@as(usize, 6), output.numFeatures());

    const tol: f32 = 1e-6;

    // Output at t=0: [0,0, 1,2, 3,4] (left padded with zeros)
    var out0: [6]f32 = undefined;
    output.readTimeStep(0, &out0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out0[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out0[1], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out0[2], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out0[3], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out0[4], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out0[5], tol);

    // Output at t=1: [1,2, 3,4, 5,6] (t0, t1, t2 stacked)
    var out1: [6]f32 = undefined;
    output.readTimeStep(1, &out1);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out1[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out1[1], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out1[2], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out1[3], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out1[4], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out1[5], tol);

    // Output at t=3: [5,6, 7,8, 0,0] (right padded with zeros)
    var out3: [6]f32 = undefined;
    output.readTimeStep(3, &out3);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out3[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out3[1], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), out3[2], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), out3[3], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out3[4], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out3[5], tol);
}

test "ConvolveLayer no padding center" {
    const allocator = std.testing.allocator;

    // ni=3, half_x=1 (window of 3) -> no=9
    const conv = ConvolveLayer.init(3, 1, 0);
    try std.testing.expectEqual(@as(usize, 9), conv.no);

    // Input: width=5, features=3
    var input = try NetworkIO.init(allocator, 5, 3, false);
    defer input.deinit();
    input.writeTimeStep(0, &[_]f32{ 1.0, 2.0, 3.0 });
    input.writeTimeStep(1, &[_]f32{ 4.0, 5.0, 6.0 });
    input.writeTimeStep(2, &[_]f32{ 7.0, 8.0, 9.0 });
    input.writeTimeStep(3, &[_]f32{ 10.0, 11.0, 12.0 });
    input.writeTimeStep(4, &[_]f32{ 13.0, 14.0, 15.0 });

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try conv.forward(&input, &output, allocator);

    const tol: f32 = 1e-6;

    // Center positions (t=1, t=2, t=3) should have no zeros -- all from valid input
    // t=2: [t1, t2, t3] = [4,5,6, 7,8,9, 10,11,12]
    var out2: [9]f32 = undefined;
    output.readTimeStep(2, &out2);
    for (out2) |v| {
        try std.testing.expect(v != 0.0);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out2[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out2[1], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out2[2], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), out2[3], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), out2[4], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), out2[5], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), out2[6], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), out2[7], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), out2[8], tol);

    // t=1 and t=3 also should have all nonzero values (they're center enough)
    var out1: [9]f32 = undefined;
    output.readTimeStep(1, &out1);
    for (out1) |v| {
        try std.testing.expect(v != 0.0);
    }

    var out3: [9]f32 = undefined;
    output.readTimeStep(3, &out3);
    for (out3) |v| {
        try std.testing.expect(v != 0.0);
    }
}

test "MaxpoolLayer downsamples" {
    const allocator = std.testing.allocator;

    // ni=2, x_scale=2
    const mp = MaxpoolLayer.init(2, 2, 1);
    try std.testing.expectEqual(@as(usize, 2), mp.no);

    // Input: width=4, features=2
    //   t=0: [1, 4], t=1: [3, 2], t=2: [5, 8], t=3: [7, 6]
    var input = try NetworkIO.init(allocator, 4, 2, false);
    defer input.deinit();
    input.writeTimeStep(0, &[_]f32{ 1.0, 4.0 });
    input.writeTimeStep(1, &[_]f32{ 3.0, 2.0 });
    input.writeTimeStep(2, &[_]f32{ 5.0, 8.0 });
    input.writeTimeStep(3, &[_]f32{ 7.0, 6.0 });

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try mp.forward(&input, &output, allocator);

    // Output: width=2
    try std.testing.expectEqual(@as(usize, 2), output.width());
    try std.testing.expectEqual(@as(usize, 2), output.numFeatures());

    const tol: f32 = 1e-6;

    // out_t=0: max([1,4], [3,2]) = [3, 4]
    var out0: [2]f32 = undefined;
    output.readTimeStep(0, &out0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out0[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out0[1], tol);

    // out_t=1: max([5,8], [7,6]) = [7, 8]
    var out1: [2]f32 = undefined;
    output.readTimeStep(1, &out1);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), out1[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), out1[1], tol);
}

test "MaxpoolLayer odd width" {
    const allocator = std.testing.allocator;

    // ni=2, x_scale=2
    const mp = MaxpoolLayer.init(2, 2, 1);

    // Input: width=5, features=2
    //   t=0: [1, 2], t=1: [3, 4], t=2: [5, 6], t=3: [7, 8], t=4: [9, 10]
    var input = try NetworkIO.init(allocator, 5, 2, false);
    defer input.deinit();
    input.writeTimeStep(0, &[_]f32{ 1.0, 2.0 });
    input.writeTimeStep(1, &[_]f32{ 3.0, 4.0 });
    input.writeTimeStep(2, &[_]f32{ 5.0, 6.0 });
    input.writeTimeStep(3, &[_]f32{ 7.0, 8.0 });
    input.writeTimeStep(4, &[_]f32{ 9.0, 10.0 });

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try mp.forward(&input, &output, allocator);

    // Output: width = ceil(5/2) = 3
    try std.testing.expectEqual(@as(usize, 3), output.width());
    try std.testing.expectEqual(@as(usize, 2), output.numFeatures());

    const tol: f32 = 1e-6;

    // out_t=0: max([1,2], [3,4]) = [3, 4]
    var out0: [2]f32 = undefined;
    output.readTimeStep(0, &out0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out0[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out0[1], tol);

    // out_t=1: max([5,6], [7,8]) = [7, 8]
    var out1: [2]f32 = undefined;
    output.readTimeStep(1, &out1);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), out1[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), out1[1], tol);

    // out_t=2: only t=4 in window -> [9, 10]
    var out2: [2]f32 = undefined;
    output.readTimeStep(2, &out2);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), out2[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), out2[1], tol);
}

test "Convolve in Layer union dispatches" {
    const allocator = std.testing.allocator;

    // Wrap ConvolveLayer in Layer, call forward, verify
    const conv = ConvolveLayer.init(2, 1, 0);
    var layer = Layer{ .convolve = conv };

    try std.testing.expectEqual(@as(usize, 6), layer.numOutputs());

    // Input: width=3, features=2
    //   t=0: [1, 2], t=1: [3, 4], t=2: [5, 6]
    var input = try NetworkIO.init(allocator, 3, 2, false);
    defer input.deinit();
    input.writeTimeStep(0, &[_]f32{ 1.0, 2.0 });
    input.writeTimeStep(1, &[_]f32{ 3.0, 4.0 });
    input.writeTimeStep(2, &[_]f32{ 5.0, 6.0 });

    var output = try NetworkIO.init(allocator, 1, 1, false);
    defer output.deinit();

    try layer.forward(&input, &output, allocator);

    try std.testing.expectEqual(@as(usize, 3), output.width());
    try std.testing.expectEqual(@as(usize, 6), output.numFeatures());

    const tol: f32 = 1e-6;

    // t=1 (center): [1,2, 3,4, 5,6]
    var out1: [6]f32 = undefined;
    output.readTimeStep(1, &out1);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out1[0], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out1[1], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out1[2], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out1[3], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out1[4], tol);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), out1[5], tol);
}

// ── Deserialization Integration Tests ────────────────────────────────────────

const tessdata = @import("tessdata.zig");

test "deserialize network from eng.traineddata" {
    const allocator = std.testing.allocator;

    // Load the model file
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

    // Parse the container
    const mgr = try tessdata.TessdataManager.init(data);
    const lstm_data = mgr.getComponent(.lstm) orelse {
        std.debug.print("Skipping test: no LSTM component\n", .{});
        return;
    };

    // Deserialize the network graph
    var reader = weights_mod.BinaryReader.init(lstm_data);
    var root = try deserializeNetwork(allocator, &reader);
    defer deinitLayer(allocator, &root);

    // 1. Root should be a Series
    try std.testing.expect(root == .series);
    const series = root.series;

    // 2. Should have multiple children
    try std.testing.expect(series.layers.len > 0);

    // 3. First child should be an Input layer
    try std.testing.expect(series.layers[0] == .input);
    const input_layer = series.layers[0].input;
    // eng.traineddata Input: height=36, depth=1
    try std.testing.expect(input_layer.height > 0);
    try std.testing.expect(input_layer.depth >= 1);

    // 4. Verify LSTM layers are present somewhere in the graph
    var has_lstm = false;
    for (series.layers) |layer| {
        switch (layer) {
            .lstm => {
                has_lstm = true;
            },
            .reversed => |r| {
                // LSTM is often inside a Reversed layer
                switch (r.child) {
                    .lstm => {
                        has_lstm = true;
                    },
                    .series => |inner_s| {
                        for (inner_s.layers) |inner_layer| {
                            if (inner_layer == .lstm) has_lstm = true;
                        }
                    },
                    else => {},
                }
            },
            .parallel => |p| {
                for (p.layers) |p_child| {
                    switch (p_child) {
                        .lstm => has_lstm = true,
                        .reversed => |pr| {
                            if (pr.child == .lstm) has_lstm = true;
                        },
                        .series => |ps| {
                            for (ps.layers) |ps_child| {
                                switch (ps_child) {
                                    .lstm => has_lstm = true,
                                    .reversed => |psr| {
                                        if (psr.child == .lstm) has_lstm = true;
                                    },
                                    else => {},
                                }
                            }
                        },
                        else => {},
                    }
                }
            },
            else => {},
        }
    }
    try std.testing.expect(has_lstm);

    // 5. Last child should be a FC/Softmax layer
    const last = series.layers[series.layers.len - 1];
    try std.testing.expect(last == .fully_connected);
    const fc_last = last.fully_connected;
    try std.testing.expect(fc_last.activation == .softmax);

    // 6. Verify reasonable dimensions
    // The eng model has 111 Unicode characters + 1 null = 112 outputs
    // (the Softmax outputs should match the unicharset size)
    try std.testing.expect(fc_last.no > 50); // at least 50 output classes
    try std.testing.expect(fc_last.no < 500); // sanity upper bound

    // 7. Verify weight matrix dimensions are non-zero
    try std.testing.expect(fc_last.weights.num_outputs > 0);
    try std.testing.expect(fc_last.weights.num_inputs > 0);

    // 8. Verify network ni/no make sense
    try std.testing.expect(series.ni > 0);
    try std.testing.expect(series.no > 0);
    try std.testing.expectEqual(fc_last.no, series.no);

    // 9. Count the layers for a sanity check
    // eng.traineddata should have Input + several LSTM-related + Softmax
    try std.testing.expect(series.layers.len >= 3);
}

test "deserialize network leaves recognizer metadata" {
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

    const mgr = try tessdata.TessdataManager.init(data);
    const lstm_data = mgr.getComponent(.lstm).?;

    var reader = weights_mod.BinaryReader.init(lstm_data);
    var root = try deserializeNetwork(allocator, &reader);
    defer deinitLayer(allocator, &root);

    // The LSTM component contains the network graph followed by LSTMRecognizer
    // metadata (spec string, training iteration, sample iteration, etc.).
    // The network deserializer should consume the graph portion, leaving the
    // recognizer metadata. For eng.traineddata this is ~81 bytes.
    const remaining = reader.remaining();
    try std.testing.expect(remaining > 0); // recognizer metadata present
    try std.testing.expect(remaining < 256); // but not too much

    // The remaining data starts with the network spec string.
    // Verify it's a valid u32 length + ASCII string.
    const spec_len = reader.readU32() catch unreachable;
    try std.testing.expect(spec_len > 0 and spec_len < 256);
    const spec_bytes = reader.readBytes(spec_len) catch unreachable;
    // The spec string should start with '[' and contain layer descriptors.
    try std.testing.expectEqual(@as(u8, '['), spec_bytes[0]);
}
