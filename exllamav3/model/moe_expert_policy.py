"""
How the routed experts that CPU MoE offload keeps in system RAM are computed. Each offloaded
layer registers with the CPU worker (moe_cpu_host.MoeCpuHost.register_layer) in one of two
execution modes, fixed for as long as the layer stays loaded:

  hybrid  the CPU worker computes the experts. During prefill, experts with at least
          EXL3_MOE_STREAM_T assignments in a chunk of at least EXL3_MOE_STREAM_MIN_ROWS rows are
          streamed to the layer's GPU and computed there instead (the default behaviour of
          -mcl / -mcs)
  stream  every active expert of every call, one-token decode included, is streamed to the
          layer's GPU and computed there. No CPU expert job is ever submitted for the layer,
          and a layer that cannot stream (missing metadata, staging slots too small) fails to
          load instead of falling back to the CPU

config.infer_params.moe_cpu_mode (EXL3_MOE_CPU_MODE, -mcm / --moe_cpu_mode) selects the mode for
every offloaded layer: "compute" gives hybrid, "stream_only" gives stream.

No framework or device imports, so tests can load this file by path.
"""

CPU_MODES = ("compute", "stream_only")

# Every accepted moe_cpu_mode names its execution mode here; a value added to CPU_MODES without an
# entry fails in execution_mode instead of silently meaning hybrid
_EXECUTION = {"compute": "hybrid", "stream_only": "stream"}


def parse_cpu_mode(value) -> str:
    if not isinstance(value, str) or value not in CPU_MODES:
        raise ValueError(f"moe_cpu_mode must be one of {', '.join(CPU_MODES)}, got {value!r}")
    return value


def execution_mode(cpu_mode) -> str:
    """The execution mode ("hybrid" or "stream") a moe_cpu_mode value gives offloaded layers"""
    return _EXECUTION[parse_cpu_mode(cpu_mode)]


def validate_streaming(spec, has_aux, slot_bytes, num_slots):
    """Admission of a stream-mode layer, before a worker job or any GPU allocation: it needs
    the packed projection layout, its GPU-side scale tensors, and staging slots that hold at
    least one expert"""
    size = spec.get("expert_bytes")
    if not spec.get("proj_dims") or not has_aux or type(size) is not int or size <= 0:
        raise RuntimeError("stream-only experts need the packed projection layout and GPU-side scale tensors "
                           "of the layer")
    if type(slot_bytes) is not int or size > slot_bytes or type(num_slots) is not int or num_slots < 1:
        raise RuntimeError(f"stream-only experts: one expert ({size} bytes) needs a weight staging slot of at "
                           f"least that size ({num_slots} slots of {slot_bytes} bytes); raise "
                           f"EXL3_MOE_CPU_WSLOT_MB / EXL3_MOE_CPU_WSLOTS")


def streamed_experts(counts, threshold, gpu_only):
    """Experts a prefill call streams to the GPU, from the per-expert assignment counts. Hybrid
    layers stream the experts at or above the threshold; stream-mode layers every expert with
    at least one assignment (never an unselected one, e.g. a split layer's GPU-resident picks)"""
    if gpu_only:
        return [e for e, count in enumerate(counts) if count > 0]
    return [e for e, count in enumerate(counts) if count >= threshold]
