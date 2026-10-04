"""
DeepSeek-V4.1 routed MoE: BlockSparseMLP plus the model's load guard
(architecture/dsv41/placement.py).

Where the routed experts live is decided exactly as for every other MoE model:
-mcl / --moe_cpu_offload, -mcs / --moe_cpu_split and the experts= of an
explicit placement (EXL3_PLACEMENT) go through BlockSparseMLP unchanged. What
V4.1 adds is a check of the layer split: a compressed layer reads the paged
pool of its kv source on its own device, so with a Cache attached the
autosplit may change device only at a layer where no pool, top-k selection or
candidate list is shared across the change (a free cut), or at the split of
the explicit placement the model was built with, where a pool replica takes
over. Every DSV41MoE asks the shared load guard at the top of its load whether
its block may land on the offered device; any other change of device raises
RuntimeError there, before anything of this module is loaded or handed to the
CPU worker.

Only DeepseekV41Model builds this class, and outside its load the guard
accepts every device.

Under EXL3_EXACT_ROWS=1 a flagged forward (params["exact_rows"]) runs the
whole layer one row per call, each the call a decode step makes: router,
expert cache lookup, routed and shared experts.
"""

from __future__ import annotations

import torch
from typing_extensions import override

from .block_sparse_mlp import BlockSparseMLP
from ..architecture.dsv41.placement import TP_REFUSAL
from ..model.math_policy import EXACT_ROWS
from ..util.tensor import forward_rows


class DSV41MoE(BlockSparseMLP):

    def __init__(self, *args, layer_idx: int, load_guard = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_idx = layer_idx
        self.load_guard = load_guard

    @override
    def load(self, device: torch.Device, **kwargs):
        # before anything of this module is loaded or handed to the CPU worker
        if self.load_guard is not None:
            self.load_guard.check(self.layer_idx, device)
        super().load(device, **kwargs)

    @override
    def make_tp_allocation(self, options: dict):
        raise NotImplementedError(TP_REFUSAL)

    # Defined only under EXL3_EXACT_ROWS=1: without it the class adds nothing to
    # BlockSparseMLP.forward, not even a frame
    if EXACT_ROWS:

        @override
        def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
            # One row per call: the router projects one row in FP32 and more rows in int8, the
            # expert kernels size their tiles from the call's (token, expert) slots, and a one-row
            # call is a decode-mode lookup of the expert cache. forward_rows copies each result
            # out of the static buffer the decode kernels return a view of
            one_row = super().forward
            if params.get("exact_rows") and x.numel() > x.shape[-1]:
                return forward_rows(lambda row: one_row(row, params, out_dtype), x)
            return one_row(x, params, out_dtype)
