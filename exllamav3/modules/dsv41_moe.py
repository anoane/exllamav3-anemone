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

Under EXL3_EXACT_ROWS=1 a flagged forward (params["exact_rows"]) gives every
row the arithmetic of the call a decode step makes: router, routed and shared
experts. Where the extension has the row-exact router and expert entry points
and the layer is one they serve (rows_native), that is one BlockSparseMLP
forward on the rows: the router projects each row with the one-row launch, the
experts of all rows run in one launch pair with the launch geometry of one row,
and the expert cache is looked up once for the call. Otherwise the whole layer
runs one row per call.
"""

from __future__ import annotations

import torch
from typing_extensions import override

from .block_sparse_mlp import BlockSparseMLP
from .block_sparse_mlp_routing import routing_sqrtsp
from ..architecture.dsv41.placement import TP_REFUSAL
from ..ext import exllamav3_ext as ext
from ..model.expert_tier_policy import DECODE
from ..model.math_policy import (
    EXACT_ROWS, EXACT_ROWS_MAX, EXACT_ROWS_CAP_ROUTER, EXACT_ROWS_CAP_MOE, exact_rows_native,
)
from ..util.tensor import forward_rows

# EXL3_EXACT_ROWS: the row-exact entry points of the extension (0 without the switch, and with an
# extension built before them: one row per call), and the two a flagged forward of the layer needs
ROWS_NATIVE = exact_rows_native(ext)
ROWS_NATIVE_MOE = EXACT_ROWS_CAP_ROUTER | EXACT_ROWS_CAP_MOE


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
            # The router projects one row in FP32 and more rows in int8, the expert kernels size
            # their tiles from the call's (token, expert) slots and group rows that picked the
            # same expert, and a one-row call is a decode-mode lookup of the expert cache. A
            # flagged call of several rows is either one forward that the row-exact entry points
            # serve (rows_native), or one forward per row. forward_rows copies each result out of
            # the static buffer the decode kernels return a view of; the single forward returns
            # that view of all rows, as every decode-sized call does
            one_row = super().forward
            if params.get("exact_rows") and x.numel() > x.shape[-1]:
                if self.rows_native(x, params):
                    return one_row(x, params, out_dtype)
                return forward_rows(lambda row: one_row(row, params, out_dtype), x)
            return one_row(x, params, out_dtype)

        # (the bound block the answer of rows_native_layer was computed for, the answer): a load
        # builds a new block, an unload drops it
        _rows_native_layer = (None, False)

        def rows_native_layer(self) -> bool:
            """
            Whether a flagged call of several rows can be one forward on this layer as loaded: the
            sqrt-softplus router on its own gate, the fused decode kernels for the routed experts
            and, if there is one, a shared expert that is a fused launch inside them, and nothing
            else around them that follows the row count (norms, latent projections, CPU experts,
            a broadcast selection, tensor-parallel reductions). A one-row call and a call of up to
            EXACT_ROWS_MAX rows then take the same branches of BlockSparseMLP.forward
            """
            return bool(
                self.routing_fn is routing_sqrtsp and self.routing_gate is not None
                and not self.router_pre_norm and not self.routed_pre_norm and not self.routed_post_norm
                and self.latent_in is None and self.latent_out is None
                and self.routing_device is None
                and not self.cpu_offload and self.cpu_split_first is None
                and not self.tp_reduce and not self.alt_residual_channel
                and self.bc is not None and self.is_quantized
                and not self.config.infer_params.no_reconstruct
                and self.intermediate_size != 0 and self.num_local_experts != 0
                and (not self.shared_experts or self.bc_sh_exp)
                and self.bc.rows_exact_ok(self.num_experts_per_tok)
            )

        def rows_native(self, x: torch.Tensor, params: dict) -> bool:
            """
            Whether this flagged call of several rows is one BlockSparseMLP forward: the extension
            has the row-exact router and expert entry points, the layer is one they serve
            (rows_native_layer, decided once per load), and the call is an ordinary decode-sized
            one whose rows the expert cache, if any, looks up in decode mode, as it does one row
            """
            if ROWS_NATIVE & ROWS_NATIVE_MOE != ROWS_NATIVE_MOE:
                return False
            bc, layer = self._rows_native_layer
            if bc is not self.bc:
                layer = self.rows_native_layer()
                self._rows_native_layer = (self.bc, layer)
            if not layer:
                return False
            rows = x.numel() // x.shape[-1]
            return (
                x.is_contiguous() and rows <= EXACT_ROWS_MAX
                and not params.get("tp_warmup") and not params.get("activate_all_experts")
                and not params.get("autosplit_measure") and not params.get("indexed_embeddings")
                and (self.tier is None or self.tier.call_mode(rows) == DECODE)
            )
