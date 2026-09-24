"""
DeepSeek-V4.1 routed MoE: BlockSparseMLP plus the model's load guard
(architecture/dsv41/placement.py).

Where the routed experts live is decided exactly as for every other MoE model:
-mcl / --moe_cpu_offload and -mcs / --moe_cpu_split go through BlockSparseMLP
unchanged. What V4.1 adds is a check of the layer split: a compressed layer
reads the paged pool of its kv source on its own device, so with a Cache
attached the autosplit may change device only at a layer where no pool, top-k
selection or candidate list is shared across the change (a free cut). Every
DSV41MoE asks the shared load guard at the top of its load whether its block
may land on the offered device; a change of device that is not a free cut
raises RuntimeError there, before anything of this module is loaded or handed
to the CPU worker.

Only DeepseekV41Model builds this class, and outside its load the guard
accepts every device.
"""

from __future__ import annotations

import torch
from typing_extensions import override

from .block_sparse_mlp import BlockSparseMLP
from ..architecture.dsv41.placement import TP_REFUSAL


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
