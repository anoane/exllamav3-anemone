"""
experts=cache for BlockSparseMLP (doc/expert_tiers.md, "The VRAM cache"): the layer's routed experts
live in its GPU's expert cache (model/expert_tier.py) instead of being loaded with the layer.

The layer is a resident layer in every other respect. Its expert Linears leave the module tree for
the load (as the CPU split's tail experts do) and get their EXL3 inner here: suh, svh, mcg / mul1
and bias resident, the trellis a view of the GPU's zeroed sentinel slot, so load_local builds its
MultiLinear pointer tables, bound classes and fused buffers exactly as for a resident layer. Every
MoE call then runs the expert cache right after routing (tier_resolve), which points the tables at
the slots holding the call's experts and holds the stream until they are there, and tells it when
the layer's compute is enqueued (tier_after_compute: a layer-mode prefill plans the next cache
layer then). The quantized fast paths read those tables on the device (the fused decode kernels,
the fused MoE kernel, the batched reconstruct tier); the per-expert dequant path gets the trellis
views explicitly (run_single_expert_dq_views). A path that would read a per-expert trellis tensor
itself (the graph and torch paths) is never reached: only layers the fused kernel covers can be
cached, and the forward raises should one be reached anyway. With prefetch=router a decode call
also applies a later cache layer's router to its own router input (tier_predict), so the disk
reads what that layer will probably miss ahead of it.
"""

from __future__ import annotations
import torch

from .quant.exl3 import LinearEXL3
from ..ext import exllamav3_ext as ext


class BlockSparseMLP_Tier:

    def _tier_init_state(self):
        self.tier = None                # TierDevice of this layer's GPU (experts=cache), else None
        self.tier_layer = None
        self._tier_modules = None
        self._tier_pred_bufs = {}       # rows -> private routing outputs of tier_predict_ids

    def _tier_plan(self):
        plan = self.placement_plan() if hasattr(self, "placement_plan") else None
        return plan if plan is not None and plan[0] == "cache" else None

    def _tier_linears(self):
        return (list(self.gates) if self.gated else []) + list(self.ups) + list(self.downs)

    def tier_maybe_attach(self, device, **kwargs) -> bool:
        """Before the module's own load: an experts=cache layer registers with its GPU's expert
        cache, takes its expert Linears out of the module tree and gives them sentinel-backed EXL3
        inners. False for any other layer"""
        plan = self._tier_plan()
        if plan is None:
            return False
        from ..model.expert_tier_config import ineligible
        tiers = getattr(self.config, "expert_tiers", None)
        if tiers is None:
            raise RuntimeError(f"placement: {self.key} is experts=cache, but this load has no expert tier (the text "
                               f"component of a layer-split load with a placement has one)")
        stc = self.config.stc
        ok = (
            (self.num_local_experts is None or self.num_local_experts == self.num_experts) and
            self.routing_first is None and self.num_experts <= 512 and
            all(l.fkey is None and stc.has_tensor(l.key + ".trellis") and l.alt_key is None
                for l in self._tier_linears())
        )
        if not ok:
            raise RuntimeError(ineligible(self.key))
        td, layer = tiers.register(self, device, plan[2], self.placement_layer_index())
        self.tier, self.tier_layer = td, layer
        try:
            lin = self._tier_linears()
            members = set(id(l) for l in lin)
            self._tier_modules = self.modules
            self.modules = [m for m in self.modules if id(m) not in members]
            dev = torch.device(device)
            E = self.num_experts
            shape_of = lambda l: tuple(stc.list_tensors(l.key)[l.key + ".trellis"]["shape"])
            shapes = [shape_of(lin[p * E]) for p in range(len(lin) // E)]
            td.set_geometry(shapes)
            for i, l in enumerate(lin):
                if shape_of(l) != shapes[i // E]:
                    raise RuntimeError(ineligible(self.key))
                self._tier_load_inner(l, dev, td.sentinel_view(i // E))
        except Exception:
            self.tier_unload()
            raise
        return True

    @staticmethod
    def _tier_load_inner(l, device, trellis):
        """Linear.load_exl3 with the trellis given (every other tensor from the checkpoint)"""
        stc = l.config.stc
        key = l.key

        def opt(subkey, dev, **kw):
            k = key + subkey
            return stc.get_tensor(k, dev, optional = True, **kw) if stc.has_tensor(k) else None

        l.device = device
        su = opt(".su", device, no_defer = True)
        suh = opt(".suh", device)
        sv = opt(".sv", device, no_defer = True)
        svh = opt(".svh", device)
        mcg = opt(".mcg", "cpu")
        mul1 = opt(".mul1", "cpu")
        bias = opt(".bias", device)
        l.inner = LinearEXL3(l.config, l.in_features, l.out_features, None, su, sv, suh, svh, trellis, mcg, mul1,
                             bias, l.out_dtype, key = key)
        l.quant_type = "exl3"
        l.used_alt_key = False

    def tier_register(self):
        """After load_local: the layer must be one the fused MoE kernel covers; its pointer tables go
        to the expert cache"""
        if self.tier is None:
            return
        from ..model.expert_tier_config import ineligible
        if not (self.support_fused and self.bc is not None and self.multi_up is not None and
                not self.config.infer_params.no_reconstruct):
            raise RuntimeError(ineligible(self.key))
        tables = ([self.multi_gate.ptrs_trellis] if self.gated else []) + \
                 [self.multi_up.ptrs_trellis, self.multi_down.ptrs_trellis]
        lin = self._tier_linears()
        shapes = [tuple(lin[p * self.num_experts].inner.trellis.shape) for p in range(len(tables))]
        self.tier.set_tables(self.tier_layer, tables, shapes)

    def tier_resolve(self, selected_experts, bsz: int, params: dict):
        """Right after routing: the call's experts in VRAM, the tables pointing at them"""
        if params.get("autosplit_measure") or self.tier.handle is None:
            return                      # the load's measuring forward runs on the sentinel
        self.tier.resolve(self.tier_layer, selected_experts, bsz)

    def tier_after_compute(self, params: dict):
        """The layer's routed compute is enqueued (layer mode: plan the next cache layer)"""
        if params.get("autosplit_measure") or self.tier.handle is None:
            return
        self.tier.after_compute(self.tier_layer)

    def tier_predict(self, z, bsz: int, params: dict):
        """prefetch=router: a decode call predicts a later cache layer's experts from this router input"""
        if params.get("autosplit_measure") or self.tier.handle is None or self.tier_layer.predict_to is None:
            return
        self.tier.predict(self.tier_layer, z, bsz)

    def tier_predict_ids(self, z) -> torch.Tensor:
        """The experts this layer's router picks for router input z (another layer's), [rows, top-k]
        int64 on this layer's device, in private buffers (the routing workspaces are shared by every
        layer of the device). DeepSeek's sqrt-softplus router runs its own selection kernel (the same
        picks this layer would make for z); other routers select on the gate logits plus the
        selection bias, which predicts but need not equal their picks"""
        from .block_sparse_mlp_routing import routing_sqrtsp, _gate_t, _esb_h, ROUTING_ACT_SQRTSP
        cfg = self.routing_cfg
        rows = z.shape[0]
        if self.routing_fn is routing_sqrtsp:
            bufs = self._tier_pred_bufs.get(rows)
            if bufs is None:
                E, K = cfg.num_experts, cfg.num_experts_per_tok
                bufs = self._tier_pred_bufs[rows] = (
                    torch.empty((rows, E), dtype = torch.half, device = self.device),
                    torch.empty((rows, K), dtype = torch.long, device = self.device),
                    torch.empty((rows, K), dtype = torch.half, device = self.device),
                )
            logits, sel, w = bufs
            _gate_t(cfg)
            ext.routing_ds3_nogroup(z, cfg.gate_tensor, logits, _esb_h(cfg), sel, w, cfg.routed_scaling_factor,
                                    cfg.gate_tensor_t, ROUTING_ACT_SQRTSP, cfg.gate_i8, cfg.gate_sb)
            return sel
        scores = torch.matmul(z.to(cfg.gate_tensor.dtype), cfg.gate_tensor).float()
        if cfg.e_score_correction_bias is not None:
            scores = scores.sigmoid() + cfg.e_score_correction_bias.float()
        return scores.topk(cfg.num_experts_per_tok, dim = -1).indices

    def tier_trellis_views(self, expert_idx: int) -> list:
        """(gate or None, up, down) trellis views of an expert, for run_single_expert_dq_views"""
        v = self.tier.trellis_views(self.tier_layer, expert_idx)
        return v if self.gated else [None] + v

    def tier_guard(self, what: str):
        raise RuntimeError(f"{self.key}: experts=cache layers compute through the fused MoE paths only; the {what} "
                           f"path would read an expert's own trellis tensor (a sentinel)")

    def tier_unload(self):
        if self._tier_modules is not None:
            for l in self._tier_linears():
                if l.inner is not None:
                    l.unload()
            self.modules = self._tier_modules
            self._tier_modules = None
        if self.tier is not None:
            self.tier.tiers.unregister(self)
        self.tier = None
        self.tier_layer = None
        self._tier_pred_bufs = {}
