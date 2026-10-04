# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for exllamav3; see NOTICE for source attribution and changes.

"""
DeepSeek-V4.1 decoder block with DELAYED hyper-connections, and the head collapse.

Derived in part from vLLM vllm/models/deepseek_v4_1/{nvidia,amd}/model.py and
vllm/model_executor/kernels/mhc/torch.py (Apache-2.0); see NOTICE.

V4.1 keeps V4's mHC arithmetic -- the same pre / post / Sinkhorn-comb mixes
from the same {key}_fn / _base / _scale tensors -- but changes WHICH pre-mix
collapses a sublayer's input. From the reference (mhc_pre_delayed_torch):

    V4    layer_input = sum_h pre_current[h] * residual[h]
    V4.1  layer_input = residual[0]                          (first sublayer)
                      = sum_h pre_CARRIED[h] * residual[h]   (every other one)
          ... and pre_current is handed on to the NEXT sublayer.

"attention uses the pre-mix carried in (identity for the first layer), the
FFN uses this layer's attention pre-mix" -- so the pre-mix runs one sublayer
behind, across sublayers AND across layers. Running V4's HyperConnection.mix
on V4.1 weights executes and stays finite, but computes the wrong function.

Two further V4.1 changes live here:

  * no learned hc_head: the checkpoint has no hc_head_* tensors, and the final
    streams are collapsed with the pre-mix produced by the LAST layer's FFN
    ("the mix the reference applies via last_layer.hc_pre(h, pre_mix)").
  * engram is injected at block ENTRY -- after the previous sublayer's post,
    before this block's attention pre -- on the full 4-stream residual, so the
    mix coefficients see the injected stream.

The carried pre-mix travels in params["dsv41_hc_pre"]. It is per token and
recomputed inside every forward (layer 0 starts from the identity), so it needs
no recurrent state -- but it MUST be reset per forward, which
DeepseekV41Model.prepare_inputs does.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing_extensions import override

from ..architecture.dsv41 import placement as dsv41_placement
from ..ext import exllamav3_ext as ext
from ..model.math_policy import (
    STABLE_ARITHMETIC, EXACT_ROWS, EXACT_ROWS_MAX, EXACT_ROWS_CAP_HC, exact_rows_native,
)
from ..util.device_copy import to_device
from ..util.tensor import g_tensor_cache, to2
from . import Module
from .dsv41_ablation import ablations
from .hyperconnections import HyperConnection
from .transformer import TransformerBlock


# EXL3_EXACT_ROWS: the row-exact entry points of the extension (0 without the switch, and with an
# extension built before them: the Python row loops)
ROWS_NATIVE = exact_rows_native(ext)


def _collapse_rows(pre: torch.Tensor, streams: torch.Tensor) -> torch.Tensor:
    """
    EXL3_EXACT_ROWS: the stream collapse (pre.unsqueeze(-1) * streams).sum(dim = 2) of pre
    (b, s, H) and streams (b, s, H, D), one token per call: (b, s, D). Every token runs the whole
    expression on its own (b, 1, ..) slices, so its sum reduces the freshly allocated (b, 1, H, D)
    product a one-token step reduces: torch chooses the layout of a reduction from its operand.
    Where the extension has the entry point, it runs this loop with the same torch operators
    (ext.hc_collapse_rows: one sequence of 2 to EXACT_ROWS_MAX tokens, FP32).
    """
    if (
        ROWS_NATIVE & EXACT_ROWS_CAP_HC and streams.shape[0] == 1 and 1 < streams.shape[1] <= EXACT_ROWS_MAX
        and pre.dtype == torch.float and streams.dtype == torch.float
    ):
        return ext.hc_collapse_rows(pre, streams)
    return torch.cat([(pre[:, j:j + 1].unsqueeze(-1) * streams[:, j:j + 1]).sum(dim = 2)
                      for j in range(streams.shape[1])], dim = 1)


class DSV41HyperConnection(HyperConnection):
    """V4's mHC mixer plus the delayed collapse V4.1 needs."""

    def mix_delayed(self, streams: torch.Tensor, params: dict, carried_pre: torch.Tensor | None,
                    own_pre: bool = False):
        """
        streams      (b, s, H, D) fp32
        carried_pre  (b, s, H) fp32 from the previous sublayer, or None for the
                     model's first sublayer (identity: select stream 0)
        own_pre      collapse with this site's own pre-mix instead (V4 semantics; the
                     EXL3_DSV41_ABLATE=v4mix negative control only)
        returns      post (b, s, H), comb (b, s, H, H), collapsed (b, s, D) fp32,
                     pre (b, s, H) -- this site's own pre-mix, for the NEXT sublayer

        post, comb and pre are V4's mixes; only the collapse differs. On a GPU they come
        from V4's fused ext.hc_mix (mix_fused), else from V4's torch body (mix_torch).

        In a flagged forward of EXL3_EXACT_ROWS (params["exact_rows"]) every token gets the
        mix of a one-token call: the fused kernel with that call's column partition, the torch
        sums one token per call. The torch body is refused there: its projection and
        reductions follow the shape of the call.
        """
        b, s, H, D = streams.shape
        exact = EXACT_ROWS and bool(params.get("exact_rows")) and b * s > 1
        if self.hc_mult == 4 and streams.is_cuda and streams.dtype == torch.float and D % 4 == 0 \
                and streams.is_contiguous():
            post, comb, pre = self.mix_fused(streams, exact)
        else:
            if exact:
                raise RuntimeError(
                    f"{self.key}: EXL3_EXACT_ROWS=1 computes the hyper-connection mix of a draft "
                    f"verification with the fused kernel only (4 contiguous FP32 CUDA streams, width a "
                    f"multiple of 4); got hc_mult {self.hc_mult}, {streams.dtype} on {streams.device}, "
                    f"width {D}, contiguous {streams.is_contiguous()}")
            post, comb, pre = self.mix_torch(streams, params)
        if own_pre:
            collapsed = _collapse_rows(pre, streams) if exact else (pre.unsqueeze(-1) * streams).sum(dim = 2)
        elif carried_pre is None:
            collapsed = streams[:, :, 0]
        elif exact:
            collapsed = _collapse_rows(carried_pre, streams)
        else:
            collapsed = (carried_pre.unsqueeze(-1) * streams).sum(dim = 2)
        return post, comb, collapsed, pre

    def mix_torch(self, streams: torch.Tensor, params: dict):
        """V4's torch mix, verbatim: (post (b, s, H), comb (b, s, H, H), pre (b, s, H))."""
        hc = self.hc_mult
        flat = self.norm.forward(streams.flatten(2), params)
        mix = F.linear(flat, self.fn)
        pre_w, post_w, comb_w = mix.split([hc, hc, hc * hc], dim = -1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_s, post_s, comb_s = self.scale.unbind(0)
        pre = torch.sigmoid(pre_w * pre_s + pre_b) + self.hc_eps
        post = 2.0 * torch.sigmoid(post_w * post_s + post_b)
        comb = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_s + comb_b.view(hc, hc)
        comb = torch.softmax(comb, dim = -1) + self.hc_eps
        comb = comb / (comb.sum(dim = -2, keepdim = True) + self.hc_eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim = -1, keepdim = True) + self.hc_eps)
            comb = comb / (comb.sum(dim = -2, keepdim = True) + self.hc_eps)
        return post.contiguous(), comb.contiguous(), pre

    def mix_fused(self, streams: torch.Tensor, exact: bool = False):
        """
        post and comb (the Sinkhorn iterations) from V4's fused ext.hc_mix; this site's pre-mix
        re-derived from the kernel's per-chunk partial dots, which the kernel leaves in its
        workspace but does not return (its own collapse uses the CURRENT pre-mix, not V4.1's
        carried one, so it is discarded). Fusing the mixing avoids the Torch body's many
        separate launches; its throughput benefit depends on the workload and hardware.

        fn stays fp32 at every row count (V4's mix switches to an fp16 copy at <= 32 rows, a
        precision step keyed on batch size). The partials workspace is sized by
        hc_mix_num_chunks: the kernel writes exactly that many chunks, which are summed here.
        Under EXL3_STABLE_ARITHMETIC=1 the workspace has one chunk at every row count (see
        below), and that chunk is the sum.

        exact (EXL3_EXACT_ROWS): every row gets the bits of a one-row call. The workspace takes
        the chunk count of one row, and the kernel takes its column partition from the workspace
        (hc_mix.cu, hc_mix_launch: n_chunks_a = partials.size(1)); a block of it reads one row,
        and a row's partials are reduced, mixed and Sinkhorn-normalized by that row's own block.
        The chunk sums are added per row here, the operand of a one-row call each (by
        ext.hc_partials_rows, the same loop with the same torch operators, where the extension
        has it).
        """
        b, s, H, D = streams.shape
        R = b * s
        st = streams.view(R, H, D)
        M1 = 2 * H + H * H + 1
        # hc_mix splits each row's projection over column chunks and the chunk count follows the
        # row count (hc_mix_num_chunks: up to 128, 80 on DeepSeek-V4.1-Flash, one from 257 rows
        # on), so the FP32 dot products of the same row are parenthesized differently in decode
        # and prefill.
        # Stable arithmetic keeps one chunk, the large-prefill partition, at every row count
        chunks = 1 if STABLE_ARITHMETIC else ext.hc_mix_num_chunks(1 if exact else R, H * D)
        dev = streams.device

        # Decode-sized calls take bucketed static workspaces, as V4's mix does (tags of their
        # own); post and comb are consumed by this site's apply_ before the next site mixes
        def ws(numel, dtype, tag):
            if R <= 32:
                return g_tensor_cache.get_bucketed(dev, numel, dtype, tag)
            return torch.empty((numel,), dtype = dtype, device = dev)
        partials = ws(R * chunks * M1, torch.float, "dsv41_hc_partials").view(R, chunks, M1)
        post = ws(R * H, torch.float, "dsv41_hc_post").view(R, H)
        comb = ws(R * H * H, torch.float, "dsv41_hc_comb").view(R, H, H)
        unused = ws(R * D, torch.half, "dsv41_hc_coll").view(R, D)
        ext.hc_mix(st, self.fn, self.base, self.scale, self.rms_eps, self.hc_eps,
                   self.sinkhorn_iters, partials, post, comb, unused)
        if STABLE_ARITHMETIC:
            p = partials[:, 0]
        elif exact and ROWS_NATIVE & EXACT_ROWS_CAP_HC and R <= EXACT_ROWS_MAX:
            p = ext.hc_partials_rows(partials)
        elif exact:
            p = torch.cat([partials[r:r + 1].sum(dim = 1) for r in range(R)], dim = 0)
        else:
            p = partials.sum(dim = 1)
        rmr = torch.rsqrt(p[:, M1 - 1:] / (H * D) + self.rms_eps)
        pre = torch.sigmoid(p[:, :H] * rmr * self.scale[0] + self.base[:H]) + self.hc_eps
        return post.view(b, s, H), comb.view(b, s, H, H), pre.view(b, s, H)

    def tp_export(self, plan, producer):
        # V4's export would rebuild a V4 HyperConnection, whose mix is not V4.1's
        raise NotImplementedError("DSV41HyperConnection: tensor-parallel loading is not supported")


def _carried(params: dict, streams: torch.Tensor) -> torch.Tensor | None:
    """The pre-mix carried in from the previous sublayer, on the streams' device."""
    pre = params.get("dsv41_hc_pre")
    if pre is not None:
        assert pre.shape == streams.shape[:-1], \
            f"carried pre-mix {tuple(pre.shape)} vs streams {tuple(streams.shape)}"
        if pre.device != streams.device:
            pre = to_device(pre, streams.device)
    return pre


class DSV41Block(TransformerBlock):
    """
    V4.1 decoder block: delayed mHC on both sublayers, engram at entry.

    The per-sublayer steps are V4's (norm, attention / MoE, the in-place
    post update); only the collapse and the pre-mix hand-off change, plus the
    engram call V4 has no equivalent of.
    """

    def __init__(self, *args, engram = None, **kwargs):
        super().__init__(*args, **kwargs)
        # A registered submodule, so Module.load/unload carry it with the block and the
        # model's iteration finds its caps: the Cache gives it recurrent state (the carried
        # n-gram ids) and Model.forward/prefill calls its prefetch. Its ~94 GiB table is NOT
        # a weight anywhere: load() opens file handles, weights_numel() excludes it, and the
        # autosplit loader measures only what the forward allocates (see dsv41_engram.py)
        self.engram = engram
        self.register_submodule(engram)

    def _engram(self, x: torch.Tensor, params: dict, skip: bool) -> torch.Tensor:
        if skip:
            # EXL3_DSV41_ABLATE=engram: the n-gram context must still advance, or clearing the
            # ablation mid-sequence would hash with stale ids
            self.engram.commit_only(params)
            return x
        return self.engram.forward(x, params)

    def tp_export(self, plan, producer):
        # V4's export would rebuild V4 mixing (and has no engram)
        raise NotImplementedError("DSV41Block: tensor-parallel loading is not supported")

    @override
    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        # EXL3_DSV41_XDEV_BF16: the FP32 residual streams cross between GPUs as BF16 and are
        # widened back to FP32 here. A BF16 input was rounded on the other side already (the
        # pipelined prefill does that at the end of its first stage). Without the switch nothing
        # is rounded or widened
        if dsv41_placement.XDEV_BF16 and x.device != self.device and x.device.type == "cuda" \
                and x.dtype in (torch.float, torch.bfloat16):
            return to_device(x.to(torch.bfloat16), self.device).float()
        return super().prepare_for_device(x, params)

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        export_state = params.get("export_state_layers")
        export_state = export_state and self.layer_idx in export_state and params.get("layer_instance", 0) == 0

        # negative controls for tests and validation runs (modules/dsv41_ablation.py)
        ablate = ablations()
        v4mix = "v4mix" in ablate

        # engram: after the previous sublayer's post, before this block's pre
        if self.engram is not None:
            x = self._engram(x, params, "engram" in ablate)

        # attention: collapse with the pre-mix carried in (identity at layer 0). The fp32
        # collapse goes straight into the norm (fp32 in, fp16 out): rounding it to fp16
        # first adds a rounding and saturates past 65504 instead of normalising
        post, comb, y, attn_pre = self.attn_hc.mix_delayed(x, params, _carried(params, x), v4mix)
        y = self.attn_norm.forward(y.contiguous(), params, out_dtype = torch.half)
        y = self.attn.forward(y, params)
        if params.get("prefill") and not export_state:
            return x
        x = self.attn_hc.apply_(x, y, post, comb, params)

        # MoE: collapse with THIS layer's attention pre-mix
        post, comb, y, ffn_pre = self.mlp_hc.mix_delayed(x, params, attn_pre, v4mix)
        y = self.mlp_norm.forward(y.contiguous(), params, out_dtype = torch.half)
        y = self.mlp.forward(y, params)
        x = self.mlp_hc.apply_(x, y, post, comb, params)

        # the next layer's attention -- or the head collapse -- consumes ffn_pre
        params["dsv41_hc_pre"] = ffn_pre

        # export_state: the block OUTPUT (the stream mean). A DSpark draft would consume the
        # block INPUTS of its target layers 37-39, after the engram; no draft class exists yet
        if export_state:
            s = params.get("export_states")
            if not s:
                s = params["export_states"] = []
            x_ = x.mean(dim = 2).half()
            x_.clamp_(-65504.0, 65504.0)
            s.append(x_)

        return to2(x, out_dtype, self.out_dtype)


class DSV41HeadCollapse(Module):
    """
    V4.1's final stream collapse. There is no learned hc_head: the streams are
    weighted by the pre-mix the LAST layer's FFN computed, then summed. Same
    output contract as V4's HyperHead -- (b, s, D) fp32 into the model norm.
    """

    def __init__(self, config, key: str, hc_mult: int):
        super().__init__(config = config, key = key, qmap = None)
        self.module_name = "DSV41HeadCollapse"
        self.hc_mult = hc_mult

    @override
    def optimizer_targets(self):
        return []

    @override
    def get_tensors(self):
        return {}

    @override
    def weights_numel(self):
        return 0

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        pre = params.get("dsv41_hc_pre")
        assert pre is not None, \
            "DSV41HeadCollapse: no pre-mix from the last layer -- did the final block run?"
        pre = to_device(pre, x.device)
        assert pre.shape == x.shape[:-1], f"pre-mix {tuple(pre.shape)} vs streams {tuple(x.shape)}"
        if EXACT_ROWS and params.get("exact_rows") and x.shape[1] > 1:
            return _collapse_rows(pre, x)
        return (pre.unsqueeze(-1) * x).sum(dim = 2)
