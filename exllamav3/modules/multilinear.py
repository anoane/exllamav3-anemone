from __future__ import annotations
import torch
from . import Linear

class MultiLinear:
    def __init__(
        self,
        device: torch.Device,
        linears: list[Linear],
        allow_bias: bool = False,
        allow_mixed_K: bool = False,
    ):
        self.device = device
        self.linears = linears
        self.num_linears = len(linears)

        assert all(l.quant_type == "exl3" for l in linears)
        # The mgemm kernels never apply biases; callers setting allow_bias handle them
        # separately (BC_BlockSparseMLP bias kernels)
        assert allow_bias or all(l.inner.bias is None for l in linears)
        assert all(not l.softcap for l in linears)
        assert all(l.post_scale == 1.0 for l in linears)

        self.in_features = linears[0].in_features
        self.out_features = linears[0].out_features
        # per-expert K, opt-in via allow_mixed_K (block-sparse experts). self.K is
        # the uniform K when all linears share one, else None; ptrs_K always carries the
        # per-linear K table (int32, mgemm K_list convention). Callers that don't opt in keep the
        # original assert, so their AssertionError-based decline paths (dsv4 fans) still work.
        ks = [l.inner.K for l in linears]
        self.K = ks[0] if all(k == ks[0] for k in ks) else None
        assert allow_mixed_K or self.K is not None
        assert all(l.in_features == self.in_features for l in linears)
        assert all(l.out_features == self.out_features for l in linears)

        self.ptrs_suh = torch.tensor([l.inner.suh.data_ptr() for l in linears], dtype = torch.long, device = device)
        self.ptrs_svh = torch.tensor([l.inner.svh.data_ptr() for l in linears], dtype = torch.long, device = device)
        self.ptrs_trellis = torch.tensor([l.inner.trellis.data_ptr() for l in linears], dtype = torch.long, device = device)
        self.ptrs_K = torch.tensor(ks, dtype = torch.int32, device = device)

        self.mcg = linears[0].inner.mcg
        assert all(l.inner.mcg == self.mcg for l in linears[1:])
        self.mul1 = linears[0].inner.mul1
        assert all(l.inner.mul1 == self.mul1 for l in linears[1:])

    def q_cb(self):
        return self.mcg, self.mul1

    def unload(self):
        pass
