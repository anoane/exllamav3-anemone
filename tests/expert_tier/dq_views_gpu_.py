"""
BC_BlockSparseMLP.run_single_expert_dq_views (the per-expert dequant path with the expert's trellis
tensors given) equals run_single_expert_dq bitwise: on a real DeepSeek-V4.1 3.0 bpw MoE layer
(DSV41_MODEL_DIR, layer 12 by default), for rows 33, 257 and 1024 and three experts, with the
module's own trellis tensors passed as views and with copies of them in other memory. The shapes,
types and devices of the given tensors are checked against the expert's.

    DSV41_MODEL_DIR=... python -m pytest -q -s tests/expert_tier/dq_views_gpu_.py      (a GPU window)
"""
import os
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL = os.environ.get("DSV41_MODEL_DIR")
DEV = torch.device("cuda", int(os.environ.get("EXL3_TIER_TEST_DEVICE", "0")))
LAYER = int(os.environ.get("EXL3_DQ_TEST_LAYER", "12"))


class DqViews(unittest.TestCase):

    def test_views_equal_own(self):
        if not MODEL or not torch.cuda.is_available():
            raise AssertionError("dq_views_gpu_: needs DSV41_MODEL_DIR and a CUDA GPU")
        from exllamav3 import Config, Model
        from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
        config = Config.from_directory(MODEL)
        model = Model.from_config(config)
        stack, m = list(model.modules), None
        while stack:
            x = stack.pop()
            if isinstance(x, BlockSparseMLP) and getattr(x, "layer_idx", None) == LAYER:
                m = x
                break
            stack.extend(getattr(x, "modules", []) or [])
        self.assertIsNotNone(m, f"layer {LAYER} has no MoE")
        m.load(DEV)
        try:
            g = torch.Generator(device = "cpu").manual_seed(5)
            I = m.intermediate_size_padded
            for rows in (33, 257, 1024):
                x = torch.randn((rows, m.hidden_size), generator = g).half().to(DEV)
                for e in (0, 17, m.num_experts - 1):
                    outs = []
                    for mode in ("own", "views", "copies"):
                        out = torch.empty((rows, m.expert_size), dtype = torch.float, device = DEV)
                        interm = torch.empty((rows * 2, I), dtype = m.interm_dtype, device = DEV)
                        interm_a = interm[:rows] if m.interm_dtype == torch.half else \
                            torch.empty((rows, I), dtype = torch.half, device = DEV)
                        yh = torch.empty((rows * 2, m.expert_size), dtype = torch.half, device = DEV)
                        tr = [l.inner.trellis for l in (m.gates[e], m.ups[e], m.downs[e])]
                        if mode == "own":
                            m.bc.run_single_expert_dq(x, e, yh, interm, interm_a, out)
                        else:
                            if mode == "copies":
                                tr = [t.clone() for t in tr]
                            m.bc.run_single_expert_dq_views(x, e, tr[0], tr[1], tr[2], yh, interm, interm_a, out)
                        outs.append(out.clone())
                    for i in (1, 2):
                        self.assertTrue(torch.equal(outs[0].view(torch.int32), outs[i].view(torch.int32)),
                                        f"rows {rows} expert {e} ({('views', 'copies')[i - 1]})")
            # a tensor of another shape is refused
            tr = [l.inner.trellis for l in (m.gates[0], m.ups[0], m.downs[0])]
            with self.assertRaisesRegex(RuntimeError, "trellis shape"):
                m.bc.run_single_expert_dq_views(x[:33], 0, tr[0], tr[2], tr[2], yh, interm, interm_a, out)
        finally:
            m.unload()


if __name__ == "__main__":
    unittest.main()
