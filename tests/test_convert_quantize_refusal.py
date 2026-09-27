import os, sys, subprocess, unittest
from unittest import mock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.conversion import convert_model


class _Config:
    architecture = "TestForCausalLM"
    model_classes = {}

class _Model:
    def __init__(self, caps):
        self.caps = caps
    def get_layout_tree(self, indent):
        return ""


def _get_base_model(caps):
    with mock.patch.object(convert_model.Config, "from_directory", return_value = _Config()), \
         mock.patch.object(convert_model.Model, "from_config", return_value = _Model(caps)):
        return convert_model.get_base_model({"in_dir": "unused", "vision_bits": 16})


class ConvertQuantizeRefusalTest(unittest.TestCase):

    def test_refused_architecture_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "Cannot quantize this model type: TestForCausalLM"):
            _get_base_model({"can_quantize": False, "uncalibrated_quantize": True})

    def test_default_unchanged(self):
        for caps in ({"uncalibrated_quantize": True}, {"can_quantize": True, "uncalibrated_quantize": True}):
            config, model, mtp_model, vision_model, tokenizer, use_reference_state = _get_base_model(caps)
            self.assertIs(model.caps, caps)
            self.assertIsNone(tokenizer)

    def test_refusal_survives_optimize_flag(self):
        # python -O strips asserts; the refusal must not depend on them
        r = subprocess.run(
            [sys.executable, "-B", "-O", "-m", "unittest", "-q", "test_convert_quantize_refusal.ConvertQuantizeRefusalTest.test_refused_architecture_raises"],
            cwd = os.path.dirname(os.path.abspath(__file__)),
            capture_output = True, text = True, timeout = 600,
        )
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
