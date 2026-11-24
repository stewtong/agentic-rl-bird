import unittest

import torch
from torch import nn

from adapter_effect import adapter_pairs, attach_adapters, resolve_modules


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        self.layers = nn.ModuleList([layer])

    def named_modules(self):
        return [("model.layers.0.self_attn.q_proj", self.layers[0].self_attn.q_proj)]


class AdapterEffectTest(unittest.TestCase):
    def test_loaded_pair_changes_output(self):
        model = ToyModel()
        pairs = adapter_pairs({
            "model.layers.0.self_attn.q_proj.lora_A.weight": torch.eye(4),
            "model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 4),
        })
        resolved = resolve_modules(model, pairs)
        inputs = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        before = model.layers[0].self_attn.q_proj(inputs)
        handles, records = attach_adapters(model, resolved, {"lora_alpha": 64, "r": 32})
        after = model.layers[0].self_attn.q_proj(inputs)
        for handle in handles:
            handle.remove()
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["b_nonzero"])
        self.assertGreater((after - before).abs().max().item(), 0)


if __name__ == "__main__":
    unittest.main()
