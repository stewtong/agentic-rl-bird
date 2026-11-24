"""Regression for vision/text ambiguity using the installed Bridge wrapper."""
import torch
from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from bird_evaluator.trainer_checks import CHECKS, unique_suffix

model = torch.nn.Module()
model.vision_model = torch.nn.Module()
model.vision_model.decoder = torch.nn.Module()
model.vision_model.decoder.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])
vision = model.vision_model.decoder.layers[3]
vision.self_attention = torch.nn.Module()
vision.self_attention.linear_proj = torch.nn.Linear(8, 4, bias=False)
model.language_model = torch.nn.Module()
model.language_model.embedding = torch.nn.Module()
model.language_model.embedding.word_embeddings = torch.nn.Embedding(4, 8)
model.language_model.decoder = torch.nn.Module()
model.language_model.decoder.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])
text = model.language_model.decoder.layers[3]
text.self_attention = torch.nn.Module()
text.self_attention.linear_proj = AdapterWrapper(torch.nn.Linear(8, 4, bias=False), torch.nn.Identity())
names = ["module.module." + name for name, _ in model.named_parameters()]
selected = [unique_suffix(names, suffix) for suffix, _, _ in CHECKS]
assert selected == ["module.module.language_model.embedding.word_embeddings.weight", "module.module.language_model.decoder.layers.3.self_attention.linear_proj.to_wrap.weight"]
try:
    unique_suffix([name for name in names if ".language_model." not in name], CHECKS[1][0])
except RuntimeError:
    pass
else:
    raise AssertionError("A vision projection must never substitute for the requested text projection")
print("PASS: installed AdapterWrapper registers text base under to_wrap; exact language selectors exclude vision")
print(selected)
