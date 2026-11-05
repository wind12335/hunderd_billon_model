#!/usr/bin/env python3
"""快速 sanity：检查转换出的 checkpoint 结构。"""
import torch

CKPT = "/ws/iluvatar_framework/generated/ckpt_13b/iter_0000001/mp_rank_00/model_optim_rng.pt"

sd = torch.load(CKPT, map_location="cpu", weights_only=True)
m = sd["model"]
print("version", sd["checkpoint_version"], "iter", sd["iteration"])
print("total keys", len(m))
ks = sorted(k for k in m if not k.endswith("_extra_state"))
print("param keys", len(ks))
for k in ks:
    if ".0." in k or "embedding" in k or "output" in k or "final" in k:
        print(" ", k, tuple(m[k].shape), m[k].dtype)
print("extra_state entries:", sum(1 for k in m if k.endswith("_extra_state")))

# 校验总参数量
total = sum(m[k].numel() for k in ks)
print("total params", f"{total/1e9:.3f}B")
