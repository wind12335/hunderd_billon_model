#!/usr/bin/env python3
"""逐层对比 HF 与 mcore 的 hidden state，定位等价性偏差在哪一层发散。"""
from __future__ import annotations

import sys
from pathlib import Path

_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

from megatron_musa_bootstrap import bootstrap  # noqa: E402

bootstrap(verbose=False)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from megatron.core import parallel_state  # noqa: E402
from megatron.core.models.gpt import GPTModel  # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer import TransformerConfig  # noqa: E402

HF_DIR = "/ws/model"
CKPT = "/ws/musa_framework/generated/ckpt_13b/iter_0000001/mp_rank_00/model_optim_rng.pt"
SEQ = 8
VOCAB = 69104
HIDDEN = 5120
FFN = 13824
NLAYERS = 40
NHEADS = 40


def main() -> int:
    from transformers import LlamaForCausalLM

    hf = LlamaForCausalLM.from_pretrained(HF_DIR, torch_dtype=torch.bfloat16)
    hf.eval()

    dist.init_process_group("gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:29996")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)

    config = TransformerConfig(
        num_layers=NLAYERS, hidden_size=HIDDEN, ffn_hidden_size=FFN,
        num_attention_heads=NHEADS, num_query_groups=NHEADS,
        normalization="RMSNorm", layernorm_epsilon=1e-6,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        add_bias_linear=False, params_dtype=torch.bfloat16,
    )
    if not hasattr(config, "offload_moe_fc1_input"):
        config.offload_moe_fc1_input = False
    model = GPTModel(
        config, transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=VOCAB, max_sequence_length=4096,
        position_embedding_type="rope", rotary_base=10000,
    )
    model.cuda().to(torch.bfloat16)
    model.eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True)["model"], strict=True)

    layer_outs: dict[int, torch.Tensor] = {}

    def make_hook(i: int):
        def hook(_m, _inp, out):
            layer_outs[i] = out[0] if isinstance(out, tuple) else out
        return hook

    for i in range(NLAYERS):
        model.decoder.layers[i].register_forward_hook(make_hook(i))

    torch.manual_seed(1234)
    input_ids = torch.randint(0, VOCAB, (1, SEQ), dtype=torch.long)

    with torch.no_grad():
        hf_out = hf(input_ids=input_ids, output_hidden_states=True)
        hf_hidden = hf_out.hidden_states  # 41 entries: [embed, l0, ..., l39]
        hf_norm_out = hf.model.norm(hf_hidden[-1])

        device = torch.device("musa:0")
        ids = input_ids.to(device)
        position_ids = torch.arange(SEQ, dtype=torch.long, device=device).unsqueeze(0)
        attn_mask = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)
        mc_logits = model(input_ids=ids, position_ids=position_ids, attention_mask=attn_mask)

    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(a.float().double(), b.float().double(), dim=-1).mean().item()

    print(f"  {'layer':>5} {'cosine':>10} {'norm_ratio(mc/hf)':>18}")
    # layer -1 = embedding output
    print(f"  {'emb':>5} {cos(hf_hidden[0], model.embedding.word_embeddings(ids).cpu()):>10.6f}")
    for i in range(NLAYERS):
        hf_h = hf_hidden[i + 1]           # after layer i
        mc_h = layer_outs[i].cpu()        # after layer i
        ratio = mc_h.float().norm().item() / hf_h.float().norm().item()
        print(f"  {i:>5} {cos(hf_h, mc_h):>10.6f} {ratio:>18.4f}")
    # final norm input is layer_outs[39]; compare final_layernorm outputs
    mc_final = model.decoder.final_layernorm(layer_outs[39])
    print(f"  {'fnorm':>5} {cos(hf_norm_out, mc_final.cpu()):>10.6f}")
    # output layer / lm_head
    print(f"  {'logit':>5} {cos(hf_out.logits, mc_logits.cpu()):>10.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
