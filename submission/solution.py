from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# 权重半精度开关：只作用于 4 个大矩阵的权重，其余数值路径保持 fp32。
# "fp16" 或 "bf16"；"all"=qkv+out+fc_in+fc_out，"mlp"=只 fc_in/fc_out（更保守）
WEIGHT_DTYPE = "fp16"
HALF_WEIGHT_SCOPE = "all"

# 诊断开关（v4 首轮 DCU 一致率崩坏后加入）：
#   1) ALLOW_FP16_REDUCED_PRECISION=False：禁止 fp16/bf16 矩阵乘使用低精度累加。
#      PyTorch 默认 True；在 hipBLAS 上可能退化为 fp16 累加，误差足以让 argmax 全错。
#   2) COMPILE_MODEL=False：关闭 torch.compile，用于判断误差是否与编译/融合有关。
ALLOW_FP16_REDUCED_PRECISION = False
COMPILE_MODEL = True


class _HalfWeightLinear(nn.Module):
    """权重为半精度、输入/输出/偏置保持 fp32 的 Linear。

    x(fp32) -> cast 到 weight dtype -> F.linear -> cast 回 fp32 -> + bias(fp32)
    只有矩阵乘这一步用低精度，后续 LN/attention/softmax/logits 仍为 fp32。
    """

    def __init__(self, linear: nn.Linear, weight_dtype: torch.dtype):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight_dtype = weight_dtype
        self.weight = nn.Parameter(
            linear.weight.detach().to(weight_dtype).contiguous(), requires_grad=False
        )
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().float().contiguous(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x.to(self.weight_dtype), self.weight)
        out = out.float()
        if self.bias is not None:
            out = out + self.bias
        return out


class Solution:
    team_name = "starter_fp32_baseline"

    def __init__(self, model_dir: str, tokenizer_path: str, device: str = "cuda:0"):
        import torch
        from tokenizers import Tokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer_path = Path(tokenizer_path).resolve()
        self.model_dir = Path(model_dir).resolve()

        # 关闭半精度矩阵乘的低精度累加（默认 True 时 hipBLAS 可能用 fp16 累加）
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = ALLOW_FP16_REDUCED_PRECISION
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = ALLOW_FP16_REDUCED_PRECISION
            print(f"reduced-precision reduction: fp16/bf16 = {ALLOW_FP16_REDUCED_PRECISION}")
        except Exception as e:
            print(f"设置 reduced-precision reduction 失败: {e}")

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DCU/HIP is not visible through torch.cuda")

        import os
        _submission_dir = os.path.dirname(os.path.abspath(__file__))
        if _submission_dir not in sys.path:
            sys.path.insert(0, _submission_dir)

        asset_root = self.tokenizer_path.parent
        if str(asset_root) not in sys.path:
            sys.path.append(str(asset_root))

        from models.progen.modeling_progen import ProGenForCausalLM

        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        token_ids = [self.tokenizer.token_to_id(letter) for letter in AMINO_ACIDS]
        if any(token_id is None for token_id in token_ids):
            raise RuntimeError("Tokenizer is missing a canonical amino-acid token")
        self.allowed_ids = torch.tensor(
            [int(token_id) for token_id in token_ids],
            dtype=torch.long,
            device=self.device,
        )

        self.model = ProGenForCausalLM.from_pretrained(
            str(self.model_dir),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.model.eval().to(self.device)

        # 权重半精度必须在 torch.compile 之前完成（compile 看到的就是 fp16 权重图）
        self._install_half_weights()

        if COMPILE_MODEL:
            try:
                self.model = torch.compile(self.model, mode="default")
                print("torch.compile 已开启")
            except Exception as e:
                print(f"torch.compile 失败，回退到原始模型: {e}")
        else:
            print("torch.compile 已关闭（COMPILE_MODEL=False）")

    def _install_half_weights(self) -> None:
        dtype = torch.float16 if WEIGHT_DTYPE.lower() == "fp16" else torch.bfloat16
        converted = 0
        for block in self.model.transformer.h:
            if HALF_WEIGHT_SCOPE in ("all", "attn"):
                block.attn.qkv_proj = _HalfWeightLinear(block.attn.qkv_proj, dtype)
                block.attn.out_proj = _HalfWeightLinear(block.attn.out_proj, dtype)
                converted += 2
            if HALF_WEIGHT_SCOPE in ("all", "mlp"):
                block.mlp.fc_in = _HalfWeightLinear(block.mlp.fc_in, dtype)
                block.mlp.fc_out = _HalfWeightLinear(block.mlp.fc_out, dtype)
                converted += 2
        self._half_linear_count = converted
        print(f"权重半精度已开启: {dtype}, 替换 {converted} 个 Linear（scope={HALF_WEIGHT_SCOPE}）")

    def reset(self) -> None:
        pass

    @staticmethod
    def _check_prefixes(prefixes: Sequence[str]) -> list[str]:
        if not isinstance(prefixes, (list, tuple)) or not prefixes:
            raise ValueError("prefixes must be a non-empty list of strings")
        values = [str(prefix) for prefix in prefixes]
        if any(not value for value in values):
            raise ValueError("prefixes cannot contain an empty string")
        return values

    @property
    def metadata(self) -> dict:
        return {
            "team_name": self.team_name,
            "precision": "fp32 activations + %s weights" % WEIGHT_DTYPE,
            "description": "compile + static kv + half-precision weights (sensitive ops kept fp32)",
            "half_linear_count": getattr(self, "_half_linear_count", 0),
        }

    def generate(self, prefixes: list[str], max_new_tokens: int) -> list[str]:
        torch = self.torch
        prefixes = self._check_prefixes(prefixes)
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        encoded = [self.tokenizer.encode(prefix).ids for prefix in prefixes]
        if len({len(row) for row in encoded}) != 1:
            raise ValueError("all prefixes in one request must have equal token length")
        input_ids = torch.tensor(encoded, dtype=torch.long, device=self.device)

        generated_steps = []
        past_key_values = None
        next_input = input_ids

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=next_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                logits = outputs.logits[:, -1, :]
                restricted = logits.index_select(-1, self.allowed_ids)
                local_index = restricted.argmax(dim=-1)
                next_token = self.allowed_ids.index_select(0, local_index).unsqueeze(-1)
                generated_steps.append(next_token)
                past_key_values = outputs.past_key_values
                next_input = next_token

        generated = torch.cat(generated_steps, dim=1).detach().cpu().tolist()
        return [
            "".join(self.tokenizer.id_to_token(int(token_id)) for token_id in row)
            for row in generated
        ]
