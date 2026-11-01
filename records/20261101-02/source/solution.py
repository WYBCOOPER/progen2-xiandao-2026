from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class Solution:
    team_name = "starter_fp32_baseline"

    def __init__(self, model_dir: str, tokenizer_path: str, device: str = "cuda:0"):
        import torch
        from tokenizers import Tokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer_path = Path(tokenizer_path).resolve()
        self.model_dir = Path(model_dir).resolve()

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

        try:
            self.model = torch.compile(self.model, mode="default")
            print("torch.compile 已开启")
        except Exception as e:
            print(f"torch.compile 失败，回退到原始模型: {e}")

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
            "precision": "fp32",
            "description": "Upstream attention with dynamic KV cache",
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
