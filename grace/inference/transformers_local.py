from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TransformersLocalProvider:
    model_name_or_path: str
    temperature: float = 0.0
    max_new_tokens: int = 64
    device_map: str = "auto"
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    _model: Any = field(init=False, default=None, repr=False)
    _tokenizer: Any = field(init=False, default=None, repr=False)

    def _lazy_load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = self.torch_dtype
        if dtype == "auto":
            dtype_value = "auto"
        else:
            dtype_value = getattr(torch, dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
            torch_dtype=dtype_value,
        )

    def generate(self, prompt: str) -> str:
        self._lazy_load()
        import torch

        if hasattr(self._tokenizer, "apply_chat_template"):
            inputs = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt").input_ids

        inputs = inputs.to(self._model.device)
        outputs = self._model.generate(
            inputs,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs.shape[-1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

