from __future__ import annotations

from typing import Any

from .openai_compatible import OpenAICompatibleProvider
from .transformers_local import TransformersLocalProvider


def build_provider(config: dict[str, Any]):
    provider_name = config.get("provider")
    if provider_name == "openai_compatible":
        return OpenAICompatibleProvider(
            model_name=config["model_name"],
            base_url=config["base_url"],
            api_key=config.get("api_key", "EMPTY"),
            temperature=float(config.get("temperature", 0.0)),
            max_new_tokens=int(config.get("max_new_tokens", 64)),
            timeout=int(config.get("timeout", 300)),
        )
    if provider_name == "transformers":
        return TransformersLocalProvider(
            model_name_or_path=config["model_name_or_path"],
            temperature=float(config.get("temperature", 0.0)),
            max_new_tokens=int(config.get("max_new_tokens", 64)),
            device_map=config.get("device_map", "auto"),
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            torch_dtype=config.get("torch_dtype", "auto"),
        )
    raise ValueError(f"Unsupported provider: {provider_name}")

