from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass
class OpenAICompatibleProvider:
    model_name: str
    base_url: str
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_new_tokens: int = 64
    timeout: int = 300

    def generate(self, prompt: str) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]

