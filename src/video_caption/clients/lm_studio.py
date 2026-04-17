import httpx


class LMStudioClient:
    def __init__(self, base_url: str, model: str, timeout: int = 60, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = {"enable_thinking": False}

    def health_check(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/models", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def translate(self, text: str, target_lang: str, system_prompt: str | None = None) -> str:
        system = system_prompt or (
            f"You are a professional subtitle translator. "
            f"Translate the following text to {target_lang}. "
            f"Preserve the original tone and brevity. "
            f"Output only the translated text, no explanations."
        )

        last_exc: Exception | None = None
        for _ in range(self.max_retries):
            try:
                r = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": text},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 512,
                        **self.extra_body,
                    },
                    timeout=self.timeout,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exc = e

        raise RuntimeError(f"LM Studio translation failed after {self.max_retries} retries") from last_exc
