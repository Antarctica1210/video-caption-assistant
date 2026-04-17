import time

import httpx
import openai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..logger import get_logger

log = get_logger("video_caption.lm_studio")


class LMStudioClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.extra_body = {"enable_thinking": False}
        # max_retries=0: we handle retries manually so we can catch openai-level errors
        self._llm = ChatOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            model=model,
            temperature=0.3,
            max_tokens=512,
            timeout=timeout,
            max_retries=0,
            extra_body=self.extra_body,
        )

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
        messages = [SystemMessage(content=system), HumanMessage(content=text)]

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._llm.invoke(messages)
                return response.content.strip()
            except (openai.APITimeoutError, httpx.TimeoutException) as e:
                last_exc = e
                log.warning(
                    "LM Studio timeout on attempt %d/%d — retrying in %ds",
                    attempt, self.max_retries, attempt * 5,
                )
                time.sleep(attempt * 5)
            except (openai.APIConnectionError, httpx.ConnectError) as e:
                last_exc = e
                log.warning(
                    "LM Studio connection error on attempt %d/%d — retrying in %ds",
                    attempt, self.max_retries, attempt * 5,
                )
                time.sleep(attempt * 5)
            except openai.APIStatusError as e:
                # Non-retriable HTTP errors (4xx etc.) — fail immediately
                log.error("LM Studio API error (HTTP %d): %s", e.status_code, e.message)
                raise

        log.error("LM Studio failed after %d attempts: %s", self.max_retries, last_exc)
        raise RuntimeError(f"LM Studio request failed after {self.max_retries} attempts") from last_exc
