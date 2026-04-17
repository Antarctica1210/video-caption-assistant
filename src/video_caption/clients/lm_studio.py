import time
from typing import Optional

import httpx
import openai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..logger import get_logger

log = get_logger("video_caption.lm_studio")

# Global cache: key is (base_url, model, json_mode) tuple, value is ChatOpenAI instance.
# Avoids re-initialising the client on every LMStudioClient construction.
_llm_cache: dict[tuple, ChatOpenAI] = {}


def _get_llm(
    base_url: str,
    model: str,
    api_key: str | None,
    timeout: int,
    json_mode: bool = False,
) -> ChatOpenAI:
    cache_key = (base_url, model, json_mode)
    if cache_key in _llm_cache:
        log.debug("LLM cache hit: model=%s json_mode=%s", model, json_mode)
        return _llm_cache[cache_key]

    if not api_key:
        raise ValueError("Missing LM Studio API key — set LM_STUDIO_API_KEY in .env")
    if not base_url:
        raise ValueError("Missing LM Studio base URL — set LM_STUDIO_BASE_URL in .env")

    log.info("Initialising LLM client: model=%s json_mode=%s", model, json_mode)

    model_kwargs: dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    try:
        llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.3,
            max_tokens=512,
            timeout=timeout,
            max_retries=0,       # retries handled manually to catch openai-level errors
            model_kwargs=model_kwargs,
        # .bind() attaches extra_body to every invocation — constructor-level extra_body
        # is not reliably forwarded by all langchain-openai versions
        ).bind(extra_body={"enable_thinking": False})
    except Exception as e:
        raise RuntimeError(f"LLM client initialisation failed for model [{model}]: {e}") from e

    _llm_cache[cache_key] = llm
    log.info("LLM client cached: model=%s json_mode=%s", model, json_mode)
    return llm


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
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def _llm(self, json_mode: bool = False) -> ChatOpenAI:
        return _get_llm(self.base_url, self.model, self.api_key, self.timeout, json_mode)

    def health_check(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/models", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def translate(self, text: str, target_lang: str, system_prompt: Optional[str] = None) -> str:
        system = system_prompt or (
            f"You are a professional subtitle translator. "
            f"Translate the following subtitle line to {target_lang}. "
            f"Preserve the original tone and brevity. "
            f"Output only the translated text. No explanations, no punctuation changes, no extra text."
        )
        messages = [SystemMessage(content=system), HumanMessage(content=text)]
        llm = self._llm()

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = llm.invoke(messages)
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
                log.error("LM Studio API error (HTTP %d): %s", e.status_code, e.message)
                raise

        log.error("LM Studio failed after %d attempts: %s", self.max_retries, last_exc)
        raise RuntimeError(f"LM Studio request failed after {self.max_retries} attempts") from last_exc
