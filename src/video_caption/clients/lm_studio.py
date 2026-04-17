import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class LMStudioClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.extra_body = {"enable_thinking": False}
        self._llm = ChatOpenAI(
            base_url=self.base_url,
            api_key=api_key or "lm-studio",
            model=model,
            temperature=0.3,
            max_tokens=512,
            timeout=timeout,
            max_retries=max_retries,
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
        response = self._llm.invoke(messages)
        return response.content.strip()
