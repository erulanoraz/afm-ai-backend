# app/services/llm_client.py
import requests
from app.utils.config import settings


class LLMClient:
    def __init__(self):
        self.api_url = settings.LLM_API_URL.rstrip("/")  # путь берём как есть
        self.api_key = settings.LLM_API_KEY

    def chat(self, messages, temperature=None):
        """
        messages = [{"role": "system"/"user"/"assistant", "content": "..."}]
        ВАЖНО: НЕ добавляем /chat/completions автоматически.
        URL полностью задаётся в .env
        """

        payload = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None
                else settings.LLM_TEMPERATURE
            )
        }

        headers = {"Authorization": f"Bearer {self.api_key}"}

        url = self.api_url  # никаких добавлений снизу!

        try:
            r = requests.post(url, json=payload, headers=headers, timeout=settings.LLM_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            return f"[LLM ERROR]: {e}"
