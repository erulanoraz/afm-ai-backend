# app/services/llm_client.py
import requests
from app.utils.config import settings

class LLMClient:
    def __init__(self):
        self.api_url = settings.LLM_API_URL
        self.api_key = settings.LLM_API_KEY

    def chat(self, messages):
        """
        messages = [{"role": "system"/"user"/"assistant", "content": "..."}]
        """
        payload = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "temperature": settings.LLM_TEMPERATURE
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # ✅ Гарантия, что путь корректный
        url = self.api_url
        if not url.endswith("/chat/completions"):
            url = url.rstrip("/") + "/chat/completions"

        try:
            r = requests.post(url, json=payload, headers=headers, timeout=settings.LLM_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[LLM ERROR]: {e}"
