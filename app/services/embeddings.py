# app/services/embeddings.py
import requests
from app.utils.config import settings

class Embedder:
    def embed_query(self, text: str):
        payload = {"model": "nomic-embed-text", "input": text}
        headers = {"Authorization": f"Bearer {settings.LLM_API_KEY}"}
        try:
            r = requests.post(f"{settings.LLM_API_URL}/embeddings", json=payload, headers=headers, timeout=20)
            r.raise_for_status()
            data = r.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            print("[Embedder ERROR]", e)
            return []
