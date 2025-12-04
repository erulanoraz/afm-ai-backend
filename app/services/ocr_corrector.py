# app/services/ocr_corrector.py
import logging
from typing import List
import re

from app.services.llm_client import LLMClient

logger = logging.getLogger("OCR_CORRECTOR")

llm_client = LLMClient()


# ================================
# Разбиение текста
# ================================
def _split_to_chunks(text: str, max_chars: int = 6000) -> List[str]:
    """
    Улучшенный вариант:
    1) если есть маркеры страниц "--- Page X ---" → режем по ним
    2) иначе — режем по предложениям, чтобы не ломать структуру
    """

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    # 1) Разбиение по страницам
    if "--- Page" in text:
        parts = re.split(r"(--- Page \d+ ---)", text)
        merged: List[str] = []
        buf = ""

        for part in parts:
            if not part.strip():
                continue
            candidate = buf + part
            if len(candidate) > max_chars and buf:
                merged.append(buf.strip())
                buf = part
            else:
                buf += part

        if buf:
            merged.append(buf.strip())

        return merged

    # 2) Разбиение по предложениям
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""

    for s in sentences:
        if not s:
            continue
        # +1 за пробел перед предложением
        if current and len(current) + len(s) + 1 > max_chars:
            chunks.append(current.strip())
            current = s
        else:
            if current:
                current += " " + s
            else:
                current = s

    if current:
        chunks.append(current.strip())

    return chunks


# ================================
# Анти-халлюцинационная проверка
# ================================
def _is_safe_diff(before: str, after: str, threshold: float = 1.25) -> bool:
    """
    Если после LLM количество слов изменилось > 25% → считаем опасным.
    """
    b = len((before or "").split())
    a = len((after or "").split())
    if b == 0:
        return True
    return a <= b * threshold


# ================================
# Вызов LLM
# ================================
def _call_llm_ocr_corrector(chunk: str) -> str:
    if not chunk or not chunk.strip():
        return chunk

    system_prompt = (
        "Ты работаешь в режиме STRICT OCR-CORRECTOR для юридических документов.\n"
        "ТВОЯ ЗАДАЧА:\n"
        "1) Исправлять только OCR-ошибки: перепутанные буквы, разорванные/слипшиеся слова.\n"
        "2) Не менять смысл, факты, суммы, даты, имена, номера дел.\n"
        "3) Не добавлять новых фраз.\n"
        "4) Не переформулировать стилистически.\n"
        "5) Сохранять структуру: абзацы, списки, нумерация.\n"
        "6) Ответ строго: только исправленный текст."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chunk},
    ]

    try:
        response = llm_client.chat(messages, temperature=0.0)

        if not response or str(response).startswith("[LLM ERROR]"):
            logger.error(f"❌ OCR_CORRECTOR LLM error → fallback: {response}")
            return chunk

        # Поддержка dict-ответа (OpenAI-стиль)
        if isinstance(response, dict):
            try:
                response_text = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
            except Exception:
                response_text = ""
            if not response_text:
                response_text = str(response)
        else:
            response_text = str(response)

        response_text = response_text.strip()
        if not response_text:
            return chunk

        # анти-халлюцинационная проверка
        if not _is_safe_diff(chunk, response_text):
            logger.warning("⚠️ OCR_CORRECTOR: слишком сильное отличие → fallback")
            return chunk

        return response_text

    except Exception as e:
        logger.error(f"❌ Exception in _call_llm_ocr_corrector: {e}", exc_info=True)
        return chunk


# ================================
# Основная функция
# ================================
def correct_ocr_text(raw_text: str) -> str:
    if not raw_text or not raw_text.strip():
        return raw_text

    try:
        chunks = _split_to_chunks(raw_text, max_chars=6000)
        if not chunks:
            return raw_text

        corrected: List[str] = []
        total = len(chunks)

        for idx, ch in enumerate(chunks, start=1):
            logger.info(f"🧠 OCR_CORRECTOR: chunk {idx}/{total}, len={len(ch)}")
            fixed = _call_llm_ocr_corrector(ch)
            corrected.append(fixed)

        result = "\n\n".join(corrected).strip()
        return result or raw_text

    except Exception as e:
        logger.error(f"❌ correct_ocr_text fatal error: {e}", exc_info=True)
        return raw_text
