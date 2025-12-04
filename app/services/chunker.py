# =====================================================================
# 📦 Chunker 7.1 — PostgreSQL-only (no Weaviate), SentenceSplitter v15
# =====================================================================

import os
import re
import uuid
import logging
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session
from PyPDF2 import PdfReader
from transformers import GPT2TokenizerFast

from app.db.models import Chunk
from app.services.ocr_worker import (
    extract_text_from_pdf,
    run_tesseract_ocr,
)
from app.services.parser import extract_text_from_file
from app.utils.config import settings
from app.tasks.vector_tasks import enqueue_chunk_vectorization
from app.utils.sentence_splitter import split_into_sentences  # v15.0

logger = logging.getLogger("CHUNKER7")

# =====================================================================
# 🔧 UUID safety
# =====================================================================

def ensure_uuid(value) -> Optional[uuid.UUID]:
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except Exception:
        logger.error(f"❌ Некорректный UUID: {value}")
        return None


# =====================================================================
# 🔧 Normalization
# =====================================================================

def _normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r", "")

    garbage = [
        r"©\s?Все права защищены.*",
        r"сканировано\s?с\s?помощью.*",
        r"страница\s*\d+\s*из\s*\d+.*",
        r"Документ создан.*",
        r"QR[- ]?код.*",
        r"электронный документ.*",
        r"Просмотрено.*",
        r"Дата печати.*",
    ]
    for g in garbage:
        t = re.sub(g, "", t, flags=re.IGNORECASE)

    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# =====================================================================
# 🔎 Detect section
# =====================================================================

SECTION_PATTERNS = {
    "protocol": r"(ПРОТОКОЛ( ДОПРОСА)?|PROTOCOL)",
    "postanovlenie": r"(ПОСТАНОВЛЕНИЕ|ПОСТАНОВИЛ|ПОСТАНОВЛЯЮ)",
    "raport": r"(РАПОРТ|RAPORT)",
    "obiasnenie": r"(ОБЪЯСНЕНИЕ|Объяснение)",
    "prilojenie": r"(ПРИЛОЖЕНИЕ|Приложение)",
}


def detect_section(text: str) -> str:
    if not text:
        return "unknown"
    for section, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return section
    return "unknown"


# =====================================================================
# 🧠 Evidence extraction layer
# =====================================================================

# ВАЖНО: этот split_sentences — тонкая обёртка над SentenceSplitter v15,
# чтобы не ломать старый интерфейс и зависимости.

def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    return split_into_sentences(text)


def build_slg_groups(sentences: List[str]) -> List[List[str]]:
    groups: List[List[str]] = []
    current: List[str] = []

    def flush():
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for sent in sentences:
        has_date = re.search(r"\d{2}\.\d{2}\.\d{4}", sent)
        has_amount = re.search(r"\d{2,3}\s?\d{3}", sent)
        role = re.search(
            r"(потерпевш\w+|подозреваем\w+|заявител\w+|свидетел\w+|граждан\w+)",
            sent,
            flags=re.IGNORECASE,
        )

        if current and (has_date or has_amount or role):
            flush()

        current.append(sent)

        if len(" ".join(current)) > 700:
            flush()

    if current:
        flush()

    return groups or ([sentences] if sentences else [])


def extract_entities(text: str) -> Dict[str, List[str]]:
    if not text:
        return {"persons": [], "amounts": [], "dates": [], "phones": [], "cards": []}

    return {
        "persons": list(
            set(
                re.findall(
                    r"\b(потерпевш\w*|подозреваем\w*|заявител\w*|свидетел\w*|граждан\w*)\b",
                    text,
                    flags=re.IGNORECASE,
                )
            )
        ),
        "amounts": list(
            set(
                re.findall(
                    r"\b\d{2,3}\s?\d{3}(?:\s?(?:тг|тенге|KZT))?\b",
                    text,
                    flags=re.IGNORECASE,
                )
            )
        ),
        "dates": list(set(re.findall(r"\b\d{2}\.\d{2}\.\d{4}\b", text))),
        "phones": list(set(re.findall(r"\+?\d{10,15}", text))),
        "cards": list(
            set(
                re.findall(
                    r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
                    text,
                )
            )
        ),
    }


EVENT_MAP = {
    "transfer": r"(перевел|перевела|перечислил|перечислила|оплатил|оплатила|перевод средств)",
    "withdrawal": r"(вывел|вывела|снял|сняла|обналичил|обналичила)",
    "promise": r"(обещал|обещала|гарантировал|гарантировала|обещание дохода)",
    "fraud": r"(обман|ввел в заблуждение|ввела в заблуждение|мошенничеств\w+)",
}


def extract_events(text: str) -> List[str]:
    if not text:
        return []
    lowered = text.lower()
    return [event for event, pattern in EVENT_MAP.items() if re.search(pattern, lowered)]


def extract_facts(text: str) -> Dict[str, Any]:
    if not text:
        return {"date": None, "amount": None, "action": None}

    date_m = re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", text)
    amount_m = re.search(r"\b\d{2,3}\s?\d{3}\b", text)

    action = None
    for a, pattern in EVENT_MAP.items():
        if re.search(pattern, text, re.IGNORECASE):
            action = a
            break

    return {
        "date": date_m.group(0) if date_m else None,
        "amount": amount_m.group(0) if amount_m else None,
        "action": action,
    }


def build_evidence_payload(chunk_text: str, page: int, section: str, paragraph_index: int):
    sentences = split_sentences(chunk_text)
    return {
        "page": page,
        "section": section,
        "paragraph_index": paragraph_index,
        "sentences": sentences,
        "slg_groups": build_slg_groups(sentences),
        "entities": extract_entities(chunk_text),
        "events": extract_events(chunk_text),
        "facts": extract_facts(chunk_text),
        "tokens_count": len(chunk_text.split()),
    }


# =====================================================================
# 🔥 Token-based chunker 7.0
# =====================================================================

gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def count_tokens(text: str) -> int:
    return len(gpt_tokenizer.encode(text)) if text else 0


def _split_long_sentence_by_tokens(sentence: str, max_tokens: int) -> List[str]:
    words = sentence.split()
    if not words:
        return []

    chunks: List[str] = []
    buf: List[str] = []
    buf_tokens = 0

    for w in words:
        wt = count_tokens(w)
        if buf_tokens + wt > max_tokens and buf:
            chunks.append(" ".join(buf))
            buf, buf_tokens = [], 0
        buf.append(w)
        buf_tokens += wt

    if buf:
        chunks.append(" ".join(buf))

    return chunks


def advanced_page_chunker(
    page_text: str,
    page_num: int,
    target_tokens=350,
    max_tokens=420,
    min_tokens=60,
    overlap_sentences=1,
):
    text = _normalize_text(page_text)
    if not text:
        return []

    sentences = split_sentences(text)
    if not sentences:
        tok = count_tokens(text)
        return [{"start": 0, "end": len(text), "text": text, "tokens": tok}]

    chunks: List[Dict[str, Any]] = []
    cur_sentences: List[str] = []
    cur_tokens = 0
    global_offset = 0

    def flush():
        nonlocal cur_sentences, cur_tokens, global_offset
        if not cur_sentences:
            return
        chunk_text = " ".join(cur_sentences).strip()
        if chunk_text:
            tok = count_tokens(chunk_text)
            chunks.append(
                {
                    "start": global_offset,
                    "end": global_offset + len(chunk_text),
                    "text": chunk_text,
                    "tokens": tok,
                }
            )
            global_offset += len(chunk_text) + 1
        cur_sentences, cur_tokens = [], 0

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        s_tokens = count_tokens(s)

        if s_tokens > max_tokens:
            if cur_sentences:
                flush()

            parts = _split_long_sentence_by_tokens(s, max_tokens)
            for p in parts:
                tok = count_tokens(p)
                chunks.append(
                    {
                        "start": global_offset,
                        "end": global_offset + len(p),
                        "text": p,
                        "tokens": tok,
                    }
                )
                global_offset += len(p) + 1
            continue

        if cur_tokens + s_tokens > max_tokens and cur_sentences:
            flush()

            if overlap_sentences and chunks:
                tail = split_sentences(chunks[-1]["text"])[-overlap_sentences:]
                cur_sentences = tail[:]
                cur_tokens = sum(count_tokens(t) for t in cur_sentences)

        cur_sentences.append(s)
        cur_tokens += s_tokens

    if cur_sentences:
        flush()

    if len(chunks) > 1:
        filtered = [c for c in chunks if c["tokens"] >= min_tokens]
        if filtered:
            chunks = filtered

    return chunks


# =====================================================================
# 📄 OCR + Chunker 7.0 (PDF)
# =====================================================================

def process_pdf_with_smart_ocr(file_path: str, file_id, db: Session) -> int:
    file_id = ensure_uuid(file_id)
    if not file_id:
        return 0

    chunks_created = 0

    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    logger.info(f"📖 SMART OCR 7.0: страниц={total_pages}")

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            text = _normalize_text(text)

            if not text or len(text) < 50:
                logger.info(f"[SMART OCR] стр {i}: мало текста → Tesseract")
                text = run_tesseract_ocr(
                    file_path=file_path,
                    page_num=i,
                    use_preprocessing=True,
                ) or ""
                text = _normalize_text(text)

            if not text.strip():
                logger.warning(f"[SMART OCR] стр {i}: текста нет после OCR")
                continue

            section = detect_section(text)
            page_chunks = advanced_page_chunker(text, page_num=i)

            for idx, ch in enumerate(page_chunks, start=1):
                chunk_text = ch["text"]
                evidence = build_evidence_payload(
                    chunk_text,
                    page=i,
                    section=section,
                    paragraph_index=idx,
                )

                chunk = Chunk(
                    chunk_id=uuid.uuid4(),
                    file_id=file_id,
                    page=i,
                    start_offset=ch["start"],
                    end_offset=ch["end"],
                    text=chunk_text,
                    evidence=evidence,
                )
                db.add(chunk)
                chunks_created += 1

                enqueue_chunk_vectorization.delay(str(chunk.chunk_id))

        except Exception as e:
            logger.error(f"❌ SMART OCR 7.0 ошибка стр {i}: {e}", exc_info=True)
            continue

    db.flush()
    logger.info(f"SMART OCR 7.0 → создано чанков: {chunks_created}")
    return chunks_created


# =====================================================================
# 📄 Fallback OCR
# =====================================================================

def process_pdf_with_ocr(file_path: str, file_id, db: Session) -> int:
    file_id = ensure_uuid(file_id)
    if not file_id:
        return 0

    full_text = extract_text_from_pdf(file_path, dpi=300, use_preprocessing=True)
    full_text = _normalize_text(full_text)

    if not full_text.strip():
        return 0

    page_chunks = advanced_page_chunker(full_text, page_num=1)
    chunks_created = 0

    for idx, ch in enumerate(page_chunks, start=1):
        chunk_text = ch["text"]
        evidence = build_evidence_payload(
            chunk_text,
            page=idx,
            section="unknown",
            paragraph_index=idx,
        )

        chunk = Chunk(
            chunk_id=uuid.uuid4(),
            file_id=file_id,
            page=idx,
            start_offset=ch["start"],
            end_offset=ch["end"],
            text=chunk_text,
            evidence=evidence,
        )
        db.add(chunk)
        chunks_created += 1

        enqueue_chunk_vectorization.delay(str(chunk.chunk_id))

    db.flush()
    logger.info(f"📄 Fallback OCR 7.0: создано {chunks_created} чанков")
    return chunks_created


# =====================================================================
# 📑 DOCX/TXT
# =====================================================================

def process_text_into_chunks(
    file_id,
    text: str,
    db: Session,
    min_len: int = 50,
    page_start: int = 1,
) -> int:
    file_id = ensure_uuid(file_id)
    if not file_id:
        return 0

    norm = _normalize_text(text)
    if not norm:
        return 0

    page_chunks = advanced_page_chunker(norm, page_num=page_start)
    chunks_created = 0

    for idx, ch in enumerate(page_chunks, start=page_start):
        chunk_text = ch["text"]

        if len(chunk_text) < min_len and len(page_chunks) > 1:
            continue

        evidence = build_evidence_payload(
            chunk_text,
            page=idx,
            section="plain_text",
            paragraph_index=idx,
        )

        chunk = Chunk(
            chunk_id=uuid.uuid4(),
            file_id=file_id,
            page=idx,
            start_offset=ch["start"],
            end_offset=ch["end"],
            text=chunk_text,
            evidence=evidence,
        )
        db.add(chunk)
        chunks_created += 1

        enqueue_chunk_vectorization.delay(str(chunk.chunk_id))

    db.flush()
    logger.info(f"process_text_into_chunks 7.0: {chunks_created} чанков")
    return chunks_created


# =====================================================================
# 📦 Entry point
# =====================================================================

def process_any_file(file_path: str, file_id, db: Session) -> int:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        c = process_pdf_with_smart_ocr(file_path, file_id, db)
        return c if c > 0 else process_pdf_with_ocr(file_path, file_id, db)

    elif ext in [".docx", ".txt"]:
        text = extract_text_from_file(file_path) or ""
        if not text.strip():
            logger.warning(f"⚠️ Не удалось извлечь текст из {file_path} ({ext}), пропускаем.")
            return 0
        return process_text_into_chunks(file_id, text, db)

    logger.warning(f"Unsupported file type: {ext}")
    return 0


# =====================================================================
# Router compatibility helpers
# =====================================================================

def sentence_splitting(text: str) -> List[str]:
    return split_sentences(text)


def chunk_by_sentences(text: str, max_chars: int = 1500, min_chars: int = 300) -> List[str]:
    sents = split_sentences(text)
    if not sents:
        return [text]

    chunks: List[str] = []
    buf = ""

    for s in sents:
        if len(buf) + len(s) + 1 > max_chars:
            if len(buf) >= min_chars:
                chunks.append(buf.strip())
                buf = s
            else:
                buf += " " + s
        else:
            buf += " " + s

    if buf.strip():
        chunks.append(buf.strip())

    return chunks


def build_chunk_payload(
    chunk_text: str,
    page: int,
    section: str = "debug",
    paragraph_index: int = 1,
):
    return build_evidence_payload(chunk_text, page, section, paragraph_index)
