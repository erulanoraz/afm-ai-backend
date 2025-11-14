# app/services/chunker.py
import os
import re
import logging
import uuid
from typing import Optional
from PyPDF2 import PdfReader, PdfWriter
from sqlalchemy.orm import Session

from app.db.models import Chunk
from app.services.ocr_worker import extract_text_from_pdf, run_tesseract_ocr, run_tesseract_ocr_image
from app.services.parser import extract_text_from_file
logger = logging.getLogger(__name__)

# ============================================================
# 🧩 Утилита: безопасное приведение к UUID
# ============================================================

def ensure_uuid(value) -> Optional[uuid.UUID]:
    """Безопасно приводит значение к UUID, возвращает None при ошибке."""
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except Exception:
        logger.error(f"❌ Некорректный UUID: {value}")
        return None


# ============================================================
# 📄 OCR обработка PDF (усовершенствованная)
# ============================================================

def process_pdf_with_ocr(file_path: str, file_id, db: Session) -> int:
    """
    Разбивает PDF на страницы и выполняет OCR через Tesseract.
    Работает как с текстовыми PDF, так и со сканами.
    
    Улучшения:
    • DPI=300 для лучшего качества
    • OEM=1 (LSTM engine)
    • Image preprocessing с бинаризацией
    • Поддержка rus+kaz
    """
    chunks_created = 0
    file_id = ensure_uuid(file_id)
    if not file_id:
        return 0

    if not os.path.exists(file_path):
        logger.error(f"❌ Файл не найден: {file_path}")
        return 0

    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        logger.info(f"📖 Открыт PDF: {os.path.basename(file_path)}, страниц: {total_pages}")
    except Exception as e:
        logger.error(f"❌ Не удалось открыть PDF {file_path}: {e}")
        # fallback: OCR всего файла с улучшениями
        try:
            full_text = extract_text_from_pdf(file_path, dpi=300, use_preprocessing=True)
            if full_text.strip():
                return process_text_into_chunks(file_id, full_text, db, page_start=1)
        except Exception as err:
            logger.error(f"❌ Fallback также не сработал: {err}")

        # ⚡ Главное исправление: НЕЛЬЗЯ возвращать 0!
        # Если текст пустой – создаём 1 чанк с пустым текстом, чтобы файл не терялся.
        safe_text = full_text if isinstance(full_text, str) else ""
        return process_text_into_chunks(file_id, safe_text, db, page_start=1)



    if total_pages > 200:
        logger.warning(f"⚠️ PDF содержит {total_pages} страниц — обработка может занять много времени.")

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            if len(text.strip()) < 50:
                tmp_page_path = f"{file_path}_page_{page_num}.pdf"
                writer = PdfWriter()
                writer.add_page(page)
                with open(tmp_page_path, "wb") as f:
                    writer.write(f)
                # OCR с новыми параметрами: DPI=300, OEM=1, preprocessing=True
                text = run_tesseract_ocr(tmp_page_path, use_preprocessing=True)
                os.remove(tmp_page_path)

            if not text.strip():
                logger.warning(f"⚠️ OCR не нашёл текст на стр. {page_num}")
                continue

            chunk = Chunk(
                chunk_id=uuid.uuid4(),
                file_id=file_id,
                page=page_num,
                start_offset=0,
                end_offset=len(text),
                text=text.strip(),
            )
            db.add(chunk)
            chunks_created += 1

            logger.debug(f"✅ Страница {page_num}: {len(text)} символов")

        except Exception as e:
            logger.error(f"❌ Ошибка OCR страницы {page_num}: {e}", exc_info=True)
            continue

    # ✅ Вместо commit используем flush (commit делается снаружи)
    try:
        db.flush()
        logger.info(f"📄 Обработано страниц PDF: {chunks_created}/{total_pages}")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении чанков: {e}")
        db.rollback()
        return 0

    return chunks_created


# ============================================================
# 🧠 Smart OCR для PDF (универсальная функция)
# ============================================================

def process_pdf_with_smart_ocr(file_path: str, file_id, db: Session) -> int:
    """
    🧠 Универсальный OCR-процессор PDF:
    - Сначала пробует встроенный текст (extract_text)
    - Если текста нет, вызывает Tesseract OCR с улучшениями
    - Поддерживает большие PDF, но прерывает при >300 стр.
    """
    file_id = ensure_uuid(file_id)
    if not file_id:
        return 0

    if not os.path.exists(file_path):
        logger.error(f"❌ Файл не найден: {file_path}")
        return 0

    chunks_created = 0

    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        if total_pages > 300:
            logger.error(f"⛔ PDF содержит {total_pages} страниц — превышен лимит (300).")
            raise Exception("Слишком большой PDF — обработка остановлена.")

        logger.info(f"📖 SMART OCR: открыт PDF {os.path.basename(file_path)} ({total_pages} стр.)")
        logger.info(f"🚀 SMART OCR параметры: DPI=300, OEM=1, preprocessing=ON, langs=rus+kaz")

        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()

                # если текста мало — OCR
                if not text or len(text.strip()) < 30:
                    logger.info(f"[SMART OCR] Страница {i}: текста мало — запускаем Tesseract OCR")
                    try:
                        text = run_tesseract_ocr_image(page, page_num=i, use_preprocessing=True)
                    except Exception as ocr_err:
                        logger.error(f"⚠️ Ошибка Tesseract OCR на стр. {i}: {ocr_err}")
                        text = ""

                if not text or not text.strip():
                    logger.warning(f"⚠️ OCR не смог извлечь текст на стр. {i}")
                    continue

                chunk = Chunk(
                    chunk_id=uuid.uuid4(),
                    file_id=file_id,
                    page=i,
                    start_offset=0,
                    end_offset=len(text),
                    text=text.strip(),
                )
                db.add(chunk)
                chunks_created += 1

                logger.debug(f"✅ Страница {i}: {len(text)} символов")

            except Exception as e:
                logger.error(f"❌ Ошибка обработки страницы {i}: {e}", exc_info=True)
                continue

        # 🟩 FAIL-SAFE: Smart OCR ничего не дал → fallback в ocr_worker
        if chunks_created == 0:
            logger.warning(f"⚠️ SMART OCR не дал чанков. Запускаю fallback OCR…")
            try:
                fallback_text = extract_text_from_pdf(file_path, dpi=300, use_preprocessing=True)
                if fallback_text and fallback_text.strip():
                    chunk = Chunk(
                        chunk_id=uuid.uuid4(),
                        file_id=file_id,
                        page=1,
                        start_offset=0,
                        end_offset=len(fallback_text),
                        text=fallback_text.strip(),
                    )
                    db.add(chunk)
                    chunks_created = 1
                    logger.info("🟩 Создан fallback чанк из extract_text_from_pdf()")
            except Exception as fe:
                logger.error(f"❌ Fallback OCR тоже не сработал: {fe}")
                # создаём минимальный placeholder
                chunk = Chunk(
                    chunk_id=uuid.uuid4(),
                    file_id=file_id,
                    page=1,
                    start_offset=0,
                    end_offset=1,
                    text=" ",
                )
                db.add(chunk)
                chunks_created = 1
                logger.info("🟧 Создан placeholder-чанк (минимальный)")

        db.flush()
        logger.info(f"✅ SMART OCR завершён — создано {chunks_created} чанков")
        return chunks_created

    except Exception as e:
        logger.error(f"❌ Критическая ошибка SMART OCR: {e}", exc_info=True)
        db.rollback()
        return 0

    finally:
        try:
            if hasattr(reader, "stream") and reader.stream:
                reader.stream.close()
                logger.debug("📘 PdfReader закрыт корректно")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось закрыть PdfReader: {e}")


# ============================================================
# 📑 Разделение текстов (DOCX/TXT/сканы после OCR)
# ============================================================

def process_text_into_chunks(
    file_id,
    text: str,
    db: Session,
    min_len: int = 50,
    page_start: int = 1
) -> int:
    """
    Разбивает текст на смысловые чанки (абзацы или предложения) и сохраняет их в БД.
    """
    if not text or not text.strip():
        logger.warning("⚠️ Пустой текст — чанки не созданы.")
        return 0

    file_id = ensure_uuid(file_id)
    if not file_id:
        return 0

    paragraphs = []
    if "\n\n" in text:
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > min_len]
    if not paragraphs:
        paragraphs = [p.strip() for p in re.split(r'(?<=[.!?])\s+(?=[А-ЯA-Z])', text) if len(p.strip()) > min_len]
    if not paragraphs:
        logger.warning(f"⚠️ Не удалось разбить текст, возможно слишком короткий ({len(text)} символов)")
        return 0

    logger.info(f"📝 Разбито на {len(paragraphs)} чанков (min_len={min_len})")

    chunks_created = 0
    total_chars = 0

    for idx, chunk_text in enumerate(paragraphs, start=page_start):
        try:
            chunk = Chunk(
                chunk_id=uuid.uuid4(),
                file_id=file_id,
                page=idx,
                start_offset=0,
                end_offset=len(chunk_text),
                text=chunk_text
            )
            db.add(chunk)
            chunks_created += 1
            total_chars += len(chunk_text)
        except Exception as e:
            logger.error(f"❌ Ошибка при добавлении чанка {idx}: {e}", exc_info=True)
            continue

    # ✅ flush вместо commit
    try:
        db.flush()
        logger.info(f"✅ Сохранено {chunks_created} чанков, всего {total_chars} символов")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении чанков в БД: {e}")
        db.rollback()
        return 0

    return chunks_created

# ============================================================
# ⚙️ Универсальный обработчик
# ============================================================

def process_any_file(file_path: str, file_id, db: Session) -> int:
    """
    Универсальная обработка PDF, DOCX и TXT файлов.
    Работает безопасно в рамках внешней транзакции.
    
    Улучшения для PDF:
    • SMART OCR с DPI=300
    • OEM=1 (LSTM engine)
    • Image preprocessing (бинаризация)
    • Поддержка rus+kaz
    """
    if not os.path.exists(file_path):
        logger.error(f"❌ Файл не существует: {file_path}")
        return 0

    ext = os.path.splitext(file_path)[1].lower()
    chunks_created = 0

    try:
        if ext == ".pdf":
            logger.info(f"📄 Обработка PDF: {os.path.basename(file_path)}")

            # ✅ Используем SMART OCR по умолчанию, он устойчивее
            chunks_created = process_pdf_with_smart_ocr(file_path, file_id, db)

            # если не сработал, fallback на обычный OCR
            if chunks_created == 0:
                logger.warning("⚠️ SMART OCR не создал чанков, пробую базовый OCR...")
                chunks_created = process_pdf_with_ocr(file_path, file_id, db)

        elif ext in [".docx", ".txt"]:
            logger.info(f"📝 Обработка текстового файла: {os.path.basename(file_path)}")
            text = extract_text_from_file(file_path)

            if not text.strip():
                logger.warning(f"⚠️ Не удалось извлечь текст, пробую OCR...")
                # OCR с улучшенными параметрами
                text = extract_text_from_pdf(file_path, dpi=300, use_preprocessing=True)

            if not text.strip():
                logger.warning(f"⚠️ Текст не найден даже после OCR — файл пропущен")
                return 0

            chunks_created = process_text_into_chunks(file_id, text, db)

        else:
            logger.warning(f"⛔ Неподдерживаемое расширение: {ext}")
            return 0

    except Exception as e:
        # ❌ Ошибка критического уровня — не закрываем транзакцию, просто логируем
        logger.error(f"❌ Критическая ошибка обработки {file_path}: {e}", exc_info=True)
        return 0

    if chunks_created == 0:
        logger.warning(f"⚠️ Не создано чанков для {file_path}")

    return chunks_created


# ============================================================
# 🧪 Тестовая функция для отладки
# ============================================================

def test_chunker(file_path: str, db: Session):
    """
    Локальный тест обработчика (запускать вручную для проверки OCR и нарезки).
    Используются улучшенные параметры OCR.
    """
    test_file_id = uuid.uuid4()
    logger.info(f"🧪 Тестирование chunker для: {file_path}")
    logger.info(f"🆔 Тестовый file_id: {test_file_id}")
    logger.info(f"🚀 Тестовые параметры OCR: DPI=300, OEM=1, preprocessing=ON, langs=rus+kaz")

    try:
        chunks = process_any_file(file_path, test_file_id, db)
        db.commit()  # ✅ вручную коммитим тестовую сессию

        logger.info(f"📊 Результат: создано {chunks} чанков")

        saved_chunks = db.query(Chunk).filter(Chunk.file_id == test_file_id).all()
        logger.info(f"✅ В БД сохранено: {len(saved_chunks)} чанков")

        for i, chunk in enumerate(saved_chunks[:3], 1):  # Покажем первые 3 чанка
            logger.info(
                f"  Chunk {i}: page={chunk.page}, length={len(chunk.text)}, preview={chunk.text[:100]}..."
            )

        return chunks

    except Exception as e:
        db.rollback()
        logger.error(f"❌ Ошибка теста chunker: {e}", exc_info=True)
        return 0