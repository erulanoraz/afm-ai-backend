# backend/scripts/diagnose_vectors.py
"""
ДИАГНОСТИЧЕСКИЙ СКРИПТ
Проверяет состояние Vector Store и Celery очередей.
"""

import os
import sys
import redis
import logging
from pathlib import Path

# ============================================================
# 1) Правильная настройка PYTHONPATH
#    (чтобы import app.* работал из scripts/)
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent      # .../backend/scripts
BACKEND_DIR = SCRIPT_DIR.parent                  # .../backend

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

from app.utils.config import settings
from app.db.session import SessionLocal
from app.db.models import Chunk, File
from app.search.vector_client import get_vector_client


# ============================================================
# 🔧 Вспомогательная функция: COUNT в Weaviate
# ============================================================

def get_weaviate_count(vc) -> int:
    """
    Возвращает количество объектов класса Chunk в Weaviate.
    Предполагается, что vc.client — это weaviate.Client.
    """
    try:
        result = (
            vc.client.query
            .aggregate("Chunk")
            .with_meta_count()
            .do()
        )
        return result["data"]["Aggregate"]["Chunk"][0]["meta"]["count"]
    except Exception as e:
        logger.error(f"❌ Ошибка получения count из Weaviate: {e}")
        return 0


# ============================================================
# 🔍 ДИАГНОСТИКА
# ============================================================

def diagnose():
    """Полная диагностика системы."""
    
    logger.info("=" * 60)
    logger.info("🔍 НАЧАЛО ДИАГНОСТИКИ VECTOR STORE")
    logger.info("=" * 60)
    
    # --------- 1. PostgreSQL --------- 
    logger.info("\n1️⃣ POSTGRESQL ДИАГНОСТИКА")
    try:
        db = SessionLocal()
        
        file_count = db.query(File).count()
        chunk_count = db.query(Chunk).count()
        
        logger.info(f"✔ БД подключена")
        logger.info(f"  • Файлы: {file_count}")
        logger.info(f"  • Чанки: {chunk_count}")
        
        if chunk_count == 0:
            logger.error("  ❌ ПРОБЛЕМА: В БД нет чанков! Нужно загрузить документы.")
        else:
            logger.info(f"  ✔ Чанки есть в БД")
            
            # Проверим первый чанк
            first_chunk = db.query(Chunk).first()
            if first_chunk:
                logger.info(
                    f"    Пример чанка: id={first_chunk.chunk_id}, "
                    f"len={len(first_chunk.text) if first_chunk.text else 0}"
                )
        
        db.close()
        
    except Exception as e:
        logger.error(f"❌ ОШИБКА PostgreSQL: {e}")
        return False
    
    # --------- 2. Weaviate --------- 
    logger.info("\n2️⃣ WEAVIATE ДИАГНОСТИКА")
    try:
        vc = get_vector_client()
        
        # Проверяем schema
        schema_ok = vc.client.schema.get()
        logger.info(f"✔ Weaviate подключён")
        logger.info(f"  • URL: {settings.WEAVIATE_URL}")
        
        classes = [c["class"] for c in schema_ok.get("classes", [])]
        if "Chunk" in classes:
            logger.info(f"  ✔ Schema 'Chunk' существует")
        else:
            logger.error(f"  ❌ Schema 'Chunk' НЕ найдена! (есть: {classes})")
            return False
        
        # Подсчёт количества объектов через aggregate
        weav_count = get_weaviate_count(vc)
        logger.info(f"  • Объектов в Weaviate (Chunk): {weav_count}")
        
        if weav_count == 0:
            if chunk_count > 0:
                logger.error("  ❌ ПРОБЛЕМА: Weaviate ПУСТОЙ, но в БД есть чанки!")
            else:
                logger.warning("  ⚠️ Weaviate пустой — и в БД тоже нет чанков.")
        else:
            logger.info("  ✔ В индексе Weaviate есть объекты")
                
    except Exception as e:
        logger.error(f"❌ ОШИБКА Weaviate: {e}")
        return False
    
    # --------- 3. Redis / Celery --------- 
    logger.info("\n3️⃣ REDIS / CELERY ДИАГНОСТИКА")
    try:
        r = redis.from_url(settings.REDIS_URL)
        
        # Пингуем Redis
        r.ping()
        logger.info(f"✔ Redis подключён")
        
        # Проверяем очереди
        queues = {
            "default": "Основная очередь (ingest)",
            "vectors": "Векторизация (vectorization)",
        }
        
        for queue_name, queue_desc in queues.items():
            try:
                queue_length = r.llen(queue_name)
                if queue_length > 0:
                    logger.warning(
                        f"  ⚠️ {queue_desc} ({queue_name}): "
                        f"{queue_length} задач в ожидании"
                    )
                else:
                    logger.info(f"  ✔ {queue_desc} ({queue_name}): пусто (ОК)")
            except Exception as e:
                logger.warning(f"  ⚠️ Не удалось проверить {queue_name}: {e}")
        
    except Exception as e:
        logger.error(f"❌ ОШИБКА Redis: {e}")
        # Не выходим жёстко, но помечаем как проблему
        # return False
    
    # --------- 4. ИТОГИ --------- 
    logger.info("\n" + "=" * 60)
    logger.info("📊 ИТОГИ ДИАГНОСТИКИ:")
    logger.info("=" * 60)
    
    # Ещё раз считаем, чтобы вывести красиво
    db = SessionLocal()
    chunk_count = db.query(Chunk).count()
    db.close()
    
    vc = get_vector_client()
    weav_count = get_weaviate_count(vc)
    
    logger.info(f"\n✔ PostgreSQL: {chunk_count} чанков")
    logger.info(f"✔ Weaviate: {weav_count} объектов (Chunk)")
    
    if chunk_count > 0 and weav_count == 0:
        logger.error("\n🔴 КРИТИЧЕСКАЯ ПРОБЛЕМА:")
        logger.error("   Чанки есть в БД, но они НЕ ИНДЕКСИРОВАНЫ в Weaviate!")
        logger.error("   Решение:")
        logger.error("   1. Проверить что запущены workers: celery -Q vectors")
        logger.error("   2. Проверить логи: tail -f logs/celery-vectors.log")
        logger.error("   3. Убедиться что flush() вызывается в vector_tasks.py")
        return False
    
    elif chunk_count > 0 and weav_count > 0:
        logger.info("\n✅ СИСТЕМА РАБОТАЕТ КОРРЕКТНО!")
        logger.info(f"   Чанков синхронизировано (по count): {weav_count}/{chunk_count}")
        return True
    
    else:
        logger.warning("\n⚠️ НЕ НАЙДЕНО ДАННЫХ")
        logger.warning("   Загрузите документы через API /upload")
        return True


# ============================================================
# 🔧 ТЕСТИРОВАНИЕ ИНДЕКСАЦИИ
# ============================================================

def test_vectorization():
    """Тест: индексирует ли система новый чанк?"""
    
    logger.info("\n" + "=" * 60)
    logger.info("🧪 ТЕСТ ИНДЕКСАЦИИ")
    logger.info("=" * 60)
    
    db = SessionLocal()
    
    # Берём первый чанк
    first_chunk = db.query(Chunk).first()
    
    if not first_chunk:
        logger.error("❌ Нет чанков для тестирования")
        db.close()
        return False
    
    logger.info(f"📦 Тестовый чанк: {first_chunk.chunk_id}")
    text_preview = (first_chunk.text or "")[:100]
    logger.info(f"   Текст (первые 100 символов): {text_preview}...")
    
    # Проверяем есть ли в Weaviate
    vc = get_vector_client()
    try:
        result = vc.search(
            query_text=(first_chunk.text or "")[:50],
            limit=5
        )
    except Exception as e:
        logger.error(f"❌ Ошибка при поиске в Weaviate: {e}")
        db.close()
        return False
    
    hits = result.get("data", {}).get("Get", {}).get("Chunk", [])
    
    found = any(h.get("chunk_id") == str(first_chunk.chunk_id) for h in hits)
    
    if found:
        logger.info(f"✔ Чанк НАЙДЕН в Weaviate")
    else:
        logger.error(f"❌ Чанк НЕ НАЙДЕН в Weaviate")
        logger.error("   Возможные причины:")
        logger.error("   1. Чанк не был отправлен в очередь 'vectors'")
        logger.error("   2. Worker 'vectors' не запущен")
        logger.error("   3. flush() не был вызван")
    
    db.close()
    return found


# ============================================================
# 🚀 MAIN
# ============================================================

if __name__ == "__main__":
    try:
        success = diagnose()
        
        logger.info("\n")
        try:
            answer = input("Запустить тест индексации? (y/n): ").strip().lower()
        except EOFError:
            answer = "n"
        
        if answer == 'y':
            test_vectorization()
        
    except KeyboardInterrupt:
        logger.info("\n⏹ Диагностика прервана")
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка: {e}", exc_info=True)
