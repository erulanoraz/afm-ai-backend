import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# Возможные типы документов (можно расширять)
DOCUMENT_TYPES = [
    "protocol_interrogation",   # протокол допроса
    "victim_statement",         # заявление/объяснение потерпевшего
    "raport",                   # рапорт
    "resolution",               # постановление
    "bank_statement",           # банковская выписка
    "transaction_table",        # таблица операций (эксель/таблица)
    "contract",                 # договор, расписка
    "chat_screenshot",          # скрин переписки
    "wallet_screenshot",        # скрин криптокошелька / кабинета
    "expert_opinion",           # заключение эксперта
    "other_evidence",           # прочие вещественные доказательства / приложения
    "unknown",
]


def classify_document(
    filename: str,
    file_bytes: bytes,
    content_type: Optional[str] = None,
    text_hint: Optional[str] = None,
) -> str:
    """
    Лёгкий классификатор типа документа.

    Использует:
    - имя файла
    - content_type (если есть)
    - text_hint (если есть)
    - простые регулярки / ключевые слова

    НЕ использует LLM.
    """

    fn = filename.lower()
    text = (text_hint or "").lower()

    # -------------------------------
    # 1) По имени файла
    # -------------------------------
    # Протокол допроса
    if any(sub in fn for sub in ["протокол_допроса", "протокол допроса", "допрос_потерпевшего", "допрос потерпевшего"]):
        return "protocol_interrogation"

    # Заявление / объяснение / жалоба
    if any(sub in fn for sub in ["заявление", "объяснение", "жалоба", "обращение"]):
        return "victim_statement"

    # Рапорт
    if "рапорт" in fn:
        return "raport"

    # Постановление
    if "постановление" in fn:
        return "resolution"

    # Банковская выписка
    if any(sub in fn for sub in ["выписка", "statement", "bank"]):
        return "bank_statement"

    # Договор / расписка
    if any(sub in fn for sub in ["договор", "расписка", "contract"]):
        return "contract"

    # Экспертное заключение
    if any(sub in fn for sub in ["заключение эксперта", "экспертиза", "экспертное заключение"]):
        return "expert_opinion"

    # Скриншоты чатов / кабинетов / кошельков
    if any(sub in fn for sub in ["screenshot", "скрин", "screen"]):
        # попытаемся грубо разделить чат/кошелёк
        if any(sub in fn for sub in ["chat", "whatsapp", "telegram", "ватсап", "телеграм"]):
            return "chat_screenshot"
        if any(sub in fn for sub in ["wallet", "usdt", "binance", "cabinet", "личный кабинет"]):
            return "wallet_screenshot"
        # неизвестный скриншот
        return "other_evidence"

    # Если по имени не сработало — смотрим content_type
    if content_type:
        ct = content_type.lower()
        if "image" in ct:
            # пробуем по имени/тексту определить, чат это или кошелёк
            if any(sub in fn for sub in ["chat", "whatsapp", "telegram"]) or "чат" in text:
                return "chat_screenshot"
            if any(sub in fn for sub in ["wallet", "usdt", "binance", "kabinet"]) or "usdt" in text:
                return "wallet_screenshot"
            return "other_evidence"

        if "excel" in ct or "spreadsheet" in ct:
            return "transaction_table"

        if "pdf" in ct and any(sub in fn for sub in ["выписка", "statement"]):
            return "bank_statement"

    # -------------------------------
    # 2) По содержимому (если есть text_hint)
    # -------------------------------
    if text:
        # Протокол допроса
        if re.search(r"протокол\s+допроса", text) or "допрошен" in text:
            return "protocol_interrogation"

        # Заявление
        if "заявление" in text and "прошу" in text:
            return "victim_statement"

        # Рапорт
        if "рапорт" in text:
            return "raport"

        # Постановление
        if "постановление" in text and ("возбудить" in text or "отказать" in text):
            return "resolution"

        # Банковская выписка / операции
        if any(word in text for word in ["касса", "kaspi", "банковский счет", "банковский счёт", "остаток", "операции по счету"]):
            return "bank_statement"

        # Договор / расписка
        if "договор" in text or "расписка" in text:
            return "contract"

        # Экспертиза
        if "заключение эксперта" in text or "экспертиза" in text:
            return "expert_opinion"

        # Скрины
        if any(word in text for word in ["чат", "переписка", "сообщение", "whatsapp", "telegram"]):
            return "chat_screenshot"
        if any(word in text for word in ["usdt", "binance", "кошелек", "кошелёк", "личный кабинет"]):
            return "wallet_screenshot"

    # -------------------------------
    # 3) Фолбэк
    # -------------------------------
    logger.debug(f"📂 document_classifier: filename={filename} → unknown")
    return "unknown"
