from __future__ import annotations
import uuid
import re
from typing import List

from app.services.facts.fact_models import LegalFact, FactToken, SourceRef
from app.utils.sentence_splitter import split_into_sentences


class FactTokenizer:
    """
    FactTokenizer v17.0 — Universal Investigator Tokenizer
    Полностью совместим с Router 6.2 Investigator Edition.

    НОВОЕ:
    ✓ Универсальный fraud/invest/economic/crypto extraction
    ✓ Новые токены под Router 6.2: crypto, scheme_marker, deception_context,
      admin_action, digital_transfer, economic_action
    ✓ Защита от ложных ФИО 2-го уровня
    ✓ Каналы: Kaspi, OKX, Binance, Qiwi, Visa, MasterCard, IBAN, криптосети
    ✓ Поддержка любых финансовых преступлений
    """

    VERSION = "17.0"

    # =====================================================================
    # СТОП-ЛИСТ ПЕРСОН (полностью переделан)
    # =====================================================================
    PERSON_STOPWORDS = {
        # служебные и паразитные слова
        "после", "примерно", "кроме", "также", "когда", "однако",
        "далее", "заявление", "программа", "данная", "документ",
        "платформа", "потерпевший", "потерпевшая", "сотрудник",
        "счет", "аккаунт", "дата", "вывод", "снятие", "вклад",
        "администратор", "группа", "перевод", "проверка",
        "заявка", "заявки", "подтверждение",
    }

    # =====================================================================
    # Паттерны сущностей
    # =====================================================================

    # ФИО (строгое распознавание)
    _fio = re.compile(
        r"\b([А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?)\s+([А-ЯЁ][а-яё]+)"
        r"(?:\s+([А-ЯЁ][а-яё]+))?\b"
    )

    # Инициалы
    _fio_initials = re.compile(
        r"\b([А-ЯЁ][а-яё]+)\s+[А-ЯЁ]\.[А-ЯЁ]\.\b"
    )

    # Деньги
    _amount = re.compile(
        r"\b\d[\d\s.,]{0,18}\s*(?:₸|тенге|тг|kzt|rub|₽|usd|usdt|eur|сом|доллар)\b",
        re.IGNORECASE,
    )

    # Даты
    _date = re.compile(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?:\s*г\.?)?\b",
        re.IGNORECASE,
    )

    # Телефоны
    _phone = re.compile(
        r"\b(?:\+?7|8)[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b"
    )

    # IBAN/карты
    _iban = re.compile(r"\bKZ\d{18}\b", re.IGNORECASE)
    _card = re.compile(r"\b(?:\d[ -]?){12,20}\b")

    # Статьи
    _article_ref = re.compile(
        r"\bст\.?\s*\d{1,3}(?:[-–]\d+)?(?:\s*(?:ук|упк|гк)\s*рк)?\b",
        re.IGNORECASE,
    )

    # Крипто-адреса
    _crypto_addr = re.compile(
        r"\b(0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b"
    )

    # Ключевые слова по направлениям
    _fraud_kw = [
        "обман", "обманным путем", "мошеннич", "ввел в заблуждение",
        "ввела в заблуждение", "никто не сможет вывести",
        "незаконно завладел", "похитил", "похитила", "обогащение",
        "финансовая пирамида", "пирамид"
    ]

    _investment_kw = [
        "инвестиц", "проценты", "дивиденды", "доход", "вклад",
        "депозит", "прибыль", "умножение", "ежедневный доход"
    ]

    _economic_kw = [
        "получил", "перевел", "снял", "пополнил", "отправил",
        "зачислил", "вывел", "оплатил", "приобрел"
    ]

    _admin_kw = [
        "администратор", "модератор", "создавал группы",
        "удалял сообщения", "рекламировал", "координировал"
    ]

    _scheme_kw = [
        "схема", "организованная группа", "привлечение вкладчиков",
        "вовлечение", "создание приложения", "фальшивые акции",
        "ввод в заблуждение", "механизм", "принцип действия"
    ]

    _crypto_kw = [
        "usdt", "ether", "bitcoin", "eth", "btc", "binance", "okx",
        "bybit", "кошелек", "wallet", "crypto"
    ]

    _channel_kw = [
        "kaspi", "halyk", "qiwi", "sber", "forte", "visa",
        "mastercard", "iban", "swift", "okx", "binance", "bybit"
    ]

    # Ролевые слова
    _role_kw = {
        "потерпевш": "victim",
        "заявител": "applicant",
        "подозреваем": "suspect",
        "обвиняем": "suspect",
        "организатор": "organizer",
        "свидетел": "witness",
    }

    # =====================================================================
    # Основной метод
    # =====================================================================
    def tokenize(self, docs: List[dict]) -> List[LegalFact]:
        all_facts = []

        for doc in docs:
            file_id = doc.get("file_id")
            text = (doc.get("text") or "").strip()
            page = doc.get("page", 1)

            if not text or not file_id:
                continue

            sentences = split_into_sentences(text)
            windows = self._context_windows(sentences)

            for idx, (before, sent, after) in enumerate(windows):
                sent = sent.strip()
                if not sent:
                    continue

                tokens = self._extract_tokens(sent, file_id, page)
                if not tokens:
                    continue

                fact = LegalFact(
                    fact_id=str(uuid.uuid4()),
                    tokens=tokens,
                    source_refs=[SourceRef(file_id=file_id, page=page)],
                    span_text=sent,
                    text=sent,
                    sentence_index=idx,
                    context_before=before.strip(),
                    context_after=after.strip(),
                )

                fact.role = self._detect_role(fact, sent)
                fact.event_type = fact.role
                fact.article_hints = self._article_hints(sent)
                fact.confidence = self._confidence_v4(fact)

                all_facts.append(fact)

        return all_facts

    # =====================================================================
    # Контекстные окна
    # =====================================================================
    def _context_windows(self, sentences):
        return [
            (
                sentences[i - 1] if i > 0 else "",
                sentences[i],
                sentences[i + 1] if i + 1 < len(sentences) else ""
            )
            for i in range(len(sentences))
        ]

    # =====================================================================
    # Извлечение сущностей
    # =====================================================================
    def _extract_tokens(self, sent: str, file_id: str, page: int):
        tokens = []
        seen = set()
        src = SourceRef(file_id=file_id, page=page)
        low = sent.lower()

        def add(tp, val):
            key = (tp, val.lower())
            if key not in seen:
                seen.add(key)
                tokens.append(FactToken(type=tp, value=val, source=src))

        # деньги
        for m in self._amount.findall(sent):
            add("amount", m)

        # даты
        for m in self._date.findall(sent):
            add("date", m)

        # ФИО
        for m in self._fio.findall(sent):
            full = " ".join([p for p in m if p])
            if full.lower() not in self.PERSON_STOPWORDS:
                add("person", full)

        # инициалы
        for m in self._fio_initials.findall(sent):
            if m[0].lower() not in self.PERSON_STOPWORDS:
                add("person", " ".join(m))

        # телефоны
        for m in self._phone.findall(sent):
            add("phone", m)

        # карты и IBAN
        for m in self._iban.findall(sent):
            add("account", m)
        for m in self._card.findall(sent):
            digits = re.sub(r"\D", "", m)
            if len(digits) >= 12:
                add("account", m)

        # крипто-адреса
        for m in self._crypto_addr.findall(sent):
            add("crypto", m)

        # статьи
        for m in self._article_ref.findall(sent):
            add("article_ref", m)

        # каналы
        for kw in self._channel_kw:
            if kw in low:
                add("channel", kw)

        # крипто-маркеры
        for kw in self._crypto_kw:
            if kw in low:
                add("crypto_flag", kw)

        # fraud
        for kw in self._fraud_kw:
            if kw in low:
                add("fraud_flag", kw)

        # invest
        for kw in self._investment_kw:
            if kw in low:
                add("invest_flag", kw)

        # economic
        for kw in self._economic_kw:
            if kw in low:
                add("economic_flag", kw)

        # admin
        for kw in self._admin_kw:
            if kw in low:
                add("admin_flag", kw)

        # scheme
        for kw in self._scheme_kw:
            if kw in low:
                add("scheme_flag", kw)

        # role words
        for raw, label in self._role_kw.items():
            if raw in low:
                add("role_label", label)

        return tokens

    # =====================================================================
    # Детектор роли (под Router 6.2)
    # =====================================================================
    def _detect_role(self, fact: LegalFact, sent: str):
        types = {t.type for t in fact.tokens}
        low = sent.lower()

        # fraud
        if "fraud_flag" in types and ("amount" in types or "economic_flag" in types):
            return "fraud_action"
        if "fraud_flag" in types:
            return "fraud_event"

        # suspect actions
        if "economic_flag" in types and "suspect" in low:
            return "suspect_action"

        # victim losses
        if "amount" in types and "victim" in low:
            return "victim_loss"

        # investment
        if "invest_flag" in types and ("amount" in types or "economic_flag" in types):
            return "investment_context"
        if "invest_flag" in types:
            return "investment_event"

        # money transfer
        if "amount" in types and ("economic_flag" in types or "channel" in types or "account" in types):
            return "money_transfer"

        # crypto
        if "crypto_flag" in types or "crypto" in types:
            return "crypto_operation"

        # admin/organizational actions
        if "admin_flag" in types:
            return "admin_action"

        # scheme markers
        if "scheme_flag" in types:
            return "scheme_marker"

        # digital transfer markers
        if "channel" in types or "account" in types:
            return "digital_transfer"

        # generic economic actions
        if "economic_flag" in types:
            return "economic_action"

        # role statements
        if "person" in types and "role_label" in types:
            return "role_statement"

        return "generic_fact"

    # =====================================================================
    # Подсказки статей
    # =====================================================================
    def _article_hints(self, text: str):
        t = text.lower()
        hints = []

        if "мошеннич" in t or "обман" in t:
            hints.append("190")
        if any(x in t for x in ["инвестиц", "пирамид", "вклад"]):
            hints.append("217")
        if any(x in t for x in ["присвоил", "растрата"]):
            hints.append("189")
        if any(x in t for x in ["налог", "уклонение"]):
            hints.append("245")

        return hints

    # =====================================================================
    # Новый confidence v4 — под Investigator Mode
    # =====================================================================
    def _confidence_v4(self, fact: LegalFact):
        score = 0.0
        types = {t.type for t in fact.tokens}

        weights = {
            "amount": 0.50,
            "date": 0.32,
            "economic_flag": 0.40,
            "fraud_flag": 0.45,
            "invest_flag": 0.38,
            "crypto_flag": 0.42,
            "crypto": 0.45,
            "account": 0.35,
            "channel": 0.30,
            "admin_flag": 0.28,
            "scheme_flag": 0.40,
            "role_label": 0.15,
            "person": 0.12,
        }

        for t in fact.tokens:
            score += weights.get(t.type, 0.08)

        # доп бонусы
        if "fraud_flag" in types and "amount" in types:
            score += 0.25
        if "invest_flag" in types and "amount" in types:
            score += 0.20
        if "crypto" in types and "amount" in types:
            score += 0.25

        return min(1.0, round(score, 3))
