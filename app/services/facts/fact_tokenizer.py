from __future__ import annotations
import uuid
import re
from typing import List, Set

from app.services.facts.fact_models import LegalFact, FactToken, SourceRef
from app.utils.sentence_splitter import split_into_sentences


class FactTokenizer:
    """
    FactTokenizer v32.0 — HUMAN-LOGIC FIXED

    Цели v32.0:
    • НЕ считать ФИО любые странные формы вроде «Военнообязанный Наличие», «Нет Отношение»,
      «Республика Казахстан», «Республики Казахстан» и т.п.;
    • Жёстко фильтровать person-токены и оставлять только реальные ФИО;
    • Сохранить всю криминальную логику v31.0 (amount / scheme / crypto / roles / confidence и т.п.).
    """

    VERSION = "32.0"

    # ======================================================================
    # Шумовые фразы, которые не могут быть именами
    # ======================================================================
    NOISE_PERSON_PHRASES: Set[str] = {
        "военнообязанный", "военнообязанная",
        "военнообязанный наличие", "наличие", "отношение", "нет отношение",
        "нет отношения", "нет сведений",
        "республика казахстан", "республики казахстан",
        "республика", "казахстан",
        "гражданин", "гражданка",
        "заявитель", "потерпевший", "потерпевшая",
        "подозреваемый", "подозреваемая",
        "организация", "компания", "банк", "проект",
        "улица", "проспект", "переулок", "микрорайон", "мкр",
        "область", "район", "город",
        "населенный пункт", "населённый пункт",
        "адрес", "место жительства",
        "он", "она", "они", "оно",
        "в наличии", "без определенного места жительства",
        "без определённого места жительства",
        "сведения отсутствуют",
    }

    # ======================================================================
    # Country / status patterns
    # ======================================================================
    COUNTRY_PATTERNS = [
        r"\bреспублика казахстан\b",
        r"\bреспублики казахстан\b",
        r"\bрк\b",
        r"\bказахстан\b",
    ]

    STATUS_PATTERNS = [
        r"\bвоеннообяз\w+\b",
        r"\bбез определенн\w+ места жительства\b",
        r"\bнет отношени\w*\b",
        r"\bне имеет отношения\b",
        r"\bналичие сведений\b",
        r"\bотсутствие сведений\b",
        r"\bгражданин\b",
        r"\bгражданка\b",
        r"\bнесовершеннолетн\w+\b",
        r"\bинвалид\b",
        r"\bбезработн\w+\b",
    ]

    # ======================================================================
    # Stop-слова для person-кандидатов
    # ======================================================================
    PERSON_STOPWORDS = {
        "после", "примерно", "кроме", "когда", "далее", "заявление",
        "данная", "данные", "документ", "платформа", "потерпевший",
        "потерпевшая", "подозреваемый", "подозреваемая", "сотрудник",
        "счет", "счёт", "аккаунт", "дата", "перевод", "заявка",
        "организация", "компания", "проект", "фонд", "банк",
        "улица", "проспект", "город", "микрорайон", "район", "область",
    }

    # ======================================================================
    # FIRST PERSON MARKERS
    # ======================================================================
    FIRST_PERSON = [
        "я", "мне", "меня", "мной",
        "мы", "нам", "нас",
        "узнал", "узнала", "понимаю", "понял", "поняла",
        "считаю", "думаю",
    ]

    # ======================================================================
    # SUSPECT ACTION VERBS
    # ======================================================================
    SUSPECT_VERBS = [
        "получил", "получила",
        "присвоил", "присвоила",
        "привлек", "привлёк", "привлекла",
        "вывел", "вывела",
        "взял", "взяла",
        "забрал", "забрала",
    ]

    # ======================================================================
    # REGEX PATTERNS
    # ======================================================================
    _fio = re.compile(
        r"\b([А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?)\s+([А-ЯЁ][а-яё]+)"
        r"(?:\s+([А-ЯЁ][а-яё]+))?\b"
    )

    _fio_initials = re.compile(
        r"\b([А-ЯЁ][а-яё]+)\s+[А-ЯЁ]\.[А-ЯЁ]\.\b"
    )

    _entity = re.compile(
        r"\b("
        r"проект|компания|организация|платформа|фонд|инвестиционная платформа|"
        r"финансовая платформа|инвестиционный проект|сервис|приложение|группа"
        r")\s+"
        r"(\"([^\"]+)\"|'([^']+)'|«([^»]+)»|“([^”]+)”|([A-Za-z0-9А-Яа-яёЁ_.-]{3,60}))",
        re.IGNORECASE,
    )

    _amount = re.compile(
        r"\b\d[\d\s.,]{0,18}\s*"
        r"(₸|тенге|тг|kzt|rub|₽|usd|usdt|eur|сом|доллар)\b",
        re.IGNORECASE,
    )

    _date = re.compile(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?:\s*г\.?)?\b",
        re.IGNORECASE,
    )

    _phone = re.compile(
        r"\b(?:\+?7|8)[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b"
    )

    _iban = re.compile(r"\bKZ\d{18}\b", re.IGNORECASE)
    _card = re.compile(r"\b(?:\d[ -]?){12,20}\b")

    _article_ref = re.compile(
        r"\bст\.?\s*\d{1,3}(?:[-–]\d+)?\s*(ук|упк|гк)?\s*рк\b",
        re.IGNORECASE,
    )

    _crypto_addr = re.compile(
        r"\b(0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b"
    )

    # ======================================================================
    # KEYWORD GROUPS
    # ======================================================================
    FRAUD = [
        "обман", "мошеннич", "ввел в заблуждение", "ввёл в заблуждение",
        "заблужден", "заблуждён", "хищен"
    ]

    INVEST = [
        "инвестиц", "проценты", "дивиденды", "доход",
        "влож", "вклад", "депозит"
    ]

    ECONOMIC = [
        "получил", "получила", "перевел", "перевела",
        "снял", "сняла", "пополнил", "пополнила",
        "отправил", "отправила", "зачислил", "зачислила",
        "вывел", "вывела", "оплатил", "оплатила"
    ]

    ADMIN = [
        "администратор", "модератор", "рекламировал", "рекламировала",
        "создавал группы", "создал группу", "удалял сообщения"
    ]

    SCHEME = [
        "схема", "организованная группа", "структура", "механизм",
        "привлечение вкладчиков", "вовлечение вкладчиков", "финансовая пирамида",
        "инвестиционная пирамида", "признаки пирамиды"
    ]

    CRYPTO = [
        "usdt", "bitcoin", "btc", "eth", "binance", "okx", "bybit",
        "wallet", "кошелек", "кошелёк"
    ]

    CHANNEL = [
        "kaspi", "halyk", "qiwi", "visa", "mastercard",
        "fortе", "forte", "iban", "swift", "счет", "счёт"
    ]

    # ======================================================================
    # PROCESSUAL PHRASES
    # ======================================================================
    PROCESSUAL = [
        "разъяснены права",
        "уведомлен об уголовной ответственности",
        "уведомлена об уголовной ответственности",
        "составлен протокол",
        "составлено постановление",
        "составил рапорт",
        "зарегистрирован в ердр",
        "зарегистрировано в ердр",
        "зарегистрировано в куи",
        "изъяты документы",
        "изъял документы",
        "приобщены к материалам дела",
        "назначена экспертиза",
        "дело направлено в суд",
        "уголовное дело возбуждено",
        "начато досудебное расследование",
        "досудебное расследование начато",
    ]

    # ======================================================================
    # ROLE KEYWORDS
    # ======================================================================
    ROLE_LABELS = {
        "потерпевш": "victim",
        "вкладчик": "victim",
        "клиент": "victim",
        "подозреваем": "suspect",
        "обвиняем": "suspect",
        "организатор": "organizer",
        "руководитель": "organizer",
        "свидетел": "witness",
        "заявител": "applicant",
    }

    # ======================================================================
    # ADDRESS MARKERS
    # ======================================================================
    ADDRESS_MARKERS = [
        "ул.", "улица", "пр.", "проспект", "микрорайон",
        "мкр.", "мкр", "переулок", "шоссе", "бульвар"
    ]

    # ======================================================================
    # УСИЛЕННАЯ ПРОВЕРКА "НАСТОЯЩЕГО ФИО"
    # ======================================================================
    def _is_valid_real_fio(self, full: str) -> bool:
        """
        Жёсткая проверка настоящего ФИО:
        - 2–3 слова
        - каждое слово: только буквы, формат 'Иванов', 'Петров', 'Куаныш'
        - нет стоп-слов
        - нет шумовых фраз (республика, военнообязанный, наличие, отношение и т.д.)
        """

        full = (full or "").strip()
        if not full:
            return False

        low_full = full.lower()

        # моментально отбрасываем явно шумовые фразы
        for noise in self.NOISE_PERSON_PHRASES:
            if noise == low_full or noise in low_full:
                return False

        # страны / статусы
        for pattern in self.COUNTRY_PATTERNS:
            if re.search(pattern, low_full):
                return False
        for pattern in self.STATUS_PATTERNS:
            if re.search(pattern, low_full):
                return False

        parts = full.split()
        if len(parts) not in (2, 3):
            return False

        for p in parts:
            # только буквы, первая заглавная, остальное строчные
            if not re.match(r"^[А-ЯЁ][а-яё]+$", p):
                return False
            if p.lower() in self.PERSON_STOPWORDS:
                return False

        return True

    # ======================================================================
    # MAIN TOKENIZATION PIPELINE
    # ======================================================================
    def tokenize(self, docs: List[dict]) -> List[LegalFact]:
        facts: List[LegalFact] = []

        for doc in docs:
            file_id = doc.get("file_id")
            text = (doc.get("text") or "").strip()
            page = doc.get("page", 1)

            if not text:
                continue

            sentences = split_into_sentences(text)
            windows = self._context_windows(sentences)

            for idx, (before, sent, after) in enumerate(windows):
                sent = sent.strip()
                if not sent:
                    continue

                # 1) мягко фильтруем вопросы следователя
                if self._is_pure_question(sent):
                    continue

                tokens = self._extract_tokens(sent, file_id, page)
                if not tokens:
                    continue

                # 2) фильтр victim-first-person ТОЛЬКО для субъективных реплик без фактов
                if self._is_pure_victim_subjective(sent, tokens):
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
                fact.confidence = self._confidence(fact)

                if fact.confidence <= 0:
                    continue

                facts.append(fact)

        return facts

    # ======================================================================
    # CONTEXT WINDOWS
    # ======================================================================
    def _context_windows(self, sentences: List[str]):
        return [
            (
                sentences[i - 1] if i > 0 else "",
                sentences[i],
                sentences[i + 1] if i + 1 < len(sentences) else ""
            )
            for i in range(len(sentences))
        ]

    # ======================================================================
    # TOKEN EXTRACTION
    # ======================================================================
    def _extract_tokens(self, sent: str, file_id: str, page: int) -> List[FactToken]:
        tokens: List[FactToken] = []
        seen = set()
        src = SourceRef(file_id=file_id, page=page)
        low = sent.lower()

        def add(tp: str, val: str):
            if not val:
                return
            key = (tp, val.lower())
            if key not in seen:
                seen.add(key)
                tokens.append(FactToken(type=tp, value=val, source=src))

        # ENTITIES / PROJECT / PLATFORM
        for g in self._entity.findall(sent):
            name = g[2] or g[3] or g[4] or g[5] or g[6]
            ent_type = g[0].lower()
            if name:
                name_clean = name.strip()
                add("entity", name_clean)
                if "проект" in ent_type:
                    add("project", name_clean)
                if "платформа" in ent_type:
                    add("platform", name_clean)
                if ("компания" in ent_type or
                        "организация" in ent_type or
                        "фонд" in ent_type):
                    add("organization", name_clean)

        # AMOUNTS
        for m in self._amount.finditer(sent):
            add("amount", m.group(0))

        # DATES
        for m in self._date.findall(sent):
            add("date", m)

        # FIO — с жёсткой проверкой, что это РЕАЛЬНОЕ ФИО
        for m in self._fio.finditer(sent):
            full = " ".join([p for p in m.groups() if p])
            if not full:
                continue

            # sanity check
            if not self._is_valid_real_fio(full):
                # возможно, это страна/орган/статус — просто игнорируем как person
                continue

            start = m.start()
            left = sent[max(0, start - 30):start].lower()

            if any(mark in left for mark in self.ADDRESS_MARKERS):
                add("address", full)
            else:
                add("person", full)

        # FIO with initials — хотя бы фамилию ловим
        for m in self._fio_initials.finditer(sent):
            full = " ".join(m.groups())
            if not full:
                continue

            low_full = full.lower()
            if any(noise in low_full for noise in self.NOISE_PERSON_PHRASES):
                continue

            start = m.start()
            left = sent[max(0, start - 30):start].lower()
            if any(mark in left for mark in self.ADDRESS_MARKERS):
                add("address", full)
            else:
                add("person", full)

        # PHONES
        for m in self._phone.findall(sent):
            add("phone", m)

        # ACCOUNTS
        for m in self._iban.findall(sent):
            add("account", m)

        for m in self._card.findall(sent):
            digits = re.sub(r"\D", "", m)
            if len(digits) >= 12:
                add("account", m)

        # CRYPTO ADDRESSES
        for m in self._crypto_addr.findall(sent):
            add("crypto", m)

        # ARTICLE REFS
        for m in self._article_ref.findall(sent):
            if isinstance(m, tuple):
                add("article_ref", m[0])
            else:
                add("article_ref", m)

        # CHANNELS
        for kw in self.CHANNEL:
            if kw in low:
                add("channel", kw)

        # FLAGS
        for kw in self.CRYPTO:
            if kw in low:
                add("crypto_flag", kw)

        for kw in self.FRAUD:
            if kw in low:
                add("fraud_flag", kw)

        for kw in self.INVEST:
            if kw in low:
                add("invest_flag", kw)

        for kw in self.ECONOMIC:
            if kw in low:
                add("economic_flag", kw)

        for kw in self.ADMIN:
            if kw in low:
                add("admin_flag", kw)

        for kw in self.SCHEME:
            if kw in low:
                add("scheme_flag", kw)

        # PROCESSUAL FLAG
        for kw in self.PROCESSUAL:
            if kw in low:
                add("processual_flag", kw)

        # ROLE LABELS
        for raw, label in self.ROLE_LABELS.items():
            if raw in low:
                add("role_label", label)

        return tokens

    # ======================================================================
    # QUESTION FILTER (мягкий)
    # ======================================================================
    def _is_pure_question(self, sent: str) -> bool:
        """
        Режем только ЧИСТЫЕ вопросы следователя:
        - начинаются с 'Вопрос:' / 'Вопрос :'
        - содержат '?'
        - И ПРИ ЭТОМ нет денег, инвестиций, крипты, сущностей, переводов.
        Всё, что содержит amount / fraud / invest / economic / crypto / entity — сохраняем.
        """
        low = sent.lower()

        if "вопрос:" not in low and "вопрос :" not in low and "?" not in sent:
            return False

        has_fact_marker = any(
            kw in low
            for kw in (
                [
                    "тенге", "usdt", "доллар", "перевел", "перевела",
                    "влож", "вклад", "финансовая пирамида",
                    "инвестиционная пирамида", "платформ", "проект"
                ]
            )
        )
        if has_fact_marker:
            return False

        return True

    # ======================================================================
    # VICTIM FIRST PERSON FILTER (умеренный)
    # ======================================================================
    def _is_pure_victim_subjective(self, sent: str, tokens: List[FactToken]) -> bool:
        """
        Отбрасываем ТОЛЬКО чистые субъективные реплики потерпевшего:
        «я понял, что это пирамида», «я считаю, что меня обманули»
        БЕЗ суммы, БЕЗ перевода, БЕЗ платформы, БЕЗ схемы.

        Если есть:
        - amount / economic_flag / invest_flag / fraud_flag / scheme_flag / crypto_flag / channel / account / entity
        → ОСТАВЛЯЕМ, даже с «я / мне / нас».
        """
        low = sent.lower()
        types = {t.type for t in (tokens or [])}

        if not any(w in low for w in self.FIRST_PERSON):
            return False

        # если есть явный подозреваемый / организатор → факт важен
        if "подозреваем" in low or "обвиняем" in low or "организатор" in low:
            return False
        if any(t.type == "role_label" and str(t.value).lower().startswith(("suspect", "organizer"))
               for t in (tokens or [])):
            return False

        # если есть сильные криминальные маркеры — НЕ режем
        CRIMINAL_TYPES = {
            "amount", "economic_flag", "fraud_flag", "invest_flag",
            "scheme_flag", "crypto_flag", "crypto", "channel",
            "account", "entity", "project", "platform", "organization",
        }
        if types.intersection(CRIMINAL_TYPES):
            return False

        # здесь остаются только чистые оценки «я понял», «я считаю» и т.п.
        return True

    # ======================================================================
    # ROLE DETECTION
    # ======================================================================
    def _detect_role(self, fact: LegalFact, sent: str) -> str:
        types = {t.type for t in fact.tokens}
        low = sent.lower()

        # 1) ultra suspect action (деньги + активный глагол подозреваемого)
        if "amount" in types and any(v in low for v in self.SUSPECT_VERBS):
            return "suspect_action"

        # 2) fraud + деньги
        if "fraud_flag" in types and "amount" in types:
            return "fraud_action"

        # 3) investment
        if "invest_flag" in types and ("amount" in types or "economic_flag" in types):
            return "investment_event"
        if "invest_flag" in types:
            return "investment_context"

        # 4) scheme / project
        if "scheme_flag" in types:
            return "scheme_marker"
        if "entity" in types or "project" in types or "platform" in types or "organization" in types:
            return "entity_reference"

        # 5) admin
        if "admin_flag" in types:
            return "admin_action"

        # 6) crypto/digital/economic
        if "crypto_flag" in types or "crypto" in types:
            return "crypto_operation"
        if "channel" in types or "account" in types:
            return "digital_transfer"
        if "economic_flag" in types:
            return "economic_action"

        # 7) victim / role / generic
        if "victim" in "".join(str(t.value).lower() for t in fact.tokens if t.type == "role_label"):
            if "amount" in types or "economic_flag" in types or "invest_flag" in types:
                return "victim_loss"
            return "victim_statement"

        if "потерпев" in low:
            if "amount" in types or "economic_flag" in types or "invest_flag" in types:
                return "victim_loss"
            return "victim_statement"

        if "role_label" in types:
            return "role_statement"

        return "generic_fact"

    # ======================================================================
    # ARTICLE HINTS FOR 190 / 217
    # ======================================================================
    def _article_hints(self, t: str):
        t = t.lower()
        hints = []
        if "мошеннич" in t or "обман" in t:
            hints.append("190")
        if "пирамид" in t or "инвестиц" in t or "вклад" in t:
            hints.append("217")
        return hints

    # ======================================================================
    # CONFIDENCE MODEL
    # ======================================================================
    def _confidence(self, fact: LegalFact) -> float:
        score = 0.0
        types = {t.type for t in fact.tokens}
        text = (fact.text or "").lower()

        weights = {
            "entity": 0.40,
            "project": 0.40,
            "platform": 0.40,
            "organization": 0.35,
            "amount": 0.40,
            "economic_flag": 0.35,
            "fraud_flag": 0.45,
            "invest_flag": 0.40,
            "scheme_flag": 0.40,
            "crypto": 0.35,
            "crypto_flag": 0.30,
            "channel": 0.25,
            "account": 0.25,
            "date": 0.12,
            "person": 0.10,
            "address": 0.05,
        }

        for t_type in types:
            score += weights.get(t_type, 0.05)

        # ultra boost — подозреваемый действует с суммой
        if getattr(fact, "role", None) == "suspect_action" and "amount" in types:
            if any(v in text for v in self.SUSPECT_VERBS):
                return 1.0

        # fraud / investment + сумма
        if getattr(fact, "role", None) in ("fraud_action", "investment_event") and "amount" in types:
            score += 0.3

        # лёгкий штраф за субъективные victim-реплики без суммы (но НЕ убиваем)
        if any(w in text for w in self.FIRST_PERSON) and "fraud_flag" in types and "amount" not in types:
            score *= 0.8

        score = min(1.0, round(score, 3))
        if score < 0.05:
            return 0.0
        return score
