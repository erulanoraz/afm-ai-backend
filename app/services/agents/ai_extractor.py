# app/services/agents/ai_extractor.py
"""
AI Extractor 3.0 (с SUPER PRE-FILTER)

Цель:
• НЕ удалять фабулу, но удалять:
    — диалоги (Вопрос / Ответ)
    — анкетные блоки (ФИО, гражданство…)
    — служебный мусор ("допрос окончен", подписи)
    — строки потерпевших, которые бесполезны
• давать ai_qualifier'у ЧИСТЫЕ предложения
• не ломать существующую логику
"""

import re
import logging
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================
# 🔥 SUPER PRE-FILTER 2.0 — детерминированная очистка текста
# ============================================================

DIALOG_QUESTIONS = [
    r"вопрос следовател[яй]:?",
    r"вопрос:?",
    r"спросили:?",
]

DIALOG_ANSWERS = [
    r"ответ подозреваем[а-яё]:?",
    r"ответ потерпевш[а-яё]:?",
    r"ответ свидетел[яй]:?",
    r"ответ:?",
]

SERVICE_GARBAGE = [
    r"на этом допрос .*? оконч[её]н",
    r"документ подготовил.*",
    r"документ подписан.*",
    r"ордер №.*",
    r"приложени[ея].*",
    r"протокол допроса.*",
    r"уведомлен.*?",
]

PERSON_TECH_LINES = [
    r"фамилия[,:\s]",
    r"имя[,:\s]",
    r"отчество[,:\s]",
    r"гражданство[,:\s]",
    r"национальность[,:\s]",
    r"семейное положение[,:\s]",
    r"место работы.*?:",
    r"место жительства.*?:",
    r"место рождения.*?:",
    r"дата рождения.*?:",
]

VICTIM_NOISE = [
    r"потерпевш[а-яё]*:?$",
    r"на этом допрос потерпевшего оконч[её]н",
]

def super_pre_filter(text: str) -> list[str]:
    """
    Принимает сырой текст чанка.
    Возвращает список очищенных предложений.
    """
    if not text or len(text.strip()) < 3:
        return []

    t = text.strip()

    # 1) Удаляем служебный мусор
    for pattern in SERVICE_GARBAGE + VICTIM_NOISE:
        t = re.sub(pattern, "", t, flags=re.IGNORECASE)

    # 2) Удаляем диалоговые маркеры
    for pattern in DIALOG_QUESTIONS + DIALOG_ANSWERS:
        t = re.sub(pattern, "", t, flags=re.IGNORECASE)

    # 3) Удаляем анкетные строки
    for pattern in PERSON_TECH_LINES:
        t = re.sub(pattern, "", t, flags=re.IGNORECASE)

    # 4) Разбиваем на предложения
    sentences = re.split(r"(?<=[\.\?\!])\s+", t)

    cleaned = []
    for s in sentences:
        s = s.strip()

        if not s or len(s) < 5:
            continue

        # Удаляем строки типа "Иванов И.И."
        if re.fullmatch(r"[А-ЯA-ZЁӘӨҚҮҰҢ][а-яa-zёәөқүұң]+ [А-ЯA-Z]\.[А-ЯA-Z]\.", s):
            continue

        cleaned.append(s)

    return cleaned


# ============================================================
# 🧠 БАЗОВЫЕ РЕГУЛЯРКИ
# ============================================================

DATE_REGEX = r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b"
AMOUNT_REGEX = r"\b(\d[\d\s]{0,15}\s?(?:тенге|тг|₸|kzt|usd|usdt|eur|руб(?:\.|лей)?))\b"

# • Фамилия Имя Отчество
# • Фамилия Имя
# • Фамилия И.О.
# • ФамилияИ.О. (с инициалами слитно)
PERSON_REGEX = r"""
(
    [A-ZА-ЯӘІҢҒҮҰҚӨҺ][a-zа-яәіңғүұқөһё]+          # Фамилия
    \s+
    [A-ZА-ЯӘІҢҒҮҰҚӨҺ][a-zа-яәіңғүұқөһё]*          # Имя
    (?:\s+[A-ZА-ЯӘІҢҒҮҰҚӨҺ][a-zа-яәіңғүұқөһё]*)?  # Отчество
)
|
(
    [A-ZА-ЯӘІҢҒҮҰҚӨҺ][a-zа-яәіңғүұқөһё]+          # Фамилия
    \s*
    [A-ZА-ЯӘІҢҒҮҰҚӨҺ]\.[A-ZА-ЯӘІҢҒҮҰҚӨҺ]\.        # Инициалы (К.Т.)
)
"""

# ============================================================
# 🔥 КАТЕГОРИИ ДЕЙСТВИЙ (универсальные триггеры)
# ============================================================

CRIMINAL_ACTIONS = {
    "money_transfer": [
        "перевел", "перевёл", "перевела", "перечислил", "перечислила",
        "отправил", "отправила", "получил", "получила",
        "перевод", "получение средств", "зачисление", "поступление средств",
    ],
    "withdrawal": [
        "вывел", "вывела", "снял", "сняла",
        "вывод средств", "снятие наличных", "обналичил", "обналичила",
    ],
    "investment": [
        "внес", "внёс", "внесла", "вложил", "вложила", "инвестировал",
        "инвестировала", "пополнил", "пополнила",
        "баланс", "инвестиция", "пополнение", "депозит",
    ],
    "fraud_signals": [
        "обман", "обманул", "обманула", "ввел в заблуждение", "ввёл в заблуждение",
        "ввел в заблуждени", "не вернули", "не получил", "не получила",
        "деньги пропали", "отказали в выводе", "отказали в возврате",
        "обманным путем", "обманным путём",
    ],
    "pyramid_activity": [
        "платформ", "приложени", "задание",
        "инвест", "группа", "чат", "вознаграждени", "реферальн",
        "пирамида", "платежные поручения", "структурные подразделения",
    ],
}

# ============================================================
# 🚫 ТЕХНИЧЕСКИЙ МУСОР (минимальный, безопасный)
# ============================================================

BANNED_PATTERNS = [
    "qr-код", "qr код",
    "эцп", "ecp",
    "электронный документ",
    "код документа",
    "датой и временем подписания",
    "подпись наложена",
    "просмотрено", "дата печати",
]

# ============================================================
# 🛠️ ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ
# ============================================================

def _split_sentences(text: str) -> list[str]:
    """
    Простое разбиение по предложениям.
    Не идеально, но главное — НЕ уничтожать текст.
    """
    if not text:
        return []
    # точки, ?, ! + переносы
    parts = re.split(r"(?<=[\.\?!])\s+", text)
    return [p.strip() for p in parts if p and len(p.strip()) > 2]


def _is_technical_noise(sentence: str) -> bool:
    """
    Убираем чисто технический мусор (QR, ЭЦП, служебные строки).
    Никаких жёстких фильтров по фабуле!
    """
    lt = sentence.lower()
    if len(lt) < 5:
        return True

    if any(p in lt for p in BANNED_PATTERNS):
        return True

    # супер-тех строки типа "стр. 1 из 5"
    if re.search(r"страниц[аы]?\s+\d+\s+из\s+\d+", lt):
        return True

    return False


# ============================================================
# 👤 НОРМАЛИЗАЦИЯ ФИО
# ============================================================

def normalize_persons(persons: list[str]) -> dict:
    """
    Группируем ФИО по фамилиям, чтобы не плодить дубликаты.
    """
    clusters = defaultdict(list)
    for p in persons:
        clean = re.sub(r"\s+", " ", p).strip()
        if not clean:
            continue
        base = clean.lower()
        key = base.split()[0]  # фамилия как ключ
        clusters[key].append(clean)
    return {k: list(set(v)) for k, v in clusters.items()}


# ============================================================
# 🧩 РОЛИ
# ============================================================

ROLE_MAP = {
    "suspect": [
        "подозреваем", "подозревается", "подозреваемому",
        "обвиняем", "в отношении", "задержан", "подследствен",
        "күдікті", "күдіктінің", "күдіктіге",
    ],
    "victim": [
        "потерпевш", "жәбірленуш",
    ],
    "witness": [
        "свидетел", "куәгер",
    ],
}

def extract_roles(facts: list[dict], persons: list[str]) -> dict:
    """
    Пытаемся привязать ФИО к ролям по окружению текста.
    """
    roles = defaultdict(list)
    normalized = normalize_persons(persons)

    for f in facts:
        txt = (f.get("text") or "").lower()
        if not txt:
            continue

        for key, variants in normalized.items():
            if not variants:
                continue

            # если фамилия фигурирует в тексте — анализируем, кто он
            if key in txt:
                for role, markers in ROLE_MAP.items():
                    if any(m in txt for m in markers):
                        roles[role].extend(variants)
                        break

    # если кого-то не отнесли никуда — OTHER
    for key, variants in normalized.items():
        already = set()
        for rlist in roles.values():
            for v in rlist:
                already.add(v)
        for v in variants:
            if v not in already:
                roles["other"].append(v)

    # убираем дубликаты
    return {r: list(sorted(set(vs))) for r, vs in roles.items() if vs}


# ============================================================
# 🔍 ПОИСК ПОДОЗРЕВАЕМОГО
# ============================================================

def detect_suspect(all_sentences: list[str], persons_from_facts: list[str]) -> str | None:
    """
    Логика:
        1) Ищем предложение, где есть "подозреваем"/"обвиняем"/"в отношении".
        2) В нём ищем ФИО.
        3) Если не нашли — берём первого человека из roles['suspect'] (позже).
    """
    markers = [
        "подозреваем", "подозревается", "подозреваемому",
        "обвиняем", "в отношении", "задержан",
        "в качестве подозреваемого",
        "күдікті", "күдіктінің", "күдіктіге",
    ]

    # 1-й проход — ищем ФИО в "подозреваемых" предложениях
    for s in all_sentences:
        lt = s.lower()
        if not any(m in lt for m in markers):
            continue

        persons = re.findall(PERSON_REGEX, s, flags=re.VERBOSE)
        if not persons:
            continue

        for group in persons:
            for item in group:
                if item.strip():
                    cand = re.sub(r"\s+", " ", item.strip())
                    logger.info(f"🔎 detect_suspect: найден кандидат из текста: {cand}")
                    return cand

    # 2-й проход — fallback: если список persons_from_facts маленький, берём первого
    if persons_from_facts:
        cand = re.sub(r"\s+", " ", persons_from_facts[0]).strip()
        logger.info(f"🔎 detect_suspect: fallback по первому лицу: {cand}")
        return cand

    logger.warning("⚠️ detect_suspect: подозреваемый не найден")
    return None


# ============================================================
# 🧱 ИЗВЛЕЧЕНИЕ СОБЫТИЙ
# ============================================================

def extract_events(sentences: list[str]) -> list[dict]:
    """
    Для каждого предложения делаем объект события:
        {
            "text": str,
            "action": one of CRIMINAL_ACTIONS keys or None,
            "amounts": [..],
            "persons": [..raw..],
            "date": "DD.MM.YYYY" or None
        }
    НИЧЕГО не удаляем, даже если action = None.
    """
    events: list[dict] = []

    for s in sentences:
        if not s or _is_technical_noise(s):
            continue

        lt = s.lower()

        dates = re.findall(DATE_REGEX, s)
        amounts = re.findall(AMOUNT_REGEX, s)
        persons_raw = re.findall(PERSON_REGEX, s, flags=re.VERBOSE)

        # выпрямляем PERSON_REGEX результаты
        persons: list[str] = []
        for group in persons_raw:
            for item in group:
                item = item.strip()
                if item:
                    persons.append(re.sub(r"\s+", " ", item))

        action = None
        for action_type, words in CRIMINAL_ACTIONS.items():
            if any(w in lt for w in words):
                action = action_type
                break

        events.append({
            "text": s.strip(),
            "action": action,
            "amounts": amounts,
            "persons": persons,
            "date": dates[0] if dates else None,
        })

    logger.info(f"📌 extract_events: событий={len(events)}")
    return events


# ============================================================
# 🔗 FLOW (ступени развития события)
# ============================================================

def build_crime_flow(events: list[dict]) -> list[dict]:
    """
    Преобразуем "сырой" список событий в понятные шаги:
        вложение → перевод → попытка вывода → обман/невозврат → ...
    """
    flow: list[dict] = []

    mapping = {
        "investment": "вложение средств",
        "money_transfer": "перевод средств",
        "withdrawal": "попытка вывода",
        "fraud_signals": "обман / невозврат",
        "pyramid_activity": "участие в финансовой схеме",
    }

    for e in events:
        step = mapping.get(e["action"])
        if not step:
            continue

        flow.append({
            "step": step,
            "amount": ", ".join(e["amounts"]),
            "text": e["text"],
            "date": e["date"],
        })

    logger.info(f"📌 build_crime_flow: шагов={len(flow)}")
    return flow


# ============================================================
# 📅 ТАЙМЛАЙН
# ============================================================

def build_timeline(events: list[dict]) -> list[dict]:
    """
    Сортируем события по дате (если указана).
    """
    result: list[tuple[datetime, dict]] = []

    for e in events:
        d = e.get("date")
        if not d:
            continue
        try:
            dt = datetime.strptime(d, "%d.%m.%Y")
            result.append((dt, e))
        except Exception:
            # если формат странный — пропускаем
            continue

    result.sort(key=lambda x: x[0])
    timeline = [e for _, e in result]
    logger.info(f"📌 build_timeline: событий с датой={len(timeline)}")
    return timeline


# ============================================================
# ⚖️ ЮРИДИЧЕСКИ ЗНАЧИМЫЕ ФАКТЫ
# ============================================================

def extract_legal_facts(events: list[dict], roles: dict) -> dict:
    legal = {
        "subject": roles.get("suspect", []),
        "objective_side": [],
        "damage": [],
        "method": [],
        "intent": None,
        "motive": None,
    }

    for e in events:
        txt = (e.get("text") or "").lower()

        if e.get("amounts"):
            legal["damage"].extend(e["amounts"])

        if e.get("action"):
            legal["objective_side"].append(
                f"{e['action']} ({', '.join(e['amounts'])})".strip("() ")
            )

        if "с целью" in txt and not legal["motive"]:
            after = txt.split("с целью", 1)[1]
            legal["motive"] = after[:150].strip()

        if "предварительн" in txt and not legal["intent"]:
            legal["intent"] = "прямой умысел (по описанию событий)"

    # убираем дубликаты
    legal["damage"] = list(sorted(set(legal["damage"])))
    legal["objective_side"] = list(sorted(set(legal["objective_side"])))

    logger.info(
        f"📌 extract_legal_facts: subject={legal['subject']}, "
        f"damage={len(legal['damage'])}, obj_side={len(legal['objective_side'])}"
    )
    return legal


# ============================================================
# 🔥 ОПРЕДЕЛЕНИЕ ТИПА ПРЕСТУПЛЕНИЯ
# ============================================================

def detect_crime_type(events: list[dict]) -> str:
    blob = " ".join(e.get("text", "").lower() for e in events)

    if any(w in blob for w in CRIMINAL_ACTIONS["fraud_signals"]):
        return "мошенничество"

    if any(w in blob for w in CRIMINAL_ACTIONS["pyramid_activity"]):
        return "финансовая схема"

    if any(w in blob for w in CRIMINAL_ACTIONS["investment"]):
        return "незаконное привлечение средств"

    if any(w in blob for w in CRIMINAL_ACTIONS["withdrawal"]):
        return "препятствие выводу средств / отказ в выводе"

    if any(w in blob for w in CRIMINAL_ACTIONS["money_transfer"]):
        return "движение денежных средств"

    return "неопределено"

# ============================================================
# 🔍 Определение важности предложения
# ============================================================

def is_meaningful(sentence: str) -> bool:
    """
    Мягкий фильтр важности.
    НЕ удаляет фабулу, только шум.
    """
    if not sentence:
        return False

    s = sentence.lower().strip()
    if len(s) < 5:
        return False

    # Технический шум — сразу в мусор
    if any(p in s for p in BANNED_PATTERNS):
        return False

    # Если содержит ключевые триггеры преступления — всегда OK
    for group in CRIMINAL_ACTIONS.values():
        if any(w in s for w in group):
            return True

    # Если есть деньги / суммы — OK
    if re.search(AMOUNT_REGEX, s):
        return True

    # Если есть ФИО — OK
    if re.search(PERSON_REGEX, s, flags=re.VERBOSE):
        return True

    # Если содержит фразы про подозреваемого — OK
    suspect_markers = [
        "подозреваем", "подозревается", "в отношении",
        "обвиняем", "задержан",
        "күдікті", "күдіктінің", "күдіктіге",
        "в качестве подозреваемого",
    ]
    if any(m in s for m in suspect_markers):
        return True

    # Потерпевшие
    if "потерпевш" in s or "жәбірленуш" in s:
        return True

    # Фразы про действия / участие
    if any(x in s for x in ["платформ", "инвест", "группа", "чат", "вывод", "влож"]):
        return True

    # Всё остальное — НЕ удаляем, но как meaningful не считаем
    return False


# ============================================================
# 🧩 ГЛАВНАЯ ФУНКЦИЯ extract_all() — ДОРАБОТАНА
# ============================================================

def extract_all(facts, persons, dates, amounts):
    """
    Главная функция.
    ДОБАВЛЕНО:
    — super_pre_filter() применяется к каждому факту
    — fact["sentences"] используется дальше
    """

    safe = []
    for f in facts or []:
        if not isinstance(f, dict):
            continue

        txt = f.get("text") or ""
        if len(txt.strip()) < 3:
            continue

        # 🔥 ДОБАВЛЕНО: PRE-FILTER
        f["sentences"] = super_pre_filter(txt)

        safe.append(f)

    facts = safe
    if not facts:
        return {
            "roles": {},
            "events": [],
            "timeline": [],
            "legal_facts": {
                "subject": [],
                "objective_side": [],
                "damage": [],
                "method": [],
                "intent": None,
                "motive": None,
            },
            "suspects": [],
            "primary_suspect": None,
            "crime_flow": [],
            "crime_type": "неопределено",
        }

    persons = persons or []
    dates = dates or []
    amounts = amounts or []

    # ---------------------------------------------------------
    # Сбор предложений
    # ---------------------------------------------------------
    all_sentences = []
    for f in facts:
        all_sentences.extend(f["sentences"])

    all_sentences = [s for s in all_sentences if isinstance(s, str) and s.strip()]

    # Мягкая фильтрация фабулы
    filtered = [s for s in all_sentences if is_meaningful(s)]
    if not filtered:
        filtered = all_sentences

    # ---------------------------------------------------------
    # Роли
    # ---------------------------------------------------------
    roles = extract_roles(facts, persons)

    suspects_list = roles.get("suspect", []) or []

    primary_suspect = detect_suspect(all_sentences, persons)
    if primary_suspect and primary_suspect not in suspects_list:
        suspects_list.append(primary_suspect)

    suspects_list = list(dict.fromkeys(suspects_list))

    # ---------------------------------------------------------
    # События → таймлайн → flow → юр.факты → тип
    # ---------------------------------------------------------
    events = extract_events(filtered) or extract_events(all_sentences)
    timeline = build_timeline(events)
    crime_flow = build_crime_flow(events)
    legal_facts = extract_legal_facts(events, roles)
    crime_type = detect_crime_type(events)

    return {
        "roles": roles,
        "events": events,
        "timeline": timeline,
        "legal_facts": legal_facts,
        "suspects": suspects_list,
        "primary_suspect": primary_suspect,
        "crime_flow": crime_flow,
        "crime_type": crime_type,
    }
