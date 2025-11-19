# app/services/agents/ai_extractor.py

import re
import logging
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================
# 🧠 БАЗОВЫЕ РЕГУЛЯРКИ
# ============================================================

DATE_REGEX = r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b"
AMOUNT_REGEX = r"\b(\d[\d\s]{0,12}\s?(?:тенге|тг|₸|kzt|usd|usdt))\b"

# Ловит:
# • Фамилия Имя Отчество
# • Фамилия Имя
# • Фамилия И.О.
# • ФамилияИ.О.
PERSON_REGEX = r"""
(
    [A-ZА-ЯӘІҢҒҮҰҚӨҺ][a-zа-яәіңғүұқөһё]+            # Фамилия
    \s+
    [A-ZА-ЯӘІҢҒҮҰҚӨҺ][a-zа-яәіңғүұқөһё]*            # Имя
    (?:\s+[A-ZА-ЯӘІҢҒҮҰҚӨҺ][a-zа-яәіңғүұқөһё]*)?    # Отчество
)
|
(
    [A-ZА-ЯӘІҢҒҮҰҚӨҺ][a-zа-яәіңғүұқөһё]+            # Фамилия
    \s*
    [A-ZА-ЯӘІҢҒҮҰҚӨҺ]\.[A-ZА-ЯӘІҢҒҮҰҚӨҺ]\.          # Инициалы (К.Т.)
)
"""

# ============================================================
# 🔥 КАТЕГОРИИ ДЕЙСТВИЙ
# ============================================================

CRIMINAL_ACTIONS = {
    "money_transfer": [
        "перевел", "перевела", "перечислил", "перечислила",
        "отправил", "отправила", "получил", "получила",
        "перевод", "получение средств"
    ],
    "withdrawal": [
        "вывел", "вывела", "снял", "сняла",
        "вывод средств"
    ],
    "investment": [
        "внес", "внесла", "вложил", "вложила", "инвестировал",
        "инвестировала", "пополнил", "пополнила",
        "баланс", "инвестиция", "пополнение"
    ],
    "fraud_signals": [
        "обман", "обманул", "ввел в заблуждение",
        "не вернули", "не получил", "не получила",
        "деньги пропали", "отказали в выводе"
    ],
    "pyramid_activity": [
        "платформ", "приложени", "задание",
        "инвест", "группа", "чат", "вознаграждение"
    ],
}

# ============================================================
# 🚫 ТЕХНИЧЕСКИЙ МУСОР
# ============================================================

BANNED_PATTERNS = [
    "qr", "эцп", "подпись",
    "служебн", "кабинет №",
    "дата печати", "просмотрено",
]

# ============================================================
# 👤 НОРМАЛИЗАЦИЯ ФИО
# ============================================================

def normalize_persons(persons: list[str]) -> dict:
    clusters = defaultdict(list)
    for p in persons:
        clean = re.sub(r"\s+", " ", p).strip()
        base = clean.lower()
        key = base.split()[0]
        clusters[key].append(clean)
    return {k: list(set(v)) for k, v in clusters.items()}

# ============================================================
# 🧩 РОЛИ
# ============================================================

ROLE_MAP = {
    "suspect": [
        "подозреваем", "подозревается", "подозреваемому",
        "обвиняем", "задержан", "подследствен",
        "күдікті", "күдіктінің", "күдіктіге"
    ],
    "victim": ["потерпевш", "жәбірленуш"],
    "witness": ["свидетел", "куәгер"],
}

def extract_roles(facts: list[dict], persons: list[str]) -> dict:
    roles = defaultdict(list)
    normalized = normalize_persons(persons)

    for variants in normalized.values():
        blob = " ".join(variants).lower()
        assigned = False

        for role, keys in ROLE_MAP.items():
            if any(k in blob for k in keys):
                roles[role].extend(variants)
                assigned = True
                break

        if not assigned:
            roles["other"].extend(variants)

    return dict(roles)

# ============================================================
# 🔍 КРИТЕРИЙ ВАЖНОСТИ ФРАЗЫ
# ============================================================

def is_meaningful(sentence: str) -> bool:
    lt = sentence.lower().strip()
    if not lt:
        return False
    if "протокол допроса подозреваем" in lt:
        return True


    suspect_markers = [
        "подозреваем", "обвиняем", "в отношении",
        "задержан", "подследствен",
        "совершил", "совершила",
        "причаст", "администрировал", "руководил",
        "күдікті", "күдіктінің", "күдіктіге",
        "күдікті ретінде"
    ]
    if any(w in lt for w in suspect_markers):
        return True

    if re.search(AMOUNT_REGEX, lt):
        return True

    for group in CRIMINAL_ACTIONS.values():
        if any(w in lt for w in group):
            return True

    if any(x in lt for x in ["платформ", "приложени", "инвест", "задани", "группа"]):
        return True

    if "потерпевш" in lt or "свидетел" in lt:
        return True

    if any(b in lt for b in BANNED_PATTERNS):
        return False

    return False

# ============================================================
# 👤 ПОИСК ПОДОЗРЕВАЕМОГО
# ============================================================

def detect_suspect(all_sentences: list[str]) -> str | None:
    markers = [
        "подозреваем", "подозревается", "подозреваемому",
        "обвиняем", "в отношении", "задержан",
        "күдікті", "күдіктінің", "күдіктіге"
    ]

    for s in all_sentences:
        lt = s.lower()

        # если нет маркера — пропускаем
        if not any(m in lt for m in markers):
            continue

        # ищем ФИО только в предложениях с маркером
        persons = re.findall(PERSON_REGEX, s, flags=re.VERBOSE)
        if not persons:
            continue

        for group in persons:
            for item in group:
                if item.strip():
                    return item.strip()

    return None


# ============================================================
# 🧱 ИЗВЛЕЧЕНИЕ СОБЫТИЙ
# ============================================================

def extract_events(sentences: list[str]) -> list[dict]:
    events = []

    for s in sentences:
        lt = s.lower()

        date = re.findall(DATE_REGEX, s)
        amounts = re.findall(AMOUNT_REGEX, s)
        persons = re.findall(PERSON_REGEX, s, flags=re.VERBOSE)

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
            "date": date[0] if date else None,
        })

    return events

# ============================================================
# 🔗 FLOW
# ============================================================

def build_crime_flow(events: list[dict]) -> list[dict]:
    flow = []

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

    return flow

# ============================================================
# 📅 ТАЙМЛАЙН
# ============================================================

def build_timeline(events: list[dict]) -> list[dict]:
    result = []
    for e in events:
        if not e["date"]:
            continue
        try:
            dt = datetime.strptime(e["date"], "%d.%m.%Y")
            result.append((dt, e))
        except Exception:
            continue
    result.sort(key=lambda x: x[0])
    return [e for _, e in result]

# ============================================================
# ⚖️ ЮРИДИЧЕСКИЕ ФАКТЫ
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
        txt = e["text"].lower()

        if e["amounts"]:
            legal["damage"].extend(e["amounts"])

        if e["action"]:
            legal["objective_side"].append(
                f"{e['action']} ({', '.join(e['amounts'])})".strip("() ")
            )

        if "с целью" in txt and not legal["motive"]:
            after = txt.split("с целью", 1)[1]
            legal["motive"] = after[:100]

        if "предварительн" in txt:
            legal["intent"] = "Прямой умысел"

    return legal

# ============================================================
# 🔥 ОПРЕДЕЛЕНИЕ ТИПА ПРЕСТУПЛЕНИЯ
# ============================================================

def detect_crime_type(events: list[dict]) -> str:
    blob = " ".join(e["text"].lower() for e in events)

    if any(w in blob for w in CRIMINAL_ACTIONS["fraud_signals"]):
        return "мошенничество"

    if any(w in blob for w in CRIMINAL_ACTIONS["pyramid_activity"]):
        return "финансовая схема"

    if any(w in blob for w in CRIMINAL_ACTIONS["investment"]):
        return "незаконное привлечение средств"

    if any(w in blob for w in CRIMINAL_ACTIONS["withdrawal"]):
        return "препятствие выводу средств"

    return "неопределено"

# ============================================================
# 🧩 ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def extract_all(facts: list[dict], persons: list[str], dates: list[str], amounts: list[str]) -> dict:
    sentences = [f.get("text", "") for f in facts if f.get("text")]
    filtered_sentences = [s for s in sentences if is_meaningful(s)]

    if not filtered_sentences:
        filtered_sentences = sentences

    roles = extract_roles(facts, persons)

    all_sentences_raw = sentences
    suspect = detect_suspect(all_sentences_raw)

    events = extract_events(filtered_sentences)
    timeline = build_timeline(events)
    crime_flow = build_crime_flow(events)
    legal_facts = extract_legal_facts(events, roles)
    crime_type = detect_crime_type(events)

    return {
        "roles": roles,
        "events": events,
        "timeline": timeline,
        "legal_facts": legal_facts,
        "suspect": suspect,
        "crime_flow": crime_flow,
        "crime_type": crime_type,
    }
