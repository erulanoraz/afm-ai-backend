# app/services/facts/fact_filter.py
import logging
import re
from typing import List

from app.services.facts.fact_models import LegalFact

logger = logging.getLogger(__name__)


class FactFilter:
    """
    FactFilter 6.0 — ИСПРАВЛЕННАЯ
    
    ГЛАВНАЯ ОШИБКА БЫЛА: убивали ВСЕ факты!
    ТЕПЕРЬ: оставляем факты которые имеют значение.
    """

    # ============================================================
    # Процессуальные ПАТТЕРНЫ (которые реально убираем)
    # ============================================================
    PROCESSUAL_KEYWORDS = [
        "разъяснены права",
        "ему разъяснены права",
        "ей разъяснены права",
        "данное постановление может быть обжаловано",
        "ознакомлен под роспись",
        "ознакомлена под роспись",
        "предупрежден об ответственности",
        "предупреждена об ответственности",
        "уведомлен об уголовной ответственности",
        "уведомлена об уголовной ответственности",
        "язык судопроизводства",
    ]

    # ============================================================
    # КРИМИНАЛЬНЫЕ ТОКЕНЫ (ОСТАВЛЯЕМ!)
    # ============================================================
    CRIME_TOKEN_TYPES = {
        "amount", "fraud_flag", "invest_flag", "scheme_flag",
        "economic_flag", "admin_flag", "crypto_flag", "crypto",
        "channel", "account", "person", "date", "action",
        "role_label", "article_ref"
    }

    # ============================================================
    # Главные РОЛИ (оставляем эти)
    # ============================================================
    IMPORTANT_ROLES = {
        "fraud_action", "fraud_event",
        "suspect_action", "money_transfer",
        "victim_loss", "investment_event",
        "investment_context", "crypto_operation",
        "economic_action", "admin_action",
        "scheme_marker", "digital_transfer",
    }

    MAX_FACTS = 100  # Больше оставляем (было 80)

    # ============================================================
    # ГЛАВНЫЙ МЕТОД
    # ============================================================
    def filter_for_qualifier(self, facts: List[LegalFact]) -> List[LegalFact]:
        """
        1) Удаляем ТОЛЬКО процессуальку
        2) Оставляем ВСЕ криминальные факты
        3) Сортируем по важности
        """
        if not facts:
            return []

        before_total = len(facts)

        # Шаг 1: Удаляем ТОЛЬКО явно процессуальные
        non_proc = [f for f in facts if not self._is_pure_processual(f)]

        logger.info(f"🧹 FactFilter: было={before_total}, после удаления процессуалки={len(non_proc)}")

        if not non_proc:
            logger.warning("⚠️ FactFilter: все факты оказались процессуальными!")
            return facts[:self.MAX_FACTS]

        # Шаг 2: Сортируем по приоритету
        sorted_facts = sorted(non_proc, key=self._score_fact, reverse=True)

        # Шаг 3: Берём топ
        result = sorted_facts[:self.MAX_FACTS]

        logger.info(f"✅ FactFilter: итоговое количество = {len(result)}")
        return result

    # ============================================================
    # ПРОВЕРКА: это ЧИСТО процессуальный факт?
    # ============================================================
    def _is_pure_processual(self, fact: LegalFact) -> bool:
        """
        СТРОГО: только если это 100% процессуальное действие
        без фактических данных.
        """
        text = (fact.text or fact.span_text or "").lower().strip()
        tokens = fact.tokens or []

        # 🔴 Чистая процессуалка
        for kw in self.PROCESSUAL_KEYWORDS:
            if kw in text:
                # НО: если есть криминальные токены — оставляем!
                if any(t.type in self.CRIME_TOKEN_TYPES for t in tokens):
                    return False
                return True

        return False

    # ============================================================
    # ОЦЕНКА ФАКТА
    # ============================================================
    def _score_fact(self, fact: LegalFact) -> int:
        """Чем выше оценка, тем важнее факт."""
        score = 0
        
        role = (fact.role or "").lower()
        tokens = fact.tokens or []
        token_types = {t.type.lower() for t in tokens}

        # 1) Роль
        role_scores = {
            "fraud_action": 100,
            "fraud_event": 95,
            "suspect_action": 90,
            "money_transfer": 85,
            "victim_loss": 80,
            "investment_event": 75,
            "crypto_operation": 75,
            "scheme_marker": 80,
            "economic_action": 70,
            "digital_transfer": 70,
            "admin_action": 60,
            "investment_context": 65,
        }
        score += role_scores.get(role, 10)

        # 2) Токены (более важные = больше очков)
        token_scores = {
            "amount": 15,
            "fraud_flag": 20,
            "invest_flag": 15,
            "crypto": 18,
            "crypto_flag": 16,
            "scheme_flag": 15,
            "economic_flag": 12,
            "channel": 10,
            "account": 10,
            "admin_flag": 8,
            "date": 5,
            "person": 3,
            "action": 8,
            "role_label": 5,
        }

        for t_type in token_types:
            score += token_scores.get(t_type, 1)

        # 3) Уверенность
        conf = fact.confidence or 0.0
        if conf > 0.5:
            score += 10

        return score