# app/services/rag_router.py
from __future__ import annotations
from typing import List, Optional
from app.services.facts.fact_models import LegalFact


class RAGRouter:
    """
    RAGRouter v9.0 — Full Criminal Context Routing Engine

    Главные улучшения:
    • удерживает максимум криминального материала (деньги, переводы, крипта, проект, схема)
    • роль + токены + confidence → сильная приоритезация
    • мягкое исключение мусора, НО не режет полезные факты
    • специальный режим для статей 190/217 (мошенничество / финансовая пирамида)
    """

    # Ограничения итогового объёма для LLM
    MAX_TOTAL = 240
    MAX_PRIMARY = 130
    MAX_SECONDARY = 80
    MAX_RESERVE = 30

    # Пороги уверенности
    CONF_STRONG = 0.30
    CONF_WEAK = 0.12

    # Разрешённые (ядерные) роли
    ALLOWED = {
        "suspect_action",
        "fraud_action",
        "fraud_event",

        "investment_event",
        "investment_context",

        "scheme_marker",
        "entity_reference",

        "crypto_operation",
        "digital_transfer",
        "economic_action",

        "victim_loss",
        "money_transfer",

        "admin_action",
    }

    # Роли, почти всегда бесполезные
    BLOCKED = {
        "generic_fact",
        "role_statement",
        "victim_statement",
        "address_only",
    }

    # Нормализация входящих ролей
    NORMALIZE = {
        "fraud_flag": "fraud_event",
        "invest_flag": "investment_event",
        "role_suspect": "suspect_action",
        "role_organizer": "suspect_action",
        "role_victim": "victim_loss",
        "role_witness": "role_statement",

        "entity": "entity_reference",
        "project": "entity_reference",
        "platform": "entity_reference",
        "organization": "entity_reference",

        "crypto": "crypto_operation",
        "crypto_flag": "crypto_operation",

        "account": "digital_transfer",
        "channel": "digital_transfer",

        "scheme": "scheme_marker",
        "scheme_flag": "scheme_marker",

        "economic_flag": "economic_action",
        "money": "money_transfer",
    }

    # Приоритет ролей (меньше = важнее)
    ROLE_PRIORITY = {
        "suspect_action": 1,
        "fraud_action": 2,
        "fraud_event": 3,

        "scheme_marker": 4,
        "investment_event": 5,
        "investment_context": 6,

        "money_transfer": 7,
        "crypto_operation": 7,
        "economic_action": 8,
        "digital_transfer": 8,

        "victim_loss": 9,
        "admin_action": 10,

        "entity_reference": 11,
    }

    # Вес токенов для усиления приоритета
    BOOST_TOKENS = {
        "amount": 18,
        "fraud_flag": 14,
        "invest_flag": 12,
        "scheme_flag": 12,
        "crypto_flag": 12,
        "crypto": 12,
        "channel": 8,
        "account": 8,
        "economic_flag": 8,
        "admin_flag": 6,
        "entity": 10,
        "project": 10,
        "platform": 10,
        "organization": 8,
        "article_ref": 5,
        "date": 3,
    }

    def _token_boost(self, f: LegalFact) -> int:
        score = 0
        for t in f.tokens or []:
            score += self.BOOST_TOKENS.get(t.type, 0)
        return score

    # ======================================================================
    # ГЛАВНАЯ ФУНКЦИЯ
    # ======================================================================
    def route_for_qualifier(
        self,
        facts: List[LegalFact],
        target_article: Optional[str] = None,
    ) -> List[LegalFact]:

        if not facts:
            return []

        # ------------------------------------------------------------
        # 1. Нормализация ролей
        # ------------------------------------------------------------
        for f in facts:
            if f.role in self.NORMALIZE:
                f.role = self.NORMALIZE[f.role]

        # ------------------------------------------------------------
        # 2. Drop BLOCKED роли (но мягко)
        # ------------------------------------------------------------
        filtered: List[LegalFact] = []

        for f in facts:
            role = f.role or ""

            # мусор
            if role in self.BLOCKED:
                continue

            # Если роль неизвестна, но есть токены денег / проекта / платформы / крипты
            # → оставляем как полезный контекст (особенно под 190 / 217)
            if role not in self.ALLOWED:
                token_types = {t.type for t in f.tokens or []}
                if token_types.intersection({"amount", "project", "platform", "crypto", "crypto_flag",
                                             "channel", "account", "scheme_flag", "fraud_flag", "invest_flag"}):
                    filtered.append(f)
                    continue
                if target_article in ("190", "217"):
                    filtered.append(f)
                continue

            filtered.append(f)

        if not filtered:
            return []

        # ------------------------------------------------------------
        # 3. Strong / Weak
        # ------------------------------------------------------------
        strong, weak = [], []

        for f in filtered:
            conf = getattr(f, "confidence", 0.0) or 0.0
            if conf >= self.CONF_STRONG:
                strong.append(f)
            else:
                weak.append(f)

        # ------------------------------------------------------------
        # 4. PRIMARY — роль + токены + confidence
        # ------------------------------------------------------------
        def primary_score(f: LegalFact):
            rp = self.ROLE_PRIORITY.get(f.role, 99)
            tb = -self._token_boost(f)
            cp = -(getattr(f, "confidence", 0.0) or 0.0) * 10
            return (rp, tb, cp)

        primary = sorted(strong, key=primary_score)[: self.MAX_PRIMARY]

        # ------------------------------------------------------------
        # 5. SECONDARY — полезный контекст
        # ------------------------------------------------------------
        def secondary_score(f: LegalFact):
            return -self._token_boost(f)

        secondary = sorted(weak, key=secondary_score)[: self.MAX_SECONDARY]

        # ------------------------------------------------------------
        # 6. RESERVE — запас ширины фабулы
        # ------------------------------------------------------------
        reserve = []
        for f in weak[len(secondary):]:
            if len(reserve) < self.MAX_RESERVE:
                reserve.append(f)

        # ------------------------------------------------------------
        # 7. MERGE
        # ------------------------------------------------------------
        result = primary + secondary + reserve

        return result[: self.MAX_TOTAL]
