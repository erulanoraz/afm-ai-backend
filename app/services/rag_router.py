from __future__ import annotations
from typing import List, Optional
from app.services.facts.fact_models import LegalFact


class RAGRouter:
    """
    RAG Router 6.3 — Anti-Garbage Investigator Edition
    Пропускает ТОЛЬКО реальные криминальные факты.
    """

    MAX_PRIMARY = 60
    MAX_SECONDARY = 40
    MAX_RESERVE = 10
    MAX_TOTAL = 110

    CONF_STRONG = 0.35
    CONF_WEAK = 0.15

    # ---------------------------------------------------------------
    # РАЗРЕШЁННЫЕ РОЛИ (только реальные криминальные)
    # ---------------------------------------------------------------
    ALLOWED_ROLES = {
        "fraud_action",
        "fraud_event",
        "suspect_action",
        "victim_loss",
        "money_transfer",
        "investment_event",
        "investment_context",
        "crypto_operation",
        "economic_action",
        "digital_transfer",
        "admin_action",
        "scheme_marker",
        "deception_context",
    }

    # запрещаем generic/role_statement
    BLOCKED_ROLES = {"generic_fact", "role_statement", "amount_related"}

    NORMALIZE = {
        "fraud_flag": "fraud_event",
        "invest_flag": "investment_event",
        "role_suspect": "suspect_action",
        "role_organizer": "suspect_action",
        "role_victim": "victim_loss",
        "role_witness": "role_statement",
        "action": "economic_action",
        "channel": "digital_transfer",
        "account": "digital_transfer",
        "crypto": "crypto_operation",
        "scheme": "scheme_marker",
    }

    ROLE_PRIORITY = {
        "fraud_action": 1,
        "fraud_event": 2,
        "suspect_action": 3,
        "scheme_marker": 3,
        "investment_event": 4,
        "investment_context": 4,
        "admin_action": 4,
        "crypto_operation": 5,
        "money_transfer": 5,
        "economic_action": 6,
        "digital_transfer": 6,
        "victim_loss": 7,
    }

    # ---------------------------------------------------------------
    # Главный метод
    # ---------------------------------------------------------------
    def route_for_qualifier(
        self, facts: List[LegalFact], target_article: Optional[str] = None
    ) -> List[LegalFact]:

        if not facts:
            return []

        # нормализация ролей
        for f in facts:
            if f.role in self.NORMALIZE:
                f.role = self.NORMALIZE[f.role]

        # убираем generic_fact, role_statement, amount_related
        filtered = [
            f for f in facts
            if (f.role in self.ALLOWED_ROLES)
        ]

        if not filtered:
            return []

        # strong/weak
        strong = []
        weak = []

        for f in filtered:
            if f.confidence >= self.CONF_STRONG:
                strong.append(f)
            elif f.confidence >= self.CONF_WEAK:
                # слабые но криминальные
                if f.role in ("fraud_event", "scheme_marker", "investment_event"):
                    strong.append(f)
                else:
                    weak.append(f)

        # baseline: только криминальные
        primary = sorted(
            strong,
            key=lambda f: self.ROLE_PRIORITY.get(f.role, 99)
        )[:self.MAX_PRIMARY]

        secondary = weak[:self.MAX_SECONDARY]
        reserve = weak[self.MAX_SECONDARY:self.MAX_SECONDARY + self.MAX_RESERVE]

        result = primary + secondary + reserve
        return result[:self.MAX_TOTAL]

