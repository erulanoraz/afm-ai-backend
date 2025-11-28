from __future__ import annotations
from typing import List, Tuple, Dict, Set
from app.services.facts.fact_models import LegalFact, FactToken, SourceRef


class FactGraph:

    # =====================================================================
    # 📘 Основная точка входа
    # =====================================================================
    def build(self, facts: List[LegalFact]) -> List[LegalFact]:
        """
        Группировка по роли, затем аккуратное семантическое объединение.
        """
        if not facts:
            return []

        merged: List[LegalFact] = []
        bucket: Dict[str, List[LegalFact]] = {}

        # 1. группируем по роли
        for f in facts:
            bucket.setdefault(f.role, []).append(f)

        # 2. объединяем в пределах роли
        for role, items in bucket.items():
            merged.extend(self._merge_role_facts(items))

        return merged

    # =====================================================================
    # 📘 Семантическое объединение фактов одинаковой роли
    # =====================================================================
    def _merge_role_facts(self, facts: List[LegalFact]) -> List[LegalFact]:
        """
        Новый ключ объединения:
            • набор токенов (type, value)
            • нормализованный span_text (без пробелов)
            • sentence_index (для защиты от ложного merge)
        """
        if not facts:
            return []

        unique_map: Dict[Tuple, LegalFact] = {}

        for f in facts:
            tokens_key = tuple(sorted((t.type, t.value) for t in f.tokens))
            span_key = self._normalize_span(f.span_text)
            sent_key = f.sentence_index

            merge_key = (tokens_key, span_key, sent_key)

            if merge_key not in unique_map:
                unique_map[merge_key] = f
                continue

            existing = unique_map[merge_key]

            # ------------------------------------------------------
            # 1) объединяем source_refs
            # ------------------------------------------------------
            old_src = {(s.file_id, s.page) for s in existing.source_refs}
            new_src = {(s.file_id, s.page) for s in f.source_refs}
            combined = old_src | new_src

            existing.source_refs = [
                SourceRef(file_id=fid, page=pg) for fid, pg in combined
            ]

            # ------------------------------------------------------
            # 2) объединяем токены (не допускаем дубликатов)
            # ------------------------------------------------------
            seen = {(t.type, t.value) for t in existing.tokens}
            for t in f.tokens:
                if (t.type, t.value) not in seen:
                    existing.tokens.append(t)
                    seen.add((t.type, t.value))

            # ------------------------------------------------------
            # 3) оставляем span_text как у existing (главного)
            #    context_before / after тоже оставляем
            # ------------------------------------------------------

            # ------------------------------------------------------
            # 4) объединяем article_hints
            # ------------------------------------------------------
            hints = set(existing.article_hints or []) | set(f.article_hints or [])
            existing.article_hints = list(sorted(hints))

        return list(unique_map.values())

    # =====================================================================
    # 📘 Нормализация span_text — чтобы merge был точным
    # =====================================================================
    def _normalize_span(self, span: str) -> str:
        if not span:
            return ""
        s = span.lower().strip()
        s = " ".join(s.split())
        return s
