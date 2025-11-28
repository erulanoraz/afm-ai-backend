# app/services/validation/verifier.py
from typing import Dict, List, Any

from app.utils.config import settings
from app.services.validation.rules_engine import (
    check_text_consistency,
    require_inline_citations,
)


# -------------------------------------------------------------------
# 1. Проверка источников фактов
# -------------------------------------------------------------------

def verify_facts_provenance(facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    violations: List[Dict[str, Any]] = []

    for f in facts:
        srcs = f.get("sources") or []
        conf = float(f.get("confidence", 0.0))
        is_critical = f.get("critical", False)

        if not srcs:
            violations.append({"fact_id": f.get("fact_id"), "issue": "no_sources"})
            continue

        uniq = {
            (s.get("file_id"), s.get("page"))
            for s in srcs
            if s.get("file_id") is not None
        }

        if settings.REQUIRE_TWO_SOURCES and len(uniq) < 2:
            violations.append(
                {"fact_id": f.get("fact_id"), "issue": "only_one_source"}
            )

        if is_critical and conf < settings.CONF_THRESH_CRITICAL:
            violations.append(
                {
                    "fact_id": f.get("fact_id"),
                    "issue": "low_confidence_critical",
                    "value": conf,
                    "threshold": settings.CONF_THRESH_CRITICAL,
                }
            )

        if not is_critical and conf < settings.CONF_THRESH_DEFAULT:
            violations.append(
                {
                    "fact_id": f.get("fact_id"),
                    "issue": "low_confidence_default",
                    "value": conf,
                    "threshold": settings.CONF_THRESH_DEFAULT,
                }
            )

        for s in srcs:
            if not s.get("file_id"):
                violations.append(
                    {"fact_id": f.get("fact_id"), "issue": "source_missing_file_id"}
                )
            if s.get("page") is None:
                violations.append(
                    {"fact_id": f.get("fact_id"), "issue": "source_missing_page"}
                )

    return {
        "ok": not violations,
        "violations": violations,
    }


# -------------------------------------------------------------------
# 2. Проверка текстов
# -------------------------------------------------------------------

def verify_output_texts(ustanovil_text: str, postanov_text: str) -> Dict[str, Any]:
    checks = {
        "ustanovil": check_text_consistency(ustanovil_text),
        "post": check_text_consistency(postanov_text),
    }
    warnings: List[Dict[str, Any]] = []
    citations_ok = True

    if settings.ENFORCE_CITATIONS:
        ok_u, warn_u = require_inline_citations(ustanovil_text)
        ok_p, warn_p = require_inline_citations(postanov_text)
        citations_ok = ok_u and ok_p
        warnings.extend(warn_u + warn_p)

    verdict = "OK"
    if not citations_ok or not checks["ustanovil"]["has_dates"]:
        verdict = (
            "INSUFFICIENT_EVIDENCE"
            if settings.RETURN_INSUFFICIENT_ON_FAIL
            else "REVIEW_REQUIRED"
        )

    return {
        "verdict": verdict,
        "citations_ok": citations_ok,
        "checks": checks,
        "warnings": warnings,
    }


# -------------------------------------------------------------------
# 3. Token-level проверка: used_tokens vs все токены из facts
# -------------------------------------------------------------------

def verify_token_alignment(
    used_tokens: List[str],
    all_token_ids: List[str],
) -> Dict[str, Any]:
    used = set(used_tokens or [])
    allowed = set(all_token_ids or [])

    unknown = sorted(used - allowed)
    unused = sorted(allowed - used)

    ok = len(unknown) == 0

    return {
        "ok": ok,
        "total_used": len(used),
        "total_available": len(allowed),
        "unknown_tokens": unknown,
        "unused_tokens": unused,
    }


# -------------------------------------------------------------------
# 4. Sentence ↔ Token alignment
# -------------------------------------------------------------------

def verify_sentence_token_alignment(
    sentence_map: List[Dict[str, Any]],
    used_tokens: List[str],
    all_token_ids: List[str],
) -> Dict[str, Any]:
    if not sentence_map:
        return {
            "ok": False,
            "sentence_count": 0,
            "violations": [
                {"issue": "no_sentence_map", "msg": "LLM не вернул поле 'sentences'"}
            ],
        }

    violations: List[Dict[str, Any]] = []
    all_ids_set = set(all_token_ids or [])
    from_sentences: set[str] = set()

    for idx, s in enumerate(sentence_map, start=1):
        text = (s.get("text") or "").strip()
        tokens = s.get("tokens") or []

        if not text:
            violations.append({"index": idx, "issue": "empty_sentence_text"})
            continue

        if not tokens:
            violations.append(
                {"index": idx, "issue": "sentence_without_tokens", "text": text}
            )
            continue

        for t in tokens:
            from_sentences.add(t)
            if t not in all_ids_set:
                violations.append(
                    {
                        "index": idx,
                        "issue": "unknown_token_in_sentence",
                        "token": t,
                        "text": text,
                    }
                )

    used_set = set(used_tokens or [])

    used_not_in_sentences = sorted(used_set - from_sentences)
    in_sentences_not_in_used = sorted(from_sentences - used_set)

    if used_not_in_sentences:
        violations.append(
            {
                "issue": "used_tokens_without_sentence",
                "tokens": used_not_in_sentences,
            }
        )

    if in_sentences_not_in_used:
        violations.append(
            {
                "issue": "sentence_tokens_not_marked_used",
                "tokens": in_sentences_not_in_used,
            }
        )

    ok = len(violations) == 0

    return {
        "ok": ok,
        "sentence_count": len(sentence_map),
        "mapped_tokens": sorted(from_sentences),
        "violations": violations,
    }


# -------------------------------------------------------------------
# 5. Главная точка входа
# -------------------------------------------------------------------

def run_full_verification(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    ожидаемые поля:
    - facts: List[dict]           (LegalFact как dict)
    - ustanovil / established_text: str
    - final_postanovlenie: str
    - used_tokens: List[str]
    - sentences: List[dict]
    """
    facts = payload.get("facts", []) or []

    # поддерживаем оба варианта ключей: 'ustanovil' и 'established_text'
    ustanovil_text = (
        payload.get("established_text")
        or payload.get("ustanovil")
        or ""
    )
    post_text = payload.get("final_postanovlenie", "") or ""
    used_tokens = payload.get("used_tokens", []) or []
    sentence_map = payload.get("sentences", []) or []

    # 1) источники фактов
    facts_res = verify_facts_provenance(facts)

    # 2) текстовые проверки
    text_res = verify_output_texts(ustanovil_text, post_text)

    # 3) собираем token_id из facts
    all_token_ids = [
        f.get("token_id")
        for f in facts
        if isinstance(f, dict) and f.get("token_id")
    ]

    # 4) token-level проверка
    token_res = verify_token_alignment(used_tokens, all_token_ids)

    # 5) sentence ↔ token
    sentence_res = verify_sentence_token_alignment(
        sentence_map,
        used_tokens,
        all_token_ids,
    )

    overall_ok = (
        facts_res["ok"]
        and text_res["verdict"] == "OK"
        and token_res["ok"]
        and sentence_res["ok"]
    )

    return {
        "overall_ok": overall_ok,
        "facts": facts_res,
        "texts": text_res,
        "tokens": token_res,
        "sentences": sentence_res,
    }
