# app/services/validation/verifier.py
from typing import Dict, List, Any
from app.utils.config import (
    settings,
)
settings.CONF_THRESH_CRITICAL
settings.REQUIRE_TWO_SOURCES

from app.services.validation.rules_engine import check_text_consistency, require_inline_citations


def verify_facts_provenance(facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    1) каждый факт должен иметь хотя бы 1 источник (file_id/page/offset/optional chunk_id)
    2) для критичных фактов confidence >= порога
    3) по возможности — минимум 2 независимых источника
    """
    violations = []
    for f in facts:
        srcs = f.get("sources") or []
        conf = float(f.get("confidence", 0.0))
        is_critical = f.get("critical", False)

        if not srcs:
            violations.append({"fact_id": f.get("fact_id"), "issue": "no_sources"})
            continue

        # независимость = различие file_id или page
        uniq = {(s.get("file_id"), s.get("page")) for s in srcs if s.get("file_id") is not None}
        if settings.REQUIRE_TWO_SOURCES and len(uniq) < 2:
            violations.append({"fact_id": f.get("fact_id"), "issue": "only_one_source"})

        if is_critical and conf < settings.CONF_THRESH_CRITICAL:
            violations.append({
                "fact_id": f.get("fact_id"),
                "issue": "low_confidence_critical",
                "value": conf,
                "threshold": settings.CONF_THRESH_CRITICAL
            })

        if not is_critical and conf < settings.CONF_THRESH_DEFAULT:
            violations.append({
                "fact_id": f.get("fact_id"),
                "issue": "low_confidence_default",
                "value": conf,
                "threshold": settings.CONF_THRESH_DEFAULT
            })

        # проверим обязательные поля источника
        for s in srcs:
            if not s.get("file_id"):
                violations.append({"fact_id": f.get("fact_id"), "issue": "source_missing_file_id"})
            if s.get("page") is None:
                violations.append({"fact_id": f.get("fact_id"), "issue": "source_missing_page"})

    return {
        "ok": not violations,
        "violations": violations
    }


def verify_output_texts(ustanovil_text: str, postanov_text: str) -> Dict[str, Any]:
    """
    1) требуем инлайн-цитирование
    2) базовая согласованность
    """
    checks = {
        "ustanovil": check_text_consistency(ustanovil_text),
        "post": check_text_consistency(postanov_text)
    }
    warnings = []
    citations_ok = True

    if settings.ENFORCE_CITATIONS:
        ok_u, warn_u = require_inline_citations(ustanovil_text)
        ok_p, warn_p = require_inline_citations(postanov_text)
        citations_ok = (ok_u and ok_p)
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
        "warnings": warnings
    }


def run_full_verification(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload: результат qualify_documents()
    """
    facts = payload.get("facts", [])
    ustanovil_text = payload.get("established_text", "") or ""
    post_text = payload.get("final_postanovlenie", "") or ""

    facts_res = verify_facts_provenance(facts)
    text_res = verify_output_texts(ustanovil_text, post_text)

    overall_ok = facts_res["ok"] and text_res["verdict"] == "OK"

    return {
        "overall_ok": overall_ok,
        "facts": facts_res,
        "texts": text_res
    }
