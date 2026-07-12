from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any

from .csl_task_adapter import (
    ActionDirective,
    ConstraintAtom,
    DecisionReceipt,
    DependencySpec,
    EvidenceRequirement,
    GenericTaskAdapter,
    ObjectiveSpec,
    ProgressContext,
    RecoveryAction,
    TaskIR,
    canonical_action,
    semantic_fingerprint,
)


ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")
CLICK_ASIN_RE = re.compile(r"^\s*click\[\s*([A-Z0-9]{10})\s*\]\s*$", re.I)
PRICE_RE = re.compile(r"(?:Price:\s*)?\$\s*(-?\d+(?:\.\d+)?)", re.I)
RATING_RE = re.compile(r"Rating:\s*(\d+(?:\.\d+)?)", re.I)
RATING_LABEL_RE = re.compile(r"^Rating:\s*", re.I)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "one",
    "the",
    "to",
    "with",
}

_DIMENSION_UNITS: dict[str, tuple[str, float]] = {
    "ah": ("capacity_mah", 1000.0),
    "cm": ("length_mm", 10.0),
    "count": ("count", 1.0),
    "counts": ("count", 1.0),
    "ct": ("count", 1.0),
    "ft": ("length_mm", 304.8),
    "gb": ("storage_gb", 1.0),
    "g": ("mass_g", 1.0),
    "gram": ("mass_g", 1.0),
    "grams": ("mass_g", 1.0),
    "in": ("length_mm", 25.4),
    "inch": ("length_mm", 25.4),
    "inches": ("length_mm", 25.4),
    "kg": ("mass_g", 1000.0),
    "l": ("volume_ml", 1000.0),
    "liter": ("volume_ml", 1000.0),
    "liters": ("volume_ml", 1000.0),
    "litre": ("volume_ml", 1000.0),
    "litres": ("volume_ml", 1000.0),
    "m": ("length_mm", 1000.0),
    "mah": ("capacity_mah", 1.0),
    "mb": ("storage_gb", 0.001),
    "mg": ("mass_g", 0.001),
    "ml": ("volume_ml", 1.0),
    "mm": ("length_mm", 1.0),
    "oz": ("mass_oz", 1.0),
    "ounce": ("mass_oz", 1.0),
    "ounces": ("mass_oz", 1.0),
    "pc": ("count", 1.0),
    "pcs": ("count", 1.0),
    "piece": ("count", 1.0),
    "pieces": ("count", 1.0),
    "tb": ("storage_gb", 1000.0),
}


def parse_visible_catalog_entities(action: str | None, observation: str | None) -> list[dict[str, Any]]:
    """Turn a visible separator-delimited catalog page into entity-bound facts."""
    if not observation or "[SEP]" not in observation:
        return []

    parts = [part.strip() for part in observation.split("[SEP]") if part.strip()]
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()

    for index, part in enumerate(parts):
        asin = part.upper()
        if not ASIN_RE.fullmatch(asin) or index + 2 >= len(parts):
            continue
        title = parts[index + 1]
        price = _parse_price(parts[index + 2])
        if price is None:
            continue
        entities.append(
            _product_entity(
                asin=asin,
                title=title,
                price=price,
                rating=None,
                rating_observed=False,
                detail_observed=False,
            )
        )
        seen.add(asin)

    clicked_asin = selected_asin_from_action(action)
    detail_price_index = next(
        (index for index, part in enumerate(parts) if part.lower().startswith("price:")),
        None,
    )
    if clicked_asin and detail_price_index is not None:
        price = _parse_price(parts[detail_price_index])
        rating_text = next((part for part in parts if RATING_LABEL_RE.match(part)), None)
        rating = _parse_rating(rating_text) if rating_text is not None else None
        rating_observed = rating_text is not None
        title = _detail_title(parts, detail_price_index)
        if price is not None and title:
            detail = _product_entity(
                asin=clicked_asin,
                title=title,
                price=price,
                rating=rating,
                rating_observed=rating_observed,
                detail_observed=True,
            )
            entities = [item for item in entities if item["subject"] != detail["subject"]]
            entities.append(detail)
            entities.append(
                {
                    "subject": "current_selection",
                    "status": "candidate",
                    "confidence": 0.99,
                    "properties": {
                        "selected_option_id": clicked_asin,
                        "title": title,
                        "price": price,
                        "rating": rating,
                        "rating_observed": rating_observed,
                        "detail_observed": True,
                    },
                }
            )
            seen.add(clicked_asin)

    return entities


def selected_asin_from_action(action: str | None) -> str | None:
    match = CLICK_ASIN_RE.match(action or "")
    return match.group(1).upper() if match else None


def build_candidate_ledger(
    *,
    question: str,
    materialized_state: list[Any],
    max_options: int = 8,
    progress: ProgressContext | None = None,
) -> list[str]:
    """Summarize the best visible match for each task-listed shopping option."""
    descriptions = _available_option_descriptions(question)
    products, selected = _materialized_products(materialized_state)
    if not descriptions or not products:
        return []
    visible_asins = {
        part.strip().upper()
        for part in question.split("[SEP]")
        if ASIN_RE.fullmatch(part.strip().upper())
    }
    detail_page = "[SEP] Buy Now" in question

    lines: list[str] = []
    for index, description in enumerate(descriptions[:max_options], start=1):
        ranked = _rank_product_matches(
            description,
            products,
            descriptions=descriptions,
            preferred_asins=visible_asins,
        )
        ranked = [item for item in ranked if item[1] >= 0.30 and item[0] is True]
        if not ranked:
            lines.append(f"{index}. {_clip(description, 90)} => unresolved")
            continue

        _valid, _score, _severity, _visible, detail_observed, asin, data = ranked[0]
        rating = data.get("rating")
        if isinstance(rating, (int, float)):
            rating_text = f"{rating:g}"
        elif data.get("rating_observed") is True:
            rating_text = "N.A."
        else:
            rating_text = "unobserved"
        price = data.get("price")
        price_text = f"${price:.2f}" if isinstance(price, (int, float)) else "unobserved"
        selected_text = ", selected" if asin == selected else ""
        approximate_text = ", approximate" if _unrequested_variant(description, str(data.get("title") or "")) else ""
        actionable = asin in visible_asins or (detail_page and asin == selected)
        candidate_ref = asin if actionable else "historical evidence (not currently clickable)"
        lines.append(
            f"{index}. {_clip(description, 74)} => {candidate_ref} | {price_text} | "
            f"rating={rating_text} | detail={'yes' if detail_observed else 'no'}"
            f"{selected_text}{approximate_text} | {_clip(str(data.get('title') or ''), 105)}"
        )
        alternatives = []
        for _valid, _score, severity, _visible, _detail, other_asin, other in ranked[1:3]:
            other_ref = (
                other_asin
                if other_asin in visible_asins
                else "historical evidence"
            )
            identity = "exact" if severity == 0 else "variant" if severity == 1 else "bundle-conflict"
            alternatives.append(
                f"{other_ref} ({identity}: {_clip(str(other.get('title') or ''), 70)})"
            )
        if alternatives:
            lines.append(f"{index}. alternative identity hypotheses: {'; '.join(alternatives)}")
    if progress is not None and progress.latest_stale_signal is not None:
        signal = progress.latest_stale_signal
        lines.append(
            f"progress: stale={signal.action} repeated={signal.repeat_count}; "
            "do not reuse it without a different evidence path"
        )
    return lines


def build_candidate_evidence_guidance(
    *,
    question: str,
    materialized_state: list[Any],
    progress: ProgressContext | None = None,
    task_ir: TaskIR | None = None,
) -> str | None:
    directive = build_candidate_action_directive(
        question=question,
        materialized_state=materialized_state,
        progress=progress,
        task_ir=task_ir,
    )
    return directive.display_text if directive is not None else None


def build_candidate_action_directive(
    *,
    question: str,
    materialized_state: list[Any],
    progress: ProgressContext | None = None,
    task_ir: TaskIR | None = None,
) -> ActionDirective | None:
    """Translate public WebShop state into the generic action-directive contract."""
    descriptions = _available_option_descriptions(question)
    products, selected = _materialized_products(materialized_state)
    if not descriptions:
        return None

    task = task_ir or _compile_webshop_task(question)
    progress = progress or ProgressContext(scope_id="global")

    visible_asins = {
        part.strip().upper()
        for part in question.split("[SEP]")
        if ASIN_RE.fullmatch(part.strip().upper())
    }
    detail_page = "[SEP] Buy Now" in question and "[SEP] Rating:" in question
    resolved: list[tuple[str, str, dict[str, Any], str, str | None]] = []
    metric = task.objective.metric
    requires_detail = metric == "rating"

    for description in descriptions:
        description_status, description_reason = _compatibility_status(
            task=task,
            description=description,
            title="",
            materialized_state=materialized_state,
        )
        match = _best_product_match(
            description,
            products,
            descriptions=descriptions,
            preferred_asins=visible_asins,
        )
        if match is None:
            if description_status == "incompatible":
                resolved.append((description, "", {}, description_status, description_reason))
                continue
            if detail_page:
                return _required_webshop_directive(
                    action="click[Back to Search]",
                    reason="leave the inspected detail page to find the next unresolved option",
                    semantic_target=description,
                )
            proposed = f"search[{_evidence_search_query(description)}]"
            return _recover_stale_action(
                proposed=proposed,
                question=question,
                description=description,
                progress=progress,
                reason="find this unresolved listed option",
                extra_actions=_visible_identity_recovery_actions(
                    description,
                    products,
                    descriptions=descriptions,
                    visible_asins=visible_asins,
                ),
            )

        asin, data = match
        title = str(data.get("title") or "")
        if _identity_mismatch(description, title, descriptions):
            if detail_page:
                return _required_webshop_directive(
                    action="click[Back to Search]",
                    reason="the inspected package/variant does not exactly match the listed option",
                    semantic_target=description,
                )
            proposed = f"search[{_evidence_search_query(description, mismatched_title=title)}]"
            return _recover_stale_action(
                proposed=proposed,
                question=question,
                description=description,
                progress=progress,
                reason="replace the contradicted entity with an exact listing",
                extra_actions=_visible_identity_recovery_actions(
                    description,
                    products,
                    descriptions=descriptions,
                    visible_asins=visible_asins,
                ),
            )

        compatibility_status, compatibility_reason = _compatibility_status(
            task=task,
            description=description,
            title=title,
            materialized_state=materialized_state,
        )
        budget_limit = _visible_budget_limit(question)
        if (
            budget_limit is not None
            and isinstance(data.get("price"), (int, float))
            and float(data["price"]) > budget_limit
        ):
            compatibility_status = "incompatible"
            compatibility_reason = f"price ${float(data['price']):.2f} exceeds total budget ${budget_limit:.2f} by itself"
        if compatibility_status == "incompatible":
            resolved.append((description, asin, data, compatibility_status, compatibility_reason))
            continue

        approximate = _unrequested_variant(description, title)
        recovery_query = f"search[{_evidence_search_query(description, mismatched_title=title)}]"
        if approximate and not progress.is_stale(recovery_query):
            if detail_page:
                variant_action = _preferred_visible_variant_action(
                    question=question,
                    description=description,
                )
                if variant_action is not None and not progress.is_stale(variant_action):
                    return _required_webshop_directive(
                        action=variant_action,
                        reason="select the visible single/default variant before leaving the detail page",
                        semantic_target=description,
                    )
                return _required_webshop_directive(
                    action="click[Back to Search]",
                    reason="the inspected package/variant is unspecified; seek a closer identity match",
                    semantic_target=description,
                )
            return _recover_stale_action(
                proposed=recovery_query,
                question=question,
                description=description,
                progress=progress,
                reason="reduce identity uncertainty introduced by an unspecified variant",
                extra_actions=_visible_identity_recovery_actions(
                    description,
                    products,
                    descriptions=descriptions,
                    visible_asins=visible_asins,
                ),
            )

        if (
            requires_detail
            and compatibility_status != "incompatible"
            and data.get("detail_observed") is not True
        ):
            if asin in visible_asins:
                return _required_webshop_directive(
                    action=f"click[{asin}]",
                    reason="verify the exact listed candidate's detail-page price and rating",
                    semantic_target=description,
                )
            if detail_page:
                return _required_webshop_directive(
                    action="click[Back to Search]",
                    reason="leave the completed detail page before gathering the next candidate",
                    semantic_target=description,
                )
            proposed = f"search[{_evidence_search_query(description)}]"
            return _recover_stale_action(
                proposed=proposed,
                question=question,
                description=description,
                progress=progress,
                reason=f"retrieve the exact candidate for {metric} verification",
                extra_actions=_visible_identity_recovery_actions(
                    description,
                    products,
                    descriptions=descriptions,
                    visible_asins=visible_asins,
                ),
            )
        resolved.append((description, asin, data, compatibility_status, compatibility_reason))

    unknown_reasons = sorted(
        {
            str(reason)
            for _description, _asin, _data, status, reason in resolved
            if status == "unknown" and reason
        }
    )
    if unknown_reasons:
        return ActionDirective(
            action="",
            mode="advisory",
            reason=(
                "Decision status: compatibility is unresolved because the prior scope summary is missing "
                "or inconclusive; do not rank or commit current candidates yet."
            ),
        )

    ranked = [
        (float(data[metric]), asin, description)
        for description, asin, data, status, _reason in resolved
        if status == "compatible" and isinstance(data.get(metric), (int, float))
    ]
    eligible_count = sum(
        1 for _description, _asin, _data, status, _reason in resolved if status == "compatible"
    )
    if len(ranked) != eligible_count:
        return ActionDirective(
            action="",
            mode="advisory",
            reason=(
                f"Eligible candidates are identified, but at least one required {metric} remains "
                "unavailable; gather different evidence."
            ),
        )
    if not ranked:
        return ActionDirective(
            action="",
            mode="advisory",
            reason=(
                "No candidate is currently supported as compatible; recover the prior decision "
                "dependency or gather compatibility evidence."
            ),
        )
    reverse = task.objective.direction == "maximize"
    _value, winner, winner_description = sorted(ranked, reverse=reverse)[0]
    if detail_page and selected == winner:
        if progress.is_stale("click[Buy Now]"):
            return _required_webshop_directive(
                action="click[Back to Search]",
                reason=(
                    "the unchanged final transition was already blocked; leave it and gather "
                    "new identity or objective evidence"
                ),
                semantic_target=winner_description,
                forbidden_actions=("click[Buy Now]",),
            )
        return _advisory_webshop_directive(
            action="click[Buy Now]",
            reason="all listed candidates are verified and the current exact candidate wins the public objective",
        )
    if winner in visible_asins:
        return _advisory_webshop_directive(
            action=f"click[{winner}]",
            reason="open the verified objective winner before purchase",
        )
    if detail_page:
        return _advisory_webshop_directive(
            action="click[Back to Search]",
            reason="leave the non-winning detail page and return to the verified winner",
        )
    proposed = f"search[{_evidence_search_query(winner_description)}]"
    return _recover_stale_action(
        proposed=proposed,
        question=question,
        description=winner_description,
        progress=progress,
        reason="return to the verified objective winner",
        extra_actions=_visible_identity_recovery_actions(
            winner_description,
            products,
            descriptions=descriptions,
            visible_asins=visible_asins,
        ),
    )


def build_visible_decision_certificate(
    *,
    question: str,
    materialized_state: list[Any],
    selected_option_id: str | None = None,
    task_ir: TaskIR | None = None,
) -> dict[str, Any] | None:
    descriptions = _available_option_descriptions(question)
    products, current_selection = _materialized_products(materialized_state)
    selected = (selected_option_id or current_selection or "").upper()
    if not descriptions or not selected:
        return None

    task = task_ir or _compile_webshop_task(question)
    metric = task.objective.metric
    requires_detail = metric == "rating"

    evidenced = {
        asin: data
        for asin, data in products.items()
        if isinstance(data.get("price"), (int, float))
        and data.get("title")
        and (not requires_detail or data.get("detail_observed") is True)
    }
    if not evidenced:
        return None

    assignments: list[tuple[str, dict[str, Any], str]] = []
    used: set[str] = set()
    uncertainties: list[str] = []
    soft_uncertainties: list[str] = []
    for description in descriptions:
        candidates = []
        for asin, data in evidenced.items():
            if asin in used:
                continue
            title = str(data.get("title") or "")
            if _identity_mismatch(description, title, descriptions):
                continue
            score = _identity_score(description, title, descriptions)
            selected_bonus = 0.03 if asin == selected else 0.0
            severity = _variant_severity(description, title)
            candidates.append((-severity, score + selected_bonus, score, asin, data))
        if not candidates:
            uncertainties.append(f"No detailed product evidence matched option: {description}")
            continue
        _severity_rank, _rank, score, asin, data = max(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        if score < 0.34:
            uncertainties.append(f"Low-confidence product match for option: {description}")
            continue
        title = str(data.get("title") or "")
        if _identity_mismatch(description, title, descriptions):
            uncertainties.append(f"Contradicted identity attributes for option: {description}")
            continue
        used.add(asin)
        assignments.append((asin, data, description))

    expected_ids = [asin for asin, _data, _description in assignments]
    if len(assignments) != len(descriptions):
        return None
    if selected not in expected_ids:
        return None

    objective = {"metric": task.objective.metric, "direction": task.objective.direction}
    candidates = []
    observed_facts = []
    for asin, data, description in assignments:
        objective_value = data.get(metric)
        compatibility_status, compatibility_reason = _compatibility_status(
            task=task,
            description=description,
            title=str(data.get("title") or ""),
            materialized_state=materialized_state,
        )
        budget_limit = _visible_budget_limit(question)
        if (
            budget_limit is not None
            and isinstance(data.get("price"), (int, float))
            and float(data["price"]) > budget_limit
        ):
            compatibility_status = "incompatible"
            compatibility_reason = f"price ${float(data['price']):.2f} exceeds total budget ${budget_limit:.2f} by itself"
        eligible = compatibility_status != "incompatible"
        ineligible_reason = compatibility_reason if compatibility_status == "incompatible" else None
        if (
            asin == selected
            and compatibility_status == "unknown"
            and compatibility_reason is not None
        ):
            soft_uncertainties.append(compatibility_reason)
        if not isinstance(objective_value, (int, float)):
            if metric == "rating" and data.get("rating_observed") is True:
                eligible = False
                ineligible_reason = "rating was visibly unavailable"
            else:
                uncertainties.append(f"{metric} is missing for {asin}")
        if _unrequested_variant(description, str(data.get("title") or "")) and asin == selected:
            uncertainties.append(f"Selected option {asin} adds an unspecified variant")
        evidence_ids = sorted(str(item) for item in data.get("evidence_state_ids", []) if item)
        candidates.append(
            {
                "option_id": asin,
                "description": description,
                "title": data.get("title"),
                "eligible": eligible,
                "compatibility_status": compatibility_status,
                "ineligible_reason": ineligible_reason,
                "compatibility_reason": compatibility_reason,
                "objective_value": objective_value,
                "price": data.get("price"),
                "rating": data.get("rating"),
                "evidence": evidence_ids,
            }
        )
        observed_facts.append(
            {
                "option_id": asin,
                "title": data.get("title"),
                "price": data.get("price"),
                "rating": data.get("rating"),
                "rating_observed": data.get("rating_observed"),
                "detail_observed": data.get("detail_observed") is True,
                "evidence_state_ids": evidence_ids,
            }
        )

    certificate: dict[str, Any] = {
        "expected_option_ids": expected_ids,
        "evaluated_option_ids": expected_ids,
        "objective": objective,
        "candidates": candidates,
        "selected_option_id": selected,
        "uncertainties": uncertainties,
        "soft_uncertainties": soft_uncertainties,
        "grounding_receipt": {
            "validated_against_environment": not uncertainties,
            "validated_option_ids": expected_ids,
            "observed_facts": observed_facts,
        },
    }
    budget = _budget_from_question(question, selected_price=evidenced[selected].get("price"))
    if budget is not None:
        certificate["budget"] = budget
    return certificate


def _product_entity(
    *,
    asin: str,
    title: str,
    price: float,
    rating: float | None,
    rating_observed: bool,
    detail_observed: bool,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "asin": asin,
        "title": title,
        "price": price,
        "rating_observed": rating_observed,
        "detail_observed": detail_observed,
    }
    if rating is not None:
        properties["rating"] = rating
    return {
        "subject": f"product {asin}",
        "status": "candidate",
        "confidence": 0.99 if detail_observed else 0.9,
        "properties": properties,
    }


def _parse_price(text: str) -> float | None:
    match = PRICE_RE.fullmatch(text.strip()) or PRICE_RE.search(text)
    return float(match.group(1)) if match else None


def _parse_rating(text: str) -> float | None:
    match = RATING_RE.search(text)
    return float(match.group(1)) if match else None


def _detail_title(parts: list[str], price_index: int) -> str:
    if price_index <= 0:
        return ""
    return parts[price_index - 1].strip()


def _available_option_descriptions(question: str) -> list[str]:
    match = re.search(
        r"\*\*Available Options:\*\*\s*(.+?)(?:\n\s*-{5,}|\n\s*\[SEP\])",
        question,
        flags=re.I | re.S,
    )
    if not match:
        return []
    return [
        item.strip()
        for item in re.findall(r"(?m)^\s*-\s+(.+?)\s*$", match.group(1))
        if item.strip()
    ]


def _materialized_products(states: list[Any]) -> tuple[dict[str, dict[str, Any]], str | None]:
    products: dict[str, dict[str, Any]] = {}
    selected: str | None = None
    for state in states:
        data = asdict(state) if is_dataclass(state) else dict(state)
        subject = str(data.get("subject") or "")
        variable = str(data.get("variable") or "")
        value = data.get("value")
        if subject == "current_selection" and variable == "selected_option_id":
            selected = str(value).upper()
            continue
        match = re.fullmatch(r"product\s+([A-Z0-9]{10})", subject, flags=re.I)
        if not match:
            continue
        asin = match.group(1).upper()
        product = products.setdefault(asin, {"evidence_state_ids": []})
        product[variable] = value
        state_id = data.get("state_id")
        if state_id:
            product["evidence_state_ids"].append(state_id)
    return products, selected


def _best_product_match(
    description: str,
    products: dict[str, dict[str, Any]],
    *,
    descriptions: list[str] | None = None,
    preferred_asins: set[str] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    ranked = _rank_product_matches(
        description,
        products,
        descriptions=descriptions,
        preferred_asins=preferred_asins,
    )
    if not ranked:
        return None
    _valid, _score, _severity, _visible, _detailed, asin, data = ranked[0]
    return asin, data


def _rank_product_matches(
    description: str,
    products: dict[str, dict[str, Any]],
    *,
    descriptions: list[str] | None = None,
    preferred_asins: set[str] | None = None,
) -> list[tuple[bool, float, int, bool, bool, str, dict[str, Any]]]:
    all_descriptions = descriptions or [description]
    preferred = preferred_asins or set()
    ranked = []
    for asin, data in products.items():
        title = str(data.get("title") or "")
        score = _identity_score(description, title, all_descriptions)
        mismatch = _identity_mismatch(description, title, all_descriptions)
        severity = _variant_severity(description, title)
        ranked.append(
            (
                not mismatch,
                score,
                severity,
                asin in preferred,
                data.get("detail_observed") is True,
                asin,
                data,
            )
        )
    ranked = [item for item in ranked if item[1] >= 0.28]
    return sorted(
        ranked,
        key=lambda item: (item[0], -item[2], item[3], item[1], item[4]),
        reverse=True,
    )


def _visible_identity_recovery_actions(
    description: str,
    products: dict[str, dict[str, Any]],
    *,
    descriptions: list[str],
    visible_asins: set[str],
    limit: int = 2,
) -> tuple[str, ...]:
    ranked = _rank_product_matches(
        description,
        products,
        descriptions=descriptions,
        preferred_asins=visible_asins,
    )
    actions = [
        f"click[{asin}]"
        for valid, score, _severity, visible, _detail, asin, _data in ranked
        if valid and visible and score >= 0.28
    ]
    return tuple(dict.fromkeys(actions[: max(0, limit)]))


def _title_similarity(description: str, title: str) -> float:
    left = _tokens(description)
    right = _tokens(title)
    if not left or not right:
        return 0.0
    overlap = left & right
    recall = len(overlap) / len(left)
    precision = len(overlap) / len(right)
    return 0.75 * recall + 0.25 * precision


def _identity_score(description: str, title: str, descriptions: list[str]) -> float:
    score = _title_similarity(description, title)
    contrastive = _contrastive_terms(description, descriptions)
    title_tokens = _tokens(title)
    if contrastive:
        score += 0.22 * (len(contrastive & title_tokens) / len(contrastive))
    if _identity_mismatch(description, title, descriptions):
        score -= 0.45
    elif _unrequested_variant(description, title):
        score -= 0.12
    return score


def _identity_mismatch(description: str, title: str, descriptions: list[str]) -> bool:
    return bool(
        _variant_mismatch(description, title)
        or _structured_identity_conflict(description, title)
        or _modifier_conflict(description, title)
        or _anchor_coverage_conflict(description, title, descriptions)
        or _contrastive_conflict(description, title, descriptions)
    )


def _modifier_conflict(description: str, title: str) -> bool:
    requested = _ordered_tokens(description)
    observed = _ordered_tokens(title)
    observed_set = set(observed)
    requested_set = set(requested)
    for index in range(1, len(requested) - 1):
        modifier = requested[index - 1]
        category = (requested[index], requested[index + 1])
        if modifier in _STOPWORDS or len(modifier) <= 2 or modifier in observed_set:
            continue
        for observed_index in range(1, len(observed) - 1):
            if (observed[observed_index], observed[observed_index + 1]) != category:
                continue
            observed_modifier = observed[observed_index - 1]
            if (
                observed_modifier not in _STOPWORDS
                and observed_modifier not in requested_set
                and len(observed_modifier) > 2
            ):
                return True
    return False


def _anchor_coverage_conflict(description: str, title: str, descriptions: list[str]) -> bool:
    anchors = _contrastive_terms(description, descriptions)
    if len(anchors) < 5:
        return False
    return len(anchors & _tokens(title)) / len(anchors) < 0.35


def _ordered_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))
        if len(token) > 1 or token.isdigit()
    ]


def _structured_identity_conflict(description: str, title: str) -> bool:
    requested_models = _model_tokens(description)
    observed_models = _model_tokens(title)
    if requested_models and observed_models and requested_models.isdisjoint(observed_models):
        return True

    requested_dimensions = _numbered_dimensions(description)
    observed_dimensions = _numbered_dimensions(title)
    for unit, requested_values in requested_dimensions.items():
        observed_values = observed_dimensions.get(unit)
        if observed_values and requested_values.isdisjoint(observed_values):
            return True
    return False


def _model_tokens(text: str) -> set[str]:
    normalized = text.lower().replace("_", "-")
    values = set(
        re.findall(
            r"\b(?:[a-z]{2,}[a-z0-9]*-\d+[a-z0-9-]*|\d+[a-z]+[a-z0-9-]*)\b",
            normalized,
        )
    )
    return {value for value in values if not _is_compact_dimension(value)}


def _numbered_dimensions(text: str) -> dict[str, set[float]]:
    dimensions: dict[str, set[float]] = {}
    for number, unit in re.findall(
        r"\b(\d+(?:\.\d+)?)\s*[- ]?\s*([a-zA-Z]{1,12})\b",
        text,
    ):
        family, scale = _normalize_dimension_unit(unit)
        dimensions.setdefault(family, set()).add(round(float(number) * scale, 6))
    return dimensions


def _normalize_dimension_unit(unit: str) -> tuple[str, float]:
    lowered = unit.lower()
    if lowered in _DIMENSION_UNITS:
        return _DIMENSION_UNITS[lowered]
    return lowered.rstrip("s"), 1.0


def _is_compact_dimension(value: str) -> bool:
    match = re.fullmatch(r"\d+(?:\.\d+)?([a-z]+)", value)
    return bool(match and match.group(1) in _DIMENSION_UNITS)


def _contrastive_terms(description: str, descriptions: list[str]) -> set[str]:
    own = _tokens(description)
    others: set[str] = set()
    for item in descriptions:
        if item != description:
            others.update(_tokens(item))
    return own - others


def _contrastive_conflict(description: str, title: str, descriptions: list[str]) -> bool:
    own_tokens = _tokens(description)
    title_tokens = _tokens(title)
    for other in descriptions:
        if other == description:
            continue
        other_tokens = _tokens(other)
        shared = own_tokens & other_tokens
        if len(shared) / max(min(len(own_tokens), len(other_tokens)), 1) < 0.28:
            continue
        own_only = own_tokens - other_tokens
        other_only = other_tokens - own_tokens
        if own_only and not (title_tokens & own_only) and title_tokens & other_only:
            return True
    return False


def _variant_mismatch(description: str, title: str) -> bool:
    requested_counts = _explicit_counts(description)
    observed_counts = _explicit_counts(title)
    return bool(requested_counts and observed_counts and requested_counts.isdisjoint(observed_counts))


def _unrequested_variant(description: str, title: str) -> bool:
    requested_counts = _explicit_counts(description)
    requested_piece_counts = {item for item in requested_counts if item[0] == "piece"}
    requested_collection = bool(
        requested_piece_counts
        or re.search(
            r"\b(?:assorted|bundle|kit|pack|set|variety)\b",
            description,
            re.I,
        )
    )
    if requested_collection:
        return False
    extra_collection = bool(
        "&" in title
        or re.search(r"\b(?:assorted|bundle|kit|set|variety)\b", title, re.I)
    )
    return extra_collection or any(
        unit == "piece" and value > 1 for unit, value in _explicit_counts(title)
    )


def _variant_severity(description: str, title: str) -> int:
    if not _unrequested_variant(description, title):
        return 0
    extra_collection = bool(
        "&" in title
        or re.search(r"\b(?:assorted|bundle|kit|set|variety)\b", title, re.I)
    )
    return 2 if extra_collection else 1


def _explicit_counts(text: str) -> set[tuple[str, float]]:
    aliases = {
        "ct": "piece",
        "count": "piece",
        "counts": "piece",
        "pc": "piece",
        "pcs": "piece",
        "piece": "piece",
        "pieces": "piece",
        "oz": "oz",
        "ounce": "oz",
        "ounces": "oz",
    }
    values: set[tuple[str, float]] = set()
    for number, unit in re.findall(
        r"\b(\d+(?:\.\d+)?)\s*[- ]?\s*(ct|counts?|pcs?|pieces?|oz|ounces?)\b",
        text,
        flags=re.I,
    ):
        values.add((aliases[unit.lower()], float(number)))
    return values


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))
        if (len(token) > 1 or token.isdigit()) and token not in _STOPWORDS
    }


def _clip(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _evidence_search_query(description: str, *, mismatched_title: str | None = None) -> str:
    query = re.sub(r"^(?:an?|the)\s+", "", description.strip().rstrip("."), flags=re.I)
    if mismatched_title:
        clue = _novel_visible_dimension(description, mismatched_title)
        if clue and clue.lower() not in query.lower():
            query = f"{query} {clue}"
    return _clip(query, 160)


def _novel_visible_dimension(description: str, title: str) -> str | None:
    requested_families = set(_numbered_dimensions(description))
    for match in re.finditer(
        r"\b(\d+(?:\.\d+)?)\s*[- ]?\s*"
        r"(ct|counts?|pcs?|pieces?|oz|ounces?|ml|liters?|litres?|l|g|grams?|kg|gb|tb|mah)\b",
        title,
        flags=re.I,
    ):
        number, unit = match.group(1), match.group(2)
        family, _scale = _normalize_dimension_unit(unit)
        if family == "count" or family in requested_families:
            continue
        return f"{number} {unit.lower()}"
    return None


def _preferred_visible_variant_action(*, question: str, description: str) -> str | None:
    requested_dimensions = _numbered_dimensions(description)
    candidates = []
    for raw in question.split("[SEP]"):
        value = " ".join(raw.strip().split())
        if not value or len(value) > 100:
            continue
        single_variant = bool(
            re.search(r"\b(?:pack\s+of|count|pieces?)\s*1\b|\b1\s*(?:count|pieces?)\b", value, re.I)
        )
        dimensions = _numbered_dimensions(value)
        matching_dimensions = sum(
            bool(values & dimensions.get(family, set()))
            for family, values in requested_dimensions.items()
        )
        if single_variant or matching_dimensions:
            candidates.append((matching_dimensions, single_variant, -len(value), value))
    if not candidates:
        return None
    _dimensions, _single, _length, value = max(candidates)
    return f"click[{value}]"


def _has_semantic_compatibility_rules(question: str) -> bool:
    return bool(re.search(r"Compatibility notes:|Must be compatible with", question, flags=re.I))


def _objective_from_question(question: str) -> dict[str, str]:
    lowered = question.lower()
    if re.search(r"highest[- ]priced|most expensive|maximi[sz]e[^\n]{0,30}price", lowered):
        return {"metric": "price", "direction": "maximize"}
    if re.search(r"lowest[- ]priced|cheapest|minimi[sz]e[^\n]{0,30}price", lowered):
        return {"metric": "price", "direction": "minimize"}
    if re.search(r"highest[- ]rated|maximi[sz]e[^\n]{0,30}rating", lowered):
        return {"metric": "rating", "direction": "maximize"}
    return {"metric": "price", "direction": "maximize"}


def _budget_from_question(question: str, selected_price: Any) -> dict[str, Any] | None:
    if not isinstance(selected_price, (int, float)):
        return None
    match = re.search(r"(?:total budget|combined must not exceed)[^$\d]{0,30}\$?\s*(\d+(?:\.\d+)?)", question, re.I)
    if not match:
        return None
    limit = float(match.group(1))
    history = _purchase_history_body(question)
    current_step = re.search(r"(?m)^\s*Product\s+(\d+)\s*:", question, re.I)
    if history is None and current_step is not None and int(current_step.group(1)) > 1:
        if float(selected_price) > limit:
            return {"before": 0.0, "after": float(selected_price), "limit": limit, "lower_bound_only": True}
        return None
    spent = sum(float(value) for value in re.findall(r"\$\s*(\d+(?:\.\d+)?)", history or ""))
    return {"before": spent, "after": spent + float(selected_price), "limit": limit}


def _visible_budget_limit(question: str) -> float | None:
    match = re.search(
        r"(?:total budget|combined must not exceed)[^$\d]{0,30}\$?\s*(\d+(?:\.\d+)?)",
        question,
        re.I,
    )
    return float(match.group(1)) if match else None


def _purchase_history_body(question: str) -> str | None:
    heading = re.search(
        r"(?im)(?:^|\[SEP\]|\n)\s*(?:#{1,6}\s*)?(?:\*{1,2})?"
        r"Purchase History(?:\*{1,2})?\s*:?\s*(?=\n|\[SEP\])",
        question,
    )
    if heading is None:
        return None
    tail = question[heading.end() :]
    boundary = re.search(
        r"(?im)(?:^|\[SEP\]|\n)\s*(?:-{5,}\s*)?(?:\*{1,3}\s*)?"
        r"(?:What You Need to Buy at Current Step|Product\s+\d+\s*:)",
        tail,
    )
    return tail[: boundary.start()] if boundary is not None else tail


def _compile_webshop_task(question: str) -> TaskIR:
    objective_data = _objective_from_question(question)
    objective = ObjectiveSpec(
        metric=objective_data["metric"],
        direction=objective_data["direction"],
    )
    positive = _parse_relation_map(question, "pairs well with")
    negative = _parse_relation_map(question, "avoids")
    constraints = [
        ConstraintAtom(
            field="compatibility",
            operator="allows",
            value={"source": source, "targets": targets},
            source_text=f"{source} pairs well with {', '.join(targets)}",
        )
        for source, targets in positive.items()
    ]
    constraints.extend(
        ConstraintAtom(
            field="compatibility",
            operator="forbids",
            value={"source": source, "targets": targets},
            source_text=f"{source} avoids {', '.join(targets)}",
        )
        for source, targets in negative.items()
    )
    has_prior_dependency = bool(
        (positive or negative or re.search(r"must\s+be\s+compatible", question, re.I))
        and re.search(r"\b(?:previous|prior|earlier|history)\b", question, re.I)
    )
    dependencies = []
    if has_prior_dependency:
        dependencies.append(
            DependencySpec(
                source="previous_decision",
                relation="constrains",
                target="current_decision",
                source_text="public task text requires a prior committed decision",
            )
        )
    evidence = [EvidenceRequirement(field="identity", source_level="list")]
    evidence.append(
        EvidenceRequirement(
            field=objective.metric,
            source_level="detail" if objective.metric == "rating" else "list",
        )
    )
    step_match = re.search(r"(?m)^\s*Product\s+(\d+)\s*:", question, re.I)
    return TaskIR(
        adapter_name="webshop",
        goal_id=semantic_fingerprint(question)[:16],
        options=_available_option_descriptions(question),
        objective=objective,
        constraints=constraints,
        dependencies=dependencies,
        evidence_requirements=evidence,
        commit_actions=["click[Buy Now]"],
        metadata={
            "positive_relations": positive,
            "negative_relations": negative,
            "step_index": int(step_match.group(1)) if step_match else None,
            "source": "public_task_text_and_action_schema",
        },
    )


def _parse_relation_map(question: str, phrase: str) -> dict[str, list[str]]:
    relation_map: dict[str, list[str]] = {}
    escaped = re.escape(phrase)
    pattern = re.compile(
        rf"([A-Za-z][A-Za-z0-9 '&/+-]{{0,55}}?)\s+{escaped}\s+"
        rf"(?:one\s+of\s*:\s*)?([^\.\n]+)",
        flags=re.I,
    )
    for match in pattern.finditer(question):
        source = _clean_relation_source(match.group(1))
        targets = _split_relation_targets(match.group(2))
        if source and targets:
            relation_map.setdefault(source, [])
            relation_map[source].extend(target for target in targets if target not in relation_map[source])
    return relation_map


def _clean_relation_source(value: str) -> str:
    cleaned = re.sub(r"[*#]", "", value).strip(" :-")
    if ":" in cleaned:
        cleaned = cleaned.rsplit(":", 1)[-1].strip()
    if "." in cleaned:
        cleaned = cleaned.rsplit(".", 1)[-1].strip()
    words = cleaned.split()
    return " ".join(words[-4:]).lower()


def _split_relation_targets(value: str) -> list[str]:
    cleaned = re.split(r"\*\*(?:Preference|Avoid|Constraint)", value, maxsplit=1, flags=re.I)[0]
    items = re.split(r"\s*,\s*|\s+or\s+|\s+and\s+", cleaned, flags=re.I)
    return [
        re.sub(r"[*#]", "", item).strip(" :-").lower()
        for item in items
        if re.sub(r"[*#]", "", item).strip(" :-")
    ]


def _compatibility_status(
    *,
    task: TaskIR,
    description: str,
    title: str,
    materialized_state: list[Any],
) -> tuple[str, str | None]:
    positive = task.metadata.get("positive_relations") or {}
    negative = task.metadata.get("negative_relations") or {}
    if not positive and not negative:
        return "compatible", None

    receipt_text = _prior_receipt_text(task, materialized_state)
    if not receipt_text:
        return "unknown", "prior decision scope summary is unavailable"

    active_positive = [
        target
        for source, targets in positive.items()
        if _phrase_present(source, receipt_text)
        for target in targets
    ]
    active_negative = [
        target
        for source, targets in negative.items()
        if _phrase_present(source, receipt_text)
        for target in targets
    ]
    if not active_positive and not active_negative:
        return "unknown", "prior decision does not activate a visible compatibility rule"

    candidate_text = f"{description} {title}".lower()
    denied = [target for target in active_negative if _phrase_present(target, candidate_text)]
    if denied:
        return "incompatible", f"violates public relation {', '.join(denied)}"

    if not active_positive:
        return "compatible", "no positive relation is required"
    allowed = [target for target in active_positive if _phrase_present(target, candidate_text)]
    if allowed:
        return "compatible", f"matches public relation {', '.join(allowed)}"
    if _is_visible_collection(candidate_text):
        return "unknown", "collection may contain a value allowed by the public relation"
    return "incompatible", "no required public compatibility relation is visible"


def _prior_receipt_text(task: TaskIR, materialized_state: list[Any]) -> str:
    rows = [asdict(item) if is_dataclass(item) else dict(item) for item in materialized_state]
    receipts = [row for row in rows if str(row.get("variable") or "") == "decision_receipt"]
    if not receipts:
        return ""
    step_index = task.metadata.get("step_index")
    expected_scope = f"product-{int(step_index) - 1}" if isinstance(step_index, int) and step_index > 1 else None
    if expected_scope is not None:
        scoped = [
            row
            for row in receipts
            if str((row.get("value") or {}).get("scope_id") or "") == expected_scope
        ]
        if scoped:
            receipts = scoped
    receipts.sort(key=lambda row: str(row.get("updated_at") or ""))
    value = receipts[-1].get("value") or {}
    if not isinstance(value, dict):
        return ""
    decision = value.get("committed_decision") or {}
    if not isinstance(decision, dict):
        return ""
    dependency_payload = {
        "scope_id": value.get("scope_id"),
        "status": value.get("status"),
        "confidence": value.get("confidence"),
        "selected_option_id": decision.get("selected_option_id"),
        "semantic_option": decision.get("semantic_option"),
        "entity": decision.get("entity"),
    }
    return json_text(dependency_payload)


def _phrase_present(phrase: str, text: str) -> bool:
    phrase_tokens = _tokens(phrase)
    return bool(phrase_tokens and phrase_tokens <= _tokens(text))


def _is_visible_collection(text: str) -> bool:
    return bool(
        re.search(r"\b(?:assorted|bundle|kit|multiple|set|variety)\b", text, re.I)
        or re.search(r"\b\d+\s+(?:colors?|flavors?|options?|sizes?|styles?)\b", text, re.I)
    )


def _recover_stale_action(
    *,
    proposed: str,
    question: str,
    description: str,
    progress: ProgressContext,
    reason: str,
    extra_actions: tuple[str, ...] = (),
) -> ActionDirective:
    actions: list[tuple[str, str]] = []
    if not progress.is_stale(proposed):
        actions.append((proposed, reason))

    alternative = f"search[{_contrastive_search_query(description)}]"
    if canonical_action(alternative) != canonical_action(proposed) and not progress.is_stale(alternative):
        actions.append(
            (alternative, "emphasize unresolved identity attributes with a different query")
        )
    actions.extend(
        (action, "inspect another visible identity hypothesis")
        for action in extra_actions
        if not progress.is_stale(action)
    )
    if "[SEP] Back to Search" in question and not progress.is_stale("click[Back to Search]"):
        actions.append(("click[Back to Search]", "leave the current evidence path and replan"))
    if "[SEP] Next >" in question and not progress.is_stale("click[Next >]"):
        actions.append(("click[Next >]", "inspect one different result page"))

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for action, action_reason in actions:
        key = canonical_action(action)
        if key and key not in seen:
            deduped.append((action, action_reason))
            seen.add(key)
    if deduped:
        primary, primary_reason = deduped[0]
        recoveries = tuple(
            _webshop_recovery_action(
                action=action,
                reason=action_reason,
                semantic_target=description,
            )
            for action, action_reason in deduped[1:]
        )
        forbidden = tuple(
            action
            for action in progress.stale_actions
            if action and canonical_action(action) != canonical_action(primary)
        )
        return _required_webshop_directive(
            action=primary,
            reason=primary_reason if progress.is_stale(proposed) else reason,
            semantic_target=description,
            recovery_actions=recoveries,
            forbidden_actions=forbidden,
        )

    stale = progress.latest_stale_signal
    stale_action = stale.action if stale is not None else canonical_action(proposed)
    return ActionDirective(
        action="",
        mode="advisory",
        reason=f"Do not repeat {stale_action}; choose a different legal action that can add evidence.",
    )


def _required_webshop_directive(
    *,
    action: str,
    reason: str,
    semantic_target: str,
    recovery_actions: tuple[RecoveryAction, ...] = (),
    forbidden_actions: tuple[str, ...] = (),
) -> ActionDirective:
    primary = _webshop_recovery_action(
        action=action,
        reason=reason,
        semantic_target=semantic_target,
    )
    obligation_id = f"evidence-{semantic_fingerprint(semantic_target + '|' + reason)[:12]}"
    prohibited = tuple(
        dict.fromkeys(("click[Buy Now]", *forbidden_actions))
    )
    return ActionDirective(
        action=action,
        reason=reason,
        mode="required_state_change",
        obligation_id=obligation_id,
        match_mode=primary.match_mode,
        semantic_target=semantic_target,
        required_terms=primary.required_terms,
        required_state_delta=(reason,),
        recovery_actions=recovery_actions,
        forbidden_actions=prohibited,
        metadata={
            "source": "public_webshop_protocol",
            "policy": "state_change_action_set",
        },
    )


def _webshop_recovery_action(
    *,
    action: str,
    reason: str,
    semantic_target: str,
) -> RecoveryAction:
    parts = re.match(r"\s*([A-Za-z_][A-Za-z0-9_-]*)\[", action)
    operator = parts.group(1).lower() if parts else "unknown"
    semantic_match = operator == "search"
    target_terms = tuple(
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9._'+-]*", semantic_target)
        if token.lower() not in _STOPWORDS
    )[:4]
    return RecoveryAction(
        action=action,
        operator=operator,
        reason=reason,
        expected_evidence=(semantic_target,),
        match_mode="semantic_target" if semantic_match else "exact",
        semantic_target=semantic_target if semantic_match else None,
        required_terms=target_terms if semantic_match else (),
    )


def _advisory_webshop_directive(*, action: str, reason: str) -> ActionDirective:
    return ActionDirective(
        action=action,
        reason=reason,
        mode="advisory",
        metadata={"source": "public_webshop_protocol"},
    )


def _contrastive_search_query(description: str) -> str:
    raw_tokens = re.findall(r"[A-Za-z0-9.#'+-]+", description.strip().rstrip("."))
    informative = [
        token
        for token in raw_tokens
        if len(token) > 2 and token.lower() not in _STOPWORDS
    ]
    if len(informative) > 10:
        informative = informative[:5] + informative[-5:]
    query = " ".join(informative) or _evidence_search_query(description)
    if canonical_action(f"search[{query}]") == canonical_action(
        f"search[{_evidence_search_query(description)}]"
    ):
        query = f"exact {query}"
    return _clip(query, 160)


def json_text(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=True, sort_keys=True).lower()


class WebShopTaskAdapter(GenericTaskAdapter):
    """Public WebShop protocol adapter; it contains no benchmark answers or item values."""

    name = "webshop"

    def supports(self, question: str) -> bool:
        has_protocol_page = "[SEP]" in question and (
            "[SEP] Search" in question or "[SEP] Buy Now" in question
        )
        has_public_shopping_schema = bool(
            "**Available Options:**" in question
            and re.search(r"\b(?:buy|price|product|rating|shopping)\b", question, re.I)
        )
        return has_protocol_page or has_public_shopping_schema

    def compile_task(self, question: str) -> TaskIR:
        return _compile_webshop_task(question)

    def build_candidate_ledger(
        self,
        *,
        task: TaskIR,
        question: str,
        materialized_state: list[Any],
        progress: ProgressContext,
    ) -> list[str]:
        del task
        return build_candidate_ledger(
            question=question,
            materialized_state=materialized_state,
            progress=progress,
        )

    def recommend_next_action(
        self,
        *,
        task: TaskIR,
        question: str,
        materialized_state: list[Any],
        progress: ProgressContext,
    ) -> str | None:
        directive = self.build_action_directive(
            question=question,
            materialized_state=materialized_state,
            progress=progress,
            task=task,
        )
        return directive.display_text if directive is not None else None

    def build_action_directive(
        self,
        *,
        task: TaskIR,
        question: str,
        materialized_state: list[Any],
        progress: ProgressContext,
    ) -> ActionDirective | None:
        return build_candidate_action_directive(
            question=question,
            materialized_state=materialized_state,
            progress=progress,
            task_ir=task,
        )

    def build_decision_certificate(
        self,
        *,
        task: TaskIR,
        question: str,
        materialized_state: list[Any],
        selected_option_id: str | None = None,
    ) -> dict[str, Any] | None:
        return build_visible_decision_certificate(
            question=question,
            materialized_state=materialized_state,
            selected_option_id=selected_option_id,
            task_ir=task,
        )

    def build_decision_receipt(
        self,
        *,
        scope_id: str,
        materialized_state: list[Any],
        events: list[Any],
    ) -> DecisionReceipt | None:
        receipt = super().build_decision_receipt(
            scope_id=scope_id,
            materialized_state=materialized_state,
            events=events,
        )
        rows = [asdict(item) if is_dataclass(item) else dict(item) for item in materialized_state]
        event_rows = [asdict(item) if is_dataclass(item) else dict(item) for item in events]
        products, selected = _materialized_products(materialized_state)
        question = next(
            (
                str(row.get("observation") or row.get("raw_chunk") or "")
                for row in reversed(event_rows)
                if "**Available Options:**" in str(row.get("observation") or row.get("raw_chunk") or "")
            ),
            "",
        )
        task = _compile_webshop_task(question) if question else None
        descriptions = task.options if task is not None else []
        selection_basis = "observed_current_selection"
        has_success = any(row.get("is_success") is True for row in event_rows)
        has_failure = any(row.get("is_failure") is True for row in event_rows)
        if not has_success and task is not None:
            ranked: list[tuple[float, int, str, str]] = []
            budget_limit = _visible_budget_limit(question)
            for description in descriptions:
                match = _best_product_match(description, products, descriptions=descriptions)
                if match is None:
                    continue
                asin, data = match
                title = str(data.get("title") or "")
                if _identity_mismatch(description, title, descriptions):
                    continue
                value = data.get(task.objective.metric)
                if not isinstance(value, (int, float)):
                    continue
                if (
                    budget_limit is not None
                    and isinstance(data.get("price"), (int, float))
                    and float(data["price"]) > budget_limit
                ):
                    continue
                compatibility_status, _reason = _compatibility_status(
                    task=task,
                    description=description,
                    title=title,
                    materialized_state=materialized_state,
                )
                if compatibility_status != "incompatible":
                    status_rank = 1 if compatibility_status == "compatible" else 0
                    ranked.append((float(value), status_rank, asin, description))
            if ranked:
                reverse = task.objective.direction == "maximize"
                if reverse:
                    _value, _status_rank, selected, _description = max(
                        ranked,
                        key=lambda item: (item[0], item[1]),
                    )
                else:
                    _value, _status_rank, selected, _description = min(
                        ranked,
                        key=lambda item: (item[0], -item[1]),
                    )
                selection_basis = "provisional_objective_ranking"
        if selected is None:
            return receipt
        entity = dict(products.get(selected, {}))
        entity.pop("evidence_state_ids", None)
        semantic_option = None
        if descriptions:
            semantic_option = max(
                descriptions,
                key=lambda description: _identity_score(
                    description,
                    str(entity.get("title") or ""),
                    descriptions,
                ),
            )
        selected_evidence = [
            str(row.get("state_id"))
            for row in rows
            if (
                str(row.get("subject") or "").lower() == f"product {selected}".lower()
                or (
                    row.get("subject") == "current_selection"
                    and row.get("variable") in {"selected_option_id", "purchase_status"}
                )
            )
            and row.get("state_id")
        ]
        status = "committed" if has_success else "failed" if has_failure else "provisional"
        confidence = 0.95 if status == "committed" else 0.55
        if semantic_option and _unrequested_variant(semantic_option, str(entity.get("title") or "")):
            confidence = min(confidence, 0.48)
        if receipt is None:
            receipt = DecisionReceipt(
                scope_id=scope_id,
                status=status,
                committed_decision={},
                satisfied_constraints=[],
                evidence_state_ids=[],
                source_event_ids=sorted(
                    str(row.get("event_id")) for row in event_rows if row.get("event_id")
                ),
                confidence=confidence,
            )
        receipt.status = status
        receipt.confidence = confidence
        receipt.committed_decision = {
            "selected_option_id": selected,
            "semantic_option": semantic_option,
            "entity": entity,
            "selection_basis": selection_basis,
        }
        receipt.evidence_state_ids = sorted(set(selected_evidence))
        return receipt
