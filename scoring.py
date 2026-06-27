from __future__ import annotations

import math
from typing import Any

from .models import OptimizerConfig


def points(cost: int | float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def delta_points(old_cost: int, new_cost: int) -> float:
    return points(new_cost) - points(old_cost)


def fresh_rate(validation: dict[str, Any]) -> float:
    right = validation.get("fresh_right")
    total = validation.get("fresh_total")
    if right is None or total in (None, 0):
        return 0.0
    return float(right) / float(total)


def official_exact(validation: dict[str, Any]) -> bool:
    return (
        validation.get("public_wrong") == 0
        and validation.get("arc_gen_wrong") == 0
        and validation.get("cost") is not None
    )


def is_accepted(validation: dict[str, Any], incumbent_cost: int, config: OptimizerConfig) -> bool:
    cost = validation.get("cost")
    if cost is None or cost >= incumbent_cost:
        return False
    if validation.get("public_wrong") != 0 or validation.get("arc_gen_wrong") != 0:
        return False
    if config.require_exact and not validation.get("exact"):
        return False
    if config.public_arc_only:
        return True
    if config.require_fresh_exact:
        return validation.get("fresh_total") not in (None, 0) and validation.get("fresh_right") == validation.get("fresh_total")
    return fresh_rate(validation) >= config.fresh_min


def near_miss_category(validation: dict[str, Any], incumbent_cost: int, config: OptimizerConfig) -> str | None:
    cost = validation.get("cost")
    if cost is None or cost >= incumbent_cost:
        return None
    public_wrong = validation.get("public_wrong")
    arc_wrong = validation.get("arc_gen_wrong")
    rate = fresh_rate(validation)
    reduction = (incumbent_cost - cost) / max(1, incumbent_cost)
    if public_wrong == 0 and arc_wrong is not None and arc_wrong <= 5 and rate >= config.fresh_min:
        return "A"
    if public_wrong is not None and public_wrong <= 1 and arc_wrong == 0 and rate >= config.fresh_min:
        return "B"
    if public_wrong == 0 and arc_wrong == 0 and rate >= 0.80:
        return "C"
    if public_wrong is not None and arc_wrong is not None and public_wrong + arc_wrong <= 8 and reduction >= 0.20:
        return "D"
    if public_wrong == 0 and arc_wrong is not None and arc_wrong <= 10:
        return "E"
    if arc_wrong == 0 and public_wrong == 1:
        return "F"
    return None


def is_near_miss(validation: dict[str, Any], incumbent_cost: int, config: OptimizerConfig) -> bool:
    return near_miss_category(validation, incumbent_cost, config) is not None
