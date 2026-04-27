from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass(frozen=True)
class SelectionPlan:
    apis: List[str]

    @property
    def all_apis(self) -> List[str]:
        return list(self.apis)


class MultiRouletteSelector:
    def __init__(self, groups: Dict[str, List[str]]) -> None:
        _SKIP_GROUPS = {"_excluded"}
        self._groups: Dict[str, List[str]] = {
            g: list(apis)
            for g, apis in groups.items()
            if g not in _SKIP_GROUPS and apis
        }
        self._api_to_group: Dict[str, str] = {
            api: group
            for group, apis in self._groups.items()
            for api in apis
        }
        self._usage: Dict[str, int] = defaultdict(int)
        self._anomaly_apis: Set[str] = set()
        self._attempted: Set[str] = set()
        self._executed: Set[str] = set()

    def select(self, n: int = 30) -> List[str]:
        return self.select_plan(n=n).all_apis

    def select_plan(self, n: int = 30, **_: object) -> SelectionPlan:
        groups = self._groups
        if not groups or n <= 0:
            return SelectionPlan(apis=[])

        m = {g: sum(1 for a in apis if a not in self._executed) for g, apis in groups.items()}
        total_unused = sum(m.values())

        if total_unused == 0:
            quotas_real = {g: n * len(apis) / max(1, sum(len(v) for v in groups.values()))
                           for g, apis in groups.items()}
            available = {g: len(apis) for g, apis in groups.items()}
        else:
            quotas_real = {g: n * m[g] / total_unused for g in groups}
            available = {g: len(apis) for g, apis in groups.items()}

        floors = {g: math.floor(q) for g, q in quotas_real.items()}
        for g in groups:
            floors[g] = min(floors[g], available[g])
        remainder_budget = n - sum(floors.values())
        ordered = sorted(
            groups.keys(),
            key=lambda g: -(quotas_real[g] - floors[g]),
        )
        allocations: Dict[str, int] = dict(floors)
        for g in ordered:
            if remainder_budget <= 0:
                break
            cap = available[g]
            if allocations[g] < cap:
                allocations[g] += 1
                remainder_budget -= 1

        selected: List[str] = []
        for g, apis in groups.items():
            k = min(allocations[g], len(apis))
            if k <= 0:
                continue
            selected.extend(self._roulette(list(apis), k))

        return SelectionPlan(apis=selected)

    def record_attempts(self, apis: List[str]) -> None:
        self._attempted.update(apis)

    def record_usage(self, apis: List[str], anomaly_detected: bool = False) -> None:
        self._executed.update(apis)
        for api in apis:
            if anomaly_detected:
                self._anomaly_apis.add(api)
                continue
            self._usage[api] += 1

    def stats(self) -> Dict[str, object]:
        zero_executed = sorted(
            group for group, apis in self._groups.items()
            if not any(api in self._executed for api in apis)
        )
        per_group = {}
        low_conversion = []
        for group, apis in self._groups.items():
            total = len(apis)
            attempted = sum(1 for api in apis if api in self._attempted)
            executed = sum(1 for api in apis if api in self._executed)
            attempted_cov = attempted / total if total else 0.0
            executed_cov = executed / total if total else 0.0
            conversion = executed / attempted if attempted else 0.0
            per_group[group] = {
                "total": total,
                "attempted": attempted,
                "executed": executed,
                "attempted_coverage": attempted_cov,
                "executed_coverage": executed_cov,
                "executed_attempted_ratio": conversion,
            }
            if attempted > 0 and conversion < 0.25:
                low_conversion.append(group)
        return {
            "total_unique_used": sum(1 for v in self._usage.values() if v > 0),
            "anomaly_exempt_apis": len(self._anomaly_apis),
            "per_group": per_group,
            "groups_zero_executed": zero_executed,
            "groups_low_conversion": sorted(low_conversion),
        }

    def _score(self, api: str) -> float:
        return 1.0 / (self._usage[api] + 1)

    def _roulette(self, apis: List[str], k: int) -> List[str]:
        pool = list(apis)
        chosen: List[str] = []
        for _ in range(k):
            if not pool:
                break
            scores = [self._score(a) for a in pool]
            total = sum(scores)
            if total == 0:
                idx = random.randrange(len(pool))
            else:
                r = random.uniform(0, total)
                cumulative = 0.0
                idx = len(pool) - 1
                for i, s in enumerate(scores):
                    cumulative += s
                    if cumulative >= r:
                        idx = i
                        break
            chosen.append(pool[idx])
            pool.pop(idx)
        return chosen
