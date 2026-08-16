"""Pure helpers for ranking and sequentially reducing tuning candidates."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence


class TuningError(ValueError):
    """A machine-readable tuning failure."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, object]:
        return {
            "success": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": dict(self.details),
            },
        }


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise TuningError("invalid_candidate", "candidate_id 必須是非空字串")
        if not isinstance(self.parameters, Mapping):
            raise TuningError("invalid_candidate", "parameters 必須是 mapping")
        try:
            encoded = json.dumps(dict(self.parameters), allow_nan=False)
            snapshot = json.loads(encoded)
        except (TypeError, ValueError) as error:
            raise TuningError(
                "invalid_candidate",
                "parameters 必須是有限且可序列化的 JSON 物件",
                {"candidate_id": self.candidate_id},
            ) from error
        if not isinstance(snapshot, dict):
            raise TuningError("invalid_candidate", "parameters 必須是 JSON 物件")
        object.__setattr__(self, "candidate_id", self.candidate_id.strip())
        object.__setattr__(self, "parameters", MappingProxyType(snapshot))


@dataclass(frozen=True)
class CandidateResult:
    candidate: Candidate
    samples: tuple[float, ...]
    budget: int


@dataclass(frozen=True)
class CandidateScore:
    candidate: Candidate
    budget: int
    p5: float
    mean: float
    median: float
    cv: float
    stable: bool
    sample_count: int


@dataclass(frozen=True)
class SearchRound:
    budget: int
    ranked: tuple[CandidateScore, ...]
    survivors: tuple[Candidate, ...]


@dataclass(frozen=True)
class SearchResult:
    winner: CandidateScore
    rounds: tuple[SearchRound, ...]


Evaluator = Callable[[Candidate, int], Iterable[float]]


def score_samples(
    candidate: Candidate,
    samples: Iterable[float],
    budget: int,
    *,
    max_cv: float | None = None,
) -> CandidateScore:
    """Score throughput samples using linear p5 and sample CV."""
    normalized = _validate_samples(samples)
    _validate_budget(budget)
    _validate_max_cv(max_cv)
    mean = statistics.fmean(normalized)
    cv = statistics.stdev(normalized) / mean if len(normalized) > 1 else 0.0
    return CandidateScore(
        candidate=candidate,
        budget=budget,
        p5=_linear_percentile(normalized, 0.05),
        mean=mean,
        median=statistics.median(normalized),
        cv=cv,
        stable=max_cv is None or cv <= max_cv,
        sample_count=len(normalized),
    )


def rank_candidates(
    results: Iterable[CandidateResult],
    *,
    max_cv: float | None = None,
) -> tuple[CandidateScore, ...]:
    """Rank stable candidates by p5, then median and mean throughput."""
    _validate_max_cv(max_cv)
    scores = [
        score_samples(result.candidate, result.samples, result.budget, max_cv=max_cv)
        for result in results
    ]
    if not scores:
        raise TuningError("invalid_results", "至少需要一個候選結果")
    scores.sort(
        key=lambda score: (
            not score.stable,
            -score.p5,
            -score.median,
            -score.mean,
            score.candidate.candidate_id,
        )
    )
    return tuple(scores)


def sequential_halving(
    candidates: Sequence[Candidate],
    budgets: Sequence[int],
    evaluator: Evaluator,
    reduction_factor: int = 2,
    *,
    max_cv: float | None = None,
) -> SearchResult:
    """Evaluate all candidates cheaply, then spend larger budgets on survivors."""
    active = _validate_search_inputs(candidates, budgets, evaluator, reduction_factor, max_cv)
    rounds: list[SearchRound] = []
    last_ranking: tuple[CandidateScore, ...] | None = None
    for round_index, budget in enumerate(budgets):
        results: list[CandidateResult] = []
        for candidate in active:
            try:
                samples = tuple(evaluator(candidate, budget))
            except TuningError:
                raise
            except Exception as error:
                raise TuningError(
                    "evaluation_failed",
                    "候選評估失敗",
                    {
                        "candidate_id": candidate.candidate_id,
                        "budget": budget,
                        "cause": type(error).__name__,
                    },
                ) from error
            results.append(CandidateResult(candidate, samples, budget))
        last_ranking = rank_candidates(results, max_cv=max_cv)
        survivor_count = (
            1
            if round_index == len(budgets) - 1
            else max(1, math.ceil(len(active) / reduction_factor))
        )
        active = tuple(score.candidate for score in last_ranking[:survivor_count])
        rounds.append(SearchRound(budget, last_ranking, active))
    assert last_ranking is not None
    return SearchResult(last_ranking[0], tuple(rounds))


def _validate_search_inputs(
    candidates: Sequence[Candidate],
    budgets: Sequence[int],
    evaluator: Evaluator,
    reduction_factor: int,
    max_cv: float | None,
) -> tuple[Candidate, ...]:
    if not candidates or any(not isinstance(candidate, Candidate) for candidate in candidates):
        raise TuningError("invalid_candidates", "至少需要一個有效 Candidate")
    identifiers = [candidate.candidate_id for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise TuningError("duplicate_candidate", "candidate_id 不得重複")
    if not budgets or any(not _is_positive_int(budget) for budget in budgets):
        raise TuningError("invalid_budgets", "budgets 必須是正整數序列")
    if any(current <= previous for previous, current in zip(budgets, budgets[1:])):
        raise TuningError("invalid_budgets", "budgets 必須嚴格遞增")
    if not _is_positive_int(reduction_factor) or reduction_factor < 2:
        raise TuningError("invalid_reduction_factor", "reduction_factor 必須至少為 2")
    if not callable(evaluator):
        raise TuningError("invalid_evaluator", "evaluator 必須可呼叫")
    _validate_max_cv(max_cv)
    return tuple(candidates)


def _validate_samples(samples: Iterable[float]) -> tuple[float, ...]:
    try:
        raw = tuple(samples)
    except TypeError as error:
        raise TuningError("invalid_samples", "samples 必須是數值序列") from error
    if not raw or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw
    ):
        raise TuningError("invalid_samples", "samples 必須是非空數值序列")
    normalized = tuple(float(value) for value in raw)
    if not normalized or any(not math.isfinite(value) or value <= 0 for value in normalized):
        raise TuningError("invalid_samples", "samples 必須是非空、有限且大於零")
    return normalized


def _validate_budget(budget: int) -> None:
    if not _is_positive_int(budget):
        raise TuningError("invalid_budget", "budget 必須是正整數")


def _validate_max_cv(max_cv: float | None) -> None:
    if max_cv is None:
        return
    if isinstance(max_cv, bool) or not isinstance(max_cv, (int, float)):
        raise TuningError("invalid_max_cv", "max_cv 必須是有限非負數")
    if not math.isfinite(float(max_cv)) or max_cv < 0:
        raise TuningError("invalid_max_cv", "max_cv 必須是有限非負數")


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _linear_percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
