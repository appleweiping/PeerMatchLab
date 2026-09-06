"""Reserving slots for senior reviewers, and refusing to pretend otherwise."""

from __future__ import annotations

import itertools
import random

import pytest

from peermatchlab.assignment import AssignmentEngine
from peermatchlab.config import MatchConfig
from peermatchlab.models import DataValidationError, Document, Expert
from peermatchlab.scoring import MatchScorer

THRESHOLD = 0.75
TOPICS = ("learning", "vision", "systems", "theory")


def _experts(*specs: tuple[str, float, int]) -> list[Expert]:
    return [
        Expert(
            id=name,
            name=name.upper(),
            summary="learning vision systems theory",
            capacity=capacity,
            seniority=seniority,
        )
        for name, seniority, capacity in specs
    ]


def _engine(documents: list[Document], experts: list[Expert]) -> AssignmentEngine:
    return AssignmentEngine(MatchScorer(documents, experts))


def _seniority(experts: list[Expert]) -> dict[str, bool]:
    return {expert.id: expert.seniority >= THRESHOLD for expert in experts}


# ---------------------------------------------------------------------------
# What the reservation guarantees.
# ---------------------------------------------------------------------------


def test_a_reserved_slot_goes_to_a_senior_expert() -> None:
    documents = [Document("d", "learning vision", required_experts=2)]
    experts = _experts(("senior", 0.9, 1), ("junior-a", 0.1, 1), ("junior-b", 0.2, 1))
    plan = _engine(documents, experts).assign(
        reviewers_per_document=2, minimum_senior_reviewers=1, senior_threshold=THRESHOLD
    )
    picked = {item.expert_id for item in plan.assignments}
    assert "senior" in picked
    assert len(picked) == 2


def test_an_unfillable_reservation_becomes_unmet_rather_than_junior() -> None:
    # No expert clears the threshold, so the reserved slot cannot be filled. The
    # remaining slot is still filled: the constraint withholds one seat, it does
    # not withhold the whole document.
    documents = [Document("d", "learning vision", required_experts=2)]
    experts = _experts(("junior-a", 0.1, 1), ("junior-b", 0.2, 1))
    plan = _engine(documents, experts).assign(
        reviewers_per_document=2, minimum_senior_reviewers=1, senior_threshold=THRESHOLD
    )
    assert len(plan.assignments) == 1
    assert plan.unmet["d"] == 1


def test_reserving_every_slot_admits_only_senior_experts() -> None:
    documents = [Document("d", "learning vision", required_experts=2)]
    experts = _experts(("senior", 0.8, 1), ("junior", 0.1, 5))
    plan = _engine(documents, experts).assign(
        reviewers_per_document=2, minimum_senior_reviewers=2, senior_threshold=THRESHOLD
    )
    assert {item.expert_id for item in plan.assignments} == {"senior"}
    assert plan.unmet["d"] == 1


def test_a_reservation_larger_than_the_demand_is_clamped() -> None:
    documents = [Document("d", "learning vision", required_experts=1)]
    experts = _experts(("senior", 0.9, 1), ("junior", 0.1, 1))
    plan = _engine(documents, experts).assign(
        reviewers_per_document=1, minimum_senior_reviewers=5, senior_threshold=THRESHOLD
    )
    assert [item.expert_id for item in plan.assignments] == ["senior"]
    assert not plan.unmet


def test_an_expert_never_fills_both_a_reserved_and_a_free_slot() -> None:
    # The reserved and free sides of the split meet at one capacity-one node per
    # document-expert pair, which is what stops the same expert from being
    # counted twice for one document.
    documents = [Document("d", "learning vision", required_experts=2)]
    experts = _experts(("senior", 0.9, 5))
    plan = _engine(documents, experts).assign(
        reviewers_per_document=2, minimum_senior_reviewers=1, senior_threshold=THRESHOLD
    )
    assert [item.expert_id for item in plan.assignments] == ["senior"]
    assert plan.unmet["d"] == 1


def test_the_threshold_decides_who_counts_as_senior() -> None:
    documents = [Document("d", "learning vision", required_experts=1)]
    experts = _experts(("mid", 0.6, 1), ("junior", 0.1, 1))
    engine = _engine(documents, experts)
    strict = engine.assign(
        reviewers_per_document=1, minimum_senior_reviewers=1, senior_threshold=0.75
    )
    relaxed = engine.assign(
        reviewers_per_document=1, minimum_senior_reviewers=1, senior_threshold=0.5
    )
    assert strict.unmet["d"] == 1
    assert [item.expert_id for item in relaxed.assignments] == ["mid"]


def test_zero_reservation_reproduces_the_unconstrained_plan() -> None:
    documents = [Document("d", "learning vision", required_experts=2)]
    experts = _experts(("senior", 0.9, 1), ("junior", 0.1, 1))
    engine = _engine(documents, experts)
    plain = engine.assign(reviewers_per_document=2)
    zero = engine.assign(reviewers_per_document=2, minimum_senior_reviewers=0)
    assert plain == zero


def test_the_strategy_label_records_the_reservation() -> None:
    documents = [Document("d", "learning vision", required_experts=1)]
    experts = _experts(("senior", 0.9, 1))
    engine = _engine(documents, experts)
    assert engine.assign(reviewers_per_document=1).strategy == "optimal"
    assert (
        engine.assign(reviewers_per_document=1, minimum_senior_reviewers=1).strategy
        == "optimal-senior"
    )
    assert (
        engine.assign(
            strategy="greedy", reviewers_per_document=1, minimum_senior_reviewers=1
        ).strategy
        == "greedy-senior"
    )
    assert (
        engine.assign(
            reviewers_per_document=1, minimum_senior_reviewers=1, load_balance_penalty=0.5
        ).strategy
        == "optimal-balanced-senior"
    )


def test_the_greedy_baseline_honours_the_same_floor() -> None:
    documents = [Document("d", "learning vision", required_experts=2)]
    experts = _experts(("senior", 0.8, 1), ("junior-a", 0.9 - 0.2, 1), ("junior-b", 0.1, 1))
    plan = _engine(documents, experts).assign(
        strategy="greedy",
        reviewers_per_document=2,
        minimum_senior_reviewers=1,
        senior_threshold=THRESHOLD,
    )
    seniority = _seniority(experts)
    assert sum(1 for item in plan.assignments if seniority[item.expert_id]) >= 1


# ---------------------------------------------------------------------------
# What it refuses.
# ---------------------------------------------------------------------------


def test_the_reservation_is_refused_alongside_institution_diversity() -> None:
    # Reserving senior slots gives those units their own arcs from the source,
    # which may reach only senior pairs. Institution diversity needs a
    # capacity-one gate between the document and those pairs, and a unit passing
    # through that gate no longer carries which side it came from. A network
    # merging them would satisfy one constraint and quietly relax the other.
    documents = [Document("d", "learning vision", required_experts=2)]
    experts = _experts(("senior", 0.9, 1), ("junior", 0.1, 1))
    with pytest.raises(ValueError, match="not jointly expressible"):
        _engine(documents, experts).assign(
            reviewers_per_document=2,
            minimum_senior_reviewers=1,
            require_distinct_institutions=True,
        )


@pytest.mark.parametrize("value", [-1, 1.5, "1", True, None])
def test_the_reservation_count_is_validated(value: object) -> None:
    documents = [Document("d", "learning vision")]
    experts = _experts(("senior", 0.9, 1))
    with pytest.raises(ValueError, match="minimum_senior_reviewers"):
        _engine(documents, experts).assign(minimum_senior_reviewers=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.1, 1.5, float("nan"), "high"])
def test_the_threshold_is_validated(value: object) -> None:
    documents = [Document("d", "learning vision")]
    experts = _experts(("senior", 0.9, 1))
    with pytest.raises(ValueError, match="senior_threshold"):
        _engine(documents, experts).assign(
            minimum_senior_reviewers=1,
            senior_threshold=value,  # type: ignore[arg-type]
        )


def test_the_configuration_carries_and_validates_the_reservation() -> None:
    config = MatchConfig.from_mapping({"minimum_senior_reviewers": 1, "senior_threshold": 0.6})
    assert config.minimum_senior_reviewers == 1
    assert config.senior_threshold == pytest.approx(0.6)
    for payload, message in (
        ({"minimum_senior_reviewers": -1}, "must not be negative"),
        ({"minimum_senior_reviewers": 1.5}, "must be an integer"),
        ({"minimum_senior_reviewers": True}, "must be an integer"),
        ({"senior_threshold": 2.0}, "between 0 and 1"),
        (
            {"minimum_senior_reviewers": 1, "require_distinct_institutions": True},
            "cannot be combined",
        ),
    ):
        with pytest.raises(DataValidationError, match=message):
            MatchConfig.from_mapping(payload)


# ---------------------------------------------------------------------------
# The plan is still the best one the constraint allows.
# ---------------------------------------------------------------------------


def _random_instance(seed: int) -> tuple[list[Document], list[Expert]]:
    rng = random.Random(seed)
    documents = [
        Document(f"d{index}", " ".join(rng.sample(TOPICS, 2)), required_experts=rng.choice([1, 2]))
        for index in range(rng.randint(1, 3))
    ]
    experts = [
        Expert(
            id=f"e{index}",
            name=f"E{index}",
            summary=" ".join(rng.sample(TOPICS, rng.randint(1, 3))),
            capacity=rng.randint(1, 2),
            seniority=round(rng.uniform(0.0, 1.0), 2),
        )
        for index in range(rng.randint(2, 5))
    ]
    return documents, experts


def _best_total(scorer: MatchScorer, documents, experts, floor: int) -> tuple[int, float] | None:
    """Best (assignment count, total score) over every feasible plan."""
    scores = {(s.document_id, s.expert_id): s.total for s in scorer.matrix() if s.eligible}
    capacity = {expert.id: expert.capacity for expert in experts}
    senior = {expert.id: expert.seniority >= THRESHOLD for expert in experts}
    per_document = []
    for document in documents:
        demand = document.required_experts if document.required_experts else 2
        eligible = sorted(e.id for e in experts if (document.id, e.id) in scores)
        options = []
        for size in range(min(demand, len(eligible)), -1, -1):
            for combo in itertools.combinations(eligible, size):
                juniors = sum(1 for item in combo if not senior[item])
                if juniors > demand - min(floor, demand):
                    continue
                options.append(combo)
        per_document.append(options)
    best: tuple[int, float] | None = None
    for choice in itertools.product(*per_document):
        used: dict[str, int] = {}
        total = 0.0
        count = 0
        feasible = True
        for document, combo in zip(documents, choice, strict=True):
            for expert_id in combo:
                used[expert_id] = used.get(expert_id, 0) + 1
                if used[expert_id] > capacity[expert_id]:
                    feasible = False
                    break
                total += scores[(document.id, expert_id)]
                count += 1
            if not feasible:
                break
        if not feasible:
            continue
        candidate = (count, round(total, 9))
        if best is None or candidate > best:
            best = candidate
    return best


@pytest.mark.parametrize("floor", [1, 2])
def test_the_reserved_plan_is_optimal_under_its_constraint(floor: int) -> None:
    """Satisfying a constraint is easy if score may be given up freely.

    Exhaustive search over small instances is what shows nothing better was
    available, rather than only that the floor held.
    """

    for seed in range(60):
        documents, experts = _random_instance(seed)
        scorer = MatchScorer(documents, experts)
        plan = AssignmentEngine(scorer).assign(
            reviewers_per_document=2,
            minimum_senior_reviewers=floor,
            senior_threshold=THRESHOLD,
        )
        seniority = _seniority(experts)
        for document in documents:
            demand = document.required_experts if document.required_experts else 2
            picked = [item for item in plan.assignments if item.document_id == document.id]
            juniors = sum(1 for item in picked if not seniority[item.expert_id])
            assert juniors <= demand - min(floor, demand), (seed, document.id)
        best = _best_total(scorer, documents, experts, floor)
        if best is not None:
            assert (len(plan.assignments), round(plan.total_score, 9)) >= best, seed
