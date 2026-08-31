from __future__ import annotations

import pytest

from peermatchlab.models import Document, Expert, Publication


@pytest.fixture
def documents() -> tuple[Document, ...]:
    return (
        Document(
            "paper-1",
            "Efficient neural retrieval",
            "Late interaction and compact indexes for document search",
            topics=("information retrieval", "machine learning"),
            keywords=("ranking", "indexing"),
        ),
        Document(
            "paper-2",
            "Transparent recommendation audits",
            "Measuring exposure and recommendation quality",
            topics=("recommender systems", "responsible AI"),
            keywords=("audit", "fairness"),
        ),
    )


@pytest.fixture
def experts() -> tuple[Expert, ...]:
    return (
        Expert(
            "expert-a",
            "Ari",
            "Search systems and efficient ranking",
            topics=("information retrieval",),
            keywords=("ranking", "indexing"),
            publications=(Publication("Fast ranking", "search indexes", 2025),),
            capacity=1,
            institution="North Lab",
            seniority=0.7,
            bids={"paper-1": 1.0},
        ),
        Expert(
            "expert-b",
            "Bo",
            "Recommendation evaluation and accountable machine learning",
            topics=("recommender systems", "responsible AI"),
            keywords=("fairness", "audit"),
            publications=(Publication("Auditing feeds", "recommendation exposure", 2024),),
            capacity=2,
            institution="West Lab",
            seniority=0.6,
            bids={"paper-2": 0.8},
        ),
        Expert(
            "expert-c",
            "Cy",
            "General machine learning",
            topics=("machine learning",),
            capacity=2,
            institution="West Lab",
            seniority=0.4,
        ),
    )
