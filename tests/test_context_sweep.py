"""Tests for the context sweep harness.

`eval/run_*.py` is production code here: it is what every reported number rests
on, and DL-26 found every scorer in this directory could be mutated to return
`True` with the whole suite green. A budget guard that silently never fires is
the same class of defect, because the failure it prevents is a run that reports
a partial population as though it were complete.
"""

from __future__ import annotations

from datetime import date

from agent.state import initial_state
from domain import EmployeeContext
from eval.run_context_sweep import (
    BUDGET_HEADROOM,
    DAILY_TOKEN_CAP,
    RESOLVE_MAX_TOKENS,
    affordable,
    cost_of,
)
from retrieval.store import SearchHit


def hit(citation, layer="federal", text="body"):
    return SearchHit(
        chunk_id=f"c-{citation}", citation=citation, authority_layer=layer,
        jurisdiction="US", content_status="substantive", heading="h",
        text=text, score=0.5,
    )


def state_with(hits):
    state = initial_state(
        "how much leave?",
        employee_context=EmployeeContext(state="CA"),
        as_of=date(2026, 4, 1),
    )
    state["retrieved"] = hits
    return state


# --- the budget guard -------------------------------------------------------


def test_a_call_that_fits_is_affordable() -> None:
    assert affordable(spent=0, price=5_000) is True


def test_a_call_that_would_cross_the_cap_is_refused() -> None:
    assert affordable(spent=DAILY_TOKEN_CAP, price=1) is False


def test_the_reserve_is_not_spendable() -> None:
    """The headroom exists to absorb the retries DL-37's tool-contract failures
    cost. A guard that lets the last call eat it has no reserve at all."""
    spendable = DAILY_TOKEN_CAP - BUDGET_HEADROOM
    assert affordable(spent=spendable - 100, price=100) is True
    assert affordable(spent=spendable - 100, price=101) is False


def test_the_boundary_is_inclusive_not_off_by_one() -> None:
    """Pinned as a value, not a relationship (DL-10): `<` and `<=` here differ
    by exactly one call, and one call is this project's noise floor."""
    assert affordable(spent=0, price=DAILY_TOKEN_CAP - BUDGET_HEADROOM) is True
    assert affordable(spent=0, price=DAILY_TOKEN_CAP - BUDGET_HEADROOM + 1) is False


# --- pricing a call before making it ----------------------------------------


def test_the_price_includes_the_reserved_output_budget() -> None:
    """Groq deducts the RESERVATION, not the tokens produced. Pricing on the
    prompt alone under-counts every call by the whole output budget, which is
    how a guard passes and the provider still refuses."""
    state = state_with([hit("29 CFR 825.200")])
    assert cost_of(state, cap=None) > RESOLVE_MAX_TOKENS


def test_a_tighter_cap_costs_less() -> None:
    """The entire premise of the experiment. If this does not hold, the arms
    are not measuring what DL-38 says they measure."""
    hits = [hit(f"fed-{i}", text="a passage " * 40) for i in range(6)]
    state = state_with(hits)
    assert cost_of(state, cap=2) < cost_of(state, cap=3) < cost_of(state, cap=None)


def test_pricing_a_thin_layer_is_unchanged_by_the_cap() -> None:
    """A cap above what was retrieved removes nothing, so it must not be
    reported as a saving."""
    state = state_with([hit("ca-1", layer="state")])
    assert cost_of(state, cap=3) == cost_of(state, cap=None)
