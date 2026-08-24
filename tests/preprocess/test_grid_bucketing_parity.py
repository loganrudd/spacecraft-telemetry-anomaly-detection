"""Stage 0 gate for docs/reviews/023-channel-time-grid.md: prove the rewritten
(floor + groupby) gap-preserving bucketing is value-equivalent to today's
(``resample(rule).mean().dropna()``) before stage 3.1 moves the pipeline onto it.

``resample()``'s default origin (``start_day``) anchors bucket edges to
midnight of the *first tick's calendar day*; ``groupby(index.floor(rule))``
anchors them to the UNIX epoch. Both anchors are themselves always a multiple
of 86400 seconds from the epoch, so the two rules produce identical bucket
edges for ANY dataset precisely when ``rate`` divides 86400 (every day
boundary is then also a rate boundary, regardless of which day the data
starts on). For a rate that does not divide 86400, whether a *particular*
dataset's edges happen to coincide depends on its start date — luck, not a
guarantee — which is why 0.2 makes non-divisor rates a config-time error
rather than a documented caveat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Candidates 023.2 measured and 023.3 shipped — all divide 86400.
_DIVISOR_RATES = [30, 90, 300]
# Do not divide 86400; chosen because they are empirically shown below to
# diverge for this fixture's start date (86400 % rate != 0 does not guarantee
# divergence for every start date, only that no guarantee exists).
_NON_DIVISOR_RATES = [11, 13]


def _irregular_series_with_outage(seed: int = 0) -> pd.Series:
    """Irregular ticks (5-25s apart) with a real 75-day outage, ~76 days total.

    Multiple ticks land in most buckets at every candidate rate, so a mean
    aggregation mismatch would show up as a value difference, not just an
    index difference.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2019-03-14 07:23:41.123456", tz="UTC")
    offsets1 = np.cumsum(rng.uniform(5.0, 25.0, size=3000))
    ts1 = start + pd.to_timedelta(offsets1, unit="s")
    resume = ts1[-1] + pd.Timedelta(hours=6) + pd.Timedelta(days=75)
    offsets2 = np.cumsum(rng.uniform(5.0, 25.0, size=3000))
    ts2 = resume + pd.to_timedelta(offsets2, unit="s")
    index = pd.DatetimeIndex(np.concatenate([ts1.values, ts2.values]))
    return pd.Series(rng.normal(size=len(index)), index=index).sort_index()


@pytest.mark.parametrize("rate_s", _DIVISOR_RATES)
def test_floor_matches_resample_for_rates_dividing_a_day(rate_s: int) -> None:
    assert 86400 % rate_s == 0  # the property this test hinges on
    series = _irregular_series_with_outage()
    rule = f"{rate_s}s"

    via_resample = series.resample(rule).mean().dropna()
    via_floor = series.groupby(series.index.floor(rule)).mean()

    pd.testing.assert_series_equal(
        via_resample, via_floor, check_names=False, check_freq=False, rtol=1e-10,
    )


@pytest.mark.parametrize("rate_s", _NON_DIVISOR_RATES)
def test_floor_diverges_from_resample_for_rates_not_dividing_a_day(rate_s: int) -> None:
    # This divergence is the finding, not a bug in the test: resample()'s
    # start_day origin and floor's epoch anchor fall out of phase for a rate
    # that does not divide a day, so the two bucketings disagree on this
    # dataset. 0.2 turns that into a config-time rejection.
    assert 86400 % rate_s != 0
    series = _irregular_series_with_outage()
    rule = f"{rate_s}s"

    via_resample = series.resample(rule).mean().dropna()
    via_floor = series.groupby(series.index.floor(rule)).mean()

    assert not via_resample.index.equals(via_floor.index)
