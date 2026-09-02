import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from base.trap_sequence import TrapSequence
from scripts.trap_distr_builder import (
    TableIIIRow,
    plot_trapping_on_axis,
    save_table_iii_summary,
    summarize_trap_events,
)


def _table_row(gas: str, mu: float) -> TableIIIRow:
    return TableIIIRow(
        gas=gas,
        classifier="DM",
        mu=mu,
        n_t_mean=2.0,
        n_0_mean=3.0,
        k_est=0.4,
        fit_t_min_s=1e-12,
        fit_t_max_s=1e-7,
        histogram_bins=50,
        n_fit_bins=12,
        n_trajectories=20,
        n_nonzero_events=40,
        loglog_slope=-mu,
        fit_r_squared=0.95,
        fit_p_value=1e-4,
        fit_std_err=0.1,
    )


def test_table_iii_event_metrics_average_per_trajectory_probability() -> None:
    sequences = [
        TrapSequence(np.ones(2), np.array([0.0, 1.0])),
        TrapSequence(np.ones(4), np.array([0.0, 0.0, 0.0, 1.0])),
    ]

    summary = summarize_trap_events(sequences)

    assert summary.n_t_mean == 1.0
    assert summary.n_0_mean == 2.0
    assert summary.k_est == 0.375
    assert summary.n_trajectories == 2
    assert summary.n_nonzero_events == 2


def test_table_iii_uses_positive_tail_exponent() -> None:
    rng = np.random.default_rng(42)
    times = (rng.pareto(1.5, 20_000) + 1.0) * 1e-9
    times = times[times <= 1e-6]
    fig, ax = plt.subplots()

    fit = plot_trapping_on_axis(ax, times, 1e-9, 1e-6, "TEST")
    plt.close(fig)

    assert fit.slope < 0.0
    assert fit.mu == -fit.slope


def test_table_iii_csv_json_merge_and_replace_rows(tmp_path: Path) -> None:
    ch4 = _table_row("CH4", 1.452)
    h2 = _table_row("H2", 2.285)
    save_table_iii_summary([ch4], tmp_path)
    csv_path, json_path = save_table_iii_summary([h2], tmp_path)

    with csv_path.open(encoding="utf-8", newline="") as file:
        csv_rows = list(csv.DictReader(file))
    json_rows = json.loads(json_path.read_text(encoding="utf-8"))

    assert [row["gas"] for row in csv_rows] == ["CH4", "H2"]
    assert [row["mu"] for row in json_rows] == [1.452, 2.285]

    save_table_iii_summary([replace(ch4, mu=1.5)], tmp_path)
    replaced_rows = json.loads(json_path.read_text(encoding="utf-8"))

    assert len(replaced_rows) == 2
    assert replaced_rows[0]["mu"] == 1.5
