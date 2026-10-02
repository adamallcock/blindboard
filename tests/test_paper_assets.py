"""Fixes M-4, M-2 and M5 in scripts/build_paper_assets.py.

M-4: tier certificates are anchored at min(certificate, analytic ceiling)
wherever they gate or normalize, and the calibration loader takes the value
from the LATEST calibration file (max across files is upward-biased on
re-calibration).

M-2: (model, tier) pairs whose route ignores graded effort merge every
requested-effort label into one "single-mode" cell with the standard
by-seed later-run-wins dedup.

M5: ONE canonicalization rule for every output. The table deduplicated
accuracy by seed but summed cost over every contributing run, while the
figures averaged all attempts -- so a plotted cell, its table cell and the
printed cost described three different populations. canonical_selection() is
now the single selection behind all of them, per-cell cost is the cost of the
SELECTED episodes, and all-attempt spend is reported separately as campaign
spend. The cross-output equality tests below are the regression."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_paper_assets as bpa  # noqa: E402


def _write_calibration(path: Path, table: dict[str, dict[str, float]]) -> None:
    payload = {
        "seeds": 60,
        "table": {
            tier: {
                agent: {"board_accuracy_mean": value}
                for agent, value in agents.items()
            }
            for tier, agents in table.items()
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------
# M-4: ceilings + latest-file calibration selection
# --------------------------------------------------------------------------


def test_ceilings_cover_published_tiers() -> None:
    assert bpa.CEILINGS == {
        "t3_observer": 0.875,
        "t6_observer26": 0.897436,
        "t7_perpetual": 0.942308,
        "t8_storm": 0.897436,
        "t8_storm_hard": 0.855769,
        "t9_numrow": 0.961538,
        "t9_macbook": 0.968085,
        "t10_relabel": 0.942308,
        "t11_maelstrom": 0.968085,
    }


def test_certificate_above_ceiling_anchors_at_ceiling(tmp_path: Path) -> None:
    # A seed-lucky planner certificate above t9_numrow's analytic ceiling
    # (.961538) must be anchored AT the ceiling in the gate summary, and the
    # above-ceiling count must be surfaced alongside it.
    _write_calibration(
        tmp_path / "calibration-20260718-000000.json",
        {"t9_numrow": {"planner": 0.99, "reference": 0.99}},
    )
    status, summary = bpa.tier_certificates_gate(calibration_dir=tmp_path)
    assert "min ref .962 at t9_numrow" in summary
    assert "1 certificate above its analytic ceiling (seed luck, anchored at ceiling)" in summary
    assert status == "WARN"  # every other tier is uncalibrated in this fixture


def test_certificate_below_ceiling_kept_as_is(tmp_path: Path) -> None:
    _write_calibration(
        tmp_path / "calibration-20260718-000000.json",
        {"t8_storm_hard": {"planner": 0.8282}},
    )
    _, summary = bpa.tier_certificates_gate(calibration_dir=tmp_path)
    assert "min ref .828 at t8_storm_hard" in summary
    assert "above" not in summary


def test_latest_calibration_file_wins_not_max(tmp_path: Path) -> None:
    # Older file holds the HIGHER planner value; the newer one must win.
    _write_calibration(
        tmp_path / "calibration-20260701-000000.json",
        {
            "t3_observer": {"planner": 0.90},
            "t6_gauntlet": {"reference": 0.95},
        },
    )
    _write_calibration(
        tmp_path / "calibration-20260702-000000.json",
        {
            # Reference here is higher still, but planner presence for the
            # tier means reference values are ignored entirely.
            "t3_observer": {"planner": 0.85, "reference": 0.99},
            "t6_gauntlet": {"reference": 0.90},
        },
    )
    best = bpa.load_calibration_best(tmp_path)
    assert best["t3_observer"] == 0.85  # latest planner, not max, not reference
    assert best["t6_gauntlet"] == 0.90  # never planner-run: latest reference


def test_planner_from_older_file_beats_newer_reference(tmp_path: Path) -> None:
    # A tier run under the planner once keeps its planner value even when a
    # later file only re-calibrated the reference agent.
    _write_calibration(
        tmp_path / "calibration-20260701-000000.json",
        {"t7_blitz": {"planner": 0.88}},
    )
    _write_calibration(
        tmp_path / "calibration-20260702-000000.json",
        {"t7_blitz": {"reference": 0.97}},
    )
    assert bpa.load_calibration_best(tmp_path)["t7_blitz"] == 0.88


# --------------------------------------------------------------------------
# Fixtures for the selection tests
# --------------------------------------------------------------------------

# openrouter/qwen/qwen3.6-27b lists at 0.45 / 2.70 USD per Mtok, so an episode
# billing 0 input and 1M output tokens costs exactly $2.70.
QWEN27B = "openrouter/qwen/qwen3.6-27b"
QWEN27B_OUT_PER_MTOK = 2.70


def _make_run(
    results_dir: Path,
    name: str,
    summary: dict,
    seeds: dict[int, float],
    output_tokens: int = 1_000_000,
) -> None:
    run_dir = results_dir / name
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for index, (seed, acc) in enumerate(sorted(seeds.items())):
        (run_dir / f"seed{index:04d}.json").write_text(
            json.dumps(
                {
                    "result": {
                        "seed": seed,
                        "board_accuracy": acc,
                        "api_calls": 26,
                        "usage": {"input_tokens": 0, "output_tokens": output_tokens},
                    }
                }
            ),
            encoding="utf-8",
        )


def _qwen_rerun_tree(tmp_path: Path) -> Path:
    """The shape that produced the reported contradictions: a cell whose seeds
    were partly re-run later, at a different requested-effort label, with the
    re-run billing more than the run it supersedes."""
    results = tmp_path / "llm"
    # Medium ran FIRST (earlier stamp) but sorts after "high" alphabetically;
    # later-run-wins must follow the timestamp, not the label.
    _make_run(
        results,
        "openrouter-qwen-qwen3.6-27b-medium-t6_gauntlet-20260719-010000",
        {
            "model": QWEN27B,
            "tier": "t6_gauntlet",
            "effort": "medium",
            "episodes": 2,
            "cost_usd_list_price": 5.40,
            "usage": {"output_tokens": 2_000_000},
        },
        {0: 0.2, 1: 0.3},
        output_tokens=1_000_000,
    )
    _make_run(
        results,
        "openrouter-qwen-qwen3.6-27b-high-t6_gauntlet-20260719-020000",
        {
            "model": QWEN27B,
            "tier": "t6_gauntlet",
            "effort": "high",
            "episodes": 2,
            "cost_usd_list_price": 10.80,
            "usage": {"output_tokens": 4_000_000},
        },
        {1: 0.9, 2: 0.5},
        output_tokens=2_000_000,
    )
    return results


# --------------------------------------------------------------------------
# M-2: same-config effort merge
# --------------------------------------------------------------------------


def test_same_config_merge_pools_efforts_with_later_wins_dedup(tmp_path: Path) -> None:
    results = _qwen_rerun_tree(tmp_path)
    cells, anomalies = bpa.load_run_groups(results)

    merged_key = (QWEN27B, "t6_gauntlet", "single-mode")
    assert set(cells) == {merged_key}, "medium/high cells must collapse into one"
    cell = cells[merged_key]
    assert cell["n"] == 3  # distinct seeds 0, 1, 2
    # Seed 1 appears under both labels; the later RUN (high, stamp 020000)
    # wins the dedup even though "high" < "medium" lexicographically.
    assert cell["mean"] == pytest.approx((0.2 + 0.9 + 0.5) / 3)
    assert cell["effort"] == "single-mode"
    assert cell["note"] == "pooled medium+high"
    # The intentional seed overlap across pooled labels is the documented
    # merge, not a declared-vs-recomputed anomaly.
    assert anomalies == []


def test_single_mode_ordered_after_max() -> None:
    assert bpa.EFFORT_ORDER["single-mode"] > bpa.EFFORT_ORDER["max"]
    # DeepSeek V4: medium and xhigh both map to high, the route's only level.
    deepseek = {
        (f"openrouter/deepseek/{model}", tier)
        for model in ("deepseek-v4-pro", "deepseek-v4-flash")
        for tier in ("t2_announced", "t6_gauntlet", "t7_perpetual", "t11_maelstrom")
    }
    assert set(bpa.SAME_CONFIG_MERGES) == {
        ("openrouter/qwen/qwen3.6-27b", "t6_gauntlet"),
        ("openrouter/qwen/qwen3.6-27b", "t2_announced"),
        ("openrouter/qwen/qwen3.6-35b-a3b", "t6_gauntlet"),
        ("openrouter/qwen/qwen3.6-35b-a3b", "t2_announced"),
    } | deepseek


def test_merged_cell_note_renders_plain_in_main_table(tmp_path: Path) -> None:
    results = tmp_path / "llm"
    _make_run(
        results,
        "openrouter-qwen-qwen3.6-27b-medium-t2_announced-20260719-010000",
        {
            "model": QWEN27B,
            "tier": "t2_announced",
            "effort": "medium",
            "episodes": 1,
            "cost_usd_list_price": 2.70,
            "usage": {"output_tokens": 1_000_000},
        },
        {0: 0.5},
    )
    cells, _ = bpa.load_run_groups(results)
    tex = bpa.main_results_table(cells)
    # The config cell carries only the effort label; which requested labels a
    # cell pooled is a per-ROW note in the tier column, so a group whose rows
    # pooled different label sets cannot inherit the first row's note.
    assert "/ single-mode &" in tex
    assert r"t2\_announced (pooled medium)" in tex
    assert "route ignores requested effort" not in tex
    assert "*" not in tex  # plain cell, no asterisk/footnote machinery


# --------------------------------------------------------------------------
# M5: cost follows the selection
# --------------------------------------------------------------------------


def test_cell_cost_is_the_selected_episodes_not_every_attempt(tmp_path: Path) -> None:
    """The reported defect: the table summed cost over every contributing run
    while scoring only the surviving episodes."""
    results = _qwen_rerun_tree(tmp_path)
    cell = bpa.load_run_groups(results)[0][(QWEN27B, "t6_gauntlet", "single-mode")]

    # Selected: seed 0 from the medium run (1M out) + seeds 1 and 2 from the
    # high run (2M out each) = 5M output tokens.
    assert cell["cost_usd"] == pytest.approx(5 * QWEN27B_OUT_PER_MTOK)
    assert cell["output_tokens"] == 5_000_000
    # All four attempts billed 6M output tokens; the superseded seed-1 medium
    # episode is real spend that no printed cell scores.
    assert cell["all_attempts"] == 4
    assert cell["all_attempt_cost_usd"] == pytest.approx(6 * QWEN27B_OUT_PER_MTOK)
    assert cell["superseded_cost_usd"] == pytest.approx(QWEN27B_OUT_PER_MTOK)
    # Sanity: the old rule (sum of run-level summary costs) is strictly larger.
    assert cell["cost_usd"] < 5.40 + 10.80


def test_per_episode_price_falls_back_to_the_runs_own_cost(tmp_path: Path) -> None:
    """A run with no per-episode usage telemetry must not price at zero."""
    results = tmp_path / "llm"
    run_dir = results / "openrouter-qwen-qwen3.6-27b-medium-t6_gauntlet-20260719-010000"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "model": QWEN27B,
                "tier": "t6_gauntlet",
                "effort": "medium",
                "episodes": 2,
                "cost_usd_list_price": 9.0,
                "usage": {"output_tokens": 0},
            }
        ),
        encoding="utf-8",
    )
    for index, seed in enumerate((0, 1)):
        (run_dir / f"seed{index:04d}.json").write_text(
            json.dumps({"result": {"seed": seed, "board_accuracy": 0.5}}),
            encoding="utf-8",
        )

    cells, anomalies = bpa.load_run_groups(results)
    cell = cells[(QWEN27B, "t6_gauntlet", "single-mode")]
    assert cell["cost_usd"] == pytest.approx(9.0)  # 2 episodes x an even 4.50 split
    assert any("fell back to an even split" in note for note in anomalies)


# --------------------------------------------------------------------------
# M5: cross-output equality -- every plotted cell equals its table cell
# --------------------------------------------------------------------------


def _assert_plot_matches_table(cells: dict[tuple[str, str, str], dict]) -> None:
    """Every figure value is the table value, and the printed row agrees."""
    plotted = bpa.plotted_matrix_cells(cells)
    ktok = bpa.plotted_cost_cells(cells)
    tex = bpa.main_results_table(cells)

    assert len(plotted) == sum(1 for cell in cells.values() if cell["n"])
    for (model, tier, effort), cell in cells.items():
        if not cell["n"]:
            continue
        figure_key = (tier, model, effort)
        # 1. The heatmap value IS the table's mean (exact float identity).
        assert plotted[figure_key] == cell["mean"]
        # 2. The cost panel divides the SELECTED tokens by the SELECTED n.
        assert ktok[figure_key] == cell["output_tokens"] / cell["n"] / 1000
        # 3. And the rendered table row prints those same numbers.
        row = (
            f"& {cell['n']} & {bpa.fmt_dec(plotted[figure_key])} $\\pm$ "
            f"{bpa.fmt_dec(cell['sem'])} & {bpa.fmt_money(cell['cost_usd'])} \\\\"
        )
        assert row in tex, f"no table row printing the plotted values for {figure_key}"


def test_plotted_cells_equal_table_cells_synthetic(tmp_path: Path) -> None:
    results = _qwen_rerun_tree(tmp_path)
    _make_run(
        results,
        "gpt-5.6-luna-max-t7_perpetual-20260713-090214",
        {
            "model": "gpt-5.6-luna",
            "tier": "t7_perpetual",
            "effort": "max",
            "episodes": 2,
            "cost_usd_list_price": 12.0,
            "usage": {"output_tokens": 2_000_000},
        },
        {0: 0.75, 1: 0.95},
    )
    cells, _ = bpa.load_run_groups(results)
    _assert_plot_matches_table(cells)


def test_plotted_cells_equal_table_cells_on_the_real_tree() -> None:
    cells, _ = bpa.load_run_groups(ROOT / "results" / "llm")
    assert len(cells) > 40, "expected the full pilot corpus"
    _assert_plot_matches_table(cells)


def test_previously_contradicting_cells_now_agree() -> None:
    """The three cells the 2026-07-27 review caught disagreeing.

    Before: the table printed .3846 / $9.9899 for q27b gauntlet while the
    figure plotted .3308, because the figure averaged all ten attempts and the
    cost summed both runs' summaries. Now both read one selection.
    """
    cells, _ = bpa.load_run_groups(ROOT / "results" / "llm")
    plotted = bpa.plotted_matrix_cells(cells)
    expected = {
        ("openrouter/qwen/qwen3.6-27b", "t6_gauntlet", "single-mode"): (0.3846, 2.9099),
        ("openrouter/qwen/qwen3.6-35b-a3b", "t2_announced", "single-mode"): (0.3333, 6.5741),
        ("openrouter/qwen/qwen3.6-35b-a3b", "t6_gauntlet", "single-mode"): (0.0846, 12.7783),
    }
    for (model, tier, effort), (mean, cost) in expected.items():
        cell = cells[(model, tier, effort)]
        assert round(cell["mean"], 4) == mean
        assert round(cell["cost_usd"], 4) == cost
        # The figure now plots the same number the table prints.
        assert plotted[(tier, model, effort)] == cell["mean"]
        # ...and strictly less than the old all-attempt cost, which is still
        # reported, just not as this cell's cost.
        assert cell["all_attempt_cost_usd"] > cell["cost_usd"]


def test_make_figures_does_no_selection_of_its_own() -> None:
    """The mechanism behind the equality: make_figures imports the selection
    instead of re-deriving one. Guard against a re-derivation creeping back."""
    source = (ROOT / "scripts" / "make_figures.py").read_text(encoding="utf-8")
    assert "from build_paper_assets import" in source
    assert "canonical_selection" in source
    for forbidden in ("summary.json", "board_accuracy", ".iterdir()", 'glob("seed'):
        assert forbidden not in source, f"make_figures re-derives the population via {forbidden}"


# --------------------------------------------------------------------------
# M5: campaign spend sees the no-summary partials
# --------------------------------------------------------------------------


def test_campaign_spend_prices_no_summary_partials(tmp_path: Path) -> None:
    results = _qwen_rerun_tree(tmp_path)
    partial = results / "openrouter-qwen-qwen3.6-27b-high-t2_announced-20260719-030000"
    partial.mkdir()
    (partial / "seed0000.json").write_text(
        json.dumps(
            {
                "result": {
                    "seed": 7,
                    "board_accuracy": 0.5,
                    "api_calls": 24,
                    "usage": {"input_tokens": 0, "output_tokens": 1_000_000},
                }
            }
        ),
        encoding="utf-8",
    )

    spend = bpa.tree_spend(results)
    assert spend["run_dirs"] == 3
    assert spend["no_summary_dirs"] == 1
    assert spend["no_summary_episodes"] == 1
    assert spend["no_summary_cost_usd"] == pytest.approx(QWEN27B_OUT_PER_MTOK)
    # 6M output tokens across the two summarized runs + 1M in the partial.
    assert spend["cost_usd"] == pytest.approx(7 * QWEN27B_OUT_PER_MTOK)
    assert spend["episodes"] == 5
    assert spend["unpriced"] == []

    # The summarized-subset view stays deliberately narrower, and the gap is
    # exactly the partial -- that gap is what the macros make legible.
    stats = bpa.snapshot_stats(results)
    assert stats["episodes"] == 4


def test_no_summary_partial_is_identified_from_its_directory_name() -> None:
    assert bpa.identity_from_dir_name(
        "gemini-gemini-3-flash-preview-high-t7_perpetual-20260719-214429"
    ) == ("gemini/gemini-3-flash-preview", "t7_perpetual", "high")
    assert bpa.identity_from_dir_name(
        "openrouter-deepseek-deepseek-v4-pro-medium-t6_gauntlet-20260719-013314"
    ) == ("openrouter/deepseek/deepseek-v4-pro", "t6_gauntlet", "medium")
    assert bpa.identity_from_dir_name("not-a-run-directory") == (None, None, None)


GEMINI_FLASH_PARTIALS = (
    "gemini-gemini-3-flash-preview-high-t11_maelstrom-20260719-221954",
    "gemini-gemini-3-flash-preview-high-t7_perpetual-20260719-214429",
)


def test_real_campaign_spend_exceeds_the_summarized_subset() -> None:
    """The falsifying arithmetic behind M5: the summarized-only total is not
    the campaign total, because completed no-summary partials are real.

    Deliberately relational, not a pinned dollar total -- the corpus grows.
    The two Gemini Flash partials the 2026-07-27 review priced at 8 episodes,
    237 calls and $6.971015 are checked by name, since those are historical
    artifacts that must never silently vanish from the accounting again.
    """
    results = ROOT / "results" / "llm"
    primary = bpa.tree_spend(results)
    summarized = bpa.snapshot_stats(results)
    records = {r.name: r for r in bpa.scan_run_dirs(results)}

    reviewed = [records[name] for name in GEMINI_FLASH_PARTIALS]
    assert all(not r.has_summary for r in reviewed), "these two must stay unsummarized"
    assert sum(len(r.seed_paths) for r in reviewed) == 8
    reviewed_cost = 0.0
    reviewed_calls = 0
    for record in reviewed:
        usage = {"input_tokens": 0, "output_tokens": 0}
        for path in record.seed_paths:
            result = json.loads(path.read_text(encoding="utf-8"))["result"]
            reviewed_calls += result["api_calls"]
            for field_name in usage:
                usage[field_name] += result["usage"][field_name]
        reviewed_cost += bpa.usage_cost(record.model, usage)
    assert reviewed_calls == 237
    assert reviewed_cost == pytest.approx(6.971015, abs=1e-6)

    # Campaign spend counts every stored run; the summarized subset does not.
    assert primary["unpriced"] == []
    assert primary["no_summary_dirs"] >= len(GEMINI_FLASH_PARTIALS)
    assert primary["episodes"] > summarized["episodes"]
    assert primary["api_calls"] > summarized["calls"]
    assert primary["cost_usd"] > summarized["cost_list"]
    # ...and the whole gap is the no-summary partials, nothing else.
    assert primary["cost_usd"] - primary["no_summary_cost_usd"] == pytest.approx(
        summarized["cost_list"], abs=1.0
    )


# --------------------------------------------------------------------------
# M5: the macros
# --------------------------------------------------------------------------


def test_snapshot_macros_separate_selected_from_campaign_spend(tmp_path: Path) -> None:
    results = _qwen_rerun_tree(tmp_path)
    cells, _ = bpa.load_run_groups(results)
    violations = bpa.violation_stats(results, cells)
    spend = {
        "arms": {
            name: bpa.tree_spend(results if name == "primary" else tmp_path / name)
            for name, _path in bpa.CAMPAIGN_TREES
        },
        "total_cost_usd": 0.0,
    }
    spend["total_cost_usd"] = sum(arm["cost_usd"] for arm in spend["arms"].values())
    tex = bpa.snapshot_macros_tex(bpa.snapshot_stats(results), violations, cells, spend)

    # Selected spend is what the table's cost column adds up to.
    assert "\\newcommand{\\selectedspend}{13.50}" in tex
    # Superseded spend is real money that no printed cell scores.
    assert "\\newcommand{\\supersededspend}{2.70}" in tex
    # Campaign spend is every stored run in the tree.
    assert "\\newcommand{\\campaignspendprimary}{16.20}" in tex
    for macro in (
        "\\campaignspendfragments",
        "\\campaignspendrobustness",
        "\\campaignspendstatic",
        "\\campaignspendallarms",
        "\\partialruns",
        "\\partialepisodes",
        "\\partialspend",
        "\\laddercost",
        "\\laddermodels",
    ):
        assert f"\\newcommand{{{macro}}}" in tex
    # The legacy summary-only macro survives, but is labelled as a subset so
    # nobody cites it as total collection cost again.
    assert "\\newcommand{\\snapshotcostlist}" in tex
    assert "SUMMARIZED SUBSET" in tex
    assert "Do not cite \\snapshotcostlist as total collection" in tex


def test_ladder_cost_is_one_canonical_attempt_per_cell() -> None:
    """The paper's 'replicates for about $72' claim, re-derived.

    The ladder is the gauntlet tier at each route's default effort. Its price
    must be the SELECTED episodes' cost -- the same runs whose accuracies the
    ladder prints -- not the all-attempt cost that the printed table used to
    carry. Asserted against the cells rather than a frozen dollar figure so
    the definition is what is pinned.
    """
    cells, _ = bpa.load_run_groups(ROOT / "results" / "llm")
    ladder = bpa.ladder_cost(cells)
    members = [
        cell
        for cell in cells.values()
        if cell["tier"] == bpa.LADDER_TIER
        and cell["effort"] in bpa.LADDER_DEFAULT_EFFORTS
    ]
    assert ladder["models"] == len(members) == 13
    assert ladder["episodes"] == sum(cell["n"] for cell in members)
    assert ladder["cost_usd"] == pytest.approx(sum(c["cost_usd"] for c in members))
    # Pricing the same ladder off all attempts inflates it: that gap is the
    # difference between "$73.73" and the old printed-table "$91.78".
    assert ladder["all_attempt_cost_usd"] > ladder["cost_usd"]
    assert ladder["cheapest"][0] == "openrouter/deepseek/deepseek-v4-flash"
    assert ladder["dearest"][0] == "claude-opus-4-8"


def test_real_macros_agree_with_the_functions_behind_them() -> None:
    """Every emitted spend macro must be the number its own function returns --
    no macro may carry a total that nothing recomputes."""
    results = ROOT / bpa.PRIMARY_TREE
    cells, _ = bpa.load_run_groups(results)
    spend = bpa.campaign_spend()
    tex = bpa.snapshot_macros_tex(
        bpa.snapshot_stats(results), bpa.violation_stats(results, cells), cells, spend
    )
    expected = {
        "campaignspendprimary": spend["arms"]["primary"]["cost_usd"],
        "campaignspendfragments": spend["arms"]["fragments"]["cost_usd"],
        "campaignspendrobustness": spend["arms"]["robustness"]["cost_usd"],
        "campaignspendstatic": spend["arms"]["static"]["cost_usd"],
        "campaignspendallarms": spend["total_cost_usd"],
        "selectedspend": sum(cell["cost_usd"] for cell in cells.values()),
        "supersededspend": sum(cell["superseded_cost_usd"] for cell in cells.values()),
        "partialspend": spend["arms"]["primary"]["no_summary_cost_usd"],
        "laddercost": bpa.ladder_cost(cells)["cost_usd"],
    }
    for macro, value in expected.items():
        assert f"\\newcommand{{\\{macro}}}{{{value:.2f}}}" in tex
    # The all-arms macro is the sum of its parts, so no arm can go missing.
    assert spend["total_cost_usd"] == pytest.approx(
        sum(arm["cost_usd"] for arm in spend["arms"].values())
    )
    # Selected + superseded + failed seeds' partial episodes is the
    # summarized part of primary campaign spend.
    primary = spend["arms"]["primary"]
    assert (
        expected["selectedspend"] + expected["supersededspend"] + primary["failed_seed_cost_usd"]
        == pytest.approx(primary["cost_usd"] - expected["partialspend"], abs=1e-6)
    )


# --------------------------------------------------------------------------
# Retained-first paper: harness-recorded prices, dated registry prices,
# failed-run spend, and the seed-paired harness comparison
# --------------------------------------------------------------------------


def test_episode_cost_prefers_the_harness_prices_then_the_dated_registry() -> None:
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    doc = {"result": {"usage": usage},
           "transcript": [{"role": "user"}, {"role": "assistant", "cost_usd_list": 0.25},
                          {"role": "assistant", "cost_usd_list": 0.5}]}
    assert bpa.episode_list_cost("gpt-5.6-luna", doc, "2026-09-25") == pytest.approx(0.75)
    # Without recorded prices the registry prices the usage on its own day:
    # luna's list fell 5x between the July and September snapshots.
    bare = {"result": {"usage": usage}, "transcript": [{"role": "assistant"}]}
    july = bpa.episode_list_cost("gpt-5.6-luna", bare, "2026-07-20")
    september = bpa.episode_list_cost("gpt-5.6-luna", bare, "2026-09-25")
    assert july == pytest.approx(1.0 + 6.0) and september == pytest.approx(0.2 + 1.2)
    assert bpa.run_date("gpt-5.6-luna-max-retained-t7_blitz-20260925-162302") == "2026-09-25"


def test_a_run_whose_every_episode_failed_is_still_billed(tmp_path: Path) -> None:
    tree = tmp_path / "retained"
    run = tree / "claude-opus-5-5-max-retained-t11_maelstrom-20260925-185758"
    run.mkdir(parents=True)
    (run / "failed_run.json").write_text(json.dumps(
        {"model": "claude-opus-5-5", "tier": "t11_maelstrom", "effort": "max",
         "cost_usd_list_price": 9.2805, "failed_seeds": {"0": "credit"}}), encoding="utf-8")
    spend = bpa.tree_spend(tree)
    assert spend["cost_usd"] == pytest.approx(9.2805)
    assert spend["no_summary_dirs"] == 1 and spend["unpriced"] == []
    assert bpa.identity_from_dir_name(run.name) == ("claude-opus-5-5", "t11_maelstrom", "max")


def test_harness_pairs_compare_the_same_seeds_only(tmp_path: Path) -> None:
    summary = {"model": "gpt-5.6-luna", "tier": "t6_gauntlet", "effort": "medium", "episodes": 3,
               "cost_usd_list_price": 1.0}
    _make_run(tmp_path / "retained", "gpt-5.6-luna-medium-retained-t6_gauntlet-20260925-120000",
              summary, {0: 0.8, 1: 0.6, 2: 1.0}, output_tokens=5_000)
    _make_run(tmp_path / "llm", "gpt-5.6-luna-medium-t6_gauntlet-20260712-120000",
              summary, {0: 0.4, 1: 0.6, 5: 0.2}, output_tokens=50_000)
    retained, _ = bpa.canonical_selection(tmp_path / "retained")
    visible, _ = bpa.canonical_selection(tmp_path / "llm")
    (row,) = bpa.harness_pairs(retained, visible)
    assert row["n"] == 2  # seeds 0 and 1 only
    assert row["delta"] == pytest.approx(0.2) and (row["up"], row["down"]) == (1, 0)
    assert row["visible_ktok"] / row["retained_ktok"] == pytest.approx(10.0)


def test_every_retained_call_reprices_at_its_days_list_price() -> None:
    """The primary tree's costs are the harness's per-request prices. Re-price
    every recorded call from its own usage on its collection day with the
    vendored registry: the two must agree. OpenRouter calls are exempt because
    their recorded cost is the route's own bill (usage.cost), not a registry
    price."""
    from blindboard.pricing import price_call

    checked = 0
    for record in bpa.scan_run_dirs(ROOT / bpa.PRIMARY_TREE):
        if record.model is None or record.model.startswith("openrouter/"):
            continue
        key = bpa.pricing_key(record.model)
        on = bpa.run_date(record.name)
        for path in record.seed_paths:
            doc = json.loads(path.read_text(encoding="utf-8"))
            for call in doc["transcript"]:
                if call.get("role") != "assistant" or call.get("cost_usd_list") is None:
                    continue
                expected = price_call(key, call.get("usage") or {}, on=on)
                assert expected == pytest.approx(call["cost_usd_list"], rel=1e-9, abs=1e-12), (
                    f"{record.name}/{path.name}: recorded {call['cost_usd_list']} vs {expected}"
                )
                checked += 1
    assert checked > 3000, "expected the retained corpus"


def _flip_doc(guess_triple: list[str], dual: bool = False) -> dict:
    """A 12-key board whose last rotation sent A, B, C on physical keys
    0, 6, 10 round the 3-cycle 0 -> 6 -> 10 -> 0, so the keys read C, A, B
    at the guess; the alternative the rotation could have produced is
    B, C, A. The guess is right everywhere else."""
    from blindboard.boards import BOARDS

    board = BOARDS["fingers12"]
    positions, moved_to = [0, 6, 10], {0: 6, 6: 10, 10: 0}
    after = {moved_to[p]: s for p, s in zip(positions, "ABC")}
    after.update(zip([p for p in range(board.n) if p not in positions], "DEFGHIJKL"))
    layout = {board.addresses[p]: s for p, s in after.items()}
    guess = dict(layout)
    guess.update({board.addresses[p]: g for p, g in zip(positions, guess_triple)})
    correct = sum(guess[a] == s for a, s in layout.items())
    rotation = {"mode": "announced", "positions": positions,
                "permutation": {str(k): v for k, v in moved_to.items()},
                "addresses": sorted(board.addresses[p] for p in positions)}
    return {
        "records": [{"turn": 20, "rotation": rotation, "relabel": None}],
        "result": {"config": {"dual": dual, "board": "fingers12"}, "guess": guess, "final_layout": layout,
                   "board_n": board.n, "board_accuracy": correct / board.n},
    }


def test_final_flip_is_scored_at_its_expectation() -> None:
    right, wrong = _flip_doc(["C", "A", "B"]), _flip_doc(["B", "C", "A"])
    assert bpa.final_flip(right) is True and bpa.final_flip(wrong) is False
    assert bpa.final_flip(_flip_doc(["C", "A", "A"])) is None
    assert bpa.final_flip(_flip_doc(["C", "A", "B"], dual=True)) is None
    # Called right: 12/12 becomes 10.5/12; called wrong (0 of the 3): 9/12 becomes 10.5/12.
    assert bpa.flip_neutral_accuracy(right) == pytest.approx(10.5 / 12)
    assert bpa.flip_neutral_accuracy(wrong) == pytest.approx(10.5 / 12)


def test_only_the_two_arrangements_a_derangement_can_leave_make_a_call() -> None:
    """All six arrangements of the right three symbols: the keys as they
    stood before the rotation, and the three single swaps, are wrong under
    either outcome, so they make no call and keep their raw score."""
    from itertools import permutations

    calls = {"".join(p): bpa.final_flip(_flip_doc(list(p))) for p in permutations("ABC")}
    assert calls == {"CAB": True, "BCA": False, "ABC": None, "ACB": None, "BAC": None, "CBA": None}
    for triple, called in calls.items():
        doc = _flip_doc(list(triple))
        if called is None:
            assert bpa.flip_neutral_accuracy(doc) == doc["result"]["board_accuracy"]


def test_every_stored_final_rotation_is_one_of_the_two_derangements() -> None:
    """Rebuilding the arrangement before the last rotation (relabelled
    addresses included) must give back a 3-cycle of it at the guess, in
    every stored episode of both campaigns."""
    checked = 0
    for tree in (bpa.PRIMARY_TREE, bpa.LEGACY_TREE):
        cells, _ = bpa.canonical_selection(ROOT / tree)
        for cell in cells.values():
            for ep in cell["episodes"]:
                doc = bpa._episode_doc(ep.path)
                arrangement = bpa.final_rotation(doc)
                if arrangement is None:
                    continue
                (a, b, c), after, _ = arrangement
                assert after in ([c, a, b], [b, c, a]), ep.path
                checked += 1
    assert checked > 200


def test_binomial_tail_matches_the_quoted_coin_flip_probability() -> None:
    assert bpa.binomial_upper_tail(20, 16) == pytest.approx(6196 / 2**20)
    assert bpa.binomial_upper_tail(10, 0) == pytest.approx(1.0)


def test_retention_growth_counts_replay_and_sets_unreplayed_turns_apart(tmp_path: Path) -> None:
    from types import SimpleNamespace

    def call(inp: int, out: int, **harness) -> dict:
        return {"role": "assistant", "text": "x", "usage": {"input_tokens": inp, "output_tokens": out},
                "harness": harness}

    user = {"role": "user", "text": "u" * 35}  # 10 tokens at 3.5 characters a token
    transcript = [
        user, call(100, 50),
        user, call(160, 40),  # grew 60 = 50 + 10: replayed
        user, call(180, 30, incomplete="max_output_tokens"),  # grew 20 < 40 + 10: dropped
        user, call(195, 60, rerouted=[{"usage": {"input_tokens": 90, "output_tokens": 25}}]),
        user, call(150, 5),  # after the incomplete turn: set apart
        # Re-sent to another host: its usage carries the dropped attempt's
        # 100 input tokens, and the kept prompt (160) grew only 10, the user text.
        user, call(260, 10, rerouted=[{"usage": {"input_tokens": 100, "output_tokens": 5}}]),
    ]
    path = tmp_path / "ep.json"
    path.write_text(json.dumps({"result": {}, "transcript": transcript}), encoding="utf-8")
    cells = {("gpt-5.6-luna", "t6_gauntlet", "medium"): {"episodes": [SimpleNamespace(path=path)]}}
    row = bpa.retention_growth(cells)["gpt-5.6-luna"]
    # Pairs: (1,2) passes; (2,3) fails (20 - 10 < 0.5 * 40); (3,4) is after an
    # incomplete turn; (4,5): kept attempt is 105 in / 35 out, 150 - 105 = 45 >= 10 + 17.5;
    # (5,6) fails: 260 - 100 - 150 = 10 is the user text alone, none of the 5 output.
    assert {k: row[k] for k in ("pairs", "passed", "after_incomplete")} == {"pairs": 4, "passed": 2, "after_incomplete": 1}
    assert row["reasoned"] == 0  # no call here records reasoning tokens


def test_reasoning_check_catches_a_replay_that_drops_only_the_reasoning(tmp_path: Path) -> None:
    """The red-team counterexample: 99 visible tokens and one of reasoning.
    Replaying the visible reply alone grows the prompt by 99, which passes
    the half-the-output check, but covers none of the reasoning."""
    from types import SimpleNamespace

    def call(inp: int, out: int, reasoning: int) -> dict:
        return {"role": "assistant", "text": "x",
                "usage": {"input_tokens": inp, "output_tokens": out, "reasoning_tokens": reasoning}}

    user = {"role": "user", "text": "u" * 35}  # 10 tokens
    transcript = [
        user, call(100, 100, 1),
        user, call(209, 100, 60),  # grew 109 = 10 + 99 visible: the 1 reasoning token dropped
        user, call(319, 10, 4),    # grew 110 = 10 + 40 visible + 60 reasoning: carried
    ]
    path = tmp_path / "ep.json"
    path.write_text(json.dumps({"result": {}, "transcript": transcript}), encoding="utf-8")
    cells = {("gpt-5.6-luna", "t6_gauntlet", "medium"): {"episodes": [SimpleNamespace(path=path)]}}
    row = bpa.retention_growth(cells)["gpt-5.6-luna"]
    assert row["pairs"] == 2 and row["passed"] == 2
    assert row["reasoned"] == 2 and row["reasoning_passed"] == 1 and row["reasoning_fail_max"] == 1
    assert row["reasoning_shares"] == pytest.approx([0.0, 1.0])


def test_snapshot_reasoning_reenters_except_below_the_checks_resolution() -> None:
    """The paper's reasoning check: almost every pair whose earlier call
    reasoned carries that reasoning, and the few that do not follow calls
    with less reasoning than the error in the user-text estimate."""
    cells, _ = bpa.canonical_selection(ROOT / bpa.PRIMARY_TREE)
    growth = bpa.retention_growth(cells)
    reasoned = sum(g["reasoned"] for g in growth.values())
    passed = sum(g["reasoning_passed"] for g in growth.values())
    assert reasoned > 8000 and reasoned - passed <= 10
    assert max(g["reasoning_fail_max"] for g in growth.values()) < 50


def test_legacy_cache_field_is_priced_as_a_cache_read() -> None:
    """The first retained luna runs recorded cached input as
    cached_input_tokens (red team, v0.14); priced as uncached input it
    overstated their cost. Usage copied from one affected episode."""
    legacy = {"cached_input_tokens": 109400, "input_tokens": 117531, "output_tokens": 5009,
              "reasoning_tokens": 4113, "total_tokens": 122540}
    named = {k: v for k, v in legacy.items() if k != "cached_input_tokens"} | {"cache_read_tokens": 109400}
    as_uncached = bpa.list_price_usd("gpt-5.6-luna", {k: v for k, v in named.items() if k != "cache_read_tokens"},
                                     on="2026-09-25")
    cost = bpa.usage_cost("gpt-5.6-luna", legacy, on="2026-09-25")
    assert cost == pytest.approx(bpa.usage_cost("gpt-5.6-luna", named, on="2026-09-25"))
    assert cost < as_uncached
    with pytest.raises(ValueError):
        bpa.normalize_usage({"cached_input_tokens": 5, "cache_read_tokens": 6, "input_tokens": 10})


def test_loss_split_counts_every_lost_key_once(tmp_path: Path) -> None:
    """A lost key is acquisition (never determined) or conversion
    (determined, guessed wrong): 10 keys, 6 right, 5 determined, 3 of those
    right, so 4 lost, 2 of each."""
    from types import SimpleNamespace

    path = tmp_path / "ep.json"
    path.write_text(json.dumps({
        "result": {"board_n": 10, "correct": 6, "board_accuracy": 0.6,
                   "diagnostics": {"deducible_at_guess": 5, "deduced_correct_in_guess": 3}},
        "transcript": []}), encoding="utf-8")
    cell = {"episodes": [SimpleNamespace(path=path)], "mean": 0.6, "sem": 0.0, "output_tokens": 0,
            "n": 1, "cost_usd": 0.0}
    values = bpa.cell_values(cell)
    assert values["acqlosspct"] == "50" and values["convlosspct"] == "50"


def test_only_perpetual_churn_tiers_changed_prompt_text() -> None:
    assert bpa.schedule_text_changed("t9_numrow") and bpa.schedule_text_changed("t11_maelstrom")
    assert not bpa.schedule_text_changed("t6_gauntlet") and not bpa.schedule_text_changed("t0_frozen")


def test_derangement_audit_runs_and_stays_near_a_fair_coin() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from derangement_structure_audit import run_audit

    audit = run_audit(episodes=60)
    assert audit["rotations"] > 5_000 and len(audit["tests"]) == 25
    assert audit["max_abs_z"] < 4.5


def test_certificates_reproduce_on_held_out_seeds() -> None:
    """The stored held-out run (fresh seeds) covers every played tier and
    agrees with each shipped certificate within three standard errors."""
    held = bpa.heldout_calibration()
    assert held is not None and held["seed_start"] >= 1000
    certs = bpa.load_calibration_best(ROOT / "results" / "calibration")
    assert {r["tier"] for r in held["rows"]} >= {"t2_announced", "t6_gauntlet", "t7_perpetual", "t11_maelstrom"}
    for row in held["rows"]:
        assert row["certificate"] == pytest.approx(certs[row["tier"]])
        assert abs(row["z"]) < 3, row


def test_every_retained_turn_reenters_the_next_prompt() -> None:
    """The paper says the check passes on every pair of consecutive calls in
    the snapshot; keep that true as the corpus grows."""
    cells, _ = bpa.canonical_selection(ROOT / bpa.PRIMARY_TREE)
    growth = bpa.retention_growth(cells)
    assert sum(g["pairs"] for g in growth.values()) > 8000
    failing = {model: g for model, g in growth.items() if g["passed"] != g["pairs"]}
    assert not failing, failing


def test_cell_grid_marks_pooled_labels_and_off_rule_n(tmp_path: Path) -> None:
    """The every-cell grid: one row per configuration, SEM beneath the mean,
    pooled requested labels as a superscript, and n only where it breaks the
    20-on-announced, 5-elsewhere rule."""
    summary = {"model": QWEN27B, "tier": "t6_gauntlet", "effort": "medium", "episodes": 5,
               "cost_usd_list_price": 1.0}
    _make_run(tmp_path / "llm", "openrouter-qwen-qwen3.6-27b-medium-t6_gauntlet-20260719-010000",
              summary, {s: 0.5 for s in range(5)})
    summary_high = {**summary, "effort": "high"}
    _make_run(tmp_path / "llm", "openrouter-qwen-qwen3.6-27b-high-t6_gauntlet-20260719-020000",
              summary_high, {0: 0.25})
    summary_t2 = {**summary, "tier": "t2_announced"}
    _make_run(tmp_path / "llm", "openrouter-qwen-qwen3.6-27b-medium-t2_announced-20260719-030000",
              summary_t2, {s: 1.0 for s in range(5)})
    cells, _ = bpa.load_run_groups(tmp_path / "llm")
    tex = bpa.cell_grid_table(cells)
    lines = tex.splitlines()
    rows = [i for i, line in enumerate(lines) if "qwen3.6-27b" in line]
    assert len(rows) == 2  # one row in the 12-key block, one in the 26-key block
    assert all("/ single-mode" in lines[i] for i in rows)
    assert any("$^{\\mathrm{mh}}$" in lines[i] for i in rows)  # the gauntlet cell pooled medium and high
    sem_rows = [lines[i + 1] for i in rows]
    assert "$_{n=5}$" in sem_rows[0]  # t2_announced expects 20
    assert sum(row.count("$_{n=") for row in sem_rows) == 1


def test_prose_lookups_render_the_table_values() -> None:
    """Every \\bbacc lookup the paper can cite equals the grid's printed mean."""
    cells, _ = bpa.canonical_selection(ROOT / bpa.PRIMARY_TREE)
    legacy, _ = bpa.canonical_selection(ROOT / bpa.LEGACY_TREE)
    pairs = bpa.harness_pairs(cells, legacy)
    tex = bpa.cell_macros_tex(cells, legacy, pairs)
    for (model, tier, effort), cell in cells.items():
        assert f"\\bb@def{{acc}}{{{model}}}{{{tier}}}{{{effort}}}{{{bpa.fmt3(cell['mean'])}}}" in tex
    for row in pairs:
        assert f"\\bb@def{{delta}}{{{row['model']}}}{{{row['tier']}}}{{{row['effort']}}}{{{bpa.signed(row['delta'])}}}" in tex
    assert "\\newcommand{\\twominacc}{" in tex and "\\newcommand{\\bbcert}[1]" in tex


def test_signed_never_prints_negative_zero() -> None:
    assert bpa.signed(-1e-12) == "+.000"
    assert bpa.signed_cell(-0.0004) == "$+.000$"
    assert bpa.signed(-0.0006) == "-.001"
