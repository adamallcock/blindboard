"""Fix KS-5/M12 and kill-shot K3: the snapshot freeze is an explicit hash
manifest covering EVERYTHING, not a promise to flip a Boolean later.

`build_paper_assets.py --freeze` writes paper/snapshot_manifest.json pinning
every run dir in every covered results tree -- classified `included` (has a
summary.json) or `excluded-no-summary` (a completed run that never wrote one)
-- plus the generator's git commit. With SNAPSHOT_FROZEN=True the builder
fails closed on any drift: missing dir, changed hash, changed classification,
or ANY unlisted run dir, with or without a summary.

The K3 finding this file now guards: the old manifest skipped summary-less
directories, so two completed Gemini Flash partials (8 episodes, 237 calls,
$6.97) would have been silently absent from a "frozen" snapshot."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_paper_assets as bpa  # noqa: E402


def _in_git_checkout() -> bool:
    try:
        proc = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT,
                              capture_output=True, text=True, check=False)
    except OSError:
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _make_run(
    results_dir: Path,
    name: str,
    model: str = "gpt-5.6-luna",
    tier: str = "t6_gauntlet",
    effort: str = "max",
    seeds: dict[int, float] | None = None,
    with_summary: bool = True,
) -> Path:
    run_dir = results_dir / name
    run_dir.mkdir(parents=True)
    seeds = {0: 0.9, 1: 0.8} if seeds is None else seeds
    if with_summary:
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "model": model,
                    "tier": tier,
                    "effort": effort,
                    "episodes": len(seeds),
                    "cost_usd_list_price": 1.0,
                    "usage": {"output_tokens": 1000},
                }
            ),
            encoding="utf-8",
        )
    for index, (seed, acc) in enumerate(sorted(seeds.items())):
        (run_dir / f"seed{index:04d}.json").write_text(
            json.dumps(
                {
                    "result": {
                        "seed": seed,
                        "board_accuracy": acc,
                        "api_calls": 26,
                        "usage": {"input_tokens": 1000, "output_tokens": 100_000},
                    }
                }
            ),
            encoding="utf-8",
        )
    return run_dir


def _freeze(results_dir: Path, manifest_path: Path, extra_trees=()) -> dict:
    manifest = bpa.build_snapshot_manifest(results_dir, extra_trees)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _entry(manifest: dict, dir_name: str) -> dict:
    return next(run for run in manifest["runs"] if run["dir_name"] == dir_name)


# --------------------------------------------------------------------------
# What the manifest contains
# --------------------------------------------------------------------------


def test_freeze_then_check_passes(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _make_run(results, "gpt-5.6-luna-max-t6_gauntlet-20260712-230714")
    _make_run(results, "gpt-5.6-sol-max-t6_gauntlet-20260712-230711", model="gpt-5.6-sol")
    manifest_path = tmp_path / "snapshot_manifest.json"
    manifest = _freeze(results, manifest_path)

    assert [run["dir_name"] for run in manifest["runs"]] == sorted(
        run["dir_name"] for run in manifest["runs"]
    )
    assert all(
        run["seed_file_names_sorted"] == ["seed0000.json", "seed0001.json"]
        for run in manifest["runs"]
    )
    # Unchanged tree: the freeze check passes without raising.
    bpa.enforce_snapshot_freeze(results, manifest_path)


def test_manifest_records_no_summary_runs_as_excluded(tmp_path: Path) -> None:
    """K3: a completed run with no summary.json is EVIDENCE. It is pinned with
    hashes and classified, never skipped."""
    results = tmp_path / "results"
    _make_run(results, "gpt-5.6-luna-max-t6_gauntlet-20260712-230714")
    _make_run(
        results,
        "gemini-gemini-3-flash-preview-high-t7_perpetual-20260719-214429",
        with_summary=False,
        seeds={1: 0.4, 2: 0.5, 3: 0.6, 4: 0.7},
    )
    manifest = bpa.build_snapshot_manifest(results)

    assert manifest["included"] == 1
    assert manifest["excluded_no_summary"] == 1
    partial = _entry(manifest, "gemini-gemini-3-flash-preview-high-t7_perpetual-20260719-214429")
    assert partial["status"] == "excluded-no-summary"
    assert partial["summary_sha256"] is None
    # Hashed like any other run: content drift in a partial is still drift.
    assert partial["seed_file_names_sorted"] == [
        "seed0000.json",
        "seed0001.json",
        "seed0002.json",
        "seed0003.json",
    ]
    assert len(partial["seed_files_sha256"]) == 64
    # Identified from its directory name alone, since there is no summary.
    assert partial["model"] == "gemini/gemini-3-flash-preview"
    assert partial["tier"] == "t7_perpetual"
    assert partial["effort"] == "high"
    # And priced, so the freeze names what the partial cost.
    assert partial["episodes"] == 4
    assert partial["api_calls"] == 104
    assert partial["cost_usd_list_price"] > 0


@pytest.mark.skipif(not _in_git_checkout(), reason="needs a git checkout: the manifest records the generator commit")
def test_manifest_records_generator_commit() -> None:
    """A frozen snapshot that cannot name its generator is not frozen."""
    git = bpa.generator_git_commit()
    assert set(git) == {"commit", "dirty"}
    assert git["commit"] is not None and len(git["commit"]) == 40


def test_manifest_covers_extra_trees(tmp_path: Path) -> None:
    """The robustness arm is cited by the paper, so the freeze covers it too --
    and its runs nest one level down under a campaign folder."""
    primary = tmp_path / "llm"
    robustness = tmp_path / "robustness"
    _make_run(primary, "gpt-5.6-luna-max-t6_gauntlet-20260712-230714")
    _make_run(robustness / "bb-r2-pilot", "gpt-5.6-luna-max-t7_perpetual-20260720-131945")

    manifest = bpa.build_snapshot_manifest(primary, [robustness])
    assert sorted(manifest["trees"]) == ["llm", "robustness"]
    nested = _entry(manifest, "bb-r2-pilot/gpt-5.6-luna-max-t7_perpetual-20260720-131945")
    assert nested["tree"] == "robustness"


# --------------------------------------------------------------------------
# What the freeze refuses
# --------------------------------------------------------------------------


def test_mutated_seed_file_fails_closed(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run_dir = _make_run(results, "gpt-5.6-luna-max-t6_gauntlet-20260712-230714")
    manifest_path = tmp_path / "snapshot_manifest.json"
    _freeze(results, manifest_path)

    seed_path = run_dir / "seed0000.json"
    seed_path.write_text(
        json.dumps({"result": {"seed": 0, "board_accuracy": 1.0}}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="seed file contents differ"):
        bpa.enforce_snapshot_freeze(results, manifest_path)


def test_mutated_summary_fails_closed(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run_dir = _make_run(results, "gpt-5.6-luna-max-t6_gauntlet-20260712-230714")
    manifest_path = tmp_path / "snapshot_manifest.json"
    _freeze(results, manifest_path)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary["cost_usd_list_price"] = 99.0
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"summary\.json differs"):
        bpa.enforce_snapshot_freeze(results, manifest_path)


def test_unmanifested_dir_fails_closed(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _make_run(results, "gpt-5.6-luna-max-t6_gauntlet-20260712-230714")
    manifest_path = tmp_path / "snapshot_manifest.json"
    _freeze(results, manifest_path)

    _make_run(results, "gpt-5.6-sol-max-t7_blitz-20260713-195143", model="gpt-5.6-sol", tier="t7_blitz")
    with pytest.raises(SystemExit, match="unmanifested run dir.*t7_blitz"):
        bpa.enforce_snapshot_freeze(results, manifest_path)


def test_unmanifested_dir_without_summary_fails_closed(tmp_path: Path) -> None:
    """K3, the exact hole: an unlisted run dir fails the freeze whether or not
    it has a summary.json. Previously a summary-less partial was tolerated,
    which let real completed episodes appear after a snapshot was named."""
    results = tmp_path / "results"
    _make_run(results, "gpt-5.6-luna-max-t6_gauntlet-20260712-230714")
    manifest_path = tmp_path / "snapshot_manifest.json"
    _freeze(results, manifest_path)

    _make_run(
        results,
        "gpt-5.6-sol-max-t7_blitz-20260713-195143",
        with_summary=False,
    )
    with pytest.raises(SystemExit, match="unmanifested run dir \\(excluded-no-summary\\)"):
        bpa.enforce_snapshot_freeze(results, manifest_path)


def test_partial_gaining_a_summary_fails_closed(tmp_path: Path) -> None:
    """Reclassification is drift: a dir the snapshot named as an excluded
    partial must not quietly become a scored cell."""
    results = tmp_path / "results"
    run_dir = _make_run(
        results, "gpt-5.6-luna-max-t6_gauntlet-20260712-230714", with_summary=False
    )
    manifest_path = tmp_path / "snapshot_manifest.json"
    _freeze(results, manifest_path)

    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "model": "gpt-5.6-luna",
                "tier": "t6_gauntlet",
                "effort": "max",
                "episodes": 2,
                "cost_usd_list_price": 1.0,
                "usage": {"output_tokens": 1000},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="classification changed"):
        bpa.enforce_snapshot_freeze(results, manifest_path)


def test_missing_manifested_dir_fails_closed(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run_dir = _make_run(results, "gpt-5.6-luna-max-t6_gauntlet-20260712-230714")
    manifest_path = tmp_path / "snapshot_manifest.json"
    _freeze(results, manifest_path)

    for child in run_dir.iterdir():
        child.unlink()
    run_dir.rmdir()
    with pytest.raises(SystemExit, match="manifested run dir missing"):
        bpa.enforce_snapshot_freeze(results, manifest_path)


def test_drift_in_an_extra_tree_fails_closed(tmp_path: Path) -> None:
    primary = tmp_path / "llm"
    robustness = tmp_path / "robustness"
    _make_run(primary, "gpt-5.6-luna-max-t6_gauntlet-20260712-230714")
    _make_run(robustness / "bb-r2-pilot", "gpt-5.6-luna-max-t7_perpetual-20260720-131945")
    manifest_path = tmp_path / "snapshot_manifest.json"
    _freeze(primary, manifest_path, [robustness])

    _make_run(robustness / "bb-r2-pilot", "gpt-5.6-sol-max-t11_maelstrom-20260720-131947")
    with pytest.raises(SystemExit, match="unmanifested run dir.*t11_maelstrom"):
        bpa.enforce_snapshot_freeze(primary, manifest_path, [robustness])


def test_missing_manifest_fails_closed_and_says_run_freeze(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _make_run(results, "gpt-5.6-luna-max-t6_gauntlet-20260712-230714")
    with pytest.raises(SystemExit) as excinfo:
        bpa.enforce_snapshot_freeze(results, tmp_path / "snapshot_manifest.json")
    assert "--freeze" in str(excinfo.value)


def test_freeze_cli_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results = tmp_path / "results"
    _make_run(results, "gpt-5.6-luna-max-t6_gauntlet-20260712-230714")
    _make_run(results, "gpt-5.6-sol-max-t7_blitz-20260713-195143", with_summary=False)
    manifest_path = tmp_path / "paper" / "snapshot_manifest.json"
    monkeypatch.setattr(bpa, "SNAPSHOT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(bpa, "SNAPSHOT_EXTRA_TREES", ())

    assert bpa.main(["--freeze", "--results", str(results)]) == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Both dirs pinned -- the partial is not dropped just because it has no
    # summary.json to hash.
    assert [run["dir_name"] for run in manifest["runs"]] == [
        "gpt-5.6-luna-max-t6_gauntlet-20260712-230714",
        "gpt-5.6-sol-max-t7_blitz-20260713-195143",
    ]
    assert manifest["included"] == 1 and manifest["excluded_no_summary"] == 1
    bpa.enforce_snapshot_freeze(results, manifest_path)


def test_the_snapshot_is_frozen_and_the_tree_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The single freeze gate both the builder and make_figures call. The v1
    snapshot is frozen, and the shipped results trees match its manifest;
    while a snapshot is open the gate must not raise. There must be exactly
    one switch."""
    assert bpa.SNAPSHOT_FROZEN is True
    assert bpa.SNAPSHOT_MANIFEST_PATH.is_file()
    bpa.enforce_freeze_if_frozen((ROOT / bpa.PRIMARY_TREE).resolve())
    monkeypatch.setattr(bpa, "SNAPSHOT_FROZEN", False)
    bpa.enforce_freeze_if_frozen(tmp_path / "does-not-exist")
    figures_src = (ROOT / "scripts" / "make_figures.py").read_text(encoding="utf-8")
    assert "enforce_freeze_if_frozen" in figures_src
    # No second switch: make_figures must not DEFINE any snapshot-scope
    # constant of its own (prose describing the removal is fine).
    assert not re.findall(r"(?m)^SNAPSHOT_\w+\s*=", figures_src)
