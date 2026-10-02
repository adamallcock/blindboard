# Vendored from benchkit@0.3.10 (sha256:cb59196ad80f772b). Do not hand-edit; run 'benchkit sync' to update.
"""Figure style constants + helpers (validated palette, dataviz method).

Single source of truth for paper figures across the benchmark repos:
- warm off-white surface, near-black text, recessive grid;
- a validated 4-slot categorical palette in FIXED order (assign series to
  slots by a stable order, never ad hoc);
- a single-hue sequential blue ramp (100 -> 700) for magnitude;
- thin marks, direct labels for low-contrast series (relief rule),
  never dual axes (use stacked shared-x panels instead).

Values were validated in benchmark-qwerty make_figures.py; matplotlib is
imported lazily so vendoring this module adds no hard dependency.
"""

from __future__ import annotations

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e3e2de"

# Fixed categorical order, validated (blue, green, amber, dark green).
CATEGORICAL: tuple[str, ...] = ("#2a78d6", "#1baf7a", "#eda100", "#008300")

# Single-hue sequential ramp, blue 100 -> 700.
SEQ_RAMP: tuple[str, ...] = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

SAVE_FORMATS = ("svg", "pdf", "png")
SAVE_DPI = 170


def categorical_map(keys: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Assign palette slots to series keys in the given (stable) order."""
    if len(keys) > len(CATEGORICAL):
        raise ValueError(
            f"{len(keys)} series but only {len(CATEGORICAL)} validated categorical slots"
        )
    return {key: CATEGORICAL[i] for i, key in enumerate(keys)}


def seq_cmap():
    """LinearSegmentedColormap over SEQ_RAMP (matplotlib imported lazily)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq", list(SEQ_RAMP))


def style_axes(ax) -> None:
    """House axis style: surface background, no top/right spines, recessive
    grid below the data, muted tick labels."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def save_figure(fig, out_dir, name: str, *, formats=SAVE_FORMATS, dpi: int = SAVE_DPI) -> None:
    """Save svg+pdf+png with the house export settings."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out / f"{name}.{ext}", dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
