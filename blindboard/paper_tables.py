# Vendored from benchkit@0.3.10 (sha256:3e5988c7cf405217). Do not hand-edit; run 'benchkit sync' to update.
"""Table-cell and prose-macro number formats for generated paper tables.

One convention across every table a paper generates: bounded quantities as
.xxx (no leading zero); signed deltas and interval bounds in math mode, so
the minus is a true minus ($+.167$, $-.196$); p-values as .xxx or <.001,
padded so decimal points line up in centered columns. Prose macros carry
their relation, so a paper writes $p\\macro$ and gets p=.013 or p<.001.
"""

from __future__ import annotations

from collections.abc import Sequence


def _round3(x: float) -> float:
    """Three decimals, and a value that rounds to zero is +0.0: formatting
    -0.0004 directly would print a negative zero."""
    return round(x, 3) + 0.0


def fmt3(x: float) -> str:
    """.xxx without a leading zero, sign kept: 0.419 -> .419, -0.196 -> -.196."""
    s = f"{_round3(x):.3f}"
    return s.replace("0.", ".", 1) if s.lstrip("-").startswith("0.") else s


def pval(x: float) -> str:
    """p-value for a table cell: <.001 below the three-decimal floor."""
    return "<.001" if x < 0.001 else fmt3(x)


def prose_p(x: float) -> str:
    """p-value with its relation, for a prose macro used as $p\\macro$:
    {=}.013 or {<}.001."""
    return "{<}.001" if x < 0.001 else "{=}" + fmt3(x)


def signed(x: float) -> str:
    """Signed .xxx without math delimiters: 0.167 -> +.167, -0.196 -> -.196;
    a value that rounds to zero is +.000."""
    s = f"{_round3(x):+.3f}"
    return s[0] + s[2:] if s[1:].startswith("0.") else s


def signed_cell(x: float) -> str:
    """Signed delta for a table cell, in math mode: $+.167$ / $-.196$."""
    return f"${signed(x)}$"


def ci_cell(ci: Sequence[float]) -> str:
    """A (low, high) interval for a table cell: $[-.295, -.097]$."""
    return f"$[{signed(ci[0])}, {signed(ci[1])}]$"


def p_cell(x: float) -> str:
    """p-value for a centered table column, padded to the width of '<.001'
    so decimal points line up under any header: \\phantom{<}.045, <.001."""
    s = pval(x)
    return rf"\phantom{{<}}{s}" if s.startswith(".") else s


__all__ = ["ci_cell", "fmt3", "p_cell", "prose_p", "pval", "signed", "signed_cell"]
