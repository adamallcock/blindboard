# Vendored from benchkit@0.3.10 (sha256:3c24a1df4f56a75b). Do not hand-edit; run 'benchkit sync' to update.
"""THE pricing registry (published list prices, dated) + the reporting rule.

Reporting rule (user decision 2026-09-25, refining 2026-07-11): a reported
cost is what the provider's published STANDARD-tier price list charges for
the tokens actually used, on the date they were used. That includes the
published cache-read, cache-write and long-context rates. The ONLY
correction from the real bill is the service tier: runs collected on a
discounted tier (flex) are priced at the standard tier.

Every entry records the date its price was verified and the date it took
effect. ``PRICING`` holds each model's earliest verified price and remains
the default for callers that pass no date, so historical reports reproduce
unchanged; later price changes live in ``PRICE_HISTORY``. Unknown models,
dates inside an unresolved price-change window, and token classes with no
published rate price as None -- never a guess.

Usage classes: ``input_tokens`` is the TOTAL prompt; ``cache_read_tokens``
and ``cache_write_tokens`` are subsets of it (the released obviousbench
convention). Uncached input = input - reads - writes.

Seeded 2026-07-16 from the union of the copybench table
(scripts/research/build_report_tables.py) and the blindboard client table
(blindboard/client.py). Prices are USD per 1M tokens.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriceEntry:
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    verified: str  # ISO date the list price was last checked
    note: str = ""
    # Published cache classes; None = no published rate for this class.
    cache_read_usd_per_mtok: float | None = None
    cache_write_usd_per_mtok: float | None = None
    # Published long-context tier: a REQUEST whose input exceeds the
    # threshold bills all its tokens at the long rates.
    long_context_threshold_tokens: int | None = None
    long_input_usd_per_mtok: float | None = None
    long_cache_read_usd_per_mtok: float | None = None
    long_cache_write_usd_per_mtok: float | None = None
    long_output_usd_per_mtok: float | None = None
    # ISO date the price took effect ("" = earliest price held). Dates on or
    # after ``unresolved_from`` and before ``effective_from`` fall between two
    # archived snapshots with different prices and are unpriceable.
    effective_from: str = ""
    unresolved_from: str = ""

    @property
    def pair(self) -> tuple[float, float]:
        return (self.input_usd_per_mtok, self.output_usd_per_mtok)


PRICING: dict[str, PriceEntry] = {
    # OpenAI standard tier, as archived at developers.openai.com/api/docs/
    # pricing on 2026-07-09 and 2026-07-29 (web.archive.org). GPT-5.6: cache
    # writes 1.25x and reads 0.1x input; prompts >272K input tokens bill the
    # whole request at the long rates (model page + prompt-caching guide).
    # GPT-5.4: implicit caching only, "no additional cache-write charge", so
    # written tokens bill at the uncached input rate; no long-context tier.
    "gpt-5.4-nano": PriceEntry(
        0.20, 1.25, "2026-09-25",
        cache_read_usd_per_mtok=0.02, cache_write_usd_per_mtok=0.20,
    ),
    "gpt-5.4-mini": PriceEntry(
        0.75, 4.50, "2026-09-25",
        cache_read_usd_per_mtok=0.075, cache_write_usd_per_mtok=0.75,
    ),
    "gpt-5.5": PriceEntry(5.0, 30.0, "2026-07-11"),
    "gpt-5.6-sol": PriceEntry(
        5.0, 30.0, "2026-09-25",
        cache_read_usd_per_mtok=0.50, cache_write_usd_per_mtok=6.25,
        long_context_threshold_tokens=272_000,
        long_input_usd_per_mtok=10.0, long_cache_read_usd_per_mtok=1.00,
        long_cache_write_usd_per_mtok=12.50, long_output_usd_per_mtok=45.0,
    ),
    "gpt-5.6-terra": PriceEntry(
        2.5, 15.0, "2026-09-25",
        cache_read_usd_per_mtok=0.25, cache_write_usd_per_mtok=3.125,
        long_context_threshold_tokens=272_000,
        long_input_usd_per_mtok=5.0, long_cache_read_usd_per_mtok=0.50,
        long_cache_write_usd_per_mtok=6.25, long_output_usd_per_mtok=22.50,
    ),
    # OpenAI GPT-6 standard tier, verified 2026-09-25 against the raw pricing
    # page and each model page: "Prompts with more than 272K input tokens are
    # priced at 2x input and cache rates and 1.5x output for the full
    # request. Cache writes are billed at 1.25x the uncached input token
    # rate." No gpt-6-terra exists.
    "gpt-6-astra": PriceEntry(
        10.0, 50.0, "2026-09-25",
        cache_read_usd_per_mtok=1.00, cache_write_usd_per_mtok=12.50,
        long_context_threshold_tokens=272_000,
        long_input_usd_per_mtok=20.0, long_cache_read_usd_per_mtok=2.00,
        long_cache_write_usd_per_mtok=25.0, long_output_usd_per_mtok=75.0,
    ),
    "gpt-6-sol": PriceEntry(
        2.0, 10.0, "2026-09-25",
        cache_read_usd_per_mtok=0.20, cache_write_usd_per_mtok=2.50,
        long_context_threshold_tokens=272_000,
        long_input_usd_per_mtok=4.0, long_cache_read_usd_per_mtok=0.40,
        long_cache_write_usd_per_mtok=5.0, long_output_usd_per_mtok=15.0,
    ),
    "gpt-6-luna": PriceEntry(
        0.10, 0.50, "2026-09-25",
        cache_read_usd_per_mtok=0.01, cache_write_usd_per_mtok=0.125,
        long_context_threshold_tokens=272_000,
        long_input_usd_per_mtok=0.20, long_cache_read_usd_per_mtok=0.02,
        long_cache_write_usd_per_mtok=0.25, long_output_usd_per_mtok=0.75,
    ),
    "gpt-5.6-luna": PriceEntry(
        1.0, 6.0, "2026-09-25",
        cache_read_usd_per_mtok=0.10, cache_write_usd_per_mtok=1.25,
        long_context_threshold_tokens=272_000,
        long_input_usd_per_mtok=2.0, long_cache_read_usd_per_mtok=0.20,
        long_cache_write_usd_per_mtok=2.50, long_output_usd_per_mtok=9.00,
    ),
    # Anthropic (direct API). Rates with cache classes verified 2026-09-25
    # against the raw page platform.claude.com/docs/en/about-claude/pricing.md.
    # cache_write is the 5-minute TTL rate (1.25x input); 1-hour writes (2x)
    # have no field here, so sessions that write them price as None. No
    # long-context tier: 4.6+ models get the full 1M window at these rates.
    "anthropic/claude-opus-5-5": PriceEntry(
        4.0, 20.0, "2026-09-25",
        cache_read_usd_per_mtok=0.20, cache_write_usd_per_mtok=5.0,
        note="cache hits are 0.05x input on Opus 5.5; first listed 2026-09-22/23",
    ),
    "anthropic/claude-opus-4-8": PriceEntry(
        5.0, 25.0, "2026-09-25",
        cache_read_usd_per_mtok=0.50, cache_write_usd_per_mtok=6.25,
    ),
    "anthropic/claude-opus-4-7": PriceEntry(5.0, 25.0, "2026-07-18"),
    "anthropic/claude-fable-5": PriceEntry(10.0, 50.0, "2026-07-11"),
    "anthropic/claude-sonnet-5": PriceEntry(
        2.0, 10.0, "2026-09-25",
        cache_read_usd_per_mtok=0.20, cache_write_usd_per_mtok=2.50,
        note=(
            "$2/$10 was charged every day since launch (2026-06-30); the page "
            "called it introductory until 2026-08-11 and the scheduled $3/$15 "
            "was cancelled 2026-08-12, so $3/$15 was never in effect"
        ),
    ),
    "anthropic/claude-haiku-4-5": PriceEntry(
        1.0, 5.0, "2026-09-25",
        cache_read_usd_per_mtok=0.10, cache_write_usd_per_mtok=1.25,
    ),
    # Google (direct API). gemini-3.5-pro is UNRELEASED as of 2026-07-18
    # (three missed launch deadlines; no API, no confirmed pricing).
    "gemini/gemini-3.5-flash": PriceEntry(1.5, 9.0, "2026-07-18"),
    "gemini/gemini-3.1-pro-preview": PriceEntry(2.0, 12.0, "2026-07-18"),
    "gemini/gemini-3-flash-preview": PriceEntry(
        0.5, 3.0, "2026-07-19",
        note="verified via OpenRouter models API; Adam chose this over 3.5-flash on cost",
    ),
    # Verified 2026-09-25 against the raw page ai.google.dev/gemini-api/docs/pricing
    # (identical in the 2026-09-02, -12 and -20 snapshots). Output includes
    # thinking tokens; implicit caching has no write charge; no long-context tier.
    "gemini/gemini-3.8-flash": PriceEntry(
        0.75, 3.75, "2026-09-25",
        cache_read_usd_per_mtok=0.075, effective_from="2026-09-02",
        note="rate through 2026-12-31; the page lists $1.50/$7.50 from 2027-01-01",
    ),
    # DeepSeek (direct API; api-docs.deepseek.com, cache-miss input rate)
    "deepseek/deepseek-v4-pro": PriceEntry(0.435, 0.87, "2026-07-18"),
    "deepseek/deepseek-v4-flash": PriceEntry(0.14, 0.28, "2026-07-18",
        note="OpenRouter routes cheaper (0.098/0.196) via third-party hosts"),
    # Via OpenRouter (list prices, verified against the models API on the date shown)
    "openrouter/anthropic/claude-opus-4.8": PriceEntry(5.0, 25.0, "2026-07-08"),
    "openrouter/anthropic/claude-opus-4.7": PriceEntry(5.0, 25.0, "2026-07-08"),
    "openrouter/anthropic/claude-fable-5": PriceEntry(10.0, 50.0, "2026-07-08"),
    "openrouter/google/gemini-3.5-flash": PriceEntry(1.5, 9.0, "2026-07-18"),
    "openrouter/qwen/qwen3.6-27b": PriceEntry(0.45, 2.70, "2026-07-18",
        note="drifted up from 0.285/2.40 (2026-07-08)"),
    "openrouter/qwen/qwen3.6-35b-a3b": PriceEntry(0.14, 1.00, "2026-07-18"),
    "openrouter/qwen/qwen3.6-flash": PriceEntry(0.188, 1.125, "2026-07-18",
        note="DashScope intl direct ~0.19/1.13"),
    "openrouter/qwen/qwen3.6-plus": PriceEntry(0.325, 1.95, "2026-07-18",
        note="DashScope direct is TIERED: >256K-token inputs bill ~4x"),
    "openrouter/qwen/qwen3.6-max-preview": PriceEntry(1.04, 6.24, "2026-07-18"),
    "openrouter/deepseek/deepseek-v4-pro": PriceEntry(0.435, 0.87, "2026-07-18"),
    "openrouter/deepseek/deepseek-v4-flash": PriceEntry(0.098, 0.196, "2026-07-18",
        note="drifted up from 0.084/0.168 (2026-07-08); direct list 0.14/0.28"),
}

# Later price eras, oldest first. Each takes effect on ``effective_from``;
# the window [unresolved_from, effective_from) lies between the last archived
# snapshot at the old price and the first at the new one.
PRICE_HISTORY: dict[str, tuple[PriceEntry, ...]] = {
    "gpt-5.6-luna": (
        PriceEntry(
            0.20, 1.20, "2026-09-25",
            note="cut 5x; old price last archived 2026-07-29, new first 2026-08-02",
            cache_read_usd_per_mtok=0.02, cache_write_usd_per_mtok=0.25,
            long_context_threshold_tokens=272_000,
            long_input_usd_per_mtok=0.40, long_cache_read_usd_per_mtok=0.04,
            long_cache_write_usd_per_mtok=0.50, long_output_usd_per_mtok=1.80,
            effective_from="2026-08-02", unresolved_from="2026-07-30",
        ),
    ),
    "gpt-5.6-terra": (
        PriceEntry(
            2.00, 12.00, "2026-09-25",
            note="cut 20%; old price last archived 2026-07-29, new first 2026-08-02",
            cache_read_usd_per_mtok=0.20, cache_write_usd_per_mtok=2.50,
            long_context_threshold_tokens=272_000,
            long_input_usd_per_mtok=4.00, long_cache_read_usd_per_mtok=0.40,
            long_cache_write_usd_per_mtok=5.00, long_output_usd_per_mtok=18.00,
            effective_from="2026-08-02", unresolved_from="2026-07-30",
        ),
    ),
    "gpt-5.6-sol": (
        PriceEntry(
            4.00, 20.00, "2026-09-25",
            note="cut 20%/33%; old price last archived 2026-08-02, new first 2026-09-02",
            cache_read_usd_per_mtok=0.40, cache_write_usd_per_mtok=5.00,
            long_context_threshold_tokens=272_000,
            long_input_usd_per_mtok=8.00, long_cache_read_usd_per_mtok=0.80,
            long_cache_write_usd_per_mtok=10.00, long_output_usd_per_mtok=30.00,
            effective_from="2026-09-02", unresolved_from="2026-08-03",
        ),
    ),
    "gemini/gemini-3.8-flash": (
        PriceEntry(
            1.50, 7.50, "2026-09-25",
            note="published step from 2027-01-01; re-verify once it is live",
            cache_read_usd_per_mtok=0.15, effective_from="2027-01-01",
        ),
    ),
}

# Compatibility view: {model: (input, output)} USD per 1M tokens, earliest
# verified list price (the default for callers that pass no date).
PRICES: dict[str, tuple[float, float]] = {
    model: entry.pair for model, entry in PRICING.items()
}


def price_entry(model: str, on: str | None = None) -> PriceEntry | None:
    """The published price in effect on ISO date ``on`` (None: the earliest
    verified price). None if the model is unknown or ``on`` falls in an
    unresolved price-change window."""
    base = PRICING.get(model)
    if base is None or on is None:
        return base
    chosen = base
    for era in PRICE_HISTORY.get(model, ()):
        if on >= era.effective_from:
            chosen = era
        elif era.unresolved_from and on >= era.unresolved_from:
            return None
    return chosen


def price_call(model: str, usage: dict[str, int], *, on: str | None = None) -> float | None:
    """Standard-tier list cost of ONE request's usage on date ``on``.

    Prices uncached input, cache reads, cache writes and output at the
    published rates, switching the whole request to the long-context rates
    when its input exceeds the published threshold. Returns None when the
    price is unknown or a class present in ``usage`` has no published rate.
    """
    entry = price_entry(model, on)
    if entry is None:
        return None
    inp = int(usage.get("input_tokens", 0) or 0)
    out = int(usage.get("output_tokens", 0) or 0)
    reads = int(usage.get("cache_read_tokens", 0) or 0)
    writes = int(usage.get("cache_write_tokens", 0) or 0)
    uncached = inp - reads - writes
    if uncached < 0:
        raise ValueError("cache_read_tokens + cache_write_tokens exceed input_tokens")
    long = (
        entry.long_context_threshold_tokens is not None
        and inp > entry.long_context_threshold_tokens
    )
    if long:
        rates = (
            entry.long_input_usd_per_mtok,
            entry.long_cache_read_usd_per_mtok,
            entry.long_cache_write_usd_per_mtok,
            entry.long_output_usd_per_mtok,
        )
    else:
        rates = (
            entry.input_usd_per_mtok,
            entry.cache_read_usd_per_mtok,
            entry.cache_write_usd_per_mtok,
            entry.output_usd_per_mtok,
        )
    r_in, r_read, r_write, r_out = rates
    if r_in is None or r_out is None:
        return None
    if (reads and r_read is None) or (writes and r_write is None):
        return None
    return (
        uncached * r_in
        + reads * (r_read or 0.0)
        + writes * (r_write or 0.0)
        + out * r_out
    ) / 1e6


def list_price_usd(model: str, usage: dict[str, int], *, on: str | None = None) -> float | None:
    """List cost of a usage-totals dict. Without cache classes and without
    ``on`` this is exactly the historical rule (input and output at the
    earliest verified price). Totals cannot see per-request long-context
    switches: price each request with ``price_call`` when a request may
    exceed a long-context threshold."""
    if on is None and not usage.get("cache_read_tokens") and not usage.get("cache_write_tokens"):
        prices = PRICES.get(model)
        if prices is None:
            return None
        cin = usage.get("input_tokens", 0) / 1e6 * prices[0]
        cout = usage.get("output_tokens", 0) / 1e6 * prices[1]
        return cin + cout
    entry = price_entry(model, on)
    if entry is None:
        return None
    return _price_short(entry, usage)


def _price_short(entry: PriceEntry, usage: dict[str, int]) -> float | None:
    inp = int(usage.get("input_tokens", 0) or 0)
    out = int(usage.get("output_tokens", 0) or 0)
    reads = int(usage.get("cache_read_tokens", 0) or 0)
    writes = int(usage.get("cache_write_tokens", 0) or 0)
    uncached = inp - reads - writes
    if uncached < 0:
        raise ValueError("cache_read_tokens + cache_write_tokens exceed input_tokens")
    if (reads and entry.cache_read_usd_per_mtok is None) or (
        writes and entry.cache_write_usd_per_mtok is None
    ):
        return None
    return (
        uncached * entry.input_usd_per_mtok
        + reads * (entry.cache_read_usd_per_mtok or 0.0)
        + writes * (entry.cache_write_usd_per_mtok or 0.0)
        + out * entry.output_usd_per_mtok
    ) / 1e6
