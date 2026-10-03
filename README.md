# Blindboard — public release (v1)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23121518.svg)](https://doi.org/10.5281/zenodo.23121518)
[![rebuild](https://github.com/adamallcock/blindboard/actions/workflows/rebuild.yml/badge.svg)](https://github.com/adamallcock/blindboard/actions/workflows/rebuild.yml)

Blindboard tests whether a language model can keep track of facts that keep
changing: the symbol layout of a keyboard that is hidden and keeps being
reshuffled. This is the code and data release for *Blindboard: Measuring
State Tracking Under Interference with a Hidden, Shifting Keyboard*
(Allcock, 2026).

On the public pilot seeds, every model that played the 12-key announced tier
is near perfect, while on the heaviest tier, 47 keys that never stop moving
and keep being renamed, scores at each model's top effort run from
.060 to 1.000. Against an earlier
campaign that resent only the visible conversation, scores move both ways and
output tokens fall by a median factor of 7.7; the
campaigns also differ in date and some prompt text, so this is an
association, not a measured effect.

## Quick start

None of these three commands needs an API key.

```
python3 scripts/play.py --tier t2_announced --seed 0 --agent reference   # watch the reference solver
python3 scripts/play.py --tier t0_frozen --seed 3 --agent human          # play yourself
python3 rebuild.py                                                       # rebuild every number in the paper
```

## What Blindboard is

The model plays on a keyboard whose layout is hidden. Each turn it presses a
few keys at once and is told which symbols came back, but not which key
produced which. Between turns some keys trade symbols, and the model is told
which keys moved but not where their symbols went. When it is ready, or at a
turn limit, it writes down the whole layout and is scored on the fraction of
keys it gets right. Harder versions, called tiers, add keys, keep the board
moving, rename keys mid-game, or run two boards at once.

A scripted reference solver plays every tier first, so each score can be read
against what careful, sound play achieves there. A replay tracker then sorts each missed key
into one the model's observations had determined (a conversion loss) or one
they had left unresolved (an acquisition loss). The split is relative to the
tracker: it locates a loss, not its cause.

Each model plays with its own earlier reasoning carried into every turn
through its provider's mechanism. An earlier campaign on the same seeds
resent only the visible conversation, and the paper compares the two.

## Pilot results

Read these as pilot measurements: the seeds are public and unsalted, most
cells have five seeds, and no significance tests are run. The comparison with
the earlier visible-transcript campaign is historical. The two campaigns also
differ in date and some prompt text, so their differences are associations,
not measured effects of retained reasoning.

The **easy test** pairs the 12-key `t2_announced` tier with the 26-key
`t6_gauntlet`, at requested medium effort (high on `deepseek-v4.1-flash`,
whose route runs medium as high). The **stress test** pairs
`t7_perpetual`, 26 keys that never stop moving, with `t11_maelstrom`, 47 keys
that never stop moving and keep being renamed, at each route's top effort.
Each cell is the mean fraction of keys right, with its standard error, over
20 seeds on `t2_announced` and 5 elsewhere; a single-mode route runs every
request at one setting.

| Model | Easy-test effort | announced | gauntlet | Stress-test effort | perpetual | maelstrom |
|---|---|---:|---:|---|---:|---:|
| `gpt-6-astra` | medium | 1.000 ± .000 | 1.000 ± .000 | max | .977 ± .023 | 1.000 ± .000 |
| `gpt-6-sol` | medium | .983 ± .011 | 1.000 ± .000 | max | .931 ± .028 | 1.000 ± .000 |
| `gpt-6-luna` | medium | 1.000 ± .000 | .615 ± .036 | max | .600 ± .046 | .226 ± .094 |
| `gpt-5.6-sol` | medium | -- | .815 ± .061 | max | .815 ± .107 | .885 ± .040 |
| `gpt-5.6-terra` | medium | -- | .769 ± .040 | max | .900 ± .050 | -- |
| `gpt-5.6-luna` | medium | .979 ± .015 | .600 ± .020 | max | .685 ± .087 | .553 ± .028 |
| `gpt-5.4-nano` | medium | 1.000 ± .000 | -- | -- | -- | -- |
| `claude-opus-5-5` | medium | 1.000 ± .000 | 1.000 ± .000 | max | .954 ± .028 | .962 ± .016 |
| `gemini-3.8-flash` | medium | 1.000 ± .000 | .946 ± .054 | high | .862 ± .085 | .791 ± .113 |
| `deepseek-v4.1-flash` | high | 1.000 ± .000 | .931 ± .043 | max | .554 ± .171 | .706 ± .157 |
| `deepseek-v4-pro` | single-mode | 1.000 ± .000 | .869 ± .054 | single-mode | .562 ± .076 | .217 ± .061 |
| `deepseek-v4-flash` | single-mode | 1.000 ± .000 | .746 ± .082 | single-mode | .146 ± .054 | .060 ± .023 |

12 models on 12 tiers:
455 scored episodes and 9,103 model
calls, collected for about $180.55 at list prices.
Every cell of both campaigns is in `paper/tables/cells.csv` and
`paper/tables/cells_visible.csv`.

<details>
<summary>Tokens and cost per model</summary>

Output tokens per episode on each stress tier at the route's top effort, and
the list price of the reported cells of both tests.

| Model | Tokens per perpetual episode | Tokens per maelstrom episode | Cost of both tests |
|---|---:|---:|---:|
| `gpt-6-astra` | 12k | 19k | $14.98 |
| `gpt-6-sol` | 40k | 59k | $8.64 |
| `gpt-6-luna` | 126k | 249k | $1.31 |
| `gpt-5.6-sol` | 30k | 41k | $11.66 |
| `gpt-5.6-terra` | 50k | -- | $4.40 |
| `gpt-5.6-luna` | 65k | 99k | $1.38 |
| `gpt-5.4-nano` | -- | -- | $0.44 |
| `claude-opus-5-5` | 99k | 121k | $32.69 |
| `gemini-3.8-flash` | 95k | 154k | $11.55 |
| `deepseek-v4.1-flash` | 280k | 269k | $2.85 |
| `deepseek-v4-pro` | 24k | 34k | $1.62 |
| `deepseek-v4-flash` | 11k | 24k | $0.19 |

</details>

## What's in this release

| To | Use |
|---|---|
| Play, or run a model | `blindboard/` (the environment, the reference solver, and the harness; standard library only), `scripts/play.py`, `scripts/run_llm.py` |
| Inspect a stored episode | `scripts/inspect_episode.py`, `scripts/show_reasoning.py` |
| Rebuild the paper's numbers and figures | `rebuild.py`, `scripts/build_paper_assets.py`, `scripts/make_figures.py`, `scripts/make_walkthrough_figures.py` |
| Audit | `tests/`, `scripts/audit_transcripts.py`, `scripts/derangement_structure_audit.py`, `scripts/calibrate.py`, `scripts/retention_recall_probe.py` |
| Read the data | `results/` (below), `paper/tables/` (every generated table, number, and CSV), `paper/figures/` |
| Read the policies | `docs/release-policy.md` (official seeds), `docs/implementation-notes.md` (the reference solver and harness in detail) |
| Check integrity | `SHA256SUMS` (every file), `PROVENANCE.json` (the source commit), `paper/snapshot_manifest.json` (the frozen snapshot: a hash of every run directory) |

| Results tree | What it holds |
|---|---|
| `results/retained/` | The primary population: every run of the retained-reasoning campaign, with full transcripts, reasoning summaries, usage, and per-request list prices. |
| `results/retained-superseded/` | Attempts replaced after a harness fix or a stopped account; priced, scored in no cell. |
| `results/llm/` | The visible-transcript campaign. |
| `results/diagnostics/` | The recall probe and other diagnostic runs. |
| `results/calibration/`, `results/calibration-heldout/` | The reference solver's scores, and their re-measurement on held-out seeds. |
| `results/smoke/`, `results/fragments/`, `results/static/` | Smoke runs, partial runs, and the addressing probe; priced into the campaign totals. |

Every episode stores each prompt and reply, the reasoning summaries the
provider returned, and each call's token usage. The providers' opaque
reasoning state (encrypted items and signatures) is never written to disk.

## Rebuild every number

`python3 rebuild.py` checks every file against `SHA256SUMS`, regenerates
every table, number, and CSV from `results/` into a temporary directory, and
compares each file byte for byte with `paper/tables/`. The snapshot is
frozen: the asset builder first checks every run directory against
`paper/snapshot_manifest.json` and refuses to build from a tree that has
drifted from it. It needs Python 3.11
or later and nothing else: standard library only, no network.
`readiness_gates.tex` is left out of the comparison, because it records a run
of the test suite:

```
python3 -m pip install pytest
python3 -m pytest -q
```

Two tests need a git checkout, because they ask git what it ignores and which
commit built a snapshot; elsewhere they skip. The figures redraw with
`python3 scripts/make_figures.py` and `python3 scripts/make_walkthrough_figures.py`
(these need matplotlib); font rendering differs across platforms, so a
redrawn image can differ slightly from the shipped one.

## Run a provider model

This requires an API key and incurs provider charges. The provider's key must
be in the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, or `OPENROUTER_API_KEY`), and model names follow
`results/` (for example `claude-opus-5-5`, `gemini/gemini-3.8-flash`,
`openrouter/deepseek/deepseek-v4-pro`):

```
python3 scripts/run_llm.py --model gpt-5.6-luna --tier t6_gauntlet --effort medium \
  --seeds 5 --harness retained --out results/mine
```

## Provenance and accounting

Run-directory names carry the local (US Eastern) date and time a run started,
as `YYYYMMDD-HHMMSS`; they are labels, not collection windows. The
retained-reasoning campaign ran from 2026-09-25 to 2026-09-28
(`results/retained`), with its recall probe and smoke runs through
2026-09-30. The visible-transcript campaign ran from 2026-07-12 to 2026-07-28
(`results/llm`). Costs are standard-tier list prices on the day of use, and
the reported per-request cost on OpenRouter.

Documented exceptions, fixed in the data:

1. **Superseded attempts.** No cell was re-run because of its score. Every
   attempt in `results/retained-superseded/` ended in a provider failure, a
   harness change, or a stopped account, and is priced into the campaign
   total. The largest group is `gpt-5.6-sol`'s first collection: it re-ran a
   refused episode from its seed, which keeps only the trajectories the
   provider never refused, so all eight of its cells were collected again on
   2026-09-28 with refused requests re-sent unchanged.
2. **Refused requests.** `gpt-5.6-sol` sometimes receives HTTP 400
   `invalid_prompt` for a request that succeeds when re-sent verbatim. Each
   is re-sent unchanged, and the model never sees the refusal; across the
   scored `sol` episodes that happened thirteen times, on
   nine turns.
3. **Turns without an action.** A turn cut off at the output-token limit, or
   ended by a call to DeepSeek's inert tool, scores as an empty reply and is
   not carried into the next request. The retention check leaves out the
   29 pairs of calls that follow such turns.
4. **Routing.** DeepSeek requests declare one inert tool, because DeepSeek
   keeps earlier reasoning only when a request declares tools. A host that
   dropped the replayed reasoning was routed around (3 turns, all
   `deepseek-v4-pro`). Gemini requests that the flex tier refused three times
   went out on the standard tier (22% of Gemini
   calls).
5. **Cached input in the first luna runs.** Eleven early `gpt-5.6-luna`
   episodes record cached input as `cached_input_tokens` and carry no
   per-request prices. The asset builder prices that field as cache reads.
6. **The visible-transcript campaign** recorded no routing provenance, and
   29 of the cells paired with this campaign ran
   an older prompt, `bb-r1`, whose text differs on
   8 of them. Run directories without a
   `summary.json` in either campaign are priced into its total and scored in
   no cell.

## What's withheld, and why

- **Official seed lists and their salts.** The seeds in this release are
  public and unsalted, so two chords can identify an episode and a player
  could replay it exactly; scores on them are pilot measurements, not
  official ones. Official runs bind a secret per-evaluation salt, and only a
  commitment to each list and salt is published, so a past official score can
  be checked once the list is revealed (`docs/release-policy.md`). The first
  official list was exposed and is never used for scoring.
- **Canary tiers.** Two implemented mechanics, near-QWERTY layouts and rotors,
  are held back as hidden tier parameterizations.

## License

The data, the generated tables and figures, and the documentation are licensed
under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
The code (`blindboard/`, `scripts/`, `tests/`, and `rebuild.py`) is licensed
under the Apache License, Version 2.0. See `LICENSE` and
`LICENSE-DATA-DOCS.md`.

## Citing

Please cite the paper
([doi:10.5281/zenodo.23121596](https://doi.org/10.5281/zenodo.23121596)) and this archive;
`CITATION.cff` carries both. This
version is archived on Zenodo as
[doi:10.5281/zenodo.23121518](https://doi.org/10.5281/zenodo.23121518); the concept DOI
[10.5281/zenodo.23121517](https://doi.org/10.5281/zenodo.23121517) always resolves to the latest version.
