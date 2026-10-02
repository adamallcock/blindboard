# Blindboard implementation notes

Details the paper summarises in its appendix, kept here in full.

## The reference player

**Belief core.** The reference maintains a sound belief state:
- per-key candidate sets;
- group constraints ⟨key-set, symbol-set⟩ from chords, propagated by elimination, hidden singles, and group containment.

Announced rotations do two things:
- update candidates to unions over the rotated set;
- preserve exactly the group constraints that survive, meaning those disjoint from or containing the rotated set.

Soundness (the truth is always inside every candidate set) is enforced by stepwise tests against the hidden layout across modes and seeds.

For its final guess the core answers each key with the posterior-marginal argmax over the layouts consistent with its own constraint set. Those layouts are enumerated exactly, and sampled only above an enumeration cap. This maximises expected keys correct given those constraints, which are a sound but lossy summary of the history.

**Planning layer.** Above the belief core, the shipped reference player replaces the remaining greedy heuristics with three devices:

1. **Optimal stopping for observer and storm modes.** Guess at the first state where everything is resolved except the freshest rotated set, and that set's pre-rotation assignment is fully known. There the posterior is exactly a uniform derangement, and no continuation has a higher expected score.
2. **A weighted filter over storm histories,** carried forward through chord filtering and derangement mixing.
   - Its support is guaranteed to contain the truth.
   - Its weights are approximate, because the enumeration is step-budgeted.
   - It recovers joint no-fixed-point information that the tracker's sound propagation discards.
3. **Deterministic endgame closers** for perpetual-announced play.

The planner is deterministic per (configuration, seed) and never scores below the belief core. The ceiling derivations and the stopping rule's floor property are machine-tested.

## Instrument checks

- **Relabel tiers.** Diagnostics translate guessed keys through the relabel chain, because comparing canonical addresses would overstate conversion gaps there. A regression test requires a perfect solver to convert 100% of determined keys under relabeling.
- **Relabeling and the reference.** The reference plays relabel tiers through a mechanical translation shim, and scores identically with relabeling on or off. A regression test enforces that.
- **Rules text.** The announced-mode rules text states how rotations evolve. It is derived from each tier's actual schedule, under a per-tier regression gate.
- **Prompt versions.** Prompt version `bb-r2` stamps itself into every result collected with it. The older `bb-r1` told every announced tier that rotations "become rarer as the game goes on and eventually stop". That is false on the six constant-churn tiers, and models quoted the promise while planning.

## Determinism and replay

- Certificate measurement pins the interpreter's hash seed.
- The tracker's backtracking sorts its candidate list before the seeded shuffle, so a replay does not inherit set iteration order.
- Reproducing an episode means environment replay: given a (tier, seed) pair and the stored action sequence, the environment regenerates the same board, rotations, observations, and score under `PYTHONHASHSEED=0`.
- Model calls are not deterministic and are not replayed; the stored transcripts are the record of what each model did.

## Retention mechanisms, per route

- **OpenAI Responses API.** The harness replays every output item of every earlier turn, encrypted reasoning included. Server-side storage is off, and the reasoning context is set to all turns.
- **Anthropic Messages API.** It replays each earlier turn's signed thinking blocks byte for byte.
- **Gemini.** It resends each earlier model turn with all of its parts and thought signatures.
- **OpenRouter.** It replays each turn's reasoning details. DeepSeek requests declare one inert tool, because DeepSeek ignores earlier reasoning unless the request declares tools.

Before collection, every model was checked live on its route. The next turn's billed prompt had to grow by the earlier turn's reasoning tokens, compared against a control that sends the visible text only. Claude Haiku 4.5 fails that check, because the API strips its earlier thinking.
