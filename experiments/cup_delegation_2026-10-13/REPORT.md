# Delegation study: what should Astra delegate, and does a learned System 1 do it better than simpler alternatives? (2026-10-13)

**Status: IN PROGRESS (interim, 18:15).** The replay results and the fresh-execution check of A0-A5 below are final. Still running
or queued: experiment C fresh executions, the Astra interface pilot, and the prospective test on the locked cups.

Plan: the user's three-batch plan.
1. **Mechanism:** the interface, A0-A5, update reliability, representations.
2. **Initialization:** 0 / 16 / 64 source interactions.
3. **Prospective:** E0-E3 on the two locked cups.

Tags: [E] measured, [I] interpretation.
- **Code:** `astra_explore/cup_transfer/{local_exec,batch3,batch3_offline}.py`; S1-v2 is unchanged except for record weights, a
  context-free mode and exclusions (batch 2 reproduces 33/33).
- **Configuration and rules:** `config/manifest.json` (pre-registered), `config/decisions.json` (rules and their two amendments).
- **Generated tables:** `REPORT_DATA.md`.

## 1. The delegated responsibility (one interface for every method)

Given a grasp intention and a nominal execution plan, select a reliable local execution, use feedback to revise it, and report
completion or the need for replanning. Astra keeps every change of strategy.

| Astra provides | The local executor returns |
|---|---|
| target object | success |
| nominal grasp: wrist reference + closing command (Astra's own first closing decision) | the correction executed at every attempt |
| desired outcome: the task-aligned grasp-lift-hold success of batch 2 | measured outcome and failure stage per attempt |
| permitted corrections: nominal + 30 persistent object-relative translations within +-1.5 cm | attempts used, decision / execution / reset time |
| local budget: 3 grasp attempts | completion, budget exhausted, or need for a new plan |

The action is a persistent correction of the grasp reference, applied just before closing and kept through lift and hold. It
uses batch 2's executor, unchanged. Every attempt restarts from the same pre-closing state, so this is **reset-based**
adaptation, not uninterrupted recovery.

## 2. Configuration and data split (the 7-vs-11-object discrepancy, resolved)

**What batch 2 used.** It had two source configurations, and its numbers stand as recorded:
- 7 objects (A, B, C, dev f1-f4): its round-2 online runs and its section-10 proposal;
- 11 objects (+ dev g1-g4): the final leave-one-out tables and the round-3 online runs.

**What this study fixes.** One configuration:

| split | objects | purpose |
|---|---|---|
| source | the 8 development-family cups (f1-f4, g1-g4), 2 starts x 31 corrections each = 496 recorded outcomes | initialization data |
| development | cups A (3 starts), B (10), cylinder C (3); plus the development family as additional targets, each with its own records removed (effective budget 7/8) | settings, mechanisms |
| locked | cup_t1 (6.22 x 11.21 cm), cup_t2 (6.93 x 10.75 cm), 20 fresh starts | prospective comparison only |

**Source selections.** Three reproducible nested selections (16 within 64 within 256, evenly over the 8 cups) were drawn by seed
before any target evaluation (`manifest.json`, committed 16:36).

**Scope.** A source record is: pre-closing state, then one bounded correction, then one executed grasp-and-lift, then the outcome.
No expert corrections are involved. These are **System 1 initialization budgets**, not the data behind Astra, the grasp program
or this project.

## 3. Mechanism batch so far (REPLAY = recorded outcomes of every correction at each development state)

### 3.1 Experiment A at the pre-registered settings (batch 2's kernel, h = 1)

64 interactions, 3 selections, scenes x selections:

| | first try, A/B/C | within 3, A/B/C | first try, dev family | within 3, dev family |
|---|---|---|---|---|
| A0 nominal | 6/16 | 6/16 | 8/16 | 8/16 |
| A1 one correction | 43/48 | 43/48 | 34/48 | 34/48 |
| A2 ranked search | 43/48 | 48/48 | 34/48 | 43/48 |
| A3 context-free, updating | 43/48 | 48/48 | 34/48 | 45/48 |
| A4 contextual, frozen | 45/48 | 48/48 | **28/48** | 36/48 |
| A5 contextual, updating | 45/48 | 48/48 | **28/48** | 45/48 |

- [E] **Where the contextual model fails.** On the development family it often has almost no local support: kernel mass
  0.2-2, against 5-10 for the context-free model.
  - It then falls back to its prior and the smallest correction, i.e. the nominal grasp, which fails on dev g1 (0/6 first try).
  - The context-free ranking pools all cups and picks corrections that hold on most of them (h25, h16, h04, h19).

### 3.2 Settings grid (kernel width h x representation x budget; `decisions.json`, rule fixed before the grid)

- [E] **Width.** Batch 2's kernel (h = 1) was tuned with all bank data; with 64 records it is too narrow. With h = 2-4 the
  methods converge (96 = 32 scenes x 3 selections):
  - A2 ranked search: 84/96 first try, 93/96 within 3;
  - A5 contextual updating (current features): 86/96 first try, 93/96 within 3.
- [E] **Representation.** Per-fingertip geometry does not help at 64 interactions; with a narrow kernel it hurts. It is best
  only at 256 (91/96 first try).
- **Amendment 1 (disclosed, before any fresh execution).**
  - The committed rule chose the context-free bandwidth by A3's within-3 count. That picked h = 1 on a one-scene margin, at a
    seven-scene first-try cost to A1/A2.
  - Settings are now chosen per role.
  - **Chosen:** E2 family (A1/A2) h = 2; E3 (A5) current features, h = 2.

### 3.3 Experiment B at the chosen settings (initialization curves; first / within 3 of 96)

| | 0 | 16 | 64 | 256 (reference) |
|---|---|---|---|---|
| A1 one correction | 42 / 42 | 71 / 71 | 84 / 84 | 84 / 84 |
| A2 ranked search | 42 / 45 | 71 / 86 | 84 / 93 | 84 / 93 |
| A3 context-free, updating | 42 / 48 | 71 / 87 | 84 / 92 | 84 / 93 |
| A4 contextual, frozen | 42 / 45 | 71 / 86 | 86 / 94 | 88 / 91 |
| A5 contextual, updating | 42 / 48 | 71 / 85 | 86 / 93 | 88 / 92 |

The nominal grasp alone holds 14/32.
- **Budget rule** (smallest budget within 3 of the best of 0/16/64): 64 for both families.
- **E2 rule:** ranked search (A2) beats one correction (A1) within 3: 93 against 84.

[I] At 64 interactions, in replay:
- source experience is what matters: 42 to 84-86 first-try successes;
- the contextual model adds about 2/96 first-try successes over a context-free ranking, and nothing within 3 attempts;
- updating adds little once failed candidates are skipped.

To be checked with fresh executions (running).

### 3.4 Experiment A, FRESH EXECUTIONS at the chosen settings (all 11 development cups, 2 new starts each)

Every method ran its own scene from the restored pre-closing state:
- up to 3 real grasp attempts, at 64 interactions, for each of the 3 selections;
- 412 grasp attempts in total, 0 safety violations, median 25 s per attempt (three simulators at once).

| | first try (of 66) | within 3 (of 66) | nominal-failing scenes: first / within 3 (of 36) | decision time |
|---|---|---|---|---|
| A0 nominal | 10/22 scenes | 12/22 | 0 / 2 of 12 | - |
| A1 one correction | 58 | 64 | 31 / 35 | 1.7 ms |
| A2 ranked search | **63** | 65 | **34** / 35 | 1.7 ms |
| A3 context-free, updating | 62 | **66** | 32 / **36** | 1.7 ms |
| A4 contextual, frozen | 58 | 65 | 28 / 35 | 2.2 ms |
| A5 contextual, updating | 59 | **66** | 30 / **36** | 2.3 ms |

Paired comparisons, first try (a better / b better):

| contrast | result |
|---|---|
| A1 vs A0 | 31 / 3 |
| A4 vs A2 | 1 / 6 |
| A5 vs A4 | 3 / 2 |
| A5 vs A3 | 1 / 4 |

Within 3 attempts, every pair differs by at most one scene.

- [E] The replay's ordering does not survive fresh execution. The contextual model was about 2 first-try successes ahead in
  replay; here it is 4-5 behind ranked search, mostly on dev f1.
  - On dev f1 the corrections every method prefers hold only about half of the time from the very same restored state (h19:
    6 of 11 runs at one start).
  - So single replay outcomes overstate how deterministic those cases are.
- [I] At 64 interactions, conditioning on the state does not improve selection; a context-free source ranking does as well
  or better. Updating matters only within 3 attempts, by at most one scene.

### 3.5 Experiment C, conservative update weight (development data)

- **Noisy replay.** Each attempt's outcome is drawn from the recorded executions of that correction at that state.
- **Result:** with updating, 920-930/960 succeed within 3; frozen, 810-840.
- **Chosen:** w = 0.25 (the rule's tie-break). The retry rule made no difference in replay.
- **Next:** fresh executions on cups with observed run-to-run variability (dev f1, g4, f4, g1) decide update strength and
  retry rule for E3 (amendment 2 fixed that rule before any result).

## 4. Prospective batch: how the locked results will be read (written 18:40, before any locked-cup load)

**Frozen configuration:** `config/prospective.json` (commit 81acf01, 18:34).
- 20 scenes, 2 locked cups x 10 seeded starts, each with one Astra proposal shared by all systems.
- The systems:
  - E0: Astra chooses the correction with measured geometry;
  - E1: E0 plus the 64 source records;
  - E2: ranked search;
  - E3: contextual updating S1;
  - E2, E3 and E1 all use the same 64 source records.
- Every system: at most 3 grasp attempts, the same 31 corrections, the same success test.

**Reading rules:**
- **Scale.** A difference of 1-2 of 20 scenes is *no detectable difference* (never equivalence). 3 or more with a two-sided
  sign test p < 0.05 is *supported*. 3 or more with a larger p is *suggestive*.
- **Primary outcomes:**
  - first-attempt success;
  - success within 3 attempts;
  - success within 120 / 240 s of measured wall clock, with robot time (60 / 120 s) as secondary.
- **Recommendation**, row by row of the plan's table:
  - **"One correction solves the cases reliably"** - from experiment A (A1 against A0, A2) and the locked first attempts of
    E2 / E3, which use the source's best correction first.
  - **"Source-ranked search matches the learned model"** - E2 against E3: no detectable difference on all three primaries
    -> a small experience-based search module.
  - **"Context improves first choices, but updates add little"** - E3 first-try success at least 3 above E2's, with
    within-3 equal.
  - **"Updates outperform the same search without updates"** - experiment C and A5 against A4 (development; the locked test
    has no frozen contextual arm).
  - **"Structured memory lets Astra match local methods at similar total cost"** - E1 against E2/E3: no detectable
    difference in success AND mean scene time within 25 % -> a separate learned S1 is not justified by this experiment.
  - **"Local selection matches success and reduces reasoning time"** - E2/E3 against E1 (and E0): success no worse by 3 or
    more, with fewer Astra calls and less waiting -> delegate local execution.
  - **"Learned S1 improves both success and total cost"** - E3 against E2 and E1: better success (supported or suggestive)
    and lower time.
  - **"Benefits occur only with the fixed source program"** - the scope statement: every proposal reuses cup A's grasp
    relative to the cup base, so whatever holds, holds for that skill.

## 5. Running and queued

- **Experiment A, fresh executions:** all 11 development cups x 2 new starts, A0-A5 at the chosen settings, 3 selections.
- **Experiment C, fresh executions:** frozen / conservative / current update x exclude-after-one / one confirmation retry, plus
  A2 and A3 under both rules, on 4 cups x 2 new starts.
- **Experiment D, interface pilot:** smoke tests of the hand-off with a scripted policy, then Astra's actual proposals on 6
  development scenes (cup B x 2, dev f1, dev g4, C, A). At each proposal: every correction executed, and the four compared
  systems run on four of them (Astra calls for E0/E1).
- **Then:** freeze `config/prospective.json`, and run the prospective test: 20 locked scenes x E0-E3, at most 240 grasp
  attempts.
