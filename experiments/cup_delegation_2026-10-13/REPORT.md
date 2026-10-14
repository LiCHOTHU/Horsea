# Delegation study: what should Astra delegate, and does a learned System 1 do it better than simpler alternatives? (2026-10-13)

**Status: COMPLETE** for the three pre-registered batches (mechanism, initialization, prospective; locked runs finished 19:11)
and both labelled extensions (section 9, finished 21:15).

## Answer

**Recommendation: Astra with simple experience-based search.**

- **What to delegate.** Astra delegates one bounded responsibility to a small local module: choosing the persistent
  pre-closing correction of its proposed grasp, retrying from experience, and reporting completion or an exhausted
  budget. Astra keeps the proposal and every change of strategy.
- **What the module is.** Rank the corrections tested on other cups, try them in order, and never repeat a failure.
- **The learned contextual System 1 is not justified here.** It conditions on the state and updates online, yet it made
  the same choice as the ranking in all 20 locked scenes. In development it was no better in replay and slightly worse
  in fresh executions.

| locked test: 20 scenes, 2 new cups | first try | within 3 | mean scene time (wall / robot) | Astra calls per scene |
|---|---|---|---|---|
| E0 Astra chooses, with geometry feedback | **0** | 7 | 122 s / 65 s | 5.0 |
| E1 Astra chooses, with the same experience as records | 20 | 20 | 62 s / 40 s | 3.0 |
| E2 local ranked search (experience-based) | 20 | 20 | 55 s / 33 s | 2.0 (proposal only) |
| E3 local learned System 1 (contextual, updating) | 20 | 20 | 54 s / 33 s | 2.0 (proposal only) |

**The active ingredient is experience about corrections on other cups, not who reads it.**
- Astra given the same 64 records (E1) succeeds as often as the local methods.
- Each attempt then costs one extra Astra call (about 7 s): +14 % wall clock, +22 % robot time.
- Without the records (E0), Astra fails the first attempt in every scene. It trusts its cup-A memory, then reasons toward
  the wrong direction.

**Scope.** Every Astra proposal, 26 of 26 across development and the locked test, reused cup A's grasp relative to the
cup base. All conclusions are for that one skill and this hand, on cups of this family.

Tags: [E] measured, [I] interpretation.
- **Code:** `astra_explore/cup_transfer/{local_exec,batch3,batch3_offline}.py`.
- **Configuration:** `config/manifest.json` (pre-registered 16:36, commit 1661f1c), `config/decisions.json` (rules 16:39,
  two disclosed amendments), `config/prospective.json` (frozen 18:34, before any locked-cup load).
- **Tables:** `REPORT_DATA.md` (generated). **Figures:** `figures/`.

## 1. The delegated responsibility (one interface for every method)

| Astra provides | The local executor returns |
|---|---|
| target object | success |
| nominal grasp: Astra's first closing decision (wrist reference + closures), not executed; the state before it is handed over | the correction executed at every attempt |
| desired outcome: batch 2's task-aligned grasp-lift-hold success | measured outcome and failure stage per attempt |
| permitted corrections: nominal + 30 persistent object-relative translations within +-1.5 cm | attempts used, decision / execution / reset time |
| local budget: 3 grasp attempts | completion, budget exhausted, or need for a new plan |

- After the hand-off, the executor runs Astra's closing decision, then the existing program's lift and hold.
- A correction translates the grasp reference from just before closing through lift and hold.
- Every attempt restarts from the same pre-closing state, so this is **reset-based** adaptation.
- In the no-render simulators, replaying Astra's executed proposal decisions reproduced the pre-closing state within
  0.007 cm.

## 2. Configuration and data split (the 7-vs-11-object discrepancy, resolved before anything ran)

**Batch 2's two source configurations stand as recorded:**
- 7 objects (A, B, C, dev f1-f4): its round-2 online runs and its section-10 proposal;
- 11 objects: its final tables and round-3 online runs.

**This study fixed one configuration.**

| split | objects | purpose |
|---|---|---|
| source | the 8 development-family cups (f1-f4, g1-g4): 2 starts x 31 corrections = 496 recorded outcomes | initialization data |
| development | cups A (3 starts), B (10), cylinder C (3); the development family as additional targets with their own records removed (effective budget 7/8) | settings, mechanisms |
| locked | cup_t1 (6.22 x 11.21 cm), cup_t2 (6.93 x 10.75 cm), 10 seeded starts each | the prospective comparison only |

- **Selections.** Three nested source selections (16 within 64 within 256, evenly over the 8 cups) were drawn by seed and
  committed before any target evaluation.
- **What the budgets are.** System 1 initialization budgets: pre-closing state, one bounded correction, one executed
  grasp-and-lift, its outcome. No expert corrections. They are not the data behind Astra or the grasp program.

## 3. Mechanism batch

### 3.1 Experiment A in replay at the pre-registered settings (batch 2's kernel, h = 1)

64 interactions; 16 primary scenes x 3 selections = 48.

| | first try, A/B/C | within 3, A/B/C | first try, dev family | within 3, dev family |
|---|---|---|---|---|
| A0 nominal | 6/16 | 6/16 | 8/16 | 8/16 |
| A1 one correction | 43/48 | 43/48 | 34/48 | 34/48 |
| A2 ranked search | 43/48 | 48/48 | 34/48 | 43/48 |
| A3 context-free, updating | 43/48 | 48/48 | 34/48 | 45/48 |
| A4 contextual, frozen | 45/48 | 48/48 | 28/48 | 36/48 |
| A5 contextual, updating | 45/48 | 48/48 | 28/48 | 45/48 |

- [E] **Why the contextual model fell behind.** With 64 records it often had almost no local support: kernel mass 0.2-2,
  against 5-10 for the context-free model. It then fell back to its prior and the smallest correction, i.e. the nominal
  grasp.
- **Settings grid** (kernel width h x representation x budget; the rule was fixed before the grid):
  - with h = 2-4 the methods converge;
  - **amendment 1** (disclosed, before any fresh execution) chose settings per role: E2 family h = 2; E3 S1-original's
    27 state features, h = 2.

### 3.2 Experiment A, FRESH EXECUTIONS at the chosen settings (11 cups x 2 new starts)

Every method ran its own scene with up to 3 real grasp attempts:
- 64 interactions, 3 selections;
- 412 grasp attempts in total, 0 safety violations.

| | first try (of 66) | within 3 (of 66) | nominal-failing scenes: first / within 3 (of 36) |
|---|---|---|---|
| A0 nominal | 10/22 scenes | 12/22 | 0 / 2 of 12 |
| A1 one correction | 58 | 64 | 31 / 35 |
| A2 ranked search | **63** | 65 | **34** / 35 |
| A3 context-free, updating | 62 | **66** | 32 / **36** |
| A4 contextual, frozen | 58 | 65 | 28 / 35 |
| A5 contextual, updating | 59 | **66** | 30 / **36** |

Decision time is 1.7 ms (context-free) and 2.3 ms (contextual).

| question | comparison | first try (a / b better) | within 3 | reading |
|---|---|---|---|---|
| Was the original grasp simply poorly calibrated? | A1 vs A0 | 31 / 3 | 29 / 1 | **yes** |
| Does a short list suffice? | A2 vs A1 | 6 / 1 | 1 / 0 | the list adds at most a scene within 3; first-try gaps are noise (same first pick) |
| Does conditioning on the state improve selection? | A4 vs A2 | 1 / 6 | 1 / 1 | **no**: if anything worse at this data scale |
| Does updating help beyond moving down a list? | A3 vs A2 | 1 / 2 | 1 / 0 | no detectable difference |
| Does updating the contextual model help? | A5 vs A4 | 3 / 2 | 1 / 0 | no detectable difference |
| Does context improve how feedback is generalized? | A5 vs A3 | 1 / 4 | 0 / 0 | no |

[E] The replay ranked the contextual model about 2 first-try successes ahead. Fresh execution reverses that:
- the difference comes from cases where single replay outcomes overstated determinism (section 3.3);
- on dev f1 the corrections every method prefers hold only part of the time from the same restored state.

### 3.3 Experiment C: are failure updates reliable? (fresh executions, cups with run-to-run variability)

Setup: 4 cups (dev f1, f4, g1, g4) x 2 new starts x 3 selections = 24 scenes per condition; 282 grasp attempts.

| condition | first try | within 3 | confirmation retries (held) | abandoned a correction that held in other runs at that start |
|---|---|---|---|---|
| A4 frozen, exclude after one failure | 19 | 24 | - | 5 |
| A4 frozen, one confirmation retry | 21 | 23 | 3 (2) | 1 |
| A5 conservative (w 0.25), exclude | 22 | 23 | - | 3 |
| A5 conservative (w 0.25), confirm | 21 | 23 | 1 (1) | 2 |
| A5 current (w 1), exclude | 20 | 24 | - | 4 |
| A5 current (w 1), confirm | 21 | 23 | 1 (0) | 3 |
| A2 ranked search, exclude / confirm | 21 / 21 | 24 / 24 | 3 (2) | 3 / 1 |
| A3 context-free updating, exclude / confirm | 19 / 20 | 24 / 24 | 3 (2) | 5 / 2 |

- [E] **The variability is real.** From the identical restored state:

  | cup / start | correction | held |
  |---|---|---|
  | dev f1 v04 | h19 | 7 of 25 runs |
  | dev f1 v03 | h19 | 12 of 23 |
  | dev f1 | h25 | 25 of 30 |

- **Retries.** A confirmation retry held in 7 of 11 cases.
- **Abandoning.** Excluding after one failure sometimes drops a correction that works (1-5 times per 24 scenes).
- [I] **It costs nothing within 3 attempts here**, because other corrections also hold. Success within 3 is 23-24/24
  under every update strength and retry rule.
  - The model neither learns something decisive from a failure nor overreacts in a way that matters.
  - A conservative update is not justified: the rule compares 46 against 47 and keeps the current update and exclusion.

### 3.4 Experiment D: what information System 1 needs

- [E] **Representation (replay, same data and settings).**
  - Per-finger geometry does not help at 64 interactions. With batch 2's narrow kernel it cuts support: development-family
    first try 28/48 against 38/48 with the 27 existing features. At h = 2, A5: current 86/93 against geometry 81/91 of 96.
  - Geometry helps only at 256 interactions (91/96 first try).
- **Post-correction geometry, audited.** It is computed before execution from the pre-closing fingertip positions (robot
  kinematics) and the tracked object pose and size, translated by the candidate correction, so it is admissible.
  - It was designed after seeing cup B and gave no gain; it stayed exploratory and was never selectable.
- [E] **Interface pilot (6 actual Astra proposals on development cups, recorded before any outcome).**
  - Every one handed off after 2 calls (27 s of Astra time, about 59 s wall).
  - Every one was cup A's grasp: wrist at (-14.5, -5.0, +6.3) cm from the cup base, closures (0.40, 0.45, 0.50, 0.40).
  - Astra's real proposals equal the scripted nominal, so System 1 was never tested on a different nominal by Astra itself.
    The changed-nominal extension (section 9) tests that directly.
  - Every correction was executed at each of the 6 proposals, recorded and replayed (replay drift at most 0.001 cm):
    - the nominal grasp holds at 2/6 (cup A, cylinder C);
    - A1-A5 hold on the first try in 17/18 and within 3 in 18/18 (A1, which repeats its pick: 17/18);
    - h19 held at all 6 proposals.
  - On 4 pilot scenes, results matched the locked test (first try / within 3):

    | system | first try | within 3 |
    |---|---|---|
    | E0 | 0/4 | 1/4 |
    | E1 | 3/4 | 4/4 |
    | E2 | 3/4 | 4/4 |
    | E3 | 4/4 | 4/4 |

**Batch statement (mechanism).**
- **Supported:** the original grasp is miscalibrated for these cups (A1 >> A0).
- **Contradicted at this data scale:** state conditioning improves selection.
- **Unresolved, no detectable difference:** updating, beyond a no-repeat list; conservative updates; confirmation
  retries.
- **Simplest supported implementation:** source-ranked search with no repeats (A2). One fixed correction (A1) is nearly
  as good.

## 4. Initialization batch (experiment B; replay at the chosen settings)

First try / within 3, of 96 = 32 scenes x 3 selections:

| | 0 | 16 | 64 | 256 (reference) |
|---|---|---|---|---|
| A1 one correction | 42 / 42 | 71 / 71 | 84 / 84 | 84 / 84 |
| A2 ranked search | 42 / 45 | 71 / 86 | 84 / 93 | 84 / 93 |
| A3 context-free, updating | 42 / 48 | 71 / 87 | 84 / 92 | 84 / 93 |
| A4 contextual, frozen | 42 / 45 | 71 / 86 | 86 / 94 | 88 / 91 |
| A5 contextual, updating | 42 / 48 | 71 / 85 | 86 / 93 | 88 / 92 |

The nominal grasp alone holds 14/32. See `figures/curves.png`.

- [E] **Source experience is the benefit:** first-try successes rise from 42 to 71 to 84-86.
- **The contextual model never pulls ahead of ranked search.** The gap is at most 2 of 96 first-try at any budget, and
  is 1 of 96 within 3 (e.g. 94 against 93 at 64).
- **16 interactions are not adequate.** Within 3 is 85-87 of 96 against 93, and it varies strongly across selections:
  ranked search 23, 31 and 32 of 32.
- **256 adds at most 2 first-try successes over 64**, so the model is not starved of data at 64.
- **Chosen by the pre-registered rule:** 64 for both families.
- [I] All methods improve similarly with more source data, so the main benefit is better prior experience. Context
  adds nothing measurable.

## 5. Prospective batch (experiment E, locked cups)

**Design.**
- Frozen in `config/prospective.json` (commit 81acf01, 18:34:53); the reading rules were committed at 18:35:30 (f5c0d7b).
  Both came before the first locked-cup load (18:35:55); the first locked system result was written at 18:44.
- 20 scenes, each with one Astra proposal. Every proposal handed off after 2 calls: 27 s of Astra time and 36 s wall per
  scene, 40 calls in total. Every proposal was again cup A's grasp.
- The four systems ran from the identical pre-closing state, in a seeded random order per scene, with at most 3 grasp
  attempts, the same 31 corrections and the same success test.

| | first try | within 2 | within 3 | within 120 s / 240 s wall | robot time: within 60 s / 120 s | mean scene time: wall / robot | decision time per scene | Astra calls per scene |
|---|---|---|---|---|---|---|---|---|
| E0 Astra + geometry | **0** | 1 | 7 | 4 / 7 | 2 / 7 | 122 s / 65 s | 23 s (Astra) | 4.95 |
| E1 Astra + structured memory | 20 | 20 | 20 | 20 / 20 | 20 / 20 | 62 s / 40 s | 7.1 s (Astra) | 3.0 |
| E2 ranked search | 20 | 20 | 20 | 20 / 20 | 20 / 20 | 55 s / 33 s | 2 ms | 2.0 |
| E3 contextual updating S1 | 20 | 20 | 20 | 20 / 20 | 20 / 20 | 54 s / 33 s | 2 ms | 2.0 |

- [E] **Choices.**
  - E2 and E3 chose h19 (-1.03, +0.72, -1.02 cm) in all 20 scenes.
  - E1 chose h09 (-0.56, -0.39, -1.26 cm) in all 20, citing records where it "produced stable holds on three cups ... with
    nearly identical pre-closing poses".
  - E0 always tried the nominal grasp first ("previously achieved an 8.24 cm lift and steady hold"), then h10 (forward,
    left, slightly down: "should seat the cup more deeply") in 19/20, then varied.
  - E0 per cup: cup_t1 6/10 within 3, cup_t2 1/10.
- **Astra call cost.** Correction-choice calls took a median of 7.5 s (E0, 59 calls) and 7.0 s (E1, 20 calls with about
  24k-character prompts). No invalid responses.
- **Paired tests.**
  - E0 against each of E1, E2, E3: first try 0 vs 20 discordant (p < 0.0001); within 3, 0 vs 13 (p = 0.0002). Supported.
  - E1, E2 and E3 against each other: no discordant scene on any success measure, so no detectable difference (not
    equivalence).

## 6. Recommendation, row by row (reading rules of section 4 of the plan, fixed before the locked results)

| finding (plan's table) | evidence | holds? |
|---|---|---|
| One correction solves the cases reliably → improve the existing executor | a single source-selected correction (A1): fresh first try 58/66, within 3 64/66 (it repeats); on the locked cups every experience-based first choice held (20/20) | **largely**: most of the gain is one calibrated persistent shift (toward the cup and down) |
| Source-ranked search matches the learned model → small local experience-based search module | E2 = E3 on every locked measure (identical choices, 20/20); development fresh: A2 63/65 vs A5 59/66 | **yes** |
| Context improves first choices, but updates add little → pretrained local selector, conservative online behavior | E3 first try = E2's; development fresh A4 1 vs 6 against A2 | no |
| Updates outperform the same search without updates → retain adaptive memory | experiment C: at most 1 of 24 scenes; A5 vs A4 fresh 1 / 0 within 3 | no |
| Structured memory lets Astra match local methods at similar total cost → a separate learned S1 is not justified | E1 20/20 = E2/E3; time +14 % wall, +22 % robot (within the pre-registered 25 %) | **yes** |
| Local selection matches success and reduces reasoning time → delegate local execution to S1 | E2/E3: same success as E1, one Astra call (about 7 s) fewer per attempt; E0 far worse | **yes** (for the simple module) |
| Learned S1 improves both success and total cost → retain the learned S1, extend its scope | E3 = E2 | no |
| Benefits occur only with the fixed source program → deploy for that skill, limit the claim | all 26 Astra proposals were cup A's grasp. Extension 9.1, with a grasp from above: the context-free ranking still transfers (31/33 first try with cup A's records); the contextual model needs records of the same program (28/33 mismatched, 33/33 matched) | **yes for the learned model; the simple ranking degrades gracefully.** Keep experience per grasp program |

**Operationally.** Selecting a persistent pre-closing correction from experience on other cups solves new cups of this
family on the first attempt. A ranked list or Astra reading records both work; the ranked list is faster and needs no
reasoning call.

**Mechanism the evidence supports.**
1. Cup A's grasp is miscalibrated for shorter cups: the thumb ends over the opening (batch 1).
2. One persistent shift toward the cup and down fixes nearly all of them.
3. Experience on other cups identifies that shift. With 64 records it is learned equally well with or without state
   conditioning.
4. Online updating and retry rules change little within 3 attempts.
5. Astra's own geometric reasoning, without the records, trusts its earlier success and then moves the wrist in a
   direction that does not help.

**What the plan asked:**
- **Is a learned System 1 needed?** No; a simple experience-based search does the same.
- **What responsibility should it take from Astra?** Selecting and retrying the bounded correction of a proposed grasp,
  and reporting the outcome.
- **How much initialization?** 64 recorded interactions over 8 cups; 16 is not enough and unstable.
- **Do online updates add value beyond ordinary search?** Not measurably here.

## 7. Deviations, amendments and caveats

- **Amendment 1** (16:41, commit 7888600; fresh executions started 16:42): settings chosen per role. The committed family rule would have
  given the context-free methods h = 1 on a one-scene margin at a seven-scene first-try cost. Both versions are in
  `decisions.json` and the grid table.
- **Amendment 2** (16:51, commit 3309fb5; experiment C runs started after 17:30): fixed how E3's update strength and
  retry rule follow from experiment C.
- **Order.** The remaining development banks of the interface pilot ran after the locked test. Every locked setting was
  frozen before.
- **Information given to Astra.** It saw no images in the correction-choice calls: the same numeric geometry the local
  methods use, plus its own running memory.
- **Statistical scope.**
  - Two locked geometries, both shorter than cup A (height 0.92 and 0.88 of cup A), where the nominal grasp fails. That is
    the regime the source data covers well.
  - 20 scenes is a targeted validation, not broad evidence across dexterous manipulation.
- **Timing.** Simulator execution is about 5x slower than the robot. Robot time removes that overhead; both are reported.
- **The simulator is the only noise source.** It turned out to matter: 13 of 160 batch-2 repeats and several
  fresh-execution corrections flip.

## 8. Files

| what | where |
|---|---|
| per-attempt decisions and outcomes (replay) | `mechanism/expA_replay.jsonl`, `expD_replay.jsonl`, `grid_replay.jsonl`, `expC_weight_noisy_replay.jsonl`, `initialization/expB_*_replay.jsonl` |
| fresh executions | `mechanism/online_{verify,expC}_<cup>.jsonl` |
| Astra proposals (prompts, responses, pre-closing images) | `{mechanism,prospective}/proposals/<cup>/<start>/` |
| compared systems (every Astra call) | `{mechanism,prospective}/systems_*.jsonl`, `systems_calls/` |
| summaries | `*_summary.json`; tables `REPORT_DATA.md`; figures `figures/{curves,systems,systems_pilot}.png` |
| manifests (simulator, git, settings) | `manifests/` |

## 9. Extensions (beyond the pre-registered design; the user asked to keep running)

### 9.1 Changed nominal grasp (done 20:28)

**Setup.**
- Every correction was executed at a different nominal grasp, on all 11 development cups at a new start (v11).
- The new nominal is the cup-transfer reference grasp from above (Astra's Unload-Cans grasp re-anchored to each cup),
  instead of cup A's side grasp. Its closing decision holds the wrist; as in the hand-off it became a target at the
  measured wrist.
- The methods were replayed with two sources:
  - the study's records from cup A's program (mismatched);
  - 64 records of the new program on the other cups (matched, leave-one-cup-out, 3 seeds).

**Results.** First try / within 3, 11 cups x 3 selections = 33:

| method | source: cup A's program (mismatched) | source: the same program on other cups (matched) |
|---|---|---|
| A0 nominal | 10/11 cups | - |
| A1 one correction | 31 / 31 | 33 / 33 |
| A2 ranked search | 31 / 33 | 33 / 33 |
| A3 context-free, updating | 31 / 33 | 33 / 33 |
| A4 contextual, frozen | 28 / 30 | 33 / 33 |
| A5 contextual, updating | 28 / 33 | 33 / 33 |

- [E] **The failures move with the grasp.**
  - With the grasp from above, the short cups that defeat cup A's side grasp hold: cup B, dev f1, dev f3.
  - The tall wide dev f4, where the side grasp held, fails. 17 of 31 corrections hold there; h19 holds, h25 does not.
- [E] **Mismatched records.**
  - The contextual model sees an unfamiliar pre-closing state and falls back to its prior, i.e. the nominal grasp first.
    On dev f4 that fails at all three selections, and on dev g4 it picks h12, which fails.
  - The context-free ranking still applies the global correction. It misses the first try only where h25 fails, and
    recovers with h19.
- [I] **Mechanism.**
  - The learned contextual model is calibrated to the program its records come from: perfect with matched records, the
    worst of the experience-based methods with mismatched ones.
  - The simple ranking degrades more gracefully when Astra's grasp changes.
  - With matched records, the simple ranking is just as good. This supports the recommendation and the scope statement:
    keep experience per grasp program, and keep the module simple.

### 9.2 Initialization curve in fresh executions (done 21:15)

A2 and A5 at 16 and 256 interactions on the 22 fresh development scenes; the 64 runs are those of section 3.2. 66 runs per
cell = 22 scenes x 3 selections.

| | 16 | 64 | 256 |
|---|---|---|---|
| A2 ranked search: first / within 3 | 44 / 58 | 63 / 65 | 61 / 65 |
| A5 contextual updating S1: first / within 3 | 47 / 60 | 59 / 66 | **64 / 66** |

- [E] **16 interactions are not enough in fresh executions either**, and they are unstable: ranked search 10, 15 and 19
  of 22 first-try successes across the three selections.
- [E] **At 256 the contextual model is ahead on the first try** (64 against 61). This matches the replay reference (88
  against 84 of 96).
- [I] State conditioning may start to pay off with about four times the chosen data. The difference is 3 of 66 and not
  resolved. At the chosen 64 it does not pay off, and the recommendation stands. Test this first if the task and the
  data grow.
