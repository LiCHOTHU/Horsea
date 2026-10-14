# Control diagnostics: does the current executor need a better System 0? (2026-10-14)

**Status: COMPLETE.** This is the first batch of the user's staged plan (control diagnostics and simple-feedback comparison).

**Gate result: the chosen grasp itself is unsuitable. Improve proposal selection before any motor-policy training.**
- No System 0 training: the gate does not justify it.
- The current executor (C0) carries out what it is told.
- Every failure comes from where the proposal puts the thumb, and neither feedback executor fixes it.

Tags: [E] measured, [I] interpretation.
- **Code:** `astra_explore/cup_transfer/control_diag{,_offline}.py`.
- **Frozen system and conditions:** `config/freeze.json`.
- **Episodes** (every segment instrumented): `episodes/<condition>.jsonl`.
- **Tables:** `REPORT_DATA.md`.

## 1. Setup

- **Fixed system** (recorded in `freeze.json`): the Astra interface and prompts, the observations and actions, the controller
  (IK, 1 cm / 3 deg per 30 Hz step, closure ramp 1/30 per step), physics, the task-aligned success test, and the existing
  source experience with its collection cost.
- **Splits:**
  - source: cups dev f1-f4 and A;
  - development: cup B, cylinder C, cups dev g1-g4 (and cup A as the nominal reference);
  - locked test: new objects, not built yet. The two batch-2 locked cups are used up and are not evidence of general transfer.
- **Coarse grasp proposal for every executor:** cup A's program re-anchored to the object. It is Astra's own proposal in all
  26 of its proposals so far. Each condition adds its error.
- **Shared constraints:** every executor gets the same observations, actuator limits and 400-step simulated-time budget,
  then a 40-step evaluation window.
- **12 development conditions x 2 starts** (r1, r2, shared by every executor, so every comparison is paired):
  - nominal grasps on A, C, B, g1, g4, g2;
  - pose errors: A +1 cm x, A -1 cm y, C +1 cm z, g3 12 deg yaw;
  - imperfect initial contacts: A pressed 0.8 cm into the cup; C met low and early.

| executor | what it does |
|---|---|
| C0 | the current common executor, unchanged |
| C1 motion feedback | wrist: integrates the remaining tracking shortfall once the commanded target has reached the reference (gain 0.5, at most 1.5 cm). Fingers: after closing, adds 0.5 x (commanded - measured) closure, at most +0.15 |
| C2 local grasp feedback | C0 + a bounded contact / hold loop (`C2_RULES`, documented in `freeze.json`). It checks that the object follows the wrist during the lift. On a slip: put the object back, open, retreat, adjust the grasp by documented geometric rules, re-close harder, lift slower. At most 2 regrasps |
| C2b | C2 with its slip trigger recalibrated on this batch's C0 / C1 lifts (see 4.2). The regrasp rules are unchanged |

**Diagnosis from simulator state** (evaluation only; no executor uses it beyond what its rules document):
- tracking of the intended vs. the achieved wrist and fingers at every 3-step segment;
- object motion before, during and after contact;
- contact geometry at the end of the first closing;
- the follow ratio during the lift;
- each failure classified as motion execution, planning / approach, grasp unsuitable, or contact control.
- **State estimation is not tested:** every executor receives the tracked object pose (an accurate-state prototype).

## 2. Results (72 + 24 episodes)

| executor | all | nominal grasps | pose errors / imperfect contacts | safety violations | C0 failures fixed |
|---|---|---|---|---|---|
| C0 current | 14/24 | 6/12 | 8/12 | 0 | - |
| C1 motion feedback | 14/24 | 6/12 | 8/12 | 0 | 0 of 10 |
| C2 local grasp feedback | **0/24** | 0/12 | 0/12 | 0 | 0 of 10 |
| C2b (recalibrated trigger) | 13/24 | 5/12 | 8/12 | 0 | 0 of 10 |

Per condition (successes of 2):

| condition | C0 | C1 | C2b |
|---|---|---|---|
| A, C, g2 nominal | 2, 2, 2 | 2, 2, 2 | 2, 1, 2 |
| B, g1, g4 nominal | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| A +1 cm x, A -1 cm y | 0, 0 | 0, 0 | 0, 0 |
| C +1 cm z, g3 12 deg yaw | 2, 2 | 2, 2 | 2, 2 |
| A pressed into the cup, C met low | 2, 2 | 2, 2 | 2, 2 |

## 3. What fails

**Every one of C0's 10 failures is the same fault.** At the end of closing (C0 and C1, 20 failed episodes), the thumb tip is
1.5-2.9 cm inside the cup's outer radius and 1.3-1.8 cm above its top, i.e. over the opening. In all 28 successes it is
1.0-1.1 cm outside the wall and 1.2-3.3 cm below the top.
- The hand reaches the commanded pose: the wrist ends within 0.1 cm of it (median; max 0.54 cm) before closing. The other
  fingers are at the wall (4 of 4 within 1.5 cm) and the index finger presses with 1.1-5.4 N (2.1-4.1 N in successes).
- The cup then stays on the table or slips as the wrist rises. Its rise is 0.2 x the wrist's or less, against 0.65 x or more
  in every success.
- Diagnosed category: **grasp unsuitable** in 10/10.
  - Motion execution failures: 0.
  - Pushed or tipped before closing: 0.
  - A reasonable grasp lost in contact: 0.
- [I] The proposal's thumb-over-the-top side grasp fits taller cups only; a 1 cm pose error moves cup A into the same fault.
  This is batch 1's mechanism, now confirmed as the only failure mode across 5 conditions.

## 4. Why feedback does not help

### 4.1 C1 (motion feedback)
- It changes no outcome (0 better, 0 worse in 24 pairs).
- There is little tracking error to correct: 0.1 cm at the end of the approach, about 0.5 cm in contact at the end of closing.
- Its finger term drives blocked fingers about 0.05 further, which does not matter when the thumb is over the opening.

### 4.2 C2 (local grasp feedback)
- **The first version failed every episode,** including the 14 that C0 holds. Two triggers fired on normal lifts:
  - its slip check (object rise < 0.5 x the wrist's after 1.5 cm): in successful lifts the cup lags the wrist at first, at
    0.42-0.76 of its rise;
  - its 10 deg tilt rule: successful lifts tilt the cup by up to 20 deg.
  - Every needless regrasp then failed.
- **C2b recalibrated the trigger on this batch's C0 / C1 lifts.** Once the wrist has risen at least 3 cm:
  - every success has a ratio of 0.65 or more, every failure 0.18 or less;
  - the threshold is 0.4, and the tilt rule is dropped.
- **C2b then catches exactly the failing grasps and rescues none of them** (11 episodes with 2 regrasps each):
  - **Lowering is limited by the table.** Its rule lowers the grasp when the thumb is above the top or over the opening. But
    this side grasp's little finger sits about 1 cm above the table, so lowering is capped and the thumb cannot get below
    the rim.
  - **The regrasps push the cup** 4.5-12 cm across the table.
  - One C2b episode (cylinder C, r2) failed where C0 held. It is run-to-run variability of a borderline grasp.
- **Development changes, disclosed** (`freeze.json`, `C2_RULES["smoke_fix"]`, `C2b`):
  - two mechanical fixes after smoke tests: retreat and re-approach instead of dragging the cup; never drive a fingertip into
    the table;
  - the trigger recalibration.
  - None of them looks at which correction would have worked.

## 5. Decision gate (the user's rules)

| gate condition | result |
|---|---|
| C1 resolves most failures | no (0/10) |
| C2 resolves most failures | no (C2 0/10, C2b 0/10) |
| feasible commands reach the region, but contact handling fails | no: the region itself is wrong (thumb over the opening); contact handling of reasonable grasps never failed |
| **the chosen grasp itself is unsuitable** | **yes, 10/10 failures → improve proposal selection before motor-policy training** |
| accurate-state works, estimated-state fails | not testable here (accurate state only) |

**Screening target.** C0 holds 6/12 of the nominal development cases, far below the ~90 % target, but every miss is the
proposal's grasp placement. With an experience-selected correction of the placement before closing (delegation study), the
same executor held the two locked cups 20/20. So the executor is sufficient. The decision that needs improving is where to
put the grasp, which the memory experiment targets.

**Next (per the plan):** the three-family memory stream with C0 as the fixed executor. No System 0 training pilot: the gate
does not justify it.
