# System 1 beneath Astra: local execution adaptation pilot (dex hand, 2026-10-10)

**Question.** Is a useful System 1 one that keeps a distribution over bounded adjustments to Astra's action program and
updates it from actual execution outcomes? H1: nearby variants beat repeating the original. H2 (primary): updating the
proposal distribution from previous outcomes finds reliable variants faster than a fixed distribution.

Evidence labels: **[E]** measured here, **[I]** interpretation. This is a small pilot (6 conditions x 3 operations, 3 fresh
starts per condition): it can show a negative or inconclusive result, not establish a small improvement.

## 1. Answers

| question | answer |
|---|---|
| Does execution flexibility help (H1)? | **[E] No, in this setup.** B and C never succeeded where Astra's original failed. On fresh starts B scored 41/54 vs A 42/54 (-1.9 pp, 95 % CI [-5.6, 0.0]); B needed 0.8 more attempts to a first success (censored at 10). |
| Does updating proposals from history help beyond random search (H2)? | **[E] No outcome difference: C = B everywhere** (first success, 41/54 fresh-start success each; difference 0.0 pp). **[E]** The update did what it was designed to do: C's attempts 6-9 scored higher than B's in all 6 door conditions (hardest: mean J 0.19 vs 0.15). **[I] H2 is inconclusive, not refuted:** where the original failed, the bounded program family contained no successful candidate for either arm to find. |
| Are improvements reliable on fresh starts? | **[E] There were no improvements to carry over.** **[E]** Fixed parameters are brittle in grasp: Astra's closure (selected by every arm) succeeded on the adaptation start in 6/6 conditions, but on fresh start e2 (can displaced 8 mm) the hand closed without capturing the can in 4/6 conditions, lifted it 28 cm in 2/6, and succeeded again in a separate re-execution for the video. [I] That start is a knife edge: the closure has no margin for this displacement. |
| Does the next System 1 need better search, richer motor actions, or continuous feedback control? | **[I] Continuous feedback control first, richer motor actions second, better search last.** The door failures are near-threshold shortfalls (the heavy door stops at 1.17-1.30 rad, 1.33 needed) that every bounded change of contact point, direction or speed made worse; what is missing is "keep pushing until the door is closed", i.e. termination and extension driven by the observed task variable, which no open-loop parameter search over this program can express. |

Continuation criterion (about +15 pp fresh-start success, better sample efficiency, same direction in at least two
operations): **not met** for either hypothesis.

## 2. What this tells us about the required System 1

1. **[E] Astra's program already carries the embodiment-specific behaviour for grasp and transport across the tested
   physics.** With 0.8-1.2x can mass and 0.7-1.0x contact friction, the original succeeded on the first attempt in 12/12
   conditions; 53/60 perturbed grasp candidates and 71/72 perturbed transport candidates also succeeded. [I] There is no
   room for local tuning there: the bottleneck is not the parameters of a working program.
2. **[E] Where the program fails, it fails by falling just short, and bounded variation makes it worse.** Door with 1.2x
   panel mass: nine repeats of the original reached 1.17-1.30 rad; B's and C's 12 distinct variants reached 0.74-1.20 rad.
   Only 7 of 72 perturbed door candidates succeeded in any condition (vs 53/60 grasp, 71/72 transport): Astra's push is
   a narrow optimum of this family. [I] The required change is to extend the action until the task variable reaches its
   target, a closed-loop decision, not a nearby open-loop program.
3. **[E] Execution noise is as large as the parameter effects near the threshold.** Repeating the original unchanged (arm
   A) found a success in 5/6 door conditions (attempts 1, 1, -, 2, 1, 4) because repeated executions varied by up to
   0.65 rad of final door angle (1.13-1.45 rad with 0.7x friction; 0.89-1.54 rad with 1.2x mass and 0.7x friction). B and C spent the same
   attempts on variants and found successes only where attempt 1 had already succeeded. [I] A System 1 that samples its
   own outcomes must separate noise from improvement (repeated trials per candidate), or it will select noise.
4. **[E] Short-term memory moved the proposal in the right direction but could not change outcomes.** C updated its
   proposal in 15/18 blocks (kept q0 in three grasp blocks where all scores tied) and its later proposals scored higher
   than B's in every door condition. [I] History is useful only if the action space contains the fix; here it did not.
5. **[E] A single adaptation start with a saturating score cannot detect fragility.** All successes score J = 1.25, so B
   and C selected the original by the earliest-attempt tie rule in every grasp and door block where it succeeded; the
   grasp's missing margin at start e2 (miss in 4/6 conditions, capture in 2/6 and in an independent re-execution, with no
   monotonic dependence on mass or friction) was invisible during adaptation. [I] Selection needs a robustness signal (several starts or
   a margin), which again points to feedback during execution rather than more offline-style search.

## 3. What was run

- **Environment (verified against the code before editing):** `Humanoid-Stack-Can-Into-Drawer-v0`, EgoVLA Ego-Humanoid
  benchmark, Isaac Sim 5.1 / IsaacLab 2.3, Unitree H1 + two Inspire hands, room 1 table 2 (randomize_idx 5788). Harness
  `astra_explore/isaac_env.py`: absolute wrist targets (position + finger / palm directions) tracked by the benchmark's
  differential IK (left DLS, right pseudo-inverse) along a path of at most 1 cm and 3 deg per 30 Hz step; four closure
  fractions per hand (`finger_base`, `finger_tip`, `thumb_rotate`, `thumb_bend`) mapped onto the 12 hand joints at most 1/30
  per step. Only change to the harness: an optional config hook (used to switch rendering off). Success criteria are the
  task's own (`stack_can_into_drawer.py`): can within 0.09 m (xy) and 0.02 m (z) of the plate; door bottom joint at least
  80 % closed (>= 1.326 rad of 1.658).
- **Source program:** pilot attempt 10 only (11 decisions, environment success at step 298); replayed here it still
  succeeds (can 2.6 cm from the saucer centre, door 1.50 rad). Attempt 7 was not needed.
- **Operations** (start = state after a fixed prefix of the reference program, rebuilt physically, never teleported):

| operation | start | program | window | parameters (bounds) |
|---|---|---|---|---|
| grasp and hold | after D0-D1 (open hand beside the can) | D2 close, D3 lift | 15-step hold (0.5 s) | the four right-hand closure fractions of D2-D3, reference +- 0.15 |
| transport and release | after D0-D3 (can held 28 cm up) | D4 carry, D5 lower, D6 release | 10-step settle | speed of D4-D5 [0.5, 1.25]; finger_base, finger_tip, thumb_bend held during D4-D5 (+- 0.15); release D6 unchanged |
| close the door | after D0-D7 (can on the saucer, left hand outside the open panel) | D8-D10 left-palm push | 10-step settle | left-hand contact offsets along world x and z (+- 1 cm); push-direction rotation about world z (+- 10 deg); speed [0.5, 1.25] |

  Speed s < 1 scales the per-step path limits by s and stretches each decision to ceil(steps / s) (more control steps);
  s > 1 shortens decisions but never below the steps the path needs at the existing 1 cm / 3 deg limits, which are never
  exceeded. Door offsets transform the left-hand targets as p' = p0 + Rz(phi)(p - p0) + (dx, 0, dz), p0 = the wrist at the
  operation start; orientations unchanged.
- **Physical conditions** (predetermined order): (mass, friction) = (1.0, 1.0), (0.8, 0.7), (1.2, 1.0), (1.0, 0.7),
  (0.8, 1.0), (1.2, 0.7). Grasp / transport: can mass and inertia, can friction and every right-hand link's friction
  (static and dynamic). Door: door panel ("bottom" link) mass and inertia, its friction and every left-hand link's
  friction. Applied through the PhysX views at the operation start and verified by read-back (e.g. can 0.35 -> 0.42 kg,
  friction 0.5 -> 0.35).
- **Starts:** adaptation on the nominal start; fresh evaluation starts e1-e3: grasp / transport with the can displaced
  8 mm at 0 / 120 / 240 deg before the reference prefix (12 / 12 valid; transport starts hold a physical grasp); door
  with the panel 0.03 / 0.06 / 0.09 rad less open.
- **Score:** J = S + 0.25 clip(P, -1, 1) - D - V, frozen in `manifest.json` before the formal run (definitions there).
  V: this environment has no existing constraint-violation signal (always 0; not replaced by a contact-force threshold).
- **Arms and schedule:** exactly as specified (attempt 1 = original for all; attempts 2-5 shared B = C draws from q0;
  C's CEM-style update after attempt 5; attempts 6-9 from each arm's distribution with common uniforms; best tested
  candidate frozen; three fresh starts). q0 = bounded diagonal Gaussian at the original, sigma0 = width / 4,
  inverse-CDF truncated sampling. Identical parameters on the same start share one execution.
- **Restoration:** Isaac Lab's `reset_to` restores poses and velocities but sets joint targets to the measured state; the
  snapshot therefore also restores the commanded joint position / velocity / effort targets, the env's episode counter
  and the harness's closure-command state. [E] Restored rollouts follow a continuous one within 2 mm (can) for grasp and
  transport; restores are not bit-exact (PhysX contact caches), so repeated executions differ by about 1-3 mm (can), and by up to
  0.65 rad of final door angle in the hardest door condition (contact-rich pushing of a heavy, slippery panel). Arm A's nine attempts were therefore executed, not copied.

## 4. Results

### Per operation and overall

| scope | arm | conditions with a success in 9 attempts | first-success attempt per condition (- = none, censored) | restricted mean first success (censored = 10) | fresh-start success | object loss (eval / adapt) | adaptation control steps |
|---|---|---|---|---|---|---|---|
| grasp | A | 6/6 | 1 1 1 1 1 1 | 1.00 | 14/18 (78 %) | 0 % / 0 % | 3240 |
| grasp | B | 6/6 | 1 1 1 1 1 1 | 1.00 | 14/18 (78 %) | 0 % / 0 % | 3240 |
| grasp | C | 6/6 | 1 1 1 1 1 1 | 1.00 | 14/18 (78 %) | 0 % / 0 % | 3240 |
| transport | A | 6/6 | 1 1 1 1 1 1 | 1.00 | 18/18 (100 %) | 0 % / 0 % | 4590 |
| transport | B | 6/6 | 1 1 1 1 1 1 | 1.00 | 18/18 (100 %) | 0 % / 2 % | 4732 |
| transport | C | 6/6 | 1 1 1 1 1 1 | 1.00 | 18/18 (100 %) | 0 % / 2 % | 4692 |
| door | A | 5/6 | 1 1 - 2 1 4 | 3.17 | 10/18 (56 %) | 0 % / 0 % | 5080 |
| door | B | 3/6 | 1 1 - - 1 - | 5.50 | 9/18 (50 %) | 0 % / 0 % | 5499 |
| door | C | 3/6 | 1 1 - - 1 - | 5.50 | 9/18 (50 %) | 0 % / 0 % | 5506 |
| overall | A | 17/18 | | 1.72 | 42/54 (78 %) | 0 % / 0 % | 12910 |
| overall | B | 15/18 | | 2.50 | 41/54 (76 %) | 0 % / 1 % | 13471 |
| overall | C | 15/18 | | 2.50 | 41/54 (76 %) | 0 % / 1 % | 13438 |

No initialization failure occurred. Constraint violations: signal unavailable (0 by definition). Steps to first success
(censored runs report all their adaptation steps) are in `results.json`.

| contrast | fresh-start success difference | 95 % CI (paired, operation-stratified) | per operation | attempts saved to first success (censored at 10) |
|---|---|---|---|---|
| H1: B - A | -1.9 pp | [-5.6, 0.0] | grasp 0, transport 0, door -6 | -0.8 [-1.7, 0.0] |
| H2: C - B | 0.0 pp | [0.0, 0.0] | 0 / 0 / 0 | 0.0 |
| C - A | -1.9 pp | [-5.6, 0.0] | grasp 0, transport 0, door -6 | -0.8 [-1.7, 0.0] |

### The door, condition by condition (the only operation with failures to repair)

| condition (mass, friction) | first success A / B / C | fresh starts A / B / C | final door angle over A's 9 repeats (needed 1.326) | mean J of attempts 6-9, B vs C |
|---|---|---|---|---|
| (1.0, 1.0) | 1 / 1 / 1 | 3 / 3 / 3 | 1.42-1.56 | 0.160 vs 0.175 |
| (0.8, 0.7) | 1 / 1 / 1 | 3 / 3 / 3 | 1.37-1.44 | 0.419 vs 0.430 |
| (1.2, 1.0) | - / - / - | 0 / 0 / 0 | 1.17-1.30 | 0.147 vs 0.189 |
| (1.0, 0.7) | 2 / - / - | 1 / 0 / 0 | 1.13-1.45 | 0.169 vs 0.195 |
| (0.8, 1.0) | 1 / 1 / 1 | 3 / 3 / 3 | 1.47-1.55 | 0.453 vs 0.724 |
| (1.2, 0.7) | 4 / - / - | 0 / 0 / 0 | 0.89-1.54 | 0.149 vs 0.172 |

### Reading against the pre-registered interpretations

| observation | here |
|---|---|
| B and C beat A; C does not beat B | not observed (B and C never beat A) |
| C finds success sooner and has higher fresh-start success than B | not observed (identical) |
| Adaptation success disappears on fresh starts | partly: grasp succeeds on the adaptation start everywhere but misses on the knife-edge start e2 in 4/6 conditions |
| Successful candidates exist, but C selects poorly | no: where the original failed, no B / C candidate succeeded |
| No arm succeeds | door with 1.2x mass: no arm, no candidate, no repeat succeeded |
| Benefit appears in only one operation | no benefit in any operation |

### Figures and videos

- `curves.png`: success vs attempt (cumulative discovery). B and C coincide exactly, so C's line covers B's.
- `eval.png`: fresh-start success per operation and arm.
- `trajectories.png`: paired trajectories of each arm's selected parameters on the same fresh start (grasp and transport,
  nominal physics: identical successes; door with 0.7x friction: all three stall below the closed threshold).
- `videos/` (rendered after the formal blocks; re-executions, so outcomes can differ slightly from the logged ones):
  door with 1.2x mass, original vs B's and C's attempt-6 variants (all three fall short, as logged); door with nominal
  physics on fresh start e1 (success); grasp with nominal physics on fresh starts e1 and e2 (both succeed in the
  re-execution, although the logged evaluation missed at e2: the knife-edge start of section 2); transport with 1.2x
  mass, B / C's selected slow variant on fresh start e1. `videos/videos.json` lists each re-executed outcome.

## 5. Budget and timing

| phase | time |
|---|---|
| implementation and setup (separate from the budget): interface checks, probe, code, discarded dry run | 16:37:34 - 17:00:42 (23 min) |
| formal run: startup 26 s, smoke test 5.3 min (12 / 12 starts valid; replay, parameter effects, scoring, throughput verified; horizons frozen) | 17:00:42 - 17:06:02 |
| 18 matched blocks (all three arms + their fresh-start evaluations), none skipped | 17:06:02 - 18:15:47 |
| representative videos (7, one simulator process at a time) and analysis | 18:15:51 - 18:19:04; total formal runtime 78 min of the 120 min budget |

447 formal rollouts (plus 18 smoke rollouts), 36,929 control steps, 70 min of simulation. No Astra call was made; every
arm reused the fixed attempt-10 program.

## 6. Limitations

- The feedback is a state-based evaluator (object heights and distances, door angle), richer than the visual / sparse
  success signal of the original Astra setting; it isolates execution adaptation but is not available on a real robot.
- With the predetermined grid, grasp and transport never failed on the adaptation start, so H1 and H2 could only be
  tested on the door (6 conditions); the fresh-start variations are small (8 mm can shift, 0.03-0.09 rad door angle).
- Restores are not bit-exact; near the door threshold, execution noise decides individual outcomes.
- Speed multipliers above 1 cannot exceed the existing interpolation limits, so they shorten dwell times only.
- One source program (attempt 10), one robot scene, three fresh starts per condition: exploratory, not a benchmark.

## 7. Files

`manifest.json` (configuration, program, bounds, seeds, physics, score, horizons) · `smoke.json` · `rollouts.jsonl`
(every execution with its outcome components and a trajectory) · `attempts.jsonl` (every arm x attempt with the proposal
state and C's update) · `blocks.jsonl` · `results.json` / `results.md` · `timing.json` · `SETUP_TIME.json` · figures ·
`videos/`. Code: `astra_explore/s1local/` (env, ops, search, runner, analyze, render, probe).
