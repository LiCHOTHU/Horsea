# Overnight System-1 / System-2 transfer study, 2026-10-10

Research question: can a learned policy (System 1) learn stable execution from Astra's own interaction data (System 2),
and can later Astra corrections improve that policy once Astra is switched off again?

Evidence labels: **[E]** measured in this run, **[A]** Astra's own statements (its decision records), **[I]** my inference.
All success counts are full-task successes over evaluated episodes, with the pool named. Three to five starts per cell:
every comparison below is exploratory.

## 1. Bottom line

| platform | Astra from reset (System 2) | best standalone student (System 1) | what blocked the rest |
|---|---|---|---|
| RoboTwin single-arm, place_container_plate | source 2/4, E01 1/1 dev | original pretrained student E02 2/5 dev, 3/10 test; trained on Astra data: E03 1/3, E05 0/5 dev and 1/10 test, E10 0/3 dev and 3/10 test | anchor E05 0/5 dev -> NO_STUDENT_SIGNAL gate for takeover and corrections |
| RoboTwin bimanual, lift_pot | source 0/4 | original pretrained student E02 3/3 dev, 2/10 test | no Astra success to learn from (BLOCKED_DATA) |
| Dex hand, Stack-Can-Into-Drawer | source 0/6 | no pretrained student (E02 NOT_AVAILABLE) | no Astra success to learn from (BLOCKED_DATA) |
| LIBERO-90 task 48 | E01 1/3 dev | none (BLOCKED_INTERFACE) | no compatible student pipeline |

- **[E] No channel from Astra to the student produced a working student tonight.** The data channel itself is valid on all
  three simulators (replay of the recorded native commands reproduces Astra's motion), but on the one platform where Astra
  produced successes (two), fine-tuning the pretrained student on them gave 0/5 on development starts, 1/10 on test starts
  (the unchanged student: 2/5 and 3/10) and 0/2 on the very starts it was trained on.
- **[E] The hybrid and correction channels (E08, E09, E11, E12) were never reached.** The registered gate fired on
  single-arm (anchor without a success in five development evaluations); on bimanual and dex there was no Astra success at
  all; LIBERO has no student.
- **[E] On lift_pot the original pretrained student beat Astra:** 3/3 development and 2/10 test starts, against 0/4 for
  Astra's source episodes. The drop from 3/3 to 2/10 shows how far three starts can mislead.
- **[I] The answer to "which information channel actually works" is therefore "none yet, at this data scale and with the
  registered fine-tuning recipe".** The bottleneck on single-arm is the student's fit to one or two demonstrations
  (open-loop error on the demonstrated frames fell only 10-25 %); on bimanual and dex it is the teacher's own success rate.

## 2. Where everything is

- Job: systemd user unit `s1s2-overnight` (started 06:16:37, window to 14:16:37), orchestrator `python -m s1s2.orchestrator`.
- Logs: `experiments/s1s2_overnight_2026-10-10/orchestrator.log` (human), `orchestrator.stdout.log`, per activity
  `<platform>/activities/<id>/worker.log`.
- Outputs: `experiments/s1s2_overnight_2026-10-10/` with `REPORT.md` (auto-generated matrix, answers, per-cell metrics),
  `metrics.json`, `state.json` (resumable progress), `astra_ledger.jsonl` (every Astra call and its token usage),
  `LAUNCH.md` (launch record and every rule change with its time), per platform `bank.jsonl` (chronological teacher-attempt
  bank incl. failures), `datasets/*.json` (membership and rule), `episodes/*` (per-episode transitions.npz, episode.json with
  trace, Astra prompts / images / responses per decision), `activities/*/ckpt.pt` + `optimizer.pt` (student checkpoints).
- Code: `astra_explore/s1s2/` (commits 86fe0ab through the commit that adds this report, authored as Licho, not pushed).

## 3. The 12 x 4 matrix

| cell | configuration | libero | robotwin_single | robotwin_bimanual | dexhand |
|---|---|---|---|---|---|
| E01 | Astra-only (teacher capability and interaction / computation cost) | COMPLETE 1/3 dev | PENDING_BUDGET 1/1 of 3 requested dev | PENDING_BUDGET | PENDING_BUDGET |
| E02 | Original pretrained S1, unchanged | NOT_AVAILABLE | COMPLETE 2/5 dev; test 3/10 | COMPLETE 3/3 dev; test 2/10 | NOT_AVAILABLE |
| E03 | S1 trained on D1 (first successful episode) | BLOCKED_INTERFACE | COMPLETE 1/3 dev | BLOCKED_DATA | BLOCKED_DATA |
| E04 | S1 trained on D3 (first three successful episodes) | BLOCKED_INTERFACE | BLOCKED_DATA | BLOCKED_DATA | BLOCKED_DATA |
| E05 | S1 trained on DN (first N = min(10, available) successful episodes) - main anchor | BLOCKED_INTERFACE | COMPLETE 0/5 dev; test 1/10 | BLOCKED_DATA | BLOCKED_DATA |
| E06 | E05 with a shorter executed action chunk (about half, >= 1) | BLOCKED_INTERFACE | COMPLETE 0/3 dev | BLOCKED_DATA | BLOCKED_DATA |
| E07 | S1 trained on all valid teacher trajectories up to the N-th success (successes + failures) | BLOCKED_INTERFACE | COMPLETE 0/3 dev | BLOCKED_DATA | BLOCKED_DATA |
| E08 | E05 then one scheduled Astra takeover at half the native step budget, current observation only | BLOCKED_INTERFACE | NO_STUDENT_SIGNAL | BLOCKED_DATA | BLOCKED_DATA |
| E09 | E08 plus the last three student interaction records before handover | BLOCKED_INTERFACE | NO_STUDENT_SIGNAL | BLOCKED_DATA | BLOCKED_DATA |
| E10 | E05 continued for dU updates on its original data | BLOCKED_INTERFACE | COMPLETE 0/3 dev; test 3/10 | BLOCKED_DATA | BLOCKED_DATA |
| E11 | E05 continued for dU updates on original data + Astra corrections (half / half batches) | BLOCKED_INTERFACE | NO_STUDENT_SIGNAL | BLOCKED_DATA | BLOCKED_DATA |
| E12 | E05 continued for dU updates on original data + additional Astra-from-reset demos (half / half) | BLOCKED_INTERFACE | NO_STUDENT_SIGNAL | BLOCKED_DATA | BLOCKED_DATA |

Status reasons for every non-COMPLETE cell, per-cell metrics (starts, native steps, policy inference calls, Astra calls
and tokens, wall time, updates, dataset size, retries) and the machine-readable `metrics.json` are in `REPORT.md`.

## 4. RoboTwin single-arm, place_container_plate

**Setup.** Cohort: pretrained-policy adaptation on a SEEN task (Horsea FM base step300000, trained on 50 RoboTwin tasks
including this one). Student interface: head/left/right RGB at 120x160, 14-D commanded joint vector, CLIP instruction;
predicts 16 future joint vectors, executes 8 (`take_action qpos`), 400-step budget. Astra: absolute end-effector targets
executed by RoboTwin's cuRobo planner. Labels: the commanded joint vector every 15 physics steps during Astra's motions
(RoboTwin's own demo convention). Recipe: the project's `continue_train` (decoder only, AdamW lr 1e-5, warm-up 250,
batch 64), U = 1000, dU = 500. Pools (prevalidated seeds): source 15, dev 5, adapt 10, test 10.

1. **Could Astra's behavior be converted into valid student training data?** [E] Yes. Replay of the recorded commands from
   the recorded initial state reproduced the motion (median end-effector deviation 0.0 cm, p90 0.1 cm) and the success.
   Source collection: 4 episodes, 2 successes (14 and 19 calls), 86 calls, 1,331 native steps; the two successes give
   467 labelled frames. [E] Astra's last planner motion can end past the 400-step limit (the two failures ended at 436 and
   426 steps); no success occurred after step 400.
2. **Did the standalone student improve?** [E] No. Original student E02 2/5 dev (1500018 and 1500020) and 3/10 test;
   anchor E05 (DN, N = 2) 0/5 dev (three starts plus two from the gate's expansion) and 1/10 test. On the ten paired test starts the successes were disjoint: E02 on 1500035, 1500041, 1500042;
   E05 on 1500038; E10 on 1500036, 1500037, 1500039 (E02 vs E05: 4 discordant starts, exact paired p = 0.63).
   [E] Diagnosis on the two training starts themselves: E02 0/2, E03 0/2, E05 0/2. [E] Open-loop
   error on the demonstrated frames (first 8 predicted actions vs labels, mean over arm joints): E02 0.032 / 0.036 rad,
   E03 0.024 / 0.039 rad, E05 0.029 / 0.030 rad on episodes 1500001 / 1500003. [E] Behaviour changed: on 1500001 Astra
   placed the container on the plate, E02 never touched it, and E03 slid it 9 cm along the table (ending 21 cm from the
   plate, versus 19 cm at the start); on 1500003 E03 knocked it off the table. [I] With this recipe, one or two
   demonstrations shift the student's behaviour without making it reproduce the grasp; the anchor is not a usable
   standalone policy.
3. **Did additional successful demonstrations improve stability?** [E] E03 (D1) 1/3 vs E05 (DN, N = 2) 0/3 on the same
   starts (per start 1500017 S/F, 1500018 F/F, 1500019 F/F). E04 (D3) BLOCKED_DATA: only two successes. [I] No evidence
   either way; E02 vs E03 already swap outcomes on two of three starts, which is the known noise floor of this base policy
   (38 % of scenes disagree between two noise replicates, DEVLOG section 11).
4. **Did more frequent observation help?** [E] E06 (8 -> 4 executed actions) 0/3 vs E05 0/3; 100 instead of 50 policy
   inference calls per failed episode. [I] Not informative: both zero.
5. **Did naive imitation of failed attempts help or hurt?** [E] E07 (two successes plus the failed episode between them,
   902 frames) 0/3 vs E05 0/3. [I] Not informative.
6. **Did Astra takeover help the same student?** [E] Not run: NO_STUDENT_SIGNAL (E05 0/5). [E] For reference, Astra from
   reset (E01) solved dev start 1500017 (27 calls, 348 steps), a start that E05, E06, E07 and E10 all failed.
7. **Did recent history improve takeover?** Not run (NO_STUDENT_SIGNAL).
8. **After corrective training, did the student improve with Astra disabled?** Not run: no corrections were collected
   (NO_STUDENT_SIGNAL). [E] E10, the extra-training control (500 more updates on DN), 0/3 dev and 3/10 test (its
   successes on starts that E05 failed, and vice versa: 4 discordant starts, p = 0.63).
9. **Was that improvement greater than extra training or more reset demonstrations?** Not answerable (E11, E12 not run).
10. **Smallest next experiment.** [I] A single-demonstration fit test on the training start: train the D1 student with a
    larger registered update budget (and, as a separate registered arm, unfrozen encoders), then evaluate it on start
    1500001 with several noise replicates. If it still cannot reproduce Astra's demonstrated success there, the transfer
    channel is blocked by the recipe or interface and more Astra data will not help; if it can, collect more Astra
    successes (about 20 calls each at the observed 2 of 4 rate) before re-running E05-E12.

## 5. RoboTwin bimanual, lift_pot

**Setup.** As single-arm (same student, same recipe), task lift_pot.

1. [E] The recording is valid (replay median deviation 0.1 cm, p90 0.4 cm; the failure is reproduced), but Astra produced no
   successful demonstration: 0/4 source episodes (22, 28, 36, 31 calls; 117 calls incl. one episode under the
   zero-success unlock rule). [E] Lift_pot success needs the pot above 0.82 m, both gripper tips within 3 cm of the handle
   contact points, and the pot upright. Astra raised the pot to 0.90 m (episode 3) and 1.05 m (episode 4); episode 4 ended
   with the pot rotated about 120 degrees. [I] Episode 3 most likely failed the contact-point condition (not measured
   directly). No supervised dataset exists, so E03-E12 are BLOCKED_DATA.
2. [E] The original student E02 solved 3/3 dev starts (98, 143, 97 steps) but only 2/10 test starts (1500088 and
   1500103, in 99 and 85 steps). In three test failures it knocked the pot off the table (pot z 0.04-0.06 m); in the others
   the pot stayed at 0.78-0.82 m (one at 0.819 m, just under the 0.82 m threshold). No trained student exists to compare.
3-9. Blocked (no data). E01 PENDING_BUDGET: 33 calls left, below one episode's 40-call ceiling.
10. [I] Here System 1 already outperforms System 2 (2/10 test vs 0/4 from reset), though not by much on the test pool.
    Astra data can only help after Astra's own from-reset success rate on this task rises (0/4 here, 1/3 in the earlier
    pilot); the smallest next experiment is teacher-side, not student-side.

## 6. Dex hand, Humanoid-Stack-Can-Into-Drawer

**Setup.** Cohort: distillation from scratch (no pretrained student; ACT, stateless, from the project's ACT pipeline;
chunk 25, execute 24, U = 3000, dU = 1500). Native 38-D wrist-pose + hand-joint interface shared by Astra and the student.

1. [E] The recording is valid (replay wrist tracking median 1.3 cm, final can position reproduced), but Astra produced no
   success in 6 source episodes (16-25 calls each, 118 calls incl. two unlock episodes). [E] The can never reached the
   saucer: it stayed on the table 20-39 cm away in five episodes and fell to the floor in one, in which the cabinet door
   reached 0.88 rad (success needs at least 1.33 rad, 80 % of its range). [A] Astra's final self-assessments include
   "no grasp is established" and "capture remains unconfirmed".
2-9. E02 NOT_AVAILABLE, E03-E12 BLOCKED_DATA, E01 PENDING_BUDGET (32 calls left).
10. [I] The earlier ten-attempt pilot reached its first success at attempt 7 with memory carried across attempts; a
    150-call platform reservation buys about 6-8 episodes, so three successes cannot be expected here. The next step is
    teacher-side (an Astra-only run long enough to obtain successes), before any student experiment.

## 7. LIBERO-90 task 48

1. [E] Student BLOCKED_INTERFACE: no compatible pretrained student (the Horsea base80 checkpoint was deleted on 2026-09-30
   and held task 48 out), and the surviving from-scratch FM pipeline needs processed observation keys (hand matrices,
   camera extrinsics / intrinsics) that the Astra recordings do not contain; no bridge was built overnight.
2-9. Blocked. [E] E01 (existing harness, unchanged): 1/3 dev (init 15 success with 32 calls and 285 steps; 16 and 17
   failed at the 300-step horizon with 34 and 40 calls), 106 calls.
10. [I] Build and replay-check the observation bridge to the FM pipeline; until then LIBERO only reports Astra-only cost.

## 8. Costs

Astra calls and tokens by platform and cell (from `astra_ledger.jsonl`):

| platform | cell | calls | input tokens | cached input | output tokens | reasoning tokens |
|---|---|---|---|---|---|---|
| dexhand | SRC | 118 | 2,324,737 | 658,816 | 76,598 | 19,522 |
| libero | E01 | 106 | 1,489,588 | 459,648 | 50,433 | 10,166 |
| robotwin_bimanual | SRC | 117 | 1,888,609 | 835,456 | 71,657 | 16,947 |
| robotwin_single | E01 | 27 | 414,222 | 140,544 | 16,736 | 3,317 |
| robotwin_single | SRC | 86 | 1,368,485 | 421,632 | 49,411 | 13,219 |

- Total 454 of 600 calls: single-arm 113, bimanual 117, dex 118, LIBERO 106 (cap 150 each). Every call returned a valid
  response; mean latency 17-21 s per call. No infrastructure retry was needed; one 10-second replay was interrupted by a
  planned restart and re-run (its partial output is kept as `*.interrupted_*`).
- Wall time: single-arm student evaluations 235 min (the largest item), Astra collection 43-50 min per platform,
  training under 2 min in total (1,000 updates take about 20 s).
- Preflight outside the ledger: 26 Astra calls by an orchestrator dry run before launch (one RoboTwin episode, unused).
- Existing pilot records were not reused as data (they lack synchronized student-native command labels); their original
  costs are in the pilot reports (DEVLOG sections 13-14).

## 9. Decisions and rule changes during the night

All are timestamped in `LAUNCH.md`. Two responded to observed outcomes: the 07:30 unlock rule (after bimanual's 0/3) and
the 09:42 diagnosis (after the single-arm gate, as the spec asks). None changes how a compared cell is trained or evaluated.
- 06:51 takeover call ceiling 20 (E08, E09, corrections), budget-safe episode starts, matched extra demos, final stages
  split (E01, then the locked test evaluation).
- 06:53 D3 exists only with three successes (E04 BLOCKED_DATA, not trained on a smaller set); E08 / E09 run as pairs.
- 07:30 zero-success platforms keep collecting source data until a first success, within the platform reservation.
- 08:38 dev expansion to five starts as the last stage.
- 09:42 data / interface diagnosis on the training starts after the NO_STUDENT_SIGNAL gate.
- 10:52 the deadline check scales evaluation estimates by the number of starts.
- A rule letting a platform draw on the global 600-call cap after its 150-call reservation was proposed and NOT applied
  (blocked by the permission system); every platform stayed within 150 calls. That choice is yours: with pooling,
  single-arm could have funded a takeover pair, which was not possible tonight anyway once the gate fired.

## 10. Cautions

- Three to five starts per cell: report raw paired outcomes, not rates. Cross-platform success rates are not comparable.
- Lower training loss is not task learning (E03's loss fell 0.042 -> 0.008 while it failed its own training start).
- Source-pool Astra episodes are collection data, not the matched E01 reference; E01 is only 1/1 on single-arm.
- Astra uses a planner the student bypasses (end-effector targets through cuRobo vs the student's joint targets).
- A zero-success gate is a resource decision, not proof that corrections could never help.
