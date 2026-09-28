# Horsea devlog

Fast and slow memory for a flow-matching robot policy: design history, experiments, results and lessons.
The newest status is at the top; the history follows in chronological order.
All success rates are closed-loop LIBERO simulation success unless marked "offline".

---

## Current status (2026-09-28)

**What Horsea is now: energy-based Horsea.**
- A frozen flow-matching policy (`fm_policy_S`) proposes an action chunk.
- Short memory = fast weights of a small energy network E_W(a, o, l), a learned cost over candidate
  actions. Empty memory gives zero energy, i.e. exactly the base policy.
- A learned writer updates W from the robot's own completed interactions: context, the command it
  executed, and the consequence it observed. No action targets and no success signals are used at
  deployment.
- A bounded final-action solver moves the proposed action toward lower energy: 3 steps, per-element
  displacement at most 0.2 per step in normalized action space.
- Offline training: the robot's exploratory actions are inputs to memory. Corrective actions at the
  states it actually visited are the targets, used only in the outer loss. Gradients flow through the
  write and the solver.

**Main result so far (stage 1, held-out hidden rotations ±30°/±50°; 10 held-out tasks; 5 seeds;
50 adaptation sequences per seed; mean success of attempts 2–5).** Horsea minus each condition is
paired, with a 95% bootstrap interval over tasks.

| condition | mean | Horsea minus this |
|---|---|---|
| **Horsea, correct history** | **41.3%** | – |
| No-history control, same corrective supervision (its best bound) | 25.7% | **+15.6 [+8.8, +22.8]** |
| Same model, memory disabled | 25.7% | +15.6 [+8.2, +23.1] |
| Last interaction only | 26.2% | +15.1 [+9.7, +20.6] |
| Shuffled action–outcome pairing | 27.7% | +13.6 [+8.4, +19.0] |
| Mismatched history (another rotation) | 17.6% | +23.7 [+15.8, +31.2] |
| Reordered history (pairs intact, chronology destroyed) | 40.7% | +0.6 [−1.3, +2.7] |
| TTT2, same data and corrective targets | 33.7% | **+7.6 [+3.5, +12.4]** |

Reading:
- Horsea uses accumulated, correctly paired action–outcome experience to act better. Chronology does
  not matter, which is expected for a fixed rotation.
- It beats a no-history model trained with the same labels, so the gain is not just extra
  supervision. It also beats TTT2 under matched data.
- Caveat: the 0.2 solver bound was selected on these conditions. Confirmation on fresh rotations,
  seeds and starts is still pending (stage 4).
- Gripper condition: the energy moves the gripper output only 0.07–0.13 and never crosses the sign
  threshold, so the executed command never changes. The identical success rates there reflect a
  too-small correction, not ignored history.

**In progress:**
- Stage 2: writer comparison (current batch writer vs sequential vs gated delta-rule-style). The first
  gated seed scores 25.5%, looking worse than the batch writer.

**Staged plan (agreed 2026-09-28):**
1. 5-seed controls at the matched bound (done).
2. Writer comparison.
3. Diagnostics: ranking, solver, gripper, and a rotation-estimation (system identification) baseline.
4. Freeze the configuration and confirm on fresh held-out conditions, reporting S1…S5 per attempt.
   The minimum worthwhile effect is fixed before looking.
5. Persistent-shift consolidation: one cycle, then a second if the first passes.
6. Then LIBERO-10.

Joint-path optimization and MeanFlow stay on hold.

---

## 0. Goal and framing (2026-09-23)

Source: `continual_memory_flow_proposal.pdf`, "Fast and slow memory for flow-matching robot policies".
- **Short memory:** fast weights that adapt a frozen flow-matching policy quickly.
- **Long memory:** the slow weights θ. Short memory is consolidated into θ, then freed.
- **Lifecycle claim:** learn into short memory → consolidate into long memory (verified) → reset → learn
  again, while never forgetting previously learned tasks.
- **Guiding user intents:**
  - Memory must *bend the denoising flow*, not act as a KV lookup that reduces to linear attention.
  - Report success rates.
  - Always track previous-task retention.
  - Keep experiments simple: one task per cycle, 2 cycles for proofs of concept.

Related-work synthesis: `research/related_work_synthesis.md`. RoboTTT (arXiv 2607.15275) is the
closest prior work: TTT layers inside a GR00T DiT action head.

## 1. Infrastructure

- **Base policy:** `fm_policy_S` (DiT velocity net, 10 Euler steps, 18.3M params), trained on the 80
  LIBERO-90 training tasks. Checkpoint: `experiments/libero/libero_90_train80/base80/fm/`.
  - About 93% on training tasks, 0–8% zero-shot on the 10 held-out LIBERO-90 tasks
    `[2, 9, 13, 25, 43, 48, 57, 66, 79, 84]`.
- **Features:** the decoder sees observations only via per-layer means of encoder tokens (encm, 4×256).
  These are precomputed per demo frame (`horsea/features.py`), so meta-training runs on GPU-resident
  features.
- **Execution mode:** receding horizon (generate 16, execute 8; `horsea/exec_check.py`).
  - It replaced temporal aggregation in protocol v2, because aggregation has no clean executed prefix.
  - Base success is unchanged (held-out 8.5% in both modes; old panel ~92%), and rollouts are 35%
    faster.
- **Protocol v2 manifest** (`horsea/manifest.py` → `experiments/manifest/protocol_v2.json`):
  - task split; 70/10 writer split;
  - 5 held-out pairs `(2,9) (13,25) (43,48) (57,66) (79,84)`;
  - start-state folds: adapt 0–9, teacher 10–29, validation 30–49;
  - 100 generated and validated test starts per held-out task (`horsea/starts.py`);
  - 20-task scene-balanced old panel; versions; checkpoint hash.
- **Job queue** (`scripts/queue.py`), running as the systemd user unit `horsea-queue`:
  - jobs file `experiments/queue/jobs.jsonl`, with priorities, dependencies, per-group caps and an
    environment budget;
  - logs in `experiments/logs/q_*.log`.

Hard-won operational lessons:
- Jobs launched from the Claude session die when the session ends (the 06:10 outage, about 6 h lost).
  The queue must run as its own systemd unit.
- LIBERO env workers must be spawned, not forked. If the parent has CUDA/EGL state, forked workers
  fail with "Offscreen framebuffer is not complete". `LiberoRunner` only switches to spawn when
  `num_parallel_envs > 1`.
- Each spawned env worker uses about 2.3 GB of RAM. Keep ≤18 envs on this 60 GB machine; one OOM kill
  happened at 24.
- The GPU (32 GB) OOMs when about 9 meta-trainings start at once. Cap writer-training jobs.
- Deep-copied policies keep instance methods bound to the original object. Pop `reset` and
  `sample_actions` before re-installing samplers.
- Double backward through the decoder needs `sdpa_kernel(SDPBackend.MATH)`.
- Keep the GPU busy: always queue the next planned step and a backfill tier (seeds, controls). Idle
  overnight GPUs happened several times and are not acceptable.

## 2. First memory arms and the adaptation-vs-K study (2026-09-23/24)

Four fast memories. All have meta-learned W₀ and inner learning rates, second-order meta-training and
about 16–18k fast weights.

| arm | where it acts | write objective |
|---|---|---|
| `ttt` / `ttt2` | TTT-MLP after each DiT block (RoboTTT-style) | KV binding on block tokens |
| `kv` | external associative memory → velocity reader | cue → content |
| `fmw` = F-write (the old "Horsea") | velocity correction on frozen decoder features | flow-matching loss on demo actions |
| `res` | final-action residual | action regression |

- **Finding: memories meta-trained on tasks the base already knows are inert** (about 8% at every K).
  Fix: the generic-instruction protocol ("complete the task"), so the demos are the only task signal
  (RoboTTT one-shot style).
- **Held-out adaptation vs K, generic protocol** (success):

| method | K=1 | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| fine-tune | 40 | 56 | 60 | 66 | 78 | 84 |
| ttt2 | 11 | 14 | 17 | 20 | 12 | 13 |
| kv | 6 | 16 | 14 | 15 | 18 | 24 |
| F-write | 4 | 21 | 26 | 21 | 34 | 28 |
| residual | 2 | 16 | 22 | 24 | 30 | 37 |

- **Consolidation with a frozen teacher is lossless** (F-write 28% → 28%, retention 88–90%).

## 3. The short → long lifecycle, demo-written memory (2026-09-24/25)

Driver: `horsea/cycles.py`, 2 cycles (task 57 → task 66).
- **Interface drift broke cycle 2.** Consolidation moved the generic-instruction channel 10× more than
  the real one. Fix: an anchor loss that keeps the generic channel and all decoder tokens equal to θ₀.
- **"Full" short memory:** write demos in chunks of 5 while held-out gain improves. Roll back the chunk
  that stops helping (a bug kept it at first).
- **Gates:** consolidation passes when memory-off success ≥ 0.9 × short-memory success, the previous
  task is within 10 points, and at most 3 rounds.
- **Complete 2-task table** (`experiments/cycles_full/REPORT.md`):

| method | after cycle 1 (57 / 66 / old) | after cycle 2 (57 / 66 / old) |
|---|---|---|
| F-write | 65 / 0 / 100 | 65 / 20 / 98 |
| Residual | 55 / 0 / 98 | 55 / 45 / 100 |
| TTT2 | 70 / 0 / 96 | 65 / 10 / 98 |
| KV | 45 / 0 / 98 | 25 / 10 / 100 |
| Fine-tune | 95 / 0 / 98 | 100 / 75 / 92 |

- **New finding: a short memory that is ON takes over every task.** Old tasks fall to 4–38%, and they
  recover after consolidation and reset.
- **Final held-out table, protocol v2, real instruction, 5 demos, 5 pairs**
  (`experiments/protocol_v2/final/TABLE.md`):

| method | new task, memory ON | old tasks, memory ON | new task after consolidation | previous task after cycle 2 | old tasks after |
|---|---|---|---|---|---|
| Horsea (writer from §5) | 6% | 95% | – | – | – |
| F-write | 28% | 32% | 50% | 28% | 96% |
| Residual | 27% | 33% | 41% | 18% | 95% |
| TTT2 | 12% | 43% | 29% | 2% | 96% |
| Fine-tune | 60% | 94% | 60% | 80% | 94% |

Lesson: user-facing process mistakes to avoid. Don't expand scope (10 cycles, multi-task cycles) when
2 single-task cycles were asked for. Always provide complete tables.

## 4. Protocol v2 and the plan addendum (2026-09-25)

The user's plan (`research/self_rollout_writer.md`) defined:
- the methods: fine-tune, TTT, F-write, residual, Horsea;
- the regimes: E (expert), R0 (own history, no labels or reward), R+ (success flag);
- the gates: teacher gain ≥10 pts; student retains ≥90% of the gain; previous task ≤10 pts drop; old
  panel ≤5 pts drop;
- forced-reset labelling;
- disjoint folds, raw-result schema and seeds.

Expert-arm lifecycle driver: `horsea/lifecycle_v2.py`.

## 5. Learning from self-rollouts: the learned writer (2026-09-25)

**Idea (user):** replace the expert FM target with a learned, outcome-conditioned velocity target
u_φ(e, x_k, t_k, c). e = (o, executed a, o′, l) and (x_k, t_k) are recorded denoising states. φ and
W₀ are meta-trained through the write (SAVN-style), with future corrective targets used only offline.

**Reading done:** RoboTTT, SAVN, EPG, Algorithm Distillation, Zo3T. Notes are in
`research/self_rollout_writer.md`.
- If the write ignores the current candidate, interleaving writes between denoising steps is just a
  schedule.
- u_φ can collapse into a second policy.
- EPG and AD need reward or goal information for ambiguous objectives.
- **RoboTTT never learns an unseen task from zero by exploration.** Its test tasks are trained tasks
  (42–57% base), it always uses expert data in training (demos, human videos, human DAgger
  corrections), and its fast weights reset every episode.

**Testbed: hidden control shifts on training tasks** (`horsea/selfplay.py`).
- The robot's commands are secretly rotated or the gripper is inverted.
- The verified corrective controller is θ₀∘g⁻¹: 75–100% success, versus 0% for θ₀ unaware.
  Takeover from reached states works only 0–75% of the time.

**Mistakes made and corrected:**
- Exploration-only learning of held-out tasks at 0% base success cannot work. With no success, no
  reward and no demo, all rollouts fail. Task 57: 0% for every method.
- I first described RoboTTT as "learning from self-rollouts instead of expert data". That is only true
  at test time.

## 6. RoboTTT-style LIBERO-10 comparison (2026-09-25/26)

Driver: `horsea/history2.py`.
- Post-train on LIBERO-10 demos, writing earlier steps of each demo into fast weights, with the loss
  on the next action.
- Test on the same tasks, writing only the robot's own history, reset every episode.
- Two-stage recipe (memory-only first, then joint), 3 seeds, 10 tasks × 10 episodes:

| method | writes on | writes off | effect of own-history writes |
|---|---|---|---|
| Plain (no memory) | 23.7% | – | – |
| TTT2 | 21.7% | 2.0% | +19.7 (every seed) |
| Horsea, velocity-residual writer | 17.3% | 14.3% | +3.0 |
| F-write / Residual (seed 0) | 2% / 0% | 19% / 18% | −17 / −18 |

- **Controls (TTT2):** last-step-only 7%, shuffled pairing 12%, bypass 11%. Accumulated, correctly
  paired history matters. Frozen W₀ (2%) is worse than no memory, so part of the +20 is the damage
  from reading an unwritten memory.
- **Consolidation attribution (2 cycles, task 3 → 5, LIBERO-10 replay):**
  - TTT2's writes are what make consolidation work: 60% vs 5–10% without them.
  - Velocity-Horsea consolidates better *without* its writes.
  - Plain, which has no memory, also dropped to 18% on old LIBERO-90 tasks and was restored by replay.
    So old-task recovery alone is not evidence of memory transfer.
- **Audit of velocity-Horsea:**
  - Offline, its writes change behaviour and depend on the experience (−19% next-action loss; shuffled
    +63%).
  - But the correction fades late in denoising, and it was trained on demo actions while tested on its
    own. Multi-probe writes made it worse (6%).
- **No memory beats Plain.** RoboTTT's headline result is not reproduced at this scale.

## 7. Energy-based Horsea (2026-09-27/28)

**Design (user):** short memory defines a cost E_{ψ,W}(a, c); actions are produced by optimizing
under it.
- Joint-path version:
  argmin E(z_K) + (1/2λ) Σ ‖z_{k+1} − T_θ(z_k)‖²/Δt, anchored to the policy's transitions.
- Supervision in a fixed action-sensitive metric d_Φ (`horsea/phi.py`, an action autoencoder;
  inspired by Perceptual Flow Matching).
- MeanFlow transitions come later. Implementation: `horsea/energy.py`, `energy_eval.py`,
  `energy_diag.py`.

**Iterations:**

| version | what changed | result |
|---|---|---|
| v1 | random-perturbation write candidates, d_Φ loss | offline −60%, but closed loop = no memory (0–1% on hard shifts). **Joint path = final-only** offline. |
| v1, easier shifts | small rotations ±20–60°, gripper | energy 20–21% vs no memory 24.6%, TTT2 31.7% |
| diagnosis | writer targets identical across rotated candidates (spread 0.0000); only the gripper flip learned | the gripper error dominates the loss |
| v2 | structured candidates (rotations, gripper flip) + bounded steps | still flat on rotations |
| v3 | rotation-only training / per-group balanced loss | **+4–7 over shuffled, +3–5 over no memory** |
| v3, 3 seeds, bound 0.05 | full control set | 31.5% vs no-history 26.0% (**+5.5 [+1.7, +9.8]**); vs self-imitation +5.8; TTT2 33.8% |
| energy diagnostic | correct history ranks the corrective target above base in 95% of queries (Spearman 0.75); mismatched 27% / −0.25; refinement moves only 0.014 | the energy is informative, the step is too small |
| step sweep (offline) | 0.2 × 3 best; more iterations diverge | select 0.2 × 3 at inference |
| **v3, bound 0.2, 5 seeds** | stage-1 table above | **41.3%, +15.6 over no-history, +7.6 over TTT2** |

**Negative results to remember:**
- Joint-path optimization never beat final-action optimization, so the path claim is unsupported.
- 20 solver iterations diverge, even when bounded.
- Energy-Horsea on LIBERO-10 with a contrastive demo-vs-own loss: writes hurt (4% vs 16.7%), and
  consolidation of a broken teacher destroyed the policy (old tasks 0%).

### 7.1 External code audit of 36ce182 and fixes (2026-09-28)

| # | finding | status |
|---|---|---|
| P1 | `selfplay.run_batch`: an episode that finished early got the batch's *last* observation as its terminal proprio, so recorded outcomes depended on the partner episode's length (and on success). | **Fixed**: each env keeps its own post-action observation. Regression test `tests/test_recorder.py` (old code gives [0,1,6], new code gives [0,1,2]). All self-rollout data re-collected into `selfplay/data_v2`. |
| P2 | `energy.adapt_seq` truncated ragged histories to the shortest one, so seq/gated writers saw less history when batched. | **Fixed**: padded chunks with a per-episode active mask (inactive episodes: no write, no decay); `WRITE_COUNT` is logged. Test `tests/test_adapt_seq.py` (batched = solo). Earlier seq/gated (GD_*) results are **invalid** and were stopped. |
| — | random write-candidate noise depended on batch composition | **Fixed**: noise drawn per episode block. |
| solver | with the learned λ≈0.026, β≈0.10 the "grad" step raises J on 100% of steps (J −0.019 → 34.4 on seed 0 at bound 0.2), yet d_Φ to the target falls 0.079 → 0.046. It is a learned bounded update, **not** an energy minimizer. | J trace added to `energy_diag`; a proximal solver (`--solver prox`) is added as a separate arm. |
| claims | targets are oracle disturbance compensation; episode boundaries depend on success; the 10 dev tasks are held out from the writer only; adaptation is between attempts only; consolidation is not wired to the energy model; TTT2 is not information-matched; the joint-path "tie" never tested the path. | Claim boundaries recorded here and in the README; to be addressed later. |

The stage-1 table above predates the P1 fix.

### 7.2 Plan from 2026-09-28 (user) and the corrected rerun

Stages, each gating the next:
1. Fix and verify: done (§7.1).
2. Reproduce hidden rotation on corrected data.
3. Normal tasks: Plain / native TTT / info-matched TTT / Horsea / outer-objective ablation. Seen-task
   adaptation and held-out-task learning are reported separately.
4. One consolidation cycle of the exact validated model, with replay, vs sequential fine-tuning with
   replay.
5. A second cycle.

Joint-path optimization stays separate.

**Stage 2 protocol (fixed before any fresh result):**
- **Data and training:** data_v2, 3 seeds. Configuration unchanged from the original run (train bound 0.05).
- **Dev tuning:** dev = 10 writer-dev tasks × {none, ±30°, ±50°}, starts 30–34. Each method gets 3
  candidates, chosen by mean over seeds:
  - Horsea: inference bound {0.05, 0.1, 0.2};
  - no-history: the same bounds;
  - TTT2: inner-lr cap {1, 3, 10}.
- **Freeze:** `horsea.v2_report select` writes `experiments/protocol_v2/v2/frozen.json` before any
  fresh run.
- **Fresh test:** {none, ±25°, ±45°, ±55°} with fresh starts 40–44.
- **Conditions:** Horsea with write, memory off, shuffled, last-only and mismatched; no-history; TTT2;
  TTT2 mismatched.
- **Metric:** mean success over attempts 2–5, plus S1…S5.
- **Pass rule:** Horsea(write) minus each of {no-history, shuffled, last-only, memory-off} has a 95%
  paired bootstrap CI above 0 on the fresh test (seeds pooled). An advantage over TTT is claimed only if
  that CI is also above 0.
- **P1 impact check:** old seed-0 model vs retrained seed 0, both at bound 0.2 on dev.
- **Exploratory backfill (not part of the claim):** prox solver, train at 0.2, seq/gated writers (seed 0).

**Stage 2 result (fresh test, 2026-09-29 12:30): PASS.** Frozen configuration: Horsea bound 0.2,
no-history bound 0.05, TTT2 lr cap 3. Setup: 3 seeds × 10 tasks × 7 fresh shifts = 210 sequences,
mean success over attempts 2–5.

| condition | success | S1…S5 | Horsea minus this [95% CI] |
|---|---|---|---|
| **Horsea, correct history** | **36.4%** | 19 40 31 39 35 | – |
| memory off | 21.1% | 20 24 18 23 19 | +15.4 [+11.9, +18.9] |
| shuffled pairing | 23.2% | 20 28 20 23 22 | +13.2 [+9.8, +16.7] |
| last interaction only | 23.3% | 19 27 20 28 19 | +13.1 [+9.6, +16.7] |
| history from another rotation | 13.2% | 21 13 15 14 11 | +23.2 [+18.9, +27.6] |
| no-history model | 22.0% | 20 27 18 25 18 | +14.4 [+11.0, +18.0] |
| TTT2 | 30.6% | 20 35 29 30 29 | +5.8 [+1.9, +9.9] |

- **Controls:** every control gap is above 0 for each seed separately, with the smallest per-seed lower
  bound at +3.9.
- **Over TTT2:** the pooled gap passes (+5.8 [+1.9, +9.9]). Per seed it is +3.6 [−3.6, +10.7],
  +10.0 [+3.9, +16.8] and +3.9 [−2.9, +11.1]. The advantage is modest and not robust per seed.
- **Other evidence:**
  - The P1 fix did not create the effect: the old seed-0 model scores 41.5% on dev, and seed 0 retrained
    on corrected data scores 42.0% (dev, bound 0.2).
  - The prox solver reaches 26.0% on dev, versus 42% for the grad solver. It does not help.
- **Scope:** oracle-corrective training targets, hidden rotation only, adaptation between attempts, and
  tasks held out from the writer only. Stage 3 (normal tasks, information-matched TTT) is next.

---

## Lessons

**Research**
1. **Separate what experience says from what to imitate.** Exploratory or failed actions are memory
   *inputs*. Appropriate subsequent actions are offline *targets*. Self-imitation of own actions
   consistently hurts: F-write/Residual −17 points on LIBERO-10, and the self-imitation control
   −5.8 points.
2. **Without information, there is nothing to learn.** Unseen tasks at 0% success with no reward and no
   demo cannot be learned from exploration. Choose testbeds where outcomes reveal something (hidden
   control shifts).
3. **Control the controls.** The key comparison is a *separately trained no-history model with the same
   supervision*. Memory-off, shuffled, last-only, mismatched and reordered each answer a different
   question.
4. **Offline metrics can mislead.** A 60% offline distance gain produced no closed-loop gain until the
   learning signal and the solver budget were fixed. Always confirm with success rates.
5. **Inspect loss balance.** A large-error component (gripper sign flip) can silently monopolize the
   writer's learning.
6. **Diagnose ranking before blaming the solver.** Here the energy ranked correctly, and the step
   budget was the bottleneck.
7. **Old-task recovery after consolidation can be pure replay.** Always compare against a no-memory
   consolidation.
8. **Selections made on development conditions must be re-confirmed** on fresh conditions, with the
   threshold fixed beforehand.

**Process**
- Keep the GPU busy; queue follow-ups before a batch drains.
- Label every number as a success rate or an offline metric.
- Smoke-test every new path before queuing long jobs.
- Keep scope exactly as asked (2 tasks, 2 cycles, single-task cycles).

## Open questions

- Does the writer comparison (stage 2) improve on the batch writer? Early gated seed: no.
- Does the +15.6 hold on fresh rotations, seeds and starts?
- A rotation-estimation baseline from the same history (system identification): how much of the
  available information does Horsea capture?
- The gripper needs corrections that cross the execution threshold. Bound, parameterization, or a
  discrete-aware solver?
- Consolidation of a persistent shift into a memoryless student, and renewed plasticity in cycle 2.
- Transfer of the validated objective to LIBERO-10 (matched supervision for Horsea, TTT2 and a
  no-history model).
- Later: joint-path optimization and MeanFlow, only if the path shows value.

## File map

| area | files |
|---|---|
| base, features, rollouts | `horsea/base.py`, `features.py`, `rollout.py`, `exec_check.py`, `starts.py`, `manifest.py` |
| first memory arms | `horsea/memory.py`, `meta_train.py`, `adapt_eval.py`, `consolidate.py` |
| lifecycle (demo-written) | `horsea/cycles.py`, `cycles_report.py`, `lifecycle_v2.py`, `final_table.py` |
| self-rollout data and writers | `horsea/selfplay.py`, `writer.py`, `r0_eval.py`, `explore_cycle.py` |
| RoboTTT-style LIBERO-10 | `horsea/history2.py`, `cycles_ttt.py`, `audit_writes.py` |
| energy Horsea | `horsea/phi.py`, `energy.py`, `energy_eval.py`, `energy_diag.py` |
| reports | `horsea/report_v2.py`; notes in `research/`; older log in `EXPERIMENTS.md` |
| results | `experiments/protocol_v2/{objective,option1*,closedloop,final,...}`, `experiments/energy/`, `experiments/history2_*`, `experiments/cycles_*` |
| queue | `scripts/queue.py`, `experiments/queue/`, `systemctl --user status horsea-queue` |
