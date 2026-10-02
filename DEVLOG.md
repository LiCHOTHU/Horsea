# Horsea devlog

Fast and slow memory for a flow-matching robot policy: design history, experiments, results and lessons.
The newest status is at the top; the history follows in chronological order.
All success rates are closed-loop LIBERO simulation success unless marked "offline".

---

## Current status (2026-10-01)

**Benchmark: RoboTwin 2.0** (LIBERO removed on 2026-09-30: too easy, the base sat at 96.5% on the
confirmation panel). Our own FM policy is trained on all 50 RoboTwin tasks (§10). Base mean success 22.1%.

**Running: the trained loop-consistency study** (§13; manifest `experiments/rt/loopc/manifest.json`):
- Does a trained FM policy benefit from internal Transformer loops?
- Should the loop graph stay constant across FM time?
- Does every FM evaluation need a loop?

Development is done:
- S* = S3 (43.1% vs N 36.2%, +6.9 [+0.0, +13.1], selection-biased).
- T1 ties S* (−1.9); Tfree is worse (−13.8 [−21.2, −6.2]).
- No 5-of-10 rule matches S*, and A5 does not beat random R5.

Confirmation on the untouched T scenes is running (3 training seeds, 2,400 episodes).

**Concluded on LIBERO (§7–9):**
- Energy Horsea passed stage 1 (+15.6 over a no-history control on hidden rotations) but failed on LIBERO-10.
- RDM with PPO did not learn.
- A frozen internal loop failed confirmation (−0.5 vs base).

---

## Earlier status (2026-09-28, LIBERO; superseded)

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

### 7.3 Stage 3, question 1: seen-task adaptation on LIBERO-10 (queued 2026-09-29)

**Bug found in the old LIBERO-10 energy runs:** 75–94% of their updates were skipped as non-finite.
- `q_EN_train_s*` skipped 1,794 / 2,248 / 2,221 of 2,400 steps.
- Two causes:
  - padded rows of ended demos kept writing a zero-change experience;
  - sequential per-step writes made the fast weights grow exponentially (0.4 → 39 → 1e35 over about
    35 writes).
- The earlier negative result (§7: 4% vs 16.7%) is therefore **invalid**. TTT2, velocity-Horsea and
  Plain were unaffected (0 skipped updates).
- **Fixes (`horsea/history2.py`):**
  - padded rows keep their memory and add an exact zero to the loss (`keep_active`, applied to every
    arm);
  - the new `energy2` arm uses the validated batch writer: re-adapt from W0 with 2 writes on at most
    16 sampled history interactions at every decision, so the number of write steps is bounded.

**Arms:** RoboTTT protocol (post-train on demos, test with the robot's own history). Same backbone,
data, steps (2400) and recipe; within-episode memory, reset each episode; episodes end at success or
520 steps.
- Plain;
- TTT (native: writes context and own action);
- **TTT-info**: TTT writing (c_{t−1} + A(c_t − c_{t−1}), a_{t−1}), the same information as Horsea;
- **TTT-info-dΦ**: the outer-objective ablation, TTT-info trained with Horsea's d_Φ outer loss;
- **Horsea (`energy2`)**: energy memory, experience writer, learned solver, d_Φ outer loss.

**Tuning:** 3 candidates per method on dev (seed 0, starts 30–34), frozen by `horsea.v3_report select`
before testing.

**Test:** starts 40–49, 3 seeds, 10 tasks × 10 episodes each. Controls: memory off, mismatched, and
last-only (Horsea).

**Pass rule:** Horsea(write) minus Plain and minus Horsea(memory off) both have 95% paired bootstrap CIs above 0 on the test starts (3 seeds pooled). A mechanism advantage needs Horsea minus TTT-info also above 0; the objective effect is TTT-info-dΦ minus TTT-info.

Question 2, learning the 10 held-out LIBERO-90 tasks, is kept separate and follows after this.

---

### 7.4 Fairness fixes for the TTT comparison (queued 2026-09-29)

Three mismatches that could change the outcome are removed in the fair variants:
1. **Decoder treatment (LIBERO-10):** the TTT arms fine-tuned the decoder, but Horsea's loss gives the
   decoder no gradient. Fair runs (`_s3f_*`) use `--lr_dec 0` for every arm.
2. **Auxiliary loss (rotation):** TTT2 had the old-task replay term λ_old = 0.5 and Horsea had none.
   Fair TTT runs (`*_nold`) use `--lam_old 0`.
3. **Domain prior in writes:** Horsea's write candidates are built from ±20–60° rotations. Fair Horsea
   variants use generic candidates: the executed chunk plus 9 random perturbations, the same count
   (`--generic_cands` in history2; `--n_cand 10` without `--structured` in energy.py).

Tuning is reused from the frozen stage-2/3 selections; this is a documented compromise. Still
unmatched: test-time compute (solver iterations) and separate eval drivers on the rotation testbed.

## 8. Recurrent-denoising memory (RDM), a new idea (2026-09-29)

The energy Horsea was dropped: it failed on normal tasks, and its rotation-testbed gain rests on a
rotation prior. The new spec (user, 2026-09-29) proposes a causal experience bank that is re-queried on
every denoising pass, with a per-decision denoising workspace. It is meta-trained with PPO on five-attempt
own-rollout metaepisodes. Code: `horsea/rdm/`; tests: `tests/test_rdm.py`.

**Implementation:**

| part | choice |
|---|---|
| backbone | frozen base80 fm_policy_S, eval mode (no dropout); K = 10 Euler steps, chunk 16, execute 8, receding horizon |
| split | modulation after decoder layer 2 of 4: g~ = (1+γ)g + β; the final projection is zero-initialised (exactly the base at init, tested) |
| workspace B | per action token, attention over the final decoder features F_j of earlier passes, plus a null token; cleared every decision |
| experience bank H | real events (c_n, p_n, executed prefix + mask, c_n+1, p_n+1, attempt/step/reset tags); 9 tokens per event from a shared 1-layer mixer; null token; no reward or success input |
| variants | plain · adapter (current observation only, no workspace) · looped (workspace, empty bank) · readonce (r_0 reused) · reread (r_k per pass) · ttt_info (TTT2 layers written with the same event information, deterministic write) |
| exploration | Gaussian σ on dims 0–5 and a Bernoulli gripper p = sigmoid(4μ_g), identical for every method. A Gaussian gripper never flipped (flip rate 0.0), so the spec's Bernoulli fallback is used |
| objective | PPO; logged-noise likelihood ratio over the 8×7 executed prefix; Monte-Carlo returns across all 5 attempts (discount 1); critic V(c, attempt, position, successes so far), training-only and the same for every variant |
| horizon | 304 steps per attempt (38 decisions × 8), running past success; 5 attempts per metaepisode |
| TTT-info replay | burn-in approximation: W_{n−8} is replayed without gradient under the current weights, then the last 8 writes with gradient |

**Acceptance tests (all pass):**
- zero-init equals the base for every variant;
- future events cannot change earlier outputs, and past events do;
- the workspace does not leak across calls;
- every variant makes 10 denoiser calls;
- logged and recomputed log-likelihoods agree to 5e-5, so the PPO ratio before an update deviates by only 1e-5;
- later-attempt reward reaches earlier decisions;
- finite gradients to every ψ parameter; θ is untouched.

**Development tasks:** 23 and 32. Their base success is 50% and 60% on training starts, from the Plain
audit `experiments/rdm/audit` (receding horizon, horizon 300). Pilot: B = 4 sequences per task per
iteration, 40 iterations, seed 0.

**Ablation ladder (user rule: validate each step before the next):**

| rung | contrast | gate |
|---|---|---|
| 0 | Plain deterministic vs Plain with σ ∈ {0.1, 0.05} | largest σ within 10 points of deterministic Plain |
| 1 | adapter (PPO) vs Plain | can PPO improve anything with this signal? |
| 2 | looped vs adapter | workspace effect |
| 3 | readonce vs looped | history effect |
| 4 | reread vs readonce | per-pass re-read effect |
| 5 | reread vs ttt_info | memory representation |

Evaluation: validation starts 30–33 per task, 5 attempts, plus a fresh-start probe (40–43) that reads a
frozen H. Metrics: S1…S5, mean S2–S5, at least one success, and probe success.

**Status labels (after the external review, 2026-09-29):**
- The current batch is a **parallel exploratory pilot**, not a sequentially gated experiment. All arms train
  at once, and intermediate rungs need not improve on their own; a workspace may only help together with
  history.
- The decisive contrasts are: re-read vs read-once; re-read vs matched TTT-info; correct history vs history
  controls; the full method vs **both** deterministic and stochastic Plain.
- One seed on two chosen development tasks can point to promising directions, not establish superiority.
- No larger sweeps or consolidation until the main contrasts replicate.

**Rung 0 (settings selection on training starts 0–7, 8 sequences × 5 attempts per task):**
- Deterministic Plain: S1…S5 = 81 62 75 69 69, mean 71.2%.
- Plain with σ = 0.05: mean 72.5%.
- Plain with σ = 0.1: 56 81 69 50 62, mean 63.7%. **σ = 0.1 is frozen for every method.**
- Both Plain baselines are reported. Beating stochastic Plain alone could just be recovering what the
  exploration noise cost.

**Reconciling the base numbers:** same checkpoint (base80, sha dc3110…), same execution (receding 16/8, no
temporal aggregation).
- The audit ran 1 episode per start on starts 0–9 (horizon 300): task 23 at 50%, task 32 at 60%.
- Rung 0 ran 5 attempts per start on starts 0–7 (horizon 304): 60% and 82.5%.
- Restricted to starts 0–7, the audit gives 50% and 75%, so the gap is sampling noise; task 32's starts 8–9
  failed in the audit.
- "Deterministic" Plain still varies across attempts from the same start, because each decision draws fresh
  FM noise ε.

**Added acceptance checks (all pass):**
- Joint log-probability equals torch.distributions (Gaussian dims 0–5 + Bernoulli gripper, summed over the
  8 × 7 prefix).
- Records keep the pre-clipping u and the initial noise ε; the bank stores clip(u); the mean is replayed
  from (ε, H_n).
- The rollout token cache and bank reset at every metaepisode start; recompute follows the current
  tokenizer parameters.
- Gradients reach TTT-info's adapters, write projections, W0, inner learning rates and gates.
- One success reward per attempt at most (asserted in `build_dataset`, R ∈ [0, 5]).
- Logged vs recomputed log-probability differ by at most ~5e-5, so the PPO ratio at the start is
  **approximately** 1, within 1e-4.

**TTT-info labelling:** it is a **deterministic-write TTT variant**, because the write uses a fixed flow
time t = 0.5 and zero noise so the likelihood can be replayed. Its training replay uses a gradient window.
- Against a full-gradient reference on one attempt, the forward pass is identical for any window.
- The gradient cosine is **0.545 with an 8-write window** and 1.000 with 32.
- The 8-write run (`ttt_info`) is therefore a distorted-gradient approximation. The main comparator is now
  `ttt_info38`, with a 38-write window (one full attempt), queued next to it.


**Pilot result (seed 0, 2026-09-30 05:21): no learning, so no memory contrast can be read.**

| arm | S1…S5 (%) | mean S2–S5 | fresh probe |
|---|---|---|---|
| Plain, deterministic | 75 25 75 62 88 | 62.5% | 75% |
| Plain, σ = 0.1 | 50 62 38 75 50 | 56.2% | 38% |
| adapter | 62 100 50 88 100 | 84.4% | 75% |
| looped | 50 50 38 38 62 | 46.9% | 50% |
| read-once | 38 38 38 75 50 | 50.0% | 75% |
| re-read | 50 62 38 75 25 | 50.0% | 62% |

- **Training curves are flat.** Mean S2–S5 in 10-iteration blocks: adapter 58→65%, looped 63→64%,
  read-once 62→57%, re-read 64→59%.
- **The trained policies are essentially the base.** The mean action moved ~0.01 from base (vs σ = 0.1),
  and history changes the mean by only ~0.001 (history of 60 events vs empty).
- **Gaps between arms are therefore noise:** 8 sequences per arm; adapter vs looped differ by 37 points
  with both at the base.
- **Next:** the learning-rate diagnostic (1e-3, adapter and re-read) must show that PPO can move the policy
  at all before any memory contrast is meaningful.

**Learning-signal diagnostics (2026-09-30):**

| run | training S2–S5 by 10-iteration block | mean action moved from base | history effect |
|---|---|---|---|
| pilot, lr 1e-4, B = 4 | flat, ~0.60 | ~0.01 | ~0.001 |
| lr 1e-3, adapter | 0.41 → 0.27 → 0.23 → 0.32 (worse than base) | 0.026 | – |
| lr 1e-3, re-read | 0.06 → 0 → 0 → 0 (diverged, KL ~1e14) | exploded | exploded |
| stabilized: tanh-bounded modulation, lr 3e-4, clip 0.5, B = 8 | adapter 0.61 → 0.62 → 0.60 → 0.61; re-read 0.63 → 0.62 → 0.64 → 0.64 | 0.004 / 0.010 | 0.0015 |

- Retry structure in Plain: P(success | previous attempt succeeded) = 0.69, P(success | previous failed) = 0.61.
  Attempts are nearly independent, so memory has headroom but no easy signal.
- **Conclusion:** at up to 16 metaepisodes per update, sparse-success PPO does not improve the policy on
  these tasks. Small steps do nothing; large steps follow the noise. TTT-info runs were stopped as
  uninformative.
- **Queued:** re-read with 32 metaepisodes per update (B = 16 per task) as the last spec-compliant scale-up.

## 9. Internal looping of DiT decoder blocks (spec 2026-09-30)

**Scope:** replaces the PPO memory pilot (§8). **Question:** does extra computation inside selected
Transformer blocks of the frozen or continued LIBERO FM policy raise closed-loop success, beyond simply
spending more compute?

**Code:** `horsea/loop/`
- `core.py`: one shared block-execution helper used by the native `forward_dec` (patched per instance)
  and by Horsea's `Flow.decode`;
- `train.py`, `evaluate.py`, `screen.py`, `diag_loss.py`, `profile.py`, `report.py`;
- immutable manifest in `experiments/loop/manifest.json`.

**Base:** the base80 checkpoint, 4 decoder blocks (width 256, 4 heads, MLP 512), K = 10, σ_min 0.001.
The native control protocol keeps temporal aggregation on (replan every step), horizon 300.

**Stage 0 (`tests/test_loop.py`, all pass):**
- R = 1 is bit-identical to the original on the native path, the wrapper and sampled chunks;
- call counts and order are correct (2,2,3,3; 50 block calls per chunk when one block is looped at every t);
- the cached-residual control equals the block;
- rows outside the loop interval are exact, with grouped execution;
- gradients flow through both repeats into the single shared block;
- encoders and the fixed time frequencies stay frozen;
- the loop's effect reaches the executed command.

**Stage 1 offline screen** (frozen, 128 held-out chunks, FM-MSE change vs the unchanged network):
- layer 0 is −0.9% / −1.6% / +2.5% (noise-side / middle / action-side);
- layer 1 is +7 / +9 / +16%;
- layer 2 is −0.4 / −0.6 / −1.4%;
- layer 3 is +15 / +31 / +21%.

**Development closed loop** (tasks 23, 32, 81 × starts 0–19 = 60 episodes per arm; paired 95% CIs vs base
resample starts within each task):

| arm | block calls | ms per chunk | success | vs base |
|---|---|---|---|---|
| base, K = 10 | 40 | 8.75 | 68.3% | – |
| frozen layer 0, middle | 43 | 9.90 | 76.7% | +8.3 [−6.7, +23.3] |
| frozen layer 0, noise-side | 44 | 10.12 | 71.7% | +3.3 |
| frozen layer 2, action-side | 43 | 9.88 | 66.7% | −1.7 |
| frozen layer 1, all t (reference) | 50 | 11.31 | 71.7% | +3.3 |
| frozen layer 0 middle, R = 4 / raw | – | – | 75.0% / 76.7% | +6.7 / +8.3 |
| ordinary K = 11 / K = 12 | 44 / 48 | 9.57 / 10.34 | 71.7% / 75.0% | +3.3 / +6.7 |
| C, ordinary continuation, 2,000 updates | 40 | 8.75 | 78.3% | +10.0 |
| L, layer-0 loop continuation, 2,000 updates | 50 | 11.31 | 78.3% | +10.0 |
| C-compute, 2,369 updates, K = 10 / K = 13 | 40 / 52 | 8.75 / 11.17 | 78.3% / 78.3% | +10.0 |
| L run without its loop (R = 1) | 40 | 8.75 | 78.3% | +10.0 |

**Continuation training** (seed 0; lr 1e-5, 100 warm-up updates, same paired data/noise/augmentation):
- A looped update costs 1.18× an ordinary one, so C-compute runs 2,369 updates.
- Held-out FM loss: C 0.01266 → 0.01109 and L 0.01296 → 0.01113, almost all within 500 updates.
- The 5,000-update extension was not triggered (the 1,000 → 2,000 drop was 0.3% / 0.1%, below 2%).

**Development verdicts:**
- **H3 (training):** no recurrence-specific benefit. L vs C is +0.0 [−8.3, +8.3], and L vs C-compute at
  matched inference cost is +0.0. The L model does equally well without its loop. The +10 over base comes
  from continuation training itself.
- **H1/H2 (frozen loop):** it passes the 5-point screening trigger over the base, but against the ordinary
  extra-step bracket it is only +1.7 [−10, +13] (vs K = 12) and +5.0 (vs K = 11).
- A **preregistered frozen confirmation** is running (`experiments/loop/confirmation_prereg.json`):
  10 panel tasks × starts 20–39 × 3 eval seeds = 600 episodes per arm, for base, frozen loop, K = 11 and
  K = 12. Primary contrasts: loop vs base and loop vs K = 12, joint 97.5% rule.

**Confirmation result (2026-10-01 06:35): the frozen loop FAILS; internal looping has not helped this
policy under the tested conditions.**

| arm (600 episodes: 10 tasks × starts 20–39 × 3 eval seeds) | success | per seed |
|---|---|---|
| base, K = 10 | 96.5% | 97.5 / 96.0 / 96.0 |
| frozen loop, layer 0, middle | 96.0% | 96.0 / 96.5 / 95.5 |
| ordinary K = 11 | 95.2% | 94.5 / 97.5 / 93.5 |
| ordinary K = 12 | 96.5% | 96.0 / 96.5 / 97.0 |

- **Primary contrasts:** loop − base = −0.5 [−2.2, +1.0] (97.5%: [−2.3, +1.3]); loop − K = 12 = −0.5
  [−2.0, +1.0] (97.5%: [−2.2, +1.2]). Neither passes, and the practical +5-point target is excluded.
- **Caveat, a design flaw:** the base is near ceiling on this panel (96.5%). The panel was chosen by a rule
  that used an audit run in the receding-horizon regime (8-step prefix). Under the checkpoint's native
  protocol (temporal aggregation, replanning every step) those tasks are nearly solved.
  - Result: the confirmation can only rule out gains larger than about +1.3 points on these tasks.
  - The development tasks, at 68% base, had room but only 60 episodes each.
- **Development-to-confirmation shrinkage:** the development +8.3 for the frozen loop did not survive.
- **Gates:** the trained loop showed no recurrence-specific benefit on development (L = C). Stage 4 (an
  input-dependent selector) is gated on a fixed-loop success, so it is not run.


## 10. Move to RoboTwin 2.0 and the base FM policy (2026-09-30/10-01)

**Why:** LIBERO was too easy for the questions asked (confirmation base 96.5%). RoboTwin 2.0 (dual-arm
aloha-agilex, 50 tasks) is far from ceiling.

**Setup** (repo `~/workspace/RoboTwin`, conda env `robotwin`; `memory/robotwin-setup.md`):
- **Data:** `aloha-agilex_clean_50`, 50 demos for each of the 50 tasks, 552k frames. 14-D joint action
  (2 × (6 arm + 1 gripper)). Demos 0–44 train; 45–49 are held out.
- **Rendering:** NVIDIA Vulkan works only with an X connection. Evaluations need DISPLAY=:0, XAUTHORITY,
  the NVIDIA ICD, and no CUDA_VISIBLE_DEVICES.
- **Concurrency:** at most 4 concurrent simulators (each ~6.7 GB of GPU memory; 6 gave vk::DeviceLost).
- **Watchdog:** every evaluation runs under a progress watchdog (`safe_fixed.sh` / `eval_safe.sh`). It
  allows 600 s without output and a 5,400 s cap, then up to 3 same-seed attempts.
- **click_bell:** the official success check passes a random policy. An opt-in strict check
  (`ROBOTWIN_STRICT=1`) scores the base at 5/13 versus 64% under the official check.

**Policy** (`horsea/rt/`, 18.1M parameters):
- ResNet-18 + FiLM on 3 cameras at 120×160, a state MLP and a CLIP ViT-B/32 instruction embedding give 5
  observation tokens.
- DiT denoiser: 4 encoder + 4 decoder blocks, width 256.
- FM with σ = 0.001 and Beta(1.5, 1) times. Euler K = 10, chunk 16, execute 8, receding horizon.
- Trained for 300k steps; loss 0.163 → 0.0021.

**Base success** (25 episodes per task, clean scenes, unseen instructions): **mean 22.1% over 50 tasks**.
- Best tasks: grab_roller 96, shake_bottle(_horizontally) 92, press_stapler 76, place_burger_fries 68.
- 13 tasks are at 0%.

## 11. Execution-path discovery on RoboTwin (Stage A, 2026-10-01)

**Design:** frozen base, K = 10. G0 is compared with 12 primary loop graphs (decoder block b ∈ 0–3 visited
twice with half of its whole-block residual each time, in the early, middle or late solver window), the
secondary graphs, and K = 11/12. Tasks: handover_mic, lift_pot, open_microwave, place_container_plate.

**Bug found and fixed:** episodes were first paired by episode index. RoboTwin's expert admission check is
not deterministic, so the same index was a different scene in different runs. Episodes are now paired by
env seed, and the noise is keyed by env seed (v0 archived).

**Result** (1,440 episodes, 41 scenes common to all arms):

| arm | success |
|---|---|
| G0 | 46.3% |
| K = 11 | 50.3% |
| K = 12 | 47.8% |
| b0_middle | 56.9% (+10.6 [−2.5, +23.3]) |
| b2_middle | 51.9% |
| late loops b0 / b1 / b2 | 40.2 / 34.5 / 37.9% |

**Noise floor:**
- Two G0 noise replicates disagree on 38% of scenes.
- Graph effects are not stable: split-half r = −0.08 and replicate r = +0.61.
- Honest (held-out) selection collapses, so the in-sample best graph is a hindsight maximum.
- One stable signal: late loops hurt handover_mic and lift_pot in both halves.

## 12. Stage B: ordinary vs multi-graph continuation (2026-10-01)

**Training:**
- **Fixed scenes:** prevalidated once per pool. D (development) holds 20 scenes per task. T (test) holds 50
  per task and is unopened. Each scene stores its instruction (RoboTwin's generator uses unseeded random).
- **C:** ordinary continuation from 300k, 5,000 updates.
- **M:** the same updates, but each minibatch draws G0 with probability 1/2 and each of the 12 loop graphs
  with probability 1/24. Shared weights, gradients through both visits.
- **Recipe:** AdamW, lr 1e-5, 250 warm-up updates, batch 64. Encoders frozen; the denoiser is trained.
- **Held-out G0 loss:** base 0.00547, C 0.00436, M 0.00438. No loop graph beats G0 on held-out loss.
  Under M the loop penalty is +0.2 to +7.9%, versus +13 to +20% for C's late loops.

**B1** (20 D scenes × 2 noise replicates × 4 tasks = 160 episodes per arm):

| arm | success |
|---|---|
| original 300k, G0 | 30.6% |
| C, G0 | 34.4% |
| C, b0_middle | 30.6% |
| C, K = 11 | 28.8% |
| M, G0 | 27.5% |
| M, b0_middle | 29.4% |

| contrast | difference [95% CI] |
|---|---|
| M_b0m − C_G0 | −5.0 [−12.5, +2.5] |
| M_b0m − C_K11 | +0.6 [−6.9, +8.1] |
| M_b0m − M_G0 | +1.9 [−4.4, +8.1] |
| interaction (M_b0m − M_G0) − (C_b0m − C_G0) | +5.6 [−3.1, +15.0] |
| M_G0 − C_G0 | −6.9 [−14.4, +0.6] |

Multi-graph training gave no detectable benefit. B2 (the same-M graph matrix on 12 D scenes) finishes as
low-priority backfill; it is superseded by §13.

## 13. Trained loop consistency (spec 2026-10-01; development done, confirmation running)

**Questions:**
- Does a trained FM policy benefit from internal loops?
- Should the loop graph stay constant across FM time?
- Does every FM evaluation need a loop?

**Fixed:** K = 10 Euler with the same grid, noise keys, chunk 16 / execute 8 and normalization. There are
no K11/K12 arms, and loops are never deleted at inference from a looping model.

**Graphs** (`horsea.rt.graph.schedule`):
- Encoding `s:<10 chars>`: character k governs FM evaluation k. '-' runs the original 0→1→2→3; a digit l
  runs block l twice with half of its whole-block residual each time (the single operator of the study).
- **Phase 1:** N = `s:----------` (40 block calls per chunk) and S_l = `s:llllllllll` (50 calls).
- **Phase 2:** schedules proposed from a held-out FM-error table E[k, l] of the Stage-B M model (same
  operator, every cell trained). 8,192 held-out frames; dropout off.
  - Tfree = per-evaluation argmin = `s:2222000002`.
  - T1 = best of the 112 schedules with at most one change = `s:2222000000`.
  - **Main table at the solver times t = k/10.** Interval-averaged tables are dominated by t → 1 in
    [0.9, 1.0), which the sampler never queries, and there T1 = S2.
  - Bootstrap stability: T1 99%, Tfree 96%.
  - In M, every repeat raises held-out FM error relative to the original graph at every evaluation.
- **Phase 3 (optional):** U5 = `s:l-l-l-l-l-`, R5 = a random 5-subset per chunk, and A5 = the 5 evaluations
  where repeating l costs least in the same table.

**Training:**
- Every model starts from 300k with its own fixed graph for 10,000 updates; the final checkpoint is u10000.
- Same recipe and the same frozen modules.
- Paired data, augmentation, FM noise/time and dropout streams.
- Fix applied: this torch build seeds its default RNGs randomly per process, so dropout is now keyed per
  update. Identical runs now reproduce exactly.

**Evaluation:** D, 20 scenes × 2 replicates × 4 tasks = 160 episodes per model.
- S* is the S_l with the best task-macro D success.
- Comparisons: S* − N, T1 − S*, Tfree − S*, Tfree − T1.
- Confirmation: 3 training seeds × 50 unopened T scenes per task (2,400 episodes) after freezing.

**Development results** (D: 20 scenes × 2 noise replicates × 4 tasks = 160 episodes per model; task-macro
success; paired scene-cluster 95% CIs):

| model | graph | block calls | success | handover / lift / microwave / container |
|---|---|---|---|---|
| N | `s:----------` | 40 | 36.2% | 42 / 57 / 12 / 32 |
| S0 | `s:0000000000` | 50 | 38.8% | 55 / 57 / 20 / 22 |
| S1 | `s:1111111111` | 50 | 42.5% | 48 / 57 / 25 / 40 |
| S2 | `s:2222222222` | 50 | 37.5% | 55 / 57 / 12 / 25 |
| **S3 = S*** | `s:3333333333` | 50 | **43.1%** | 52 / 62 / 22 / 35 |
| T1 | `s:2222000000` | 50 | 41.2% | 60 / 70 / 12 / 22 |
| Tfree | `s:2222000002` | 50 | 29.4% | 40 / 45 / 10 / 22 |
| U5 | `s:3-3-3-3-3-` | 45 | 31.9% | 50 / 38 / 15 / 25 |
| R5 | random 5 of 10 per chunk, block 3 | 45 | 38.1% | 40 / 62 / 12 / 38 |
| A5 | `s:-33333----` | 45 | 39.4% | 45 / 72 / 10 / 30 |

| contrast | difference [95% CI] |
|---|---|
| S0 / S1 / S2 / S3 − N | +2.5 / +6.2 / +1.2 / +6.9 [+0.0, +13.1] |
| S* − N | +6.9 [+0.0, +13.1] (selection-biased on D) |
| T1 − S* | −1.9 [−8.1, +5.0] |
| Tfree − S* | −13.8 [−21.2, −6.2] |
| Tfree − T1 | −11.9 [−19.4, −5.0] |
| A5 − U5 / A5 − R5 | +7.5 [−0.0, +15.0] / +1.3 [−6.2, +8.8] (A5 must beat both: not met) |
| A5 / U5 / R5 − S* | −3.8 [−11.9, +4.4] / −11.3 [−19.4, −3.8] / −5.0 [−11.9, +1.3] |

**Development reading:**
- **Do loops help?** Weakly, for some blocks only. S3 and S1 are about +7 over N (CIs at about 0); blocks 0
  and 2 are null. Block 3 was the worst zero-shot loop but is the best trained one.
- **Constant across FM time?** No evidence for time dependence. T1 ties S*; Tfree (2 switches) is worse than
  S*, T1, its own constituent constants (−8 to −9) and N.
- **Every evaluation?** No 45-call rule matched S*. U5 is significantly worse, and the selected A5 does not
  beat random R5.
- **Offline-to-closed-loop:** every trained model's own held-out FM error is within 1% of N's. M's
  per-evaluation block ranking was not reproduced by dedicated training.
- **Noise floor:** noise replicates of one model disagree on 32–38% of scenes.
- **Infrastructure:**
  - Random simulator hangs (mostly open_microwave) killed whole 20-scene jobs. Retries now resume from
    the hung scene (`HORSEA_SCENE_RESULTS`).
  - A CUDA OOM (2 trainers + a diagnostic beside 4 simulators) killed the first Tfree training. Now only
    one non-simulator GPU job runs at a time.

**Confirmation** (frozen in the manifest before any T rollout): N, S3, T1 and Tfree × training seeds 0–2 × 50
T scenes per task (2,400 episodes). Analysis: `python -m horsea.rt.loopc_confirm`.

**Code:**
- `horsea/rt/graph.py`, `continue_train.py --arm G`, `loopc_etable.py`, `loopc_analysis.py`, `loopc_confirm.py`.
- Tests: `tests/test_rt_sched.py` (7 checks: N parity, traces, schedule == window graphs, row gating,
  backprop through both calls, R5 codes, resolution) and `tests/test_rt_graph.py` (7 checks).

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
9. **Pair simulator episodes by initial state, not episode index.** RoboTwin's admission check is
   random, so the same index can be a different scene.
10. **Measure the noise floor first.** Identical RoboTwin configurations flip outcomes, and graph
    rankings had split-half r ≈ 0.
11. **Check what default RNG seeding does.** Here torch seeds randomly per process, so unkeyed dropout
    breaks paired training streams.

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
| RoboTwin policy and studies | `horsea/rt/{policy,data,train,graph,continue_train,loopc_etable,loopc_analysis,graphB_analysis}.py`; deploy `RoboTwin/policy/HorseaFM/` (copy in `horsea/rt/deploy/`) |
| RoboTwin results | `experiments/rt/{base,graph,graphB,loopc}/` (manifests committed) |
| queue | `scripts/queue.py`, `experiments/queue/`, `systemctl --user status horsea-queue` |
