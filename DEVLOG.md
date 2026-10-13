# Horsea devlog

Fast and slow memory for a flow-matching robot policy: design history, experiments, results and lessons.
The newest status is at the top; the history follows in chronological order.
All success rates are closed-loop simulation success (LIBERO until 2026-09-30, then RoboTwin 2.0 and Isaac Sim) unless
marked "offline".

---

## Current status (2026-10-13)

**Direction since 2026-10-09: GPT-6 Astra as System 2, and what System 1 beneath it should be.** Astra (`gpt-6-astra`
through the Codex CLI) explores a scene by observing, acting, remembering outcomes and retrying (§14). The open question
is the learned System 1 under it: what it owns, what it learns from, and whether its short-term memory helps. Platform:
the EgoVLA Ego-Humanoid benchmark (Isaac Sim 5.1, Unitree H1 + two Inspire hands).

What we know:
- **Astra alone:**
  - LIBERO task 48: 2 of 3 attempts; RoboTwin place_container_plate: 3 of 3; lift_pot: 1 of 3.
  - Dex hand: Stack-Can-Into-Drawer 2 of 10; first successes at attempt 2 (Unload-Cans) and attempt 1 (Open-Laptop);
    Insert-And-Unload-Cans 0 of 5 (§14, §17).
- **A student trained on Astra's data:** none worked in one night (§15).
- **Local search around Astra's program as System 1:** not useful (§16, §17). It rescued 0 of 66 matched blocks on 4
  tasks; fresh-start success was 119/180 for repeating, random and history-adaptive search alike.
- **Where grasps fail (§18):** the robot reaches what Astra commands (median 0.3 cm). On the dex hand:
  - 65 % of failed grasps close on air (Astra aimed wrong); 35 % touch the object and lose it.
  - A controlled test shows the can grasp tolerates wrist shifts up to 2 cm and rotations up to 10 deg.
- **Cup A → cup B → cylinder C (§20):** a memory-based System 1 (experience store + locally weighted linear model +
  short-horizon planner) kept Astra's command on 743 of 745 segments, so it never adapted.
  - Cup B failed in the grip. Astra re-used cup A's grasp and the hand reached it within 0.1 cm. The fingers touched
    cup B, but it slipped out when lifted: 0 of 15 such grasps held. Shifts of 0.2–1.2 cm did hold.
  - That fix was within System 1's bounds. It could not find it for two reasons: its memory held three identical
    successful grasps, and its goals defended Astra's command.
- **Batch 1 of the bottleneck plan (§21, 2026-10-13):**
  - Time: 76-88 % of wall time is Astra waiting (about 19 s per call). A failed cup B attempt makes 18-20 calls, about
    half of them recovery.
  - Failure: cup A's grasp leaves the thumb over cup B's opening instead of outside its wall. Controlled replays hold
    0 of 20; a 0.8 cm wrist shift holds 20 of 20.
  - Frames, action channels and execution are correct.
  - System 1 cannot make the correction: its deviations last 0.3 s before Astra's command re-aims the hand; its
    objective rejects the only rescuing actions; its state lacks the thumb's position.

**Next (to decide):** where the grasp correction should come from:
- make each fingertip's position relative to the wall and rim visible to Astra; or
- give System 1 a persistent object-relative action, a contact-geometry state, and varied grasp probes with failures.

Then Phase 5 (local execution from matched grasps). Design note: `research/system1_dexhand_design.md` (§19).

**Earlier phases:**
- LIBERO energy Horsea: +15.4 on hidden rotations; LIBERO was removed on 2026-09-30.
- RoboTwin 2.0 base FM policy and trained internal loops: ended 2026-10-02; loops do not help.

---

## Earlier status (2026-10-01, RoboTwin; superseded)

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

Confirmation (T, seeds 0 + 1): S3 − N = −0.2 [−4.8, +4.2] and Tfree − S3 = −8.3 [−13.0, −3.5]. Trained loops
do not help; the loop study ended on 2026-10-02.

**2026-10-02:** prior-guided exploration audit on open_microwave found no selection problem to solve (one option
wins almost everywhere; §13 of the RoboTwin notes in `experiments/explore/`).

**2026-10-09: GPT-6 Astra directly explores a scene** (§14): LIBERO task 48, 2 of 3 attempts succeeded (29 and 10
decisions); RoboTwin place_container_plate through RoboTwin's planner, 3 of 3 (8 / 11 / 16 decisions); bimanual lift_pot 1 of 3 (success in attempt 3 after two self-stopped attempts).
**2026-10-10:** Isaac Sim dex-hand bimanual task (EgoVLA Stack-Can-Into-Drawer, H1 + two Inspire hands): 0 of 3 in the original
budget; continued to 10 attempts at the user's request (no guidance): **2 of 10 successes (attempts 7 and 10)**, both with the
right hand placing the can on the saucer and the left palm closing the door (§14).

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

## 13. Trained loop consistency (spec 2026-10-01; ENDED 2026-10-02)

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
T scenes per task. The user stopped it at 01:13 on 2026-10-02 when the loop study ended. Seeds 0 and 1 were
complete for every model and task (1,600 episodes); seed 2 was partial and is excluded.

| model | T success (seeds 0 + 1) |
|---|---|
| N | 40.8% |
| S3 | 40.5% |
| T1 | 38.5% |
| Tfree | 32.2% |

| contrast | T [95% CI] | D |
|---|---|---|
| S3 − N | −0.2 [−4.8, +4.2] | +6.9 |
| T1 − S3 | −2.0 [−6.5, +2.8] | −1.9 |
| Tfree − S3 | **−8.3 [−13.0, −3.5]** | −13.8 |
| Tfree − T1 | **−6.2 [−10.8, −1.8]** | −11.9 |
| Tfree − N | **−8.5 [−13.0, −4.2]** | −6.9 |

**Conclusion:**
- Fine-tuning with a trained loop (25% more decoder compute) does not raise success. The development gain of
  S3 was a selection effect.
- Switching the looped block twice across FM time costs about 8 points; one switch is neutral.
- Not due to training length: the loop penalty (+9.4% held-out error zero-shot) is gone by 2,500 updates,
  both models plateau, and S3 − N held-out error stays at −0.2 to −0.4% from 5k to 10k updates on all seeds.
- Untested: loops trained from scratch, more than two repeats, or larger learning rates.
- N is kept as the controller going forward.

**Code:**
- `horsea/rt/graph.py`, `continue_train.py --arm G`, `loopc_etable.py`, `loopc_analysis.py`, `loopc_confirm.py`.
- Tests: `tests/test_rt_sched.py` (7 checks: N parity, traces, schedule == window graphs, row gating,
  backprop through both calls, R5 codes, resolution) and `tests/test_rt_graph.py` (7 checks).

## 14. GPT-6 Astra explores a scene directly (2026-10-09)

**Question (user):** with no policy, planner of its own or weight update, how far does Astra get by observing, acting,
remembering outcomes and retrying? Prior knowledge = Astra's pretraining; short-term memory = its recorded interactions.
Harness: `astra_explore/` (Codex CLI `gpt-6-astra`, saved ChatGPT login, one fresh `codex exec` per decision under a Landlock
filesystem sandbox, tools disabled, strict JSON responses with assessment / intent / hypothesis / expected change / references /
bounded memory rewrite; the simulator pauses during inference; retries never duplicate execution; evaluation-only logs kept
apart from the prompt). Reports: `experiments/astra_explore/run_2026-10-09_astra_t48/REPORT.md` (LIBERO) and
`run_2026-10-09_astra_rt_place_container_plate/REPORT.md` (RoboTwin).

**LIBERO-90 task 48 "pick up the ketchup and put it in the basket"** (held-out split, init state 0, native 7-D OSC deltas at
20 Hz, 300-step horizon, 3 attempts, 40/120 calls):

| attempt | outcome | decisions | steps |
|---|---|---|---|
| 1 | success | 29 | 265 |
| 2 | success | 10 | 87 |
| 3 | stopped by Astra, no success | 39 | 296 |

- Every grasp was treated as a hypothesis and verified by a lift (13 of 15 closures); all 15 verdicts matched the bottle's
  recorded motion; it never claimed success and ended attempt 3 with "stop without claiming success".
- Dominant failure: depth. At 12 of 15 closures the hand was 3.6-13.8 cm in front of the bottle along x, which the front camera
  cannot show and the wrist camera loses at grasp height; corrections were 2-3 cm with alternating signs.
- Memory reused attempt 1's release coordinates (transport in 2 moves instead of 5); replaying attempt 2's grasp coordinates
  failed because that grasp had followed an accidental push of the bottle.
- 78 calls, median latency 17.8 s, 1.09 M input tokens.

**RoboTwin `place_container_plate`** (seed 1700000, expert-admitted; RoboTwin's `ee` interface: absolute target poses per arm
executed by RoboTwin's planner, disclosed as planner-assisted; 3 attempts, 40/120 calls):

| attempt | outcome | decisions | grasp attempts |
|---|---|---|---|
| 1 | success | 8 | 1 |
| 2 | success | 11 | 2 |
| 3 | success | 16 | 3 |

- Every planned target reached within 0.7 cm; all 6 lift-test verdicts correct; 35 calls, median latency 15.5 s.
- Holds were off-centre wall pinches (separation 5.4-5.5 cm); centred closures slid off (4.9 cm). Astra stated and confirmed
  the "one finger inside, one outside" rule in attempt 3.
- Replaying a previous grasp pose failed twice; replaying the release pose worked twice (1 cm placement).

**RoboTwin `lift_pot`, bimanual** (seed 1700002, expert-admitted; both grippers must hold the two side handles and lift the pot
above 0.82 m upright; 3 attempts, 40/120 calls): attempt 1 stopped by Astra after 39 decisions (one handle held, the pot pushed
out of the right arm's reach), attempt 2 stopped after 13 (pot flung off the table while hanging from one gripper; Astra inferred
it was "displaced outside the visible workspace"), attempt 3 **success in 9 decisions**. Learned and kept across attempts: fingers
along y to clear the lid (diagnosed from a blocked descent), a reachable approach height, settling holds before trusting a closure,
small coordinated lift tests. 17 planner failures taught the arms' reach; 61 calls total.

**Isaac Sim dex-hand, bimanual** (EgoVLA benchmark, Hongyi's project; `Humanoid-Stack-Can-Into-Drawer-v0`, Unitree H1 + two
Inspire hands; the project's first evaluation scene, room 1 table 2, randomize_idx 5788; instruction "Put can on the saucer, and
Close the drawer"; success = can upright on the saucer inside the cabinet AND the cabinet door closed >= 80 %). The task is
bimanual collaboration by construction: the left arm starts outside the open door, and the door's sweep hits the can, so the
right hand must place the can before the left hand can close the door. Interface (disclosed): absolute wrist targets (position
+ finger/palm directions) tracked by the benchmark's own differential IK along a 1 cm / 3 deg per step path; four closure
fractions per hand (knuckles, finger tips, thumb rotation, thumb bend), <= 1/30 per step; horizon 400 steps (the project's).
Feasibility check before the run (author, privileged, ~40 scripted grasps): door closable, can never lifted
(`experiments/astra_explore/probes/isaac_feasibility_2026-10-10/FEASIBILITY.md`).

| attempt | outcome | decisions | closures / lift tests | best can lift |
|---|---|---|---|---|
| 1 | stopped by Astra | 15 | 3 / 1 | 2.5 cm (pushed, tipped) |
| 2 | horizon (400 steps) | 23 | 5 / 5 | 2.9 cm |
| 3 | horizon (400 steps) | 27 | 7 / 6 | **14.5 cm, then lost** |

- 0/3 successes, 65 calls, 0 failed calls, median latency 19.6 s. The left arm never moved; the door was never touched.
- Astra found the reachable hand orientations from the reported residuals (153 deg -> 2 deg), then adopted the orientation the
  hand actually reached ("Keeping the achieved orientation may improve pose tracking"; < 1.2 deg afterwards) and reused it in
  attempt 3 from the first motion. Every closure was verified by a lift test; 14 of 15 closures had no contact.
- Attempt 3 D12-D13: index contact 13.7 N, knuckles blocked, lift test raised the can 14.5 cm, the first lift of the can through
  this interface (the author's scripted grasps never got one). A fast further 7 cm lift lost it.
- Report: `experiments/astra_explore/run_2026-10-10_astra_isaac_stack_can_drawer/REPORT.md`; site case `/dexhand/`.
- **Continued to 10 attempts** (user: "maximize the attempt to 10, and also, you cannot guide the policy!"): same run resumed
  with `run_dex.py --resume` (history rebuilt from the run's records, verified identical to the recorded views; prompt text
  diffed identical). Attempts 4-10: horizon, horizon, stopped, **SUCCESS (18 decisions, step 378)**, stopped, stopped,
  **SUCCESS (11 decisions, step 298)**. 187 calls in total, 0 failed.

| attempt | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| outcome | stop | horizon | horizon | horizon | horizon | stop | **success** | stop | stop | **success** |
| best progress | – | – | can lifted 14.5 cm | lifted, flung; door 0.51 rad | carried into cabinet, dropped 11.8 cm from saucer | – | placed 3.4 cm, door 1.35 rad | – | – | placed 3.2 cm, door 1.33 rad |

- Each piece was found in one attempt and carried by Astra's own memory to the next:
  - the trackable orientation (attempt 1);
  - adopting the achieved pose (2);
  - a contact-verified grasp plus a lift test (3-5);
  - transport over the lip (5);
  - "verify retention before transport" (6);
  - the left-palm door push (4);
  - the release pose, which Astra wrote into memory at attempt 8 D02 and used as attempt 10's transport target.
- Not robust: after the first success 1 of 3 attempts succeeded, and 29 of 40 closures registered no contact.

**Reading:** the procedure (approach, correct, close, verify, carry, align, release) and the honesty were stable across
simulators; the limits were perception of depth under raw delta control and the irreproducibility of contact outcomes from
remembered coordinates. Nothing causal about memory and no success-rate estimate follows from 3 attempts on one scene.
Infrastructure note: the first LIBERO launch aborted before any model call because Codex could not read `~/.codex/rules`
inside the sandbox (`--ignore-rules` fixed it; verified with a network-denied dry run).

## 15. Overnight System-1 / System-2 transfer study (2026-10-10)

**Question (user spec):** can a learned policy (System 1) learn stable execution from Astra's own interaction data
(System 2), and can later Astra corrections improve it once Astra is switched off? Twelve registered configurations
(E01 Astra-only ... E12 extra reset demos) on four platforms, one 8-hour window, at most 600 new Astra calls (150 per
platform), one active cell, one training seed, staged order with gates. Code `astra_explore/s1s2/` (orchestrator, ledger,
platform adapters, report); run `experiments/s1s2_overnight_2026-10-10/` (`OVERNIGHT_REPORT.md`, auto `REPORT.md`,
`metrics.json`, `LAUNCH.md` with every rule change and its time); systemd user unit `s1s2-overnight`.

**Interfaces.** RoboTwin: Astra's planner-executed end-effector targets recorded at the student rate as RoboTwin's own
commanded joint vector (every 15 physics steps), the student = Horsea FM base step300000 (seen tasks; decoder-only
`continue_train`, U = 1000, dU = 500). Dex: Astra and an ACT student from scratch share the native 38-D wrist + hand step.
LIBERO: student BLOCKED_INTERFACE (no compatible student pipeline). Replay of recorded commands: 0.0 / 0.1 cm (RoboTwin
single / bimanual median end-effector deviation), 1.3 cm wrist tracking (dex).

| platform | Astra from reset | students (dev; test = 10 fresh starts) | gate / block |
|---|---|---|---|
| RoboTwin place_container_plate | source 2/4 (86 calls), E01 1/1 dev | E02 2/5, test 3/10; E03 1/3; E04 blocked (2 successes); E05 0/5, test 1/10; E06 0/3; E07 0/3; E10 0/3, test 3/10 | NO_STUDENT_SIGNAL: E08 E09 E11 E12 |
| RoboTwin lift_pot | source 0/4 (117 calls) | E02 3/3, test 2/10 | BLOCKED_DATA E03-E12; E01 PENDING_BUDGET |
| Isaac dex Stack-Can-Into-Drawer | source 0/6 (118 calls) | none (E02 NOT_AVAILABLE) | BLOCKED_DATA E03-E12; E01 PENDING_BUDGET |
| LIBERO-90 task 48 | E01 1/3 dev (106 calls) | none | BLOCKED_INTERFACE |

- Single-arm test starts: successes were disjoint across students (E02 on 35, 41, 42; E05 on 38; E10 on 36, 37, 39), i.e.
  within the base policy's known noise floor. Students trained on Astra data failed 0/2 on their own training starts;
  open-loop arm error on the demonstrated frames fell only 0.032 -> 0.024 rad (E03 on its own demo).
- Lift_pot: Astra lifted the pot past 0.82 m twice but never with both grippers at the handle contact points and the pot
  upright; the original student solved 3/3 dev starts. Dex: the can never reached the saucer in 6 episodes.
- 454 of 600 calls used (113 / 117 / 118 / 106), all valid responses, no infrastructure retries; plus 26 preflight calls.
- Not applied: pooling platform reservations into the global cap (blocked by the permission system; left to the user).

**Reading:** none of Astra's channels produced a working student tonight. The data channel is valid, but one or two Astra
successes and the registered decoder-only recipe do not move the pretrained student (and the from-scratch dex student never
got data). On lift_pot and the dex task the teacher, not the student, is the bottleneck. Smallest next experiment: a
single-demonstration fit test on the training start (larger registered update budget, separate unfrozen-encoder arm) before
any further Astra collection.

## 16. System 1 beneath Astra: local execution adaptation pilot (dex hand, 2026-10-10)

**Question (user brief):** is a useful System 1 a distribution over bounded adjustments to Astra's action program, updated
from execution outcomes?
- H1: nearby variants beat repeating the original.
- H2: updating the proposal distribution from previous outcomes finds reliable variants faster than a fixed distribution.

Setup:
- **Task and source program:** Stack-Can-Into-Drawer; Astra's successful attempt 10 (§14).
- **Operations:** grasp, transport and door, each started from a physically rebuilt prefix.
- **Conditions:** 6 per operation: the operated object's mass 0.8–1.2× and friction 0.7–1.0×.
- **Arms:** A repeats the original; B samples bounded random variants (a fixed Gaussian around the original); C is B
  plus a CEM-style update after attempt 5.
- **Budget:** 9 adaptation attempts, then the best candidate is evaluated on 3 fresh starts.
- **Code:** `astra_explore/s1local/` (no-render environment, snapshot / restore, PhysX mass and friction scaling).
- **Report:** `experiments/s1local_2026-10-10/REPORT.md`.

| | A (repeat) | B (random variants) | C (history-adaptive) |
|---|---|---|---|
| fresh-start success | 42/54 | 41/54 | 41/54 |

- **Grasp and transport were a ceiling:** the original succeeded on the first try in 12 of 12 conditions.
- **The door falls just short:** the heavy door stops at 1.17–1.30 rad; 1.326 is needed.
  - Every bounded variant did worse (0.74–1.20 rad).
  - Repeats of the original sometimes succeed by execution noise alone (final angle spread up to 0.65 rad).
- **C learned but changed nothing:** its later proposals scored higher than B's, but no candidate succeeded where the
  original failed, so C = B exactly.
- **Grasp brittleness was hidden:** with the can displaced 8 mm on a fresh start, Astra's closure missed in 4 of 6
  conditions. A single adaptation start with a saturating score could not detect this.

**Reading:** H1 not supported. H2 inconclusive: the bounded family contained no fix to find. What is missing is
closed-loop completion ("push until the door is closed"), not better open-loop search. Candidates must also be scored on
several starts.

## 17. Does the System 1 conclusion scale? Four more dex tasks (2026-10-10/11)

Same benchmark, hand and harness. Per task:
1. Astra alone: no guidance, at most 10 attempts, stop at the first success.
2. The §16 protocol on Astra's program, with task-generic code (`astra_explore/s1local/{generic,specs,runner_task,crosstask}.py`).

Report: `experiments/s1local_crosstask_2026-10-10/SCALABILITY.md`.

| task | Astra alone | pilot |
|---|---|---|
| Unload-Cans | success at attempt 2 (20 decisions; right hand only) | 3 operations, all ceiling |
| Open-Laptop | success at attempt 1 (lid 1.270 rad vs 1.257 needed) | 1 operation |
| Insert-And-Unload-Cans | 0 of 5 (attempt 5 put both cans into the tray, then ran out of time) | 4 operations on attempt 5's insert stage |
| Sort-Cans | not run (Astra workspace spend cap, 2026-10-11 00:24) | – |

Results over 4 tasks, 11 operations and 66 matched blocks:
- **0 rescues by local search.**
- Fresh-start success was 119/180 for each of A, B and C, and C = B in every operation.
- Searching only delayed the first success, by 0.3 attempts pooled.

The remaining failures are of the §16 kinds plus one new one:
- near-threshold open-loop pushes (door, laptop lid);
- grasps that work on the adaptation start but fail on fresh starts;
- long contact-rich lead-ins whose outcome does not reproduce (the right-can grasp start in Insert-And-Unload-Cans).

**Reading:** the conclusion held. The useful System 1 functions are closed-loop completion of the final motion,
closed-loop re-acquisition of the object, and scoring candidates on several starts. Open-loop parameter search is not
one of them. Sort-Cans and Insert-And-Unload-Cans attempts 6–10 were not resumed after access returned.

## 18. Why grasps fail: log diagnosis and a controlled grasp test (2026-10-11)

**Question (user):**
- When the robot fails, is Astra's spatial understanding wrong, or does the robot not do what Astra says?
- On the dex hand, what is the main cause?

Method: a log-only diagnosis of every Astra run, with privileged object state used for evaluation only. Script
`astra_explore/s1local/diagnose_grasps.py`; report `experiments/s1_diagnosis_2026-10-11/DIAGNOSIS.md`.

**Parallel grippers:** the robot does what Astra says.
- RoboTwin: 28 of 28 and 71 of 80 commanded moves reached, median error 0.26–0.27 cm. The 9 misses were out of reach.
- LIBERO: 10 of 22 closures closed on nothing, 3.0–14.3 cm off along x. That is depth, which no camera shows well.

**Dex hand, 137 closures:** 14 held and 123 failed.

| failure | grasps | share of failures |
|---|---|---|
| closed on air | 80 | 65 % |
| touched, did not hold | 21 | 17 % |
| pushed the can away | 20 | 16 % |
| lifted, then slipped | 2 | 2 % |

- In 60 of the 80 air grasps the wrist reached Astra's commanded pose (median 0.27 cm, 1.4 deg), so Astra aimed at the
  wrong place.
- 72 % of Astra's decisions came before the first lift of an attempt.

**Controlled grasp test** (`astra_explore/s1local/grasp_audit.py`): 90 trials around Astra's successful Unload-Cans grasp.

| perturbation | held |
|---|---|
| none (reference grasp) | 6/6 |
| wrist shift within ±2 cm | 34/36 |
| wrist shift of 4 cm | 2/12 |
| rotation within 10 deg | 8/8 |
| rotation beyond 10 deg | 0/8 |
| mass / friction changes | 10/10 |
| closure levels | 8/10 |

**Reading:** on the can, contact is robust once the hand is within about 2 cm and 10 deg of the reference grasp.
- About two thirds of dex failures are aiming errors. They are Astra's job, or a final-approach controller's.
- About one third are contact failures. They are System 1's job.

## 19. System 1 design note (2026-10-11)

`research/system1_dexhand_design.md`, built on four literature surveys with every reference checked. In brief:
- **What System 1 owns:** what needs feedback faster than Astra's decision period, or signals Astra cannot see: contact,
  force, finger configuration and sub-centimetre geometry. It is a contact controller conditioned on a desired object
  motion.
- **The interface ("System 1.5"):** a small contract, not text: target object, object-relative target motion,
  tolerance and success predicate.
  - It tilts System 1's distribution, π ∝ π_θ · exp(β U).
  - It comes back up as a structured report.
- **Short-term memory:** a small adaptable state on a frozen pretrained System 1. It is updated per attempt with critic
  weights and a KL bound, never by a gradient step toward one lucky action.
- **Long-term memory:** Astra's stored contracts, plus consolidation of System 1's base across tasks.
- **Plan, ablating each step:**
  1. contact controller v0, no learning;
  2. a small flow policy distilled from v0;
  3. memory ablations;
  4. Astra + System 1;
  5. generality.

## 20. Cup A → cup B → cylinder C: does a memory-based System 1 transfer execution experience? (2026-10-12)

**Question (user brief):** with Astra as System 2 and a numerical System 1 beneath it, does execution experience on cup A
transfer to a new cup B and a cylinder C? Hypotheses:
- H1: cross-object experience helps.
- H2: online updating is needed.
- H3: a numerical model adds value over Astra reading the same history.
- H4: modelling beats replay.

**Objects:**

| object | size | build |
|---|---|---|
| cup A | the benchmark can's size: D 6.37 cm, H 12.18 cm | hollow |
| cup B | 1.15× diameter, 0.90× height | hollow |
| cylinder C | 0.90× diameter, 1.10× height | solid |

- All three have mass 0.35 kg and the same friction.
- Cup B′ (hidden off-axis centre of mass) was built but not run.
- Scene: Unload-Cans. Task: lift the object at least 5 cm and hold it for 1 s.

**System 1** (`astra_explore/cup_transfer/system1.py`):
- Memory: an experience store of (state, requested and applied command, resulting change).
- Model: locally weighted ridge regression (Gaussian kernel) predicting the change over one segment.
- Planner: 32 bounded candidates × 3 segments, scored by per-mode goals (approach / close / lift / hold).
- Bounds: ±1.5 cm and ±0.15 closure per segment.
- Fallback: Astra's command whenever the data give no local support.

**Arms:**
- Astra-history: Astra with its cup A notes, no System 1.
- Nearest-neighbour replay.
- Empty-model.
- Frozen-model: cup A memory.
- Adaptive-model: cup A memory plus updates on the target.

**Protocol:**
- 2 starts per object, at most 2 attempts per start.
- 40 calls and 400 control steps per attempt.
- Source: Astra on cup A, 3 of 3 successes (121–123 steps, identical grasps).

**Run:**
- 29 attempts, plus one that the spend cap interrupted and that was rerun from scratch.
- 331 Astra calls, 254 of them on cup B.
- Cup B and cylinder C ran in parallel, on two simulators.
- The user's EgoVLA sweep was paused for the run, with approval, and restarted afterwards.
- Report: `experiments/cup_transfer_2026-10-12/REPORT.md`.

| method | cylinder C, first try | cup B, first try | cup B, within 2 tries |
|---|---|---|---|
| Astra-history | 2/2 | 1/2 | 2/2 |
| Nearest-neighbour | 2/2 | 2/2 | 2/2 |
| Empty-model | 2/2 | 0/2 | 0/2 |
| Frozen-model | 2/2 | 1/2 | 2/2 |
| Adaptive-model | 2/2 | 0/2 | 0/2 |

**System 1 almost never acted.**
- It had local support on 745 segments and kept Astra's command on 743 of them; 185 more segments fell back for lack
  of support.
- Its one-segment predictions were accurate (RMSE 0.02–0.06, normalised).
- No hypothesis was supported. The differences on cup B are Astra's variability.

**Cylinder C was a ceiling:** cup A's grasp works directly, in about 122 steps.

**Cup B failed in the grip** (per-grasp check: `cup_transfer/grasp_check.py`, `evaluation_results/run/grasp_check.md`):
- Astra re-used cup A's grasp: wrist at (−14.5, −5.0, +6.4) cm from the base, closure 0.40/0.45. The hand reached it
  within 0.1 cm.
- That grasp held cup A 3 of 3 and cylinder C 8 of 9, but cup B 0 of 15.
- The fingers touched cup B (3–6 N), and it slipped out when lifted. The cup did not move while the hand closed; on A
  and C it moved 0.4–0.7 cm and tilted 7–14 deg. Batch 1 (§21) showed that this motion is not what decides the hold;
  the thumb's final position is.
- The four upright grasps that held cup B were shifted 0.2–1.2 cm.
- Two other successes lifted a knocked-over cup; the success test has no upright check.
- Astra needed 2–5 regrasps and 290–750 control steps.

**Why System 1 could not fix it**, although the fix was within its bounds:
1. Its memory held three identical successful grasps, so it predicted no candidate to beat Astra's command.
2. Its goals defended Astra's command. "Reach Astra's pose" protects the transferred grasp. "Do not move the object
   while closing" even favoured the slipping grasp.

The linear model was not the issue.

**Correction (same evening):** the first report said cup B's grasps missed with the fingertips 3 cm outside the wall.
That compared the open hand with the cup's radius, but the open hand is just as far from cup A, where the grasp holds.

**Light rendering** (only the four prompted cameras) saves little: 10.1 GB vs 10.5 GB per simulator.

**Reading:** System 1 has a real job here, in the grip. But a System 1 that learns only from Astra's successes can only
agree with Astra. Proposed next (superseded by §21):
1. grasp-security goals: the object is drawn into the hand while closing (not valid, see §21), and it follows the hand
   at lift-off; otherwise regrasp;
2. practice data with deliberate variation and failures;
3. cup B from at least 5 starts.

The placement (Task 4) and hidden-centre-of-mass (Task 5) follow-ups were not run.

## 21. Batch 1 of the bottleneck plan: where time goes, where the first failure is, why System 1 does not fix it (2026-10-13)

**Plan (user, 2026-10-13):**
- Make the existing Astra-robot system faster and more reliable on new tasks.
- Diagnose before repairing: reproduce and instrument, then verify geometry and the action interface, then diagnose System
  1's candidates, objective and model.
- Make one minimal repair, with machine-readable gates.
- New-object transfer and delegation come later.

**Setup:**
- **Code:** `astra_explore/cup_transfer/{batch1,batch1_report,costs}.py`. The common executor gained per-segment timing and
  command lineage; System 1 gained an optional candidate trace whose selection is identical, checked on 101 of 101 logged
  states.
- **Results:** `experiments/cup_batch1_2026-10-13/` (`REPORT.md`, `REPORT_DATA.md`, `gates.json`, `manifest_*.json`).
- **GPU:** the user's EgoVLA dense sweep was paused 00:15-01:10 (approved) and restarted (`pause_dense.sh`,
  `restart_dense.sh`).

**Phase 0, cost.** Astra waiting is the cost:
- 88 % of wall time over the 29 attempts of §20.
- 57-76 % in six instrumented live attempts. These shares are lower only because three simulators shared the GPU.
- Median 19.1 s per call. System 1 planning is about 3 ms per control step.
- Cup B: 2 of 2 failures, 18-20 calls each, 7-9 of them recovery after failed lifts.
- Cups A and C: 2 of 2 successes each, 4 calls each.
- The common executor re-plans each segment from a lagging wrist, which halves free-space speed: 1.5 of 3 commanded cm
  per segment.

**Phase 1, geometry and interface:**
- **Correct:**
  - object frames and sizes;
  - every wrist channel (within 0.06 cm and 0.1 deg);
  - each closure group moves only its own joints;
  - camera projection vs depth (within 0.33 cm).
- **Replays of cup A's grasp (no Astra, no System 1):**

| object | grasp | held |
|---|---|---|
| cup A | original | 10/10 |
| cylinder C | original | 9/10 |
| cup B | original | 0/20 |
| cup B | shifted 0.8 cm (-0.5 x, +0.5 y, -0.3 z; from the cup B grasps that held in §20) | 20/20 |
| cup B | height kept relative to the rim | 0/20 |

- **Contact geometry at the end of closing:**
  - Cup B, original: the thumb tip ends over the opening, 1.85 cm inside the outer wall and 1.4 cm above the rim.
  - Every grasp that held: the thumb tip ends outside the wall (+0.5 to +1.0 cm).
- **Correction to my earlier proposal:** the shifted grasp holds without the cup moving while the hand closes, so "the
  object is drawn into the hand" is not a valid success check.

**Phases 2-3, System 1.** Six cup B decision states (two each in approach, closing and lifting). For each: System 1's full
candidate pipeline, all 32 candidates executed from the restored state twice, then a fixed continuation.
- **Selection:** System 1 chose Astra's command in all six states, and in 357 of 357 supported live segments.
- **No collapse:** every candidate was supported, and the 32 first commands stayed distinct.
- **Holding candidates:** only at mid-closing did any candidate hold (2 of 32, one run each), by pushing the cup about
  2 cm.
  - The realized-state objective ranked them 19th and 20th (the "do not move the object" term).
  - The model ranked them 17th and 27th (it predicted no motion).
  - Dropping that term still ranks them 17th and 26th, because of the deviation penalty.
- **Model ranking vs realised ranking (Spearman):** 0.53-0.77 in approach and lift, 0.18-0.37 while closing.
- **Persistent offsets** from the same states:
  - The shift holds 2 of 2 from mid-approach and from the start of closing.
  - It holds 0 of 2 from mid-closing.
  - Each single axis of it, and the rim height, holds 0 of 2 everywhere.

**Reading:**
- The earliest failing link is the geometry of the transferred grasp (the command), decided before closing.
- System 1 as built cannot correct it:
  - its deviations last 0.3 s before Astra's command re-aims the hand;
  - its objective rejects the only rescuing actions;
  - its state lacks the thumb's position relative to the wall;
  - its data have no variation.
- The minimal repair (a geometry correction) passes its diagnostic. It is a geometry / interface improvement from
  development data, not a general rule, and not memory.
- Next, to decide: make the fingertip-to-wall geometry visible to Astra, or give System 1 a persistent object-relative
  action, a contact-geometry state and varied probes.

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
12. **A learner fed only the teacher's identical successes can only agree with the teacher.** The cup System 1 kept
    Astra's command on 743 of 745 segments. Improvement needs variation and failures in its data (the same root as
    lesson 2).
13. **Goals must not penalise what helps.** "Do not move the object while closing" ranked the only candidates that
    rescued a cup B grasp 19th and 20th of 32, because they worked by pushing the cup (§21).
14. **Test a failure explanation on the case that worked.** The first cup B explanation used a metric (open-hand
    fingertip distance) that had the same value on cup A, where the grasp held.
15. **Near-threshold open-loop failures need closed-loop termination, not parameter search** (door, laptop lid).
    Score candidates on several starts: a single start with a saturating score hides brittleness.

**Process**
- Keep the GPU busy; queue follow-ups before a batch drains.
- Label every number as a success rate or an offline metric.
- Smoke-test every new path before queuing long jobs.
- Keep scope exactly as asked (2 tasks, 2 cycles, single-task cycles).
- Never match processes with `pgrep -f` on a name that your own shell's command line contains: it killed the pause
  script's own shell. Anchor on the process's own command and exclude the caller's process tree.
- Check external model access before long runs. Codex access failed for four reasons: model unsupported, logout,
  server-side revocation and the workspace spend cap.
- Measure a resource to completion. A reading cut short by a crash (6.7 GB) was not the cost (10.1 GB).
- systemd queue jobs run `/usr/bin/python3`, which has no numpy, so call the conda python explicitly.

## Open questions

**System 1 beneath Astra (2026-10-12)**
- Where should the transferred grasp's correction come from: Astra seeing the fingertips relative to the wall, or a
  System 1 with a persistent object-relative action, a contact-geometry state and varied data (§21)?
- Is there a general geometric rule for transferring a grasp between cups? Height relative to the rim failed (§21).
- How much of the dex aiming error disappears with a precise object pose and a final-approach controller, without
  learning?
- Does Astra + System 1 need fewer Astra calls per success on a new task than Astra alone?

**LIBERO era** (LIBERO was removed on 2026-09-30, so these can no longer be run as specified)

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
| Astra exploration pilots | `astra_explore/` (harness, tests), `experiments/astra_explore/run_2026-10-09_*/REPORT.md` |
| System-1 / System-2 transfer study | `astra_explore/s1s2/` (orchestrator, ledger, platform adapters, report, analyze); run `experiments/s1s2_overnight_2026-10-10/OVERNIGHT_REPORT.md` |
| Astra dex-hand harness | `astra_explore/{codex_astra,isaac_env,run_dex,schema_dex,report_dex,landlock_exec}.py`; conda env `hongyi`, cwd `egovla_release/egovla` |
| System 1 pilots (§16–18) | `astra_explore/s1local/` (env, ops, search, runner; generic task specs, crosstask; `diagnose_grasps.py`, `grasp_audit.py`); results `experiments/s1local_*`, `experiments/s1_diagnosis_2026-10-11/`, `experiments/grasp_audit_2026-10-11/` |
| System 1 design (§19) | `research/system1_dexhand_design.md` |
| cup transfer (§20) | `astra_explore/cup_transfer/` (assets, env_cups, cup_task, system1, prompt_cup, run_cup, analyze_cup, grasp_check); run `experiments/cup_transfer_2026-10-12/` |
| bottleneck batch 1 (§21) | `astra_explore/cup_transfer/{batch1,batch1_report,costs}.py`; run `experiments/cup_batch1_2026-10-13/` (`REPORT.md`, `gates.json`, `pause_dense.sh`, `restart_dense.sh`) |
| website | `astra_explore/viz/` (`build_site.py`, `build_s1local.py`, `build_transfer.py`, `launcher.py`); `python3 launcher.py restart`; served on 127.0.0.1:8765 (tunnel `ssh -L 8765:127.0.0.1:8765 licho@130.207.121.116`) |
| dex queues | `experiments/dexqueue_2026-10-10/` (`runner_q.sh`, `jobs.txt`, `pilots/`); systemd user unit `dex-pilot-queue` |
| queue | `scripts/queue.py`, `experiments/queue/`, `systemctl --user status horsea-queue` |
