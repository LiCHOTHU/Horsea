# Horsea memory study on LIBERO

Questions (from the proposal discussion, 2026-09-23):
1. **Adaptation speed.** Which fast memory adapts a frozen flow-matching policy to a *novel* task fastest (success vs number of demos)?
2. **Fast → slow.** How well does each memory's adapted behaviour transfer into the slow policy, with retention?
3. **Novel tasks.** How does it all perform on novel LIBERO tasks (held-out LIBERO-90; far-novel LIBERO-10)?

Fixed before any result was seen.

## Base policy
Your `fm_policy_S` (`imitation` repo: DiT velocity net, 4+4 blocks, width 256, 10 Euler steps, 18.27M params), trained with the **July recipe** (50 epochs, same Hydra overrides) on the **80 training tasks** of LIBERO-90. The ResNet and DiT observation encoders stay frozen everywhere afterwards. Only the denoising decoder (slow weights) or memory modules ever change.

The decoder sees the observation only through per-layer means of the encoder tokens (`_DiTDecoder`/`_FinalLayer` use `torch.mean(cond, axis=0)`). So each frame is summarised as `encm ∈ R^{4×256}` once (`horsea/features.py`), and all meta-training and distillation runs on GPU-resident summaries. `tests/test_cpu.py` checks that this reproduces `forward_dec` and `sample_actions` exactly (max error 0.0).

## Splits (`horsea/paths.py`)
- **Train:** 80 LIBERO-90 tasks.
- **Held-out novel (LIBERO-90):** `[2, 9, 13, 25, 43, 48, 57, 66, 79, 84]`. There is one per scene, each in a scene that keeps ≥3 training tasks, so each is a new instruction in a familiar scene. Task 51 (butter→basket, 0/10 for every policy in July) is avoided.
- **Far-novel:** all 10 LIBERO-10 tasks (long-horizon compositions; horizon 520).
- **Retention subset:** 20 training tasks (`TRAIN_90[::4]`).

## Memory arms (`horsea/memory.py`)
All four share one skeleton, so the comparison isolates *where memory acts* and *what the write objective is*:
- batched fast weights;
- meta-learned `W0` and per-tensor inner learning rates, each capped at 3× a calibrated stable value;
- one inner gradient step per demonstration;
- second-order meta-training;
- 15.6–18.4k fast parameters per episode.

| arm | where it acts | write objective | centred (empty memory = base exactly) |
|---|---|---|---|
| `ttt` internal TTT (RoboTTT-style) | TTT-MLP after each of the 4 DiT decoder blocks, tanh-gated | KV binding `‖f(θ_K x) − θ_V x‖²` on block tokens (noised support actions) | no (as published) |
| `kv` proposal | external associative MLP `tanh(qA+b)B+d0` → velocity reader at every solver step | cue→content association (content = 4 lowest DCT bands of the action chunk) | yes |
| `fmw` FM-write | correction field on the frozen decoder's final hidden tokens | the flow-matching loss itself on the demo's actions | yes |
| `res` fast residual | the finished action chunk, once | action regression onto the demo chunk | yes |

The inner learning rates were calibrated on CPU by sweeping step size over 10 consecutive writes. The first KV calibration (lr 1.0) diverged after 3 writes; KV is stable ≤0.10 and diverges ≥0.20, so its initial rate is 0.03. Keys, queries and values that feed the fast nets are LayerNorm-ed, so meta-learning cannot enlarge the effective step.

## Baselines
- **Zero-shot base** (K = 0).
- **`ft` direct fine-tune:** the decoder is trained on all frames of the same K demos (500 steps, batch 128, lr 5e-5, dropout on; untuned). This is the no-memory answer to "adapt to a new task".
- **Wrong-memory control:** each arm at K = 5 is written with a *different* novel task's demos. This tests whether the gain comes from memory content or just from any write.

## Protocols
- **Meta-training** (`horsea/meta_train.py`):
  - 20000 steps (raised from 8000 after measuring 0.02–0.22 s/step on the 5090); 16 training-task episodes per step.
  - K ∈ {1,2,3,5,8,10} support demos, 16 evenly spaced frames per write.
  - 32 query frames from 2 other demos of the same task; the query target is never in its own support.
  - Every 1000 steps, held-out query loss at K ∈ {0,1,5,10} is logged. It is monitoring only; there is no model selection (the final step is used).
- **Adaptation** (`horsea/adapt_eval.py`):
  - K ∈ {1,2,5,10} (TTT also K = 0), demos 0..K−1.
  - Each write uses all frames of one demo.
  - 20 closed-loop rollouts per (task, K) on init states 0–19, with the same env wrapper, temporal aggregation and horizons as the July evaluation.
- **Consolidation** (`horsea/consolidate.py`), sequential over the 10 held-out tasks with K = 5:
  1. Write the memory on the *current* slow policy.
  2. Run 20 teacher rollouts from init states 25–44 (disjoint from evaluation) and record every visited state.
  3. Endpoint FM distillation into the decoder (proposal eq. 22: teacher actions drawn fresh every step, fresh source noise), 3000 steps, mixed 1:1 with replay of old demos and of earlier consolidated tasks.
  4. Reset the memory and evaluate the memory-off student on init states 0–19.
  5. At the end, evaluate all 10 novel tasks (forgetting), the 20-task retention subset, and LIBERO-10 zero-shot.
  6. Report the transfer ratio `(student − base) / (teacher − base)`.
  - `ft` consolidates the 5 demos directly (positive control).
- **Statistics:** 95% bootstrap intervals over tasks. There is one meta-training seed (a known limitation).

## Running
```
tmux new-session -d -s horsea-exp 'bash scripts/run_pipeline.sh'   # launched this way (session 'horsea' is yours)
tail -f experiments/pipeline.log                              # orchestrator
cat experiments/pipeline_status.json                          # finished / running / failed
ls experiments/logs/                                          # one log per job
python -m horsea.report                                       # tables + plots -> experiments/report/
```
The pipeline waits for CUDA, then runs:
1. base (≈3.5 h);
2. features;
3. the 4 meta-trainings in parallel;
4. rollouts (≤3 jobs × 5 envs).

Every job is resumable and is restarted if it dies. `tests/test_cpu.py` and `tests/smoke_rollout.py` are the invariant and plumbing checks.

## Known limitations / threats
- One base, one meta-training seed.
- The `ft` recipe is untuned; so are the memory hyperparameters beyond stability calibration.
- LIBERO demos are unimodal per task (noted in July), so multimodality benefits cannot show here.
- Consolidation uses the teacher's own on-policy states. Teacher success on inits 25–44 is compared with student success on inits 0–19.
- LIBERO tasks are identified by their instruction, so consolidation into a memory-free policy is information-theoretically possible. No episodic-binding test is included.

## 2026-09-24 04:30 — protocol fix: generic-instruction (RoboTTT one-shot) setting
**Finding.** Under the original protocol the memories were meta-trained on the 80 tasks the base already solves from their instruction, so meta-training had almost no signal:
- The training-task query loss was flat in K.
- TTT's gate reached only 0.026, and its inner learning rate never moved from 0.03.
- On held-out tasks, 10 demos changed the actions by only 2.6% (TTT) and 3.0% (KV), against 24% for FM-write and residual.
- TTT closed-loop success was 0/40 at every K (first 2 held-out tasks).

RoboTTT avoids this by giving every task configuration the same prompt ("assemble circuit"), so the in-context demo is the only task signal.

**Fix.**
- Features are re-encoded with the single instruction "complete the task" (`*_generic.pt`).
- All memory arms are re-meta-trained on them (`experiments/memory_generic/`).
- Evaluation uses the same generic instruction for base, fine-tune and memories (`adapt_eval --generic`, results in `experiments/adapt_generic/`).
- TTT is replaced by `ttt2`, aligned with the reference code (lucidrains/robo_ttt): RMSNorm before qkv, plus a learned per-episode write gate.
- The original-protocol results are kept; they document the failure mode.
- Rehearsal check: after 40 meta-steps under the generic protocol, the training-task query loss already falls with K (0.135 → 0.054 at K=10). It was flat before.

## 2026-09-24 14:15: user request: larger K, and fast→slow transfer with the fixed memories
- **`gK_*`:** fine-tune plus the four generic-protocol memories at K = 20 and 40 demos, on the held-out LIBERO-90 tasks. Memories were meta-trained with ≤10 writes, so K = 20/40 also tests write extrapolation. Fine-tune keeps its 500-step recipe.
- **`gcons_*`:** sequential consolidation with the generic-protocol memories as teachers.
  - The memory is written from K = 5 demos under the generic instruction, and the teacher acts under that instruction.
  - Every visited state is recorded under both encodings, row-aligned. The memory-free student learns the teacher's actions on the **real-instruction** encoding of the same states.
  - Reason: after the reset, the slow policy must tell the 10 new tasks apart, and under a single generic instruction it cannot.
  - Replay, steps and evaluation are unchanged. Outputs go to `experiments/consolidate_generic/`.
- **Throughput fix:** jobs now run with `OMP/MKL/OPENBLAS_NUM_THREADS=2`. Load was 36 on 16 cores with the GPU at 15%.

## 2026-09-24 14:35: RoboTTT-style within-episode memory (self-rollout history), LIBERO-10
Mirrors RoboTTT's main evaluation (`horsea/history.py`). Both variants post-train the base on the LIBERO-10 demos with the same steps and data:
- **`plain`:** decoder only. This is the single-step-context baseline.
- **`ttt`:** decoder plus `ttt2` TTT layers, trained jointly on whole demo sequences.
  - One timestep = 8 env steps, i.e. one executed chunk.
  - At each timestep, the FM loss uses the fast weights written by all earlier timesteps; then that timestep's noised action tokens are written.
  - Truncated BPTT every 16 timesteps. 3000 steps, 16 sequences per step.

Test: 10 LIBERO-10 tasks × 20 rollouts. Each episode starts from W₀. The robot acts with the current fast weights, then every 8 steps writes its **own** predicted chunk and observation. No expert data at test time.

Training starts immediately; evaluation is scheduled after the larger-K (`gK_*`) jobs.

## 2026-09-25 00:50 — 2-cycle proof of concept of the short → long lifecycle (`horsea/cycles.py`, `experiments/cycles/`)
**Cycle-2 failure diagnosed.** After 2 consolidations, the velocity field on old tasks had moved 10× more under the generic instruction (4.0%) than under the real one (0.4%). Replay protects only the real-instruction channel, but the short memory learns through the generic channel.

**Fixes (identical for every memory type):**
1. Explore until full: write demos in chunks of 5 (up to 20) while held-out gain improves, and roll back the chunk that stops helping.
2. Convert fully before reset: memory-off success must reach 0.9× short-memory success, otherwise distill another round (up to 3).
3. Interface anchor during consolidation: keep the generic-instruction velocity and every decoder layer's tokens equal to θ₀'s.

**Protocol.** Cycle 1 = task 57, cycle 2 = task 66; the base scores 0% on both. 20 rollouts per task, 10 per old task.

| arm | c1 short mem (on θ₀) | c1 mem-off after reset | c2 short mem on θ₁ (same mem on θ₀) | c2 mem-off after reset | c2 previous task (57) | old tasks after c2 |
|---|---|---|---|---|---|---|
| Horsea (fmw) | 40% (45%) | 45% (2 rounds) | 25% (25%) | 40% | 65% | 94% |
| residual | 65% (65%) | 65% (2 rounds) | 30% (35%) | 35% | 40% | 98% |
| TTT2 | 65% (60%) | 55% | 0% (15%) | 5% | 70% | 98% |
| KV | 15% (15%) | 25% | 5% (5%) | 10% | 45% | 100% |
| fine-tune (20 demos → θ directly) | – | 100% | – | 85% | 100% | 98% |

## 2-task, 2-cycle test with every cell measured (2026-09-25, `experiments/cycles_full/`)

Fresh run of the 2-task test (cycle 1: task 57, cycle 2: task 66, one task per cycle), all five
algorithms. Every step measures task 57, task 66 and the 5 old tasks (memory ON rows run θ + W on
every task). The retention gate is active: previous task within 10 pts, else doubled replay, up to
3 rounds. Every intermediate θ and W is saved. Full tables: `experiments/cycles_full/REPORT.md`
(`python -m horsea.cycles_report`).

| method | after cycle 1 (57 / 66 / old) | after cycle 2 (57 / 66 / old) | notes |
|---|---|---|---|
| Horsea | 65 / 0 / 100 | 65 / 20 / 98 | c2 round 1 dropped 57 to 50 → 2× replay → back to 65 |
| Residual | 55 / 0 / 98 | 55 / 45 / 100 | c1 stayed under the gate (55 vs 59) after 3 rounds |
| TTT2 | 70 / 0 / 96 | 65 / 10 / 98 | |
| KV | 45 / 0 / 98 | 25 / 10 / 100 | c2 forgot 57 (45 → 25) even with 4× replay |
| Fine-tune | 95 / 0 / 98 | 100 / 75 / 92 | |

New finding: while a short memory is ON, it takes over every task. Old tasks fall to 4–38% and
the previous task to 0%. After consolidation and reset, they recover to 96–100%. The short memory
has no task gate, so it must only be on for its own task.
