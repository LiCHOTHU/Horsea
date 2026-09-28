# Horsea from its own rollouts: learned interaction loss (notes, 2026-09-25)

Proposal (user): keep v_{θ,W} = v_θ + r_W (θ frozen, r_W = Horsea's centred correction on frozen
θ₀ decoder features). Replace the expert FM-write target with a learned, outcome-conditioned
velocity target u_φ(e, x_k, t_k, c), where e = (o, a_executed, o′, l) and (x_k, t_k) are denoising
states recorded when that action was generated. Meta-train φ, W₀ through the write
(W⁺ = W − η∇L_write) with a future expert-correction FM loss (fresh noise; corrections never
enter the writer). Optional: writes interleaved between denoising steps. Claim: a learned
interaction loss converts self-rollout outcomes into persistent updates of the
action-generation field.

## What the papers establish (checked against the texts in scratchpad/papers)

| paper | mechanism | caveats that matter for us |
|---|---|---|
| RoboTTT (2607.15275) | TTT-MLP after attention in each of 16 DiT layers; fixed KV-binding write ‖f_W(θ_K X_t) − θ_V X_t‖² on per-timestep tokens (registers, proprio, noised action tokens); θ_K, θ_V, W₀, η learned by the outer sequence-FM loss. §3.3 DAgger Distillation: all executed actions written, FM loss only on human corrections → +36% vs +13% standard DAgger (sequence models); robot actions as targets add nothing (57% = 57%). | Same task at train and test (Pup Go Car, 100 DAgger trajectories). W reset to W₀ every rollout (no cross-attempt persistence). Future work named: robotics-specific TTT objectives; RL. |
| SAVN (1812.00971) | Learned scalar loss: 2-layer temporal conv over last k=6 LSTM states + policies, ℓ2 of output. SGD (lr 1e-4) on the whole policy every 6 steps, ≤4 updates; outer A3C; second order. Unseen scenes: 33.0 → 40.9% success. | Reward used in meta-training. Adapted weights discarded after each episode. Hand-crafted losses already 38.1/39.5%: the learned part adds ~1–3 pts. Exploited signal: visible failure (collision = near-identical frames). |
| EPG (1802.04821) | Learned loss (temporal conv over last 512–1024 transitions + 32-unit bias memory) trains a 2×64 MLP policy from random init with Adam; outer loop ES on final return; PG term annealed out (Eq. 10). | Reward-free only when dynamics vary under a fixed objective or the goal is an input: "rewards … impossible to infer from observations … cannot be completely internalized" (§6); DirectionalHopper needs reward input (§4.2). Crashes beyond the meta-trained horizon (Fig. 12). |
| AD (2210.14215) | Causal transformer, next-action NLL on multi-episode windows of RL learning histories; in-context improvement, frozen weights. | Rewards always in context (tasks chosen so the goal is only inferable from reward); no reward-free result. Expert-only data (ED) does not improve. Needs 2–4 episodes of context and ~1k+ training tasks (≤18 tasks: none). |
| Zo3T (2509.06723) | LoRA + noisy latent optimized jointly in a fixed window (t=45→30 of 50) against the sample's own frame-1 features; adapters discarded. | Interleaving matters because the loss reads the current latent. LoRA-only never tested; removing LoRA mainly costs fidelity (FID +14.5, ObjMC +1.5). |

## Design implications

1. **History-only interleaving is a schedule.** If the step-k write ignores the current candidate
   x_n^k, W_n^0..W_n^K are fixed before denoising, and the carried memory equals "all writes
   first". Interleaving only chooses which partial W each step uses. A real test of adaptation
   during denoising needs candidate-dependent writes (probes nearest x_n^k in θ₀ feature space,
   or x_n^k as a u_φ input). Plan: 3a history-only (expect a tie) vs 3b candidate-dependent.
2. **u_φ shortcut.** With (o, l, x, t) as inputs, u_φ can act as a second policy and ignore the
   experience. Controls: mismatched history, an experience-ablated writer. Give u_φ inputs only
   experience provides (surprise o′ − f̂(o, a) from a forward model, cf. TopoCut world-model
   encoder). Parameterize as a corrected endpoint â with u = (â − (1−σ)x)/(1 − (1−σ)t). Cap write
   size; warm-start from a hand-designed target (EPG Eq. 10).
3. **Information source.** Self-rollouts help only when outcomes reveal something (o, l) do not:
   visible failure (SAVN), dynamics under a fixed goal (EPG), or reward (EPG/AD); RoboTTT uses
   human corrections on the same task. A reward-free failed attempt at a new LIBERO task reveals
   arm response, object layout and missed grasps, not the goal.
4. **Our code.** Rollouts use temporal aggregation (16-step chunk generated every env step; the
   executed action averages 16 chunks): there is no clean executed prefix. Use receding-horizon
   execution (generate 16, execute 8) for self-rollout data. Budget writes per new experience, not
   per denoising step per call (≈3000 writes/episode otherwise). Naive self-write (own predicted
   chunk as FM target, `horsea/history.py`, LIBERO-10): Horsea 2% vs plain 36.5%, TTT 20%.
5. **Relevance.** Memory-ON takes over unrelated tasks (old tasks 4–38%, cycles_full). The outer
   loss needs old-task queries with the adapted memory active.

## Candidate testbeds
- **A. Hidden embodiment shift** on the 80 training tasks (per-episode action gain / xy
  rotation / bias / weak gripper; real instruction). Outcome reveals the shift (commanded Δ vs
  observed Δeef); exact DAgger corrections g⁻¹(a_θ); no reward. Test: held-out shifts; success per
  attempt 1→5; mismatched-shift history should hurt.
- **B. Retries on held-out tasks after a few demos** (current lifecycle). Meta-train corrections
  from θ with the real instruction on the training tasks; the writer must read the memory's own
  prediction; a per-attempt success flag is probably needed (AD, EPG).

## 2026-09-27 — implemented energy-Horsea objective (audit) and the objective-first comparison

Hidden-shift energy-Horsea (`horsea/energy.py`), as implemented:
- History = one whole attempt of the frozen base theta_0 under a hidden shift that is fixed within the
  sequence. Queries = decisions of another attempt with the same shift. No within-attempt writes.
- Experience = (context features, executed 8-step prefix, observed proprio change). The executed action
  is input/anchor only, never a target.
- Write: W <- W - eta grad sum_cand (E_W(cand,c) - y_phi(e,cand))^2, 2 inner steps. Candidates = executed
  chunk, 6 xy-rotations, gripper flip, 2 noisy copies.
- Outer target: g^-1(base sample from the same noise), i.e. the corrective action at the visited state
  from the privileged controller, used only in the outer loss. Loss: d_Phi or per-group normalized MSE.
- Gradients flow through the unrolled final-action solver (3 bounded steps), the differentiable write,
  the writer, the reader, W0, the inner lrs, lambda and beta. Backbone and Phi are frozen. No success
  flags anywhere.
- LIBERO-10 energy (`history2.py` mode energy) does NOT follow this objective. It uses a contrastive
  demo-vs-own-sample loss with no corrective targets, and its writes hurt (4% vs 16.7%).

Diagnostic (rotation-only model, held-out +-30/50 deg): with correct history, the corrective target
beats the base action in 95% of queries (Spearman 0.75); mismatched history gives 27% (Spearman -0.25).
Refinement moves actions only 0.014 on average and closes 22% of the distance, with no increase in
action-limit hits. So the energy is informative and the step budget limits it.

Queued comparison (rotations; gripper separately later), 3 seeds, frozen backbone:
- base;
- no-history static energy with the same corrective supervision;
- self-imitation target;
- proposed, with correct / shuffled / mismatched / last-only / no-write history;
- TTT2 trained on the same data and corrective targets, with correct / mismatched history.
Also an offline solver step-budget sweep.

## 2026-09-28 — staged plan (user) and implementation facts
Stages, in order: (1) 5-seed step-0.2 controls; (2) writer comparison (current / sequential / gated),
with memory-off, reorder, shuffled, last-only and mismatched history; (3) diagnostics: ranking, solver,
gripper threshold, rotation estimation; (4) freeze the configuration and confirm on fresh rotations,
seeds and starts; (5) persistent-shift consolidation, then a 2nd cycle; then LIBERO-10.
- The "0.2" step bound is per element, per iteration, in normalized action space ([-1,1] min-max):
  step = 0.2 tanh(beta g / 0.2). Total displacement is at most 0.6 per element after 3 iterations.
  It was changed at INFERENCE only; models were trained with 0.05.
- The gated writer is a gradient step on the squared prediction error (E_W - y)^2, i.e. a nonlinear
  delta rule, with decay W0 + alpha (W - W0) and write strength beta. It is not literally linear
  Gated DeltaNet.
- Checkpoint loading is strict except for gate weights, and only for batch-writer checkpoints.
- Unit of analysis: per seed, 50 adaptation sequences (10 tasks x 5 conditions) x 5 attempts.
  Uncertainty is bootstrapped over tasks, within seeds.
