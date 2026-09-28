# Deep understanding: "Fast and slow memory for flow-matching robot policies" (v2, 21 Sep 2026)

*How this was built.* I used the deep readings of all 18 references together with their adversarial verifications. Where a verifier corrected a reader, the verifier's version is used. From the five gap searches I kept only papers whose gap-verification returned `exists=true`, and I report the verifier's threat level for each. Figures marked ≈ were read off plots by a reader or verifier. Section and equation numbers refer to the proposal PDF.

**One correction to the brief.** The brief renders Eq. (13) as `m^{l+1} = m^l − α∇E_W(m^l; q)`. The PDF (p.10) actually says `m^{s+1} = m^s − γ∇_m[½‖m − q_t‖² + E_{W_k}(m)]` with `m^0 = q_t`. The anchor term `½‖m − q_t‖²` that the brief drops is important (see §4.2).

---

## 0. Bottom line

1. **The mechanism-level ingredients are almost all published.** Four concrete precedents:
   - **Persistent gradient-written fast weights read at every denoising step of an FM robot policy:** RoboTTT [6], WAM-TTT [5], DexWorldModel.
   - **A W₀-style centred velocity residual on a frozen FM VLA:** Proxy Policy Steering (PPS, 2609.09148). This is the only gap paper every lens rated high.
   - **A meta-trained gradient writer:** TTT layers [3]; Metalearned Neural Memory; WAM-TTT/RoboTTT.
   - **A "plastic learner → distil with protection → re-initialise" lifecycle:** Progress & Compress [16] (sleep analogy included); Reset & Distill; S&B's 2026 follow-up.

   What no verified paper combines, and what is therefore still new, is: a small, persistent, causally written, meta-trained external fast memory, read through a W₀-centred velocity residual at every solver step, consolidated into the FM backbone with source-law-preserving FM objectives, and reset only behind memory-off / retention / plasticity / episodic-exemption gates. The controlled comparison of storage × read location × transfer objective under matched budgets is also new.

2. **The retention bound (Eqs. 27–28) is a known lemma, and it is vacuous in the proposal's own multimodal regime.**
   - Gao–Huang–Jiao (JMLR 2024), Prop. 39(ii), states `W₂² ≤ (e^{2C₃}−1)/(2C₃) ∫∫‖v−ṽ‖²p_t` with ṽ C₃-Lipschitz. This is the same constant as the proposal's C_L. Albergo–Vanden-Eijnden Prop. 3, Benton et al. Thm 1, ORW-CFM-W2 Thm 3 and VGG-Flow Prop. 4 are looser versions of the same inequality.
   - For well-separated sharp modes the FM field's Jacobian at the mode boundary reaches ≈112 (σ = 0.15) or ≈361 (σ = 0.1). C_L then becomes astronomically large, and a one-sided constant does not rescue it (verified numerically by [1] and [2] verifiers).

3. **App. A's path-distillation theory also exists already.**
   - Marginal preservation for E[Ẋ_t | X_t] of any rectifiable process is rectified flow's Thm 3.3.
   - The counterexample in Eq. (33) is Hertrich–Chambolle–Delon Prop. 13, and also [13] Sec. 4.2 with γ = 0.
   - Two standard fixes exist and are absent from the proposal: latent smoothing γ(t)z, and small noise on the seeds.
   - The endpoint-vs-path experiment confounds coupling with path shape unless a same-seed (reflow) arm is added.

4. **The value/control reader (Sec. 10) is essentially VGG-Flow (NeurIPS 2025).** Adjoint Matching and Uehara et al. give the known reason it cannot realise p_base·exp(−Φ) with deterministic sampling. They also give the fixes: a memoryless SDE, or a learned source distribution. Sec. 9's Eq. (15) is HardFlow's Problem 2 and a discretised OC-Flow.

5. **Several baselines are missing and at least one could plausibly win in Stage B/E.** The strongest competitors are:
   - RoboTTT as published;
   - a PPS-style proxy residual trained on the same k experiences;
   - Gated Memory Policy (gated history cross-attention at every denoising step; its MemMimic "Match Color" task is essentially Experiment B);
   - capped retrieval guidance in velocity space (TraceFlow);
   - experience replay at matched bytes on a pretrained FM VLA, which already gives near-zero forgetting at a 20% buffer (2603.03818).

6. **Stage B as specified cannot show the benefit it is meant to show.** With at most 16 experiences, an explicit KV store (16 × 32 = 512 floats) is smaller than the 1,072-parameter fast net and retrieves exactly. TTT [3] reports that MLP hidden states are "redundant in short context". Stage B also lacks the partial-cue probe that the proposal's own biological motivation ([8]) calls for.

7. **The proposal's own maths is mostly sound**, and its hedges are mostly accurate:
   - Eqs. (11), (16), (19)–(20) and (29)–(30) check out.
   - The problems are attribution (Secs. 5, 7, 9, 10, 13, App. A), a few misattributed citations ([13], [14], [15]), missing baselines, and specific technical gaps (§4).

---

## 1. The problem, precisely

The object is π(a_k | c_k, W_k), with W_k = Write(H_k), inside a lifecycle that runs on three clocks: decisions k, generative time t, and consolidation cycles j. It decomposes into eight coupled sub-problems.

### 1.1 Storage: what holds recent experience?

**Proposal.** A persistent 16-32-16 tanh MLP W_k with 1,072 parameters (Eq. 6), plus a finite episodic store B.

**Why it is hard.**
- There is a capacity / interference / exactness triangle.
- In the small-k regime, explicit storage is cheaper *and* exact: 16 experiences × (16-d key + 16-d value) = 512 floats, fewer than 1,072 ([12] reading; verifier concurred).
- Gradient-written memories are close to attention formally:
  - A linear layer trained by gradient descent is exactly unnormalised attention over its training patterns (Irie et al. 2022, "dual form").
  - [4] Thm 5.1–5.2 rewrites the proposal's own net *exactly* into the form o = φ_{k}(q)(S₀ + Σφ_i(q_{e_i})ᵀg_i). The reader checked this to 4.4e-16 after folding d₀ in as a constant feature.
  - That form reduces to linear-attention / DeltaNet cross-attention **only if just (B, d₀) are written**. With A and b also written, the kernel is dynamic and "non-reducible" ([4] App. I, verifier correction).

**What prior work covers.** Fast-weight layers (TTT [3], DeltaNet/Gated DeltaNet, Titans, LaCT); append-only KV / modern Hopfield ([12]); nonparametric experience banks read inside FM samplers (TraceFlow, Retrieve-then-Steer); gated history attention in diffusion policies (Gated Memory Policy).

**What remains open.** Which storage wins per byte and per FLOP at k ≤ 16. The evidence conflicts:
- [3] §3.2: MLP state is "redundant in short context".
- RoboTTT Fig. 12: TTT-Linear is ≈27% worse than TTT-MLP on multi-minute robot episodes.
- One-Minute Video: Gated DeltaNet is best on 18 s clips and TTT-MLP on 63 s clips (both are ~100k-token contexts, so this is weak evidence for k ≤ 16).

The proposal's "nonlinear MLP" choice is therefore untested in its own regime.

### 1.2 Write rule: how experience becomes weights

**Proposal.** One SGD step per experience on l_assoc = ½‖f_W(q_e) − sg(y_e)‖² (Eqs. 7–8).

**Why it is hard.**
- One step per record means η_f calibration and write order matter.
- Deployment will go past the 1–16 writes seen in meta-training.
- MSE writes pull repeated cues toward the **conditional mean** of contents ([12] reading). That blends multimodal content, which is exactly what the "multimodal target family" is meant to stress.
- It is also not clear what a trained KV-binding writer actually stores. [4] reports a "memorization paradox": gradient ascent works as well as descent, queries can be replaced by keys, and more inner steps hurt. That evidence comes from Frobenius (dot-product) writers; under Frobenius loss, sign flips are absorbed exactly. The verifier showed that for the proposal's MSE writer the sign-flip test is informative, not trivially passed.

**What prior work covers.**
- [3]: learned views θ_K/θ_V/θ_Q, learned W₀ and a learned rate η(x) (−0.36 ppl), plus LN+residual (−1.22 ppl).
- Titans: momentum plus data-dependent decay. Table 5: removing weight decay drops the score from 92.68 to 85.60.
- LaCT: Muon beats momentum.
- Elastic TTT: Fisher-weighted anchors; an EMA anchor beats a global anchor in streaming.
- Miras: retention as a regulariser.
- Metalearned Neural Memory: an external MLP written by one gradient step.
- WIZARD: a hypernetwork writer for π0.5.
- ML3 [17]: learned loss.

**What remains open.**
1. **Write-count and rate extrapolation.** Three results point to fragility:
   - [4] Fig. 1: more inner steps at inference degrade NVS PSNR from ≈26 to ≈17–18 dB.
   - WAM-TTT uses inner lr 0.1 in meta-training but 0.01 at test.
   - RoboTTT: with pretraining contexts of 128/256 the model falls *below* the single-step baseline (30.8/29.0 vs 37.6).
2. **Where decay should point.** Under W₀-centring, any decay must be toward W₀, not 0. Titans and GDN decay toward 0, and Elastic TTT found a global (fixed-init) anchor worse than an EMA anchor.
3. **Whether association storage is what is used at all.**

### 1.3 Read interface inside the denoising loop

**Proposal.** A query built from the noisy candidate z_t (Eq. 3). The centred residual r = R(m_{W_k}) − R(m_{W₀}) is added to v_θ at every Euler step (Eqs. 4, 5, 9).

**Why it is hard.**
- **Sensitivity is extreme near mode boundaries.**
  - For a ±1 bimodal target with σ_min = 0, the exact FM marginal Jacobian at x = 0 is 44 at t = 0.75, 890 at t = 0.9 and 3824 at t = 15/16 ([2] reading, verified).
  - Small corrections there flip modes.
  - An unbounded additive per-step term can destabilise a sampler with few steps. TraceFlow's uncapped variant scores 24.25 task success vs 57.75 capped and 56.75 for the base.
- **Memory and backbone misfit are confounded.** The Bayes-optimal regression target of Eq. (12) splits into two parts ([1] reading; verifier-corrected to the density-free form):
  - r* = a_t[∇log p_t(x|c,W) − ∇log p_t(x|c)] + [u(x|c) − v_θ],
  - which equals (E[a|x,c,W] − E[a|x,c])/(1−t) + misfit, with a_t = (1−t)/t for CondOT.
  - The second term does not depend on history content. A shuffled- or wrong-history control is only diagnostic if it uses **the same number of writes**.
- **Timing matters in a specific way.** At t = 0 the ideal residual is a constant mean shift. Pushed through the nonlinear base flow, it re-weights modes rather than translating them: an early-only shift moved P(+mode) from 0.499 to 0.857 ([2] verifier simulation). Early-only access can therefore re-weight modes but not reshape them.

**What prior work covers.**

| Read location | Examples |
|---|---|
| Internal, per step | RoboTTT, WAM-TTT, DexWorldModel, LaCT, One-Minute Video |
| External centred velocity residual | PPS |
| Capped nonparametric velocity guidance (early steps) | TraceFlow |
| Intermediate-state prior | Retrieve-then-Steer |
| Per-step gated cross-attention to history | Gated Memory Policy |
| Gated LoRA velocity residual | FlowCorrect |
| Guidance theory | Feng et al. (ICML 2025); CFG / classifier guidance, [1] §5.2 |

**What remains open.**
- Whether a candidate-dependent read at every step beats reading once per decision (SmoLSTM, 77.5% on LIBERO-Mem), reading only at t₀, or a final-action residual, at matched reader evaluations.
- Whether an external reader matches internal TTT.
- Whether r needs a norm bound.
- Whether the reader under-uses memory, the analogue of the underfitting of vanilla guidance ([1] p.35). A memory-guidance scale w would test this.

### 1.4 Meta-training the interfaces

**Proposal.** CFM regression (Eqs. 11–12) through earlier gradient writes, second order, with θ frozen.

**Why it is hard.**
- Second-order cost, and the choice of truncation horizon.
- Training on sequences whose actions are also written needs per-chunk independent noise levels. RoboTTT without "sequence action forcing" scores ≈0.02.
- **Identification** (verifier-corrected). With a full-rank Gaussian source, p_t has full support for every t < 1, so identification in x is global. The real gap is coverage of **(c, W_k)**: memory states produced by unseen histories.
- The CFM→FM gradient equivalence for (φ, W₀) holds only if the target is the **full-history** field ū_H = E[a*−ε | x_t, t, c, H_k] ([1], [2] readings). Then L_meta = E‖v_θ + r − ū_H‖² + const. The constant is the irreducible E Var[a*−ε | x, t, c, H], and it is independent of the learned parameters.
  - Raw L_meta values are therefore not comparable across readers or history lengths unless reported relative to that floor.
  - The information-loss term E‖E[ū_H | x,c,W] − ū_H‖² measures what the memory discards.
  - The no-leakage rule is mathematically necessary here, not only good practice.

**What prior work covers.** MAML-style outer loops; [3]'s "gradients of gradients"; WAM-TTT; RoboTTT (loss masking for pure-context writes; TBPTT with W₀ gradient only through the first segment); OML; OSAKA.

**What remains open.**
- Whether freezing θ caps what a small external reader can do. WAM-TTT trains Θ_WAM jointly. RoboTTT pretrains only the new layers with GR00T frozen and then fine-tunes everything. The WAM-TTT "VLM frozen" ablation does *not* bear on this (verifier). There is no controlled evidence either way.
- Whether curated history-dependent episodes scale. [3] §4.3 flags meta-learning over curated task collections as "hard to scale".

### 1.5 Consolidation and transfer (fast → slow)

**Proposal.** A frozen teacher (backbone plus memory) is distilled into a memory-free student, either by endpoint FM with fresh noise (Eqs. 21–22) or by piecewise-linear path distillation (Eqs. 23–24). Success-filtered noise is handled explicitly.

**Why it is hard.**
- The student lacks the history. Averaging over histories under a shared context is unavoidable (App. B.4).
- The path target b* can be singular.
- The teacher-context sampler defines what gets consolidated.
- On-policy vs off-policy targets matter (SDFT).

**Theory status.**
- For a single smooth teacher ODE with fixed (c, W), trajectories do not cross ([1] Theorem 3, verifier point). So b* equals the teacher field on its support, and path distillation reproduces same-seed trajectories.
- Velocity averaging arises only from:
  - mixing memory states or histories under a shared student context; or
  - Euler maps that are not injective (h·Lip(v_θ + r) ≥ 1).
- A stochastic teacher gives yet another target. One Euler–Maruyama step yields b*(z,t) = 2εt·z/(1+2εt²), which vanishes at the knot, so knot-grid supervision under-disperses ([13] reading).

**What prior work covers.**
- P&C [16]; Reset & Distill; PLD; ReGuide; PA-RL (distil improved or steered samples).
- SDFT (a context-conditioned teacher distilled into a context-free student, on-policy).
- Rectified flow Thm 3.3 and reflow; [13] Thm 6/47.
- Masip et al. (generative distillation for diffusion continual learning).
- S&B's 2026 follow-up (LoRA consolidation of memory-conditioned reconstructions, memory-off test).

**What remains open.**
- Whether preserving intermediate evolution helps beyond preserving the teacher's noise→action **coupling**. This needs a reflow arm.
- Whether latent smoothing makes path distillation workable.
- Whether few-shot teachers are strong enough. P&C's teacher is a full column trained to convergence; this proposal's teacher comes from ≤16 one-step writes.

### 1.6 Retention of old behaviour

**Proposal.** R_old (Eq. 25) is the velocity MSE to the old field along old-policy ODE trajectories. It is added as β·R_old in Eq. (26), and the W₂ bound (27)–(28) is attached.

**Why it is hard.**
- **The forgetting regime depends on setting.** On pretrained FM VLAs with plain ER:
  - at a 20% buffer, NBT is −0.016 for π0 and 0.027 for GR00T (2603.03818);
  - at a 2% buffer it is 0.1–0.2;
  - Memory Anchors reports π0.5 random-ER NBT of 0.12 on LIBERO-Goal.
- **A naive velocity-MSE penalty has already failed once.** The same penalty against the pretrained model, evaluated on target-task data, collapsed in ConSFT (LIBERO-Goal 0.00).
- **R_old is scale-heterogeneous.** For Gaussian targets ∫κ_t² dt = π/(4σ). So the same W₂ change costs 1/σ more for near-deterministic tasks ([16] reading; consistent with [2]'s 5.236 at σ = 0.15). This is the analogue of P&C's Fisher-normalisation problem.
- **The anchor drifts.** The anchor is re-centred on the latest snapshot, as in online EWC (P&C/Huszár: "older tasks will be remembered less well").

**What prior work covers.** EWC / online EWC; ER; generative distillation (Masip); ConSFT; AnchorER (conflict-region replay selection). The bound itself is known (§3).

**What remains open.** Whether R_old beats ER at matched bytes and budgets on a pretrained FM backbone; per-task normalisation; per-task anchors (old teacher endpoints) vs the re-centred anchor across many cycles.

### 1.7 Refresh and renewed plasticity

**Proposal.** Reset W → W₀ after four gates; meta-train the memory over a mixture of accepted backbone checkpoints.

**Why it is hard.**
- A W kept across a θ update is keyed to stale features, because the query depends on h_θ. S&B 2026 makes the same point: "the stored gist … must be 'in sync' with the neocortical model".
- A writer calibrated on one checkpoint can drift on the next.
- Reset helps diverse task sequences but hurts similar ones. P&C Table 1: 127–129% transfer with re-init vs a decline to 101% without; carry-over helped very similar tasks.
- The slow networks themselves can lose plasticity over cycles.

**What prior work covers.** P&C re-init; Reset & Distill; Nested Learning's CMS (fast levels re-initialised to a meta-learned θ₀); plasticity injection.

**What remains open.** Whether rebuilding from raw or frozen-encoder records in B avoids drift; consolidation cadence (how many writes may intervene); whether the writer transfers across checkpoints (gate item 3).

### 1.8 Episodic vs reusable information

**Why it is hard.**
- Whether information can be consolidated is a property of the task distribution, not of a single record. An ordinary-looking instruction can carry a critical binding ([11] reading).
- **Full-context evaluation hides loss of cue completion.** In [8], removing CA3 recurrent plasticity left full-cue performance intact and degraded partial-cue recall (66.4% → 44.6% quadrant occupancy vs n.s. drops in controls). A memory-off student could pass gate 1 on full context while losing completion from partial cues.

**What prior work covers.**
- App. B.4.
- S&B's extended model: prediction-error-gated residual storage; the veridical DRM `id_n` tag.
- MbPA.
- TraceFlow, which "does not help where a task fails for lack of event information in the policy state".
- RoboTTT's gains appear to come mostly from episodic information, but this attribution is qualitative only (verifier).

**What remains open.** A decision procedure for gate item 4, such as record-level behavioural ablation. Also partial-cue probes in the gates.

---

## 2. Reference digest (18)

**[1] Holderrieth & Erives, MIT FM notes (PDF 18 Mar 2026; = arXiv:2506.02070 latest).**
- **What it is.** A tutorial deriving CFM, the marginalisation trick, CFM=FM up to a constant, CondOT (Alg. 3), score/denoiser conversions, SDE sampling with the same marginals, and guidance/CFG.
- **Bearing.**
  - It grounds Eq. (11), the Euler sampler and the conditional-mean identification.
  - Its classifier-guidance decomposition (eqs. 59–62) gives the direct reader a Bayes/MSE-optimal target: r* = a_t[∇log p_t(x|c,W) − ∇log p_t(x|c)] when v_θ is exact.
  - The centred residual is structurally CFG with W₀ as the null condition, which motivates a CFG baseline and a guidance-scale sweep.
- **Claim status: accurate.** This includes App. C's use of it (verifier override: [1] states the general-source recipe, App E p.84).
- **Corrections.**
  - Pin the version.
  - Write "ε ~ N(0,I) independent of (a*_k, c_k, H_k)".
  - Identification in x is global for t < 1; the gap is (c, W) coverage, not "off-support x" (verifier).
  - "No explicit optimality principle" (Sec. 8) should say "Bayes-optimal target, no control optimality". This is a framing fix, not an error.

**[2] Lipman et al., Flow Matching for Generative Modeling (ICLR 2023).**
- **What it is.** It introduces FM/CFM (Thms 1–3) and the OT conditional path with σ_min > 0, and explicitly disclaims that the marginal field is OT.
- **Bearing.**
  - Eqs. (11) and (21)–(22) are its eq. 23 in the σ_min → 0 limit. [1] supplies the exact σ_min = 0 form, so the joint citation "[1, 2]" is **accurate** (verifier override); a footnote on σ_min suffices.
  - Eqs. (29)–(30) are exactly its eq. 8 field for a Gaussian target (analytic identity).
  - Non-uniqueness of the generating field ("infinite number of vector fields generate any … path") has three consequences:
    - the Sec. 18 "integrated velocity correction" can be nonzero without any change in distribution;
    - R_old is sufficient but not necessary for retention;
    - endpoint and path students can reach the same marginals via different fields.
- **Status: accurate.** App. C's use of it for Gaussian-mixture sources is weak support, because all of [2]'s constructions start from N(0,I). Cite [13] instead.

**[3] Sun et al., Learning to (Learn at Test Time): TTT layers (ICML 2025, PMLR 267; arXiv:2407.04620).**
- **What it is.** The hidden state is a model W updated by a gradient step on a learned multi-view reconstruction ‖f(θ_K x; W) − θ_V x‖², read through a test view θ_Q x. θ_K, θ_V, θ_Q, W₀ and η(x) are learned in an outer loop through "gradients of gradients". Thm 1 shows the linear/batch-GD case is linear attention.
- **Bearing.** The proposal's writer (Eqs. 6–8 plus the Sec. 7 outer loop) *is* [3]'s framework, with these deviations:
  - nonlinear view encoders;
  - a frozen label view;
  - no LN/residual;
  - fixed η_f;
  - the write stream is experience records rather than processed tokens.

  Evidence: gains appear only at 8k–32k tokens. At short context TTT is on par with Mamba and the Transformer, and TTT-MLP is "redundant in short context".
- **Status.**
  - Sec. 21 row, Sec. 8 "strong established comparison" and Sec. 6: **accurate** (verifier overrides).
  - The **attribution gap is real**: Secs. 5 and 7 do not cite [3], although the proposal never claims the writer as novel.
  - Add the ICML 2025 venue.
  - Matched-parameter internal baselines: a single-head 32-d TTT-Linear, or 4 heads of 16×16 (1,024). **Not** 4 heads of 32×32, which is 4,096 (verifier).

**[4] Liu et al., TTT with KV Binding Is Secretly Linear Attention (ICML 2026, PMLR 306; arXiv:2602.21204).**
- **What it is.** Thm 5.1–5.3: any inner net with a linear, bias-free last layer trained by one GD step per token (plus momentum) has the form o = φ_{t+1}(q)(S₀ + Σφ_i(k_i)ᵀg_i). It also presents the "memorization paradox" experiments; the reduction ladder in which last-layer-only writes are best by tiny margins (+0.03 dB, +0.29 pp); and a parallel form for static kernels.
- **Bearing.**
  - The proposal's net fits the theorem exactly once d₀ is folded in (4.4e-16).
  - But with A and b also written, the kernel is dynamic, so the identity is a *rewrite*, not a reduction to linear attention (verifier).
  - With last-layer-only writes plus the MSE writer, the memory equals DeltaNet cross-attention with initial state S₀. That is a natural matched-state baseline.
  - The paradox evidence comes from Frobenius-loss writers on long within-sequence contexts, and transfers weakly to a few-shot, cross-stream MSE writer with fixed stop-gradient targets.
- **Status: accurate.** The Sec. 21 "distinction to test" column is a disclaimer, not a test (presentation issue). Add ICML 2026.

**[5] Feng et al., WAM-TTT (arXiv:2607.06988, preprint).**
- **What it is.** An **uncentred** TTT residual θ_O f_W(θ_Q z) is placed in the video expert of all 16 MMDiT blocks of LDA-1B.
  - One inner SGD step on human-video prediction plus a KV-reconstruction loss.
  - W_init, the slow Q/K/V/O **and Θ_WAM** are meta-trained through the inner update on the robot FM loss.
  - W_N is fixed during rollout.
- **Bearing.** This pre-empts the meta-training structure of Eq. (12) and the strict fast-only protocol. The distinctions it verifiably lacks are:
  - centring: the memory-off policy is not v_θ and is not invariant to interface training. Plain reset-to-W_init is trivially available, so reset alone is not the distinction;
  - a velocity-level readout;
  - causal multi-write deployment;
  - consolidation.
- **Evidence hygiene (verifier).** WAM-TTT should be treated as a mechanism precedent only, not as evidence that internal TTT works:
  - Table 2 appears to reuse Orig-setting cells;
  - 28 of 108 main-table cells cannot arise from the stated 25-trial rubric;
  - the gains confound backbone retraining with test-time writes (robot-only (200,0) scores 73.7 vs 74.1 for the full method);
  - the ICL comparator is untrained;
  - the evaluation tasks overlap the meta-training tasks.
- **Status.** The three table clauses are **accurate**, but the row understates the overlap. The brief's "closest robot TTT prior" is not a proposal claim (unverifiable).

**[6] Jiang et al., RoboTTT (NVIDIA; arXiv:2607.15275, preprint).**
- **What it is.** GR00T N1.7's flow-matching DiT action head with a tanh-gated (α₀ = 0.001) TTT-MLP (KV-binding loss) in each of 16 layers. The fast weights are updated once per decision on internal tokens (registers, proprio, noised action tokens), persist within a rollout, and are reset to a meta-learned W₀ each rollout. Other elements:
  - sequence FM training with loss masking for pure-context writes;
  - sequence action forcing;
  - DAgger Distillation (failures as context, corrections as targets).
- **Results.** 79% vs 42/49/56 (GR00T/Hist/GDN). One-shot imitation 6/10 vs 0/10 (Fisher p ≈ 0.011, the only clearly significant result). TTT-Linear ≈27% worse than MLP.
- **Caveats.** Figure baselines are inconsistent. With short pretraining context it falls below the single-step baseline.
- **Bearing.** This is the strongest precedent for per-step, candidate-dependent, persistent, meta-trained fast memory in the proposal's exact policy class. Its generic KVB writer extracts a goal from a single video, which challenges the need for a curated cue→content writer.
- **Status: accurate but understated.**
  - "Persist across decisions" should say "within a rollout".
  - DAgger Distillation actually meta-trains the *slow* parameters offline.
  - Whether the fast state inside a solve depends on the candidate through update-then-apply is **inferred, not stated** (verifier).
  - The Sec. 17 "same writer data" row is a design prescription. RoboTTT-as-published must be added alongside the matched-writer arm, because stripping action/register tokens (+23% / +18%) and sequence action forcing weakens it.

**[7] Li, Alim & Azizan, HardFlow (arXiv:2511.08425 v3).**
- **What it is.** Training-free, terminal-hard-constrained, cost-regularised steering of a frozen FM ODE. It is formulated as direct optimal control (Problem 2) and made tractable by one-step MPC on the posterior-mean terminal prediction.
- **Results.** 1.00 safety on manipulation (0.19 s vs 0.06 s); about 50× overhead on maze and PDE. λ_oc = 0 gives 0.10 safety; post-hoc optimisation without the prior gives 0.04.
- **Bearing.** Eq. (15)'s departure term is exactly HardFlow's Problem 2 after eliminating u, so Sec. 9 should cite it (and OC-Flow). HardFlow's own argument (App. B.B) that full-horizon coupling is ill-conditioned applies to Eq. (15).
- **Status: accurate.** The Sec. 10 comparison is fine (verifier override). The practical lesson is that the proposal needs an **oracle-constraint/cost baseline**, to show memory helps specifically when information must be inferred from experience.

**[8] Nakazawa et al., CA3 NMDA receptors in associative recall (Science 2002).**
- **What it is.** An adult-onset NR1 knockout confined to CA3 left water-maze acquisition and full-cue recall intact. With 3 of 4 distal cues removed, recall degraded: mutants 66.4% → 44.6% (chance 25%); controls showed n.s. drops; RRI P < 0.009.
- **Bearing.** It supports "cue-dependent recall / completion" and motivates the Eq. (13) auto-associative variant, though it does not decide single-pass vs iterative retrieval. It does **not** support "rapid association"; for one-shot encoding cite Nakazawa et al. 2003 (Neuron 38:305) and Cravens 2006. The lesion is mainly a write lesion with a possible read component, so "η_f = 0 with reader intact" is not a clean analogue (verifier).
- **Status: accurate but under-specified.** The design implication ("test … completion") is accurate (override), but the Stage B/C plan contains **no partial-cue probe** (Sec. 7's "ambiguous observations" and Sec. 18's "misleading memory" are different tests).

**[9] Kitamura et al., Engrams and circuits for systems consolidation (Science 2017).**
- **What it is.** Prefrontal engram cells form on Day 1 (they need MEC-Va and BLA input), mature over ≈2 weeks with hippocampal engram output, and the DG engram becomes "silent" but remains optogenetically sufficient at Day 15.
- **Corrections (verifier).**
  - Encoding-time MEC-Va→PFC inhibition **reduced** remote freezing (≈51 → 27% at D15; N = 10). It did not abolish it.
  - Many panels have N = 3–6.
  - TeTX silencing is chronic and has no behavioural readout.
  - Kitamura, Ogawa and Roy (not Okuyama) are co-first authors.
- **Bearing.** A mild tension: the proposal's lifecycle is closer to the "transfer-then-reset" standard model that [9] argues against. The proposal already labels its lifecycle an engineering choice. Its B store is closer to [9]'s silent-but-retained trace than a pure copy-then-erase.
- **Status.** Evidence column **accurate**. "Both systems can participate early" **accurate** (override). "Need not mean copying followed by erasure" is **partially accurate**: it is the authors' interpretation, silence was shown only to about Day 15, and what the hippocampal support carries is unknown.

**[10] Maingret et al., Hippocampo-cortical coupling (Nat. Neurosci. 2016).**
- **What it is.** Closed-loop cortical stimulation 20 ms after online-detected SPW-Rs rescued a weak 3-min object-place memory (DI 0.69 vs 0.46; n = 9 within-subject; Z = 2.67, P = 0.008). The same number of stimulations, delayed by 160–240 ms, did not help, even though delta and spindle rates were identical.
- **Caveats (verifier).** The random-timing control is a small between-subject group with an unreported stimulation count, and there is no implanted-unstimulated sham.
- **Bearing.** The design lesson is the **dose-matched, decoupled control**. The proposal lacks the ML analogue: a consolidation arm with the same slow-update budget but broken alignment between memory content and the slow update.
- **Status: accurate but understated.** It shows timing-specific *enhancement*. "Replay" is not measured; for replay cite Wilson & McNaughton 1994, Peyrache 2009, and Euston 2007.

**[11] Spens & Burgess, A generative model of memory construction and consolidation (Nat. Hum. Behav. 2024).**
- **What it is.** A modern Hopfield network (β = 20, one-shot storage, replay from noise) trains a VAE by teacher–student learning. The extended model stores the VAE latent plus high-error "sensory" residuals, and recall overwrites the prototype with those residuals.
- **Caveats.** Deletion, capacity limits and continual consolidation are **not simulated**, and the extended model reuses a frozen VAE.
- **Bearing.**
  - It is the conceptual template: one-shot store → replay → generative student → release when the student's error is low.
  - The same authors' 2026 follow-up (Nat. Commun., s41467-026-74357-6; verifier read the full text) is a **closer** lifecycle precedent: LoRA consolidation of memory-conditioned reconstructions into a memory-free generator, a memory-off test, measured sequential forgetting, and an explicit warning that the fast store drifts relative to the slow model.
  - W_k does not meet S&B's hippocampus requirements (high capacity, retrieval from noise, one-shot exact storage). B plus Hopfield/attention retrieval does.
- **Status: accurate.** The Sec. 2 principle "retain what the slow model cannot reconstruct" conflates S&B's dynamic prediction-error criterion with the proposal's information-theoretic non-identifiability criterion. The gate needs both.

**[12] Ramsauer et al., Hopfield Networks is All You Need (arXiv:2008.02217; ICLR 2021, venue unverified in-session).**
- **What it is.** Modern continuous Hopfield energy E = −lse(β, Xᵀξ) + ½ξᵀξ + …. The update ξ ← X softmax(βXᵀξ) is CCCP, which is exactly a unit-step gradient step on E. It is one attention step, with global-average, metastable and single-pattern regimes.
- **Bearing.**
  - It supports the attention caveat.
  - It shows Eq. (13) is **inspired by, not an instance of**, Hopfield retrieval: the anchor ½‖m − q_t‖² moves fixed points toward the cue; E_W is undefined; γ has no descent guarantee.
  - It gives the recipe for defining E_W over the fast net's own 32 key slots with CCCP guarantees.
  - The verifier corrected the reader in two places. [12] certifies a non-vanishing retrieval basin (≈Δ_i/(2M)), and small-metastable heads have the *largest* gradient (A.6.2). The capacity bounds at d = 16 are weak lower bounds, not evidence of low capacity.
- **Status: accurate.**

**[13] Albergo, Boffi & Vanden-Eijnden, Stochastic Interpolants (JMLR 26:1–80, 2025).**
- **What it is.**
  - Thm 6: the transport equation with b = E[ẋ_t | x_t] for C²-in-time interpolants I + γ(t)z under Assumption 5 (positive C² endpoint densities, moment bounds).
  - Thm 7: the regression characterisation, identical to Eq. (31).
  - Remark 11: without γ(t)z, spatial regularity is not guaranteed.
  - Sec. 4.2: with γ = 0 the density collapses to a Dirac at t = ½.
  - Thm 31: an SDE, but not an ODE, can sample from a point mass.
  - Thm 47: rectification preserves the map *if the field is perfectly learned*.
- **Status.**
  - Sec. 12 "weak continuity equation for absolutely continuous paths [13]": **partially accurate**. The general weak form should be cited to Liu 2022 (arXiv:2209.14577, eq. 14, Def. 3.1, Thm 3.2) or Ambrosio–Gigli–Savaré.
  - App. A "does not require smooth densities [13]": **mis-targeted** rather than backwards (verifier).
  - The counterexample (33) should cite [13] Sec. 4.2 and Hertrich et al. Prop. 13, and the remedy (latent smoothing) is missing.
  - App. C should cite [13] rather than [2] for Gaussian-mixture sources.
  - Add the JMLR venue. Note that [13] eq. 3.7–3.9 prints "u − s = b", which should be b = u − a·s (verifier).

**[14] Tedrake, Underactuated Robotics, Ch. 7 Dynamic Programming (living notes, © 2024).**
- **What it is.** The HJB equation, an infinite-horizon verification theorem, and the closed-form minimiser u* = −½R⁻¹f₂ᵀ∇J for control-affine dynamics with quadratic control cost.
- **Bearing.** With f₁ = v_θ, f₂ = I and R = ½I, this reproduces Eq. (19) exactly, and Eq. (20) follows (verified analytically and numerically: closed-loop cost equals V to 1e-11). The chapter itself warns "cusps are commonplace in optimal value functions". For the proposal's multimodal terminal costs V has kinks (∂_zV(0.5, 0±) = ∓1.33 in the reader's double-well example), so u* = −∇V jumps at mode boundaries.
- **Status.** Maths **correct**; citation **partially accurate**:
  - the stated theorem is infinite-horizon and time-invariant;
  - the chapter lists "finite-horizon / time-varying dynamics or cost" as an unwritten `<todo>`;
  - cite Bertsekas 2005 Vol. I Ch. 3 or Tedrake's LQR chapter ("general form"; cost positivity "not actually necessary in the finite-horizon case");
  - add the chapter number and an access date.

**[15] Marcotte et al., Regions of Reliability (ICML 2023).**
- **What it is.** A power analysis of proper scoring rules used as *evaluation tests*. With n = 30 and d ≥ 16, the energy score (ES) barely separates a bimodal law from a moment-matched Gaussian (max power 0.09–0.13 vs NLL 0.80), and it misses dependence errors.
- **Bearing.**
  - Eq. (16) is exactly the paper's unbiased ES-Full with β = 1 (verified by code and Monte Carlo). The biased V-statistic would reward under-dispersion (1-D optimum σ ≈ 0.38 at S = 2).
  - The multimodality warning is conditional. It holds at n ≈ 30 near threshold; at the proposal's n ≈ 1,000 it is borderline (roughly 700–1,400 needed); for d = 2 it is untested. [15] has no data for m < 64 (verifier).
- **Status.**
  - "Proper": **accurate** (strict propriety should be cited to Gneiting & Raftery 2007).
  - "A finite training set gives no calibration guarantee [15]": **misattributed**. [15] is about evaluation power and explicitly distinguishes itself from calibration (footnote 1).
  - ES as a mode-preserving *training* loss: **unsupported by [15]**.
  - State β.

**[16] Schwarz et al., Progress & Compress (ICML 2018).**
- **What it is.** An active column learns each task with the knowledge base (KB) frozen and reuses KB features through lateral adaptors. It is then distilled into the KB (KL + online EWC with γ-decayed, per-task-normalised Fisher) and optionally re-initialised, all framed as day/night sleep.
- **Results.**
  - Omniglot: 82.84 vs 79.15 (online EWC) after 5 passes, but a **tie after 1 pass** (70.32 vs 69.99).
  - First-task retention: P&C ≈0.58 vs online EWC ≈0.67 and EWC ≈0.72 (pixel-traced, approximate; verifier). The two-phase design *costs* retention.
  - Atari forward transfer needs re-initialisation (127–129% vs a decline to 101%).
- **Status.** "Relevant mechanism" and "online EWC" sentences are **accurate**. The distinction column is **too narrow**.
- **Internal inconsistency.** Sec. 13 names EWC as "a useful alternative comparison", but the Sec. 17 table has no EWC, P&C or LwF arm.
- **Storage asymmetry.** P&C stores no past data; R_old needs D_old. Also cite Kirkpatrick 2017 and Huszár 2017.

**[17] Bechtle et al., Meta-Learning via Learned Loss (ICPR 2020; DOI 10.1109/icpr48806.2021.9412010).**
- **What it is.** A randomly initialised loss network M_φ trains an optimizee by gradient descent. φ is meta-trained through the post-update task loss (second order), with meta-train-only "shaping" information.
- **Bearing.**
  - With inputs (e, f_W(q_e)), the learned scalar loss is exactly a learned **conservative** error-vector writer (per-write update diff 2.4e-7; the head's final bias gets no gradient). The alternative is therefore *less* general than it sounds, unless the head sees W-dependent or multi-query inputs.
  - ML3's gains are confounded with learning rate: baselines use a fixed α = 0.001, while the model-based RL meta-network outputs a learning rate.
  - The unshaped learned loss can be *worse* than MSE: Reacher inverse dynamics got worse than at initialisation.
- **Status: accurate.** Nuance: "minimizing the scalar in isolation" is exactly what happens at meta-test. The accurate statement is that the head must be trained through the post-write future loss, and only dM/df matters.

**[18] De Bortoli et al., Diffusion Schrödinger Bridge (NeurIPS 2021).**
- **What it is.** A Schrödinger bridge (SB) is the KL-closest path law to a reference diffusion under both marginal constraints. It is solved by IPF (each half-step is a time reversal / Doob h-transform, trained by mean-matching regression). Lemma 19 (Girsanov): KL = ¼∫E‖b₁ − b₂‖² for two √2-diffusions that share noise.
- **Bearing.**
  - It confirms the Sec. 9 sentence and makes it constructive. Under an Euler–Maruyama reference with noise σ > 0, the departure term = σ²·(−log p_ref(Z|z₀)) + const, and E_Q[departure] = σ²KL(Q‖P_ref) + Nσ²/2. K gradient steps on one Z therefore compute a **MAP path**, not a KL-optimal law. That explains the anticipated loss of diversity.
  - The Sec. 10 value reader with σ > 0 and a log-density-ratio terminal cost is an IPF half-bridge.
  - A Girsanov retention bound for an SDE surrogate needs weight w_t ≥ t/(1−t), which diverges at t → 1, so it is not a drop-in replacement for Gronwall (verifier).
- **Status: accurate.** The FM contrast sentence is accurate (override). Write "Schrödinger bridge", not "bridge".

---

## 3. Closest prior work and novelty assessment

### 3.1 Named works

- **RoboTTT [6] (medium-high).** It pre-empts: fast weights used at every denoising step of an FM action-chunk DiT, queried with noisy action tokens; persistence across decisions within a rollout; meta-learned W₀ and projections via a causal sequence FM loss with the same interpolant and target as Eqs. (11)–(12); pure-context writes via loss masking; learning failure→correction dynamics. Requirements 1–2 and Eq. (1)'s "history acts through updated weights" are therefore **not new in robotics**. It does *not* pre-empt: an external centred reader with exact reset, memory-off consolidation into θ, retention measurement, reset gates, or meta-training across backbone checkpoints. It is the system the proposal's own falsification test ("if internal TTT ≥ external reader, drop superiority") will be run against.
- **WAM-TTT [5] (medium; high for the recipe).** It pre-empts the meta-training recipe and the KV-reconstruction writer for a robot FM model with a frozen backbone at deployment. It does not pre-empt centring, velocity-level readout, multi-write causal deployment, or consolidation. Its evidence is weak (§2).
- **TTT layers [3] (high overlap for the writer, low for the rest).** The writer, learned W₀ and second-order outer loop are [3]. An output-side fast-weight layer also has precedent: Clark et al. 2022, "a final layer of fast weights, whose initialization is trained as slow weights" (cited in [3]; medium per the gap verifier). [3] has no lifecycle.
- **KV-binding [4] (low for claims, medium for framing).** It does not show the proposal's full-write memory equals linear attention (the kernel is dynamic). It does make internal TTT or a matched linear/DeltaNet memory plausible competitors, and it casts doubt on an unqualified "fast weights store associations" story.
- **Progress & Compress [16] (high for the lifecycle concept).** Plastic learner with the slow model frozen, then distillation into the protected slow model, then re-initialisation, plus the sleep analogy. P&C also shows the risk: the two-phase design tied single-network online EWC in one pass and retained worse.
- **Spens & Burgess [11] (low alone; low-medium with the 2026 follow-up).** A conceptual template. The 2026 follow-up (LoRA consolidation of RAG reconstructions, memory-off test, forgetting under sequential consolidation) is the closest cognitive-modelling precedent for the "teacher = generator + memory, student = memory-off generator" lifecycle.
- **HardFlow [7] (low for the core; attribution for Sec. 9).** Eq. (15) is its Problem 2. The novelty of a memory-derived cost lies in *how the cost is written by experience*, not in the control machinery.
- **Proxy Policy Steering (2609.09148; verifier: high, every lens).** It adapts a frozen FM VLA with v_base + γ[v_task − v_ref], where v_task is initialised from a v_ref distilled from the base's own denoising states. The correction is zero before adaptation and γ = 0 recovers the base. Ablations: w/o ref 64 → 55, w/o tune → 48. Its PoE/density-ratio reading holds per noise level only, under an assumption (A3) the authors call "a modeling choice". It pre-empts the *centred velocity residual* as an interface. It has no memory, no causal writes, no meta-trained writer, no consolidation, and is single-task with 50 demos. The proposal's centring is motivated by exact reset; PPS's by cancelling shared approximation error. Cite both motivations.
- **TraceFlow (2609.20646; verifier: medium in three lenses, high in one).** Capped kernel-density guidance from success and failure traces, added to a frozen FM expert's velocity in early solver steps. The bank grows across rounds with no weight updates. Real robot: 21 → 39 → 47/50. The simulation aggregate is unchanged (34.92 → 34.99). Retrieval is keyed on current state, so it fails on Counting/Occlusion, the analogue of Stage B's identical-present-context design. It is the nonparametric counterpart of the direct reader.

### 3.2 Contribution-by-contribution verdict

| Proposal contribution | Pre-empted by (verified) | What remains new | Verdict |
|---|---|---|---|
| **(i) External, W₀-centred velocity reader** | PPS (centred adapted-minus-reference velocity residual, every step, frozen FM VLA); FlowCorrect (gated LoRA velocity residual, zero at Δθ = 0); CFG/classifier-guidance structure ([1] eq. 62, 65); plasticity injection (twin-minus-frozen copy; verifier: low); Clark FWL (output-side fast weights); Hinton & Plaut 1987 (additive fast weights decaying to zero); TraceFlow (capped velocity guidance from experience) | Reading out a *persistent gradient-written associative memory* into velocity, centred by W₀ of the *same* reader, so the memory-off policy equals v_θ exactly and stays invariant to reader/interface training. This enables exact reset within a lifecycle and reader swaps. | The interface is **not new**; its role in the lifecycle is. Present centring as borrowed and justified by two arguments (exact reset; shared-error cancellation). |
| **(ii) Memory access at every denoising step** | RoboTTT, WAM-TTT, DexWorldModel (internal); LaCT (write clean / read noisy chunks, video diffusion); One-Minute Video; Gated Memory Policy (gated history cross-attention at each denoising step); TraceFlow (early steps) | An *external* candidate-dependent query to fast weights that are frozen during the solve. Its value is purely empirical, and controls exist: SmoLSTM (read once per decision), Retrieve-then-Steer (t₀ only), final-action residual | **Not new.** State it as a hypothesis to test, not a feature. |
| **(iii) Meta-trained gradient writer** | TTT [3]; Metalearned Neural Memory (external MLP, one modulated GD step, meta-trained, including RL); WAM-TTT, RoboTTT (robot FM loss through the inner update); Clark FWL; TTT-E2E; VITA; OML/OSAKA; ML3 [17] for the learned-loss alternative | Minor: hetero-associative cue→content with a fixed pretrained target encoder (WAM-TTT's writer is self-associative) | **Not new.** Cite [3] and MNM in Secs. 5 and 7. |
| **(iv) FM consolidation lifecycle with gates and reset** | P&C, Reset & Distill (ICLR 2025), OSAKA (consolidation at boundaries), Nested Learning CMS (re-init to meta-learned θ₀, NeurIPS 2025, medium), S&B 2026, SlowFast-VGen (per-episode fast LoRA plus outer slow loop, no distillation; medium), ReGuide/PLD/PA-RL (steered teacher → weights), SDFT (on-policy context-conditioned self-distillation), When Robots Sleep, LM "Sleep" papers | FM-specific transfer that preserves the source law (fresh-noise re-pairing; explicit handling of selection bias); a memory-off gate with a per-task retention matrix, a renewed-plasticity check and an episodic exemption; writer meta-trained over accepted checkpoints; applied to few-shot parametric fast memory in an FM robot policy | **Concept pre-empted; the specific combination is unclaimed.** This, with the controlled factorial, is the defensible core. |
| **(v) Retention bound (27)–(28)** | Gao–Huang–Jiao Prop. 39(ii) (same constant, same proof; medium); Albergo–Vanden-Eijnden Prop. 3; Benton et al. Thm 1 (time-varying L_t); ORW-CFM-W2 Thm 3 (also uses ‖v − v_ref‖² as a regulariser; medium); VGG-Flow Prop. 4 | Only its reading as a retention certificate over consolidation cycles | **Not new, and vacuous for multimodal targets.** Demote to a cited lemma; see §4.2. |
| **(vi) Path vs endpoint distillation** | Theory: rectified flow Thm 3.3 and the nonlinear extension; Liu 2022 weak form; [13] Thm 6, Remark 11, Sec. 4.2, Thm 47; Hertrich Prop. 13 (= Eq. 33) with Thm 14/Ex. 16 (the smoothing fix). Related empirics: Masip et al. (endpoint-only generative replay "catastrophic" vs teacher field-matching, but with 2- and 10-step DDIM teachers and not path regression); Q-VGM | The empirical comparison for *adapted-teacher* consolidation in FM policies, **if** it includes coupling-controlled arms (fresh-noise endpoint / same-seed reflow / (smoothed) path) | **Theory not new; the experiment is new only once the coupling confound is fixed.** |

Two further pre-emptions sit outside the numbered claims:
- **Sec. 10 value/control reader.** It is VGG-Flow's formulation (v = v_base + ṽ, cost λ/2∫‖ṽ‖² − r(x₁), ṽ* = −(1/λ)∇V) with a memory-dependent running cost added. Adjoint Matching proves naive deterministic control does not produce the tilted distribution ("when … σ(t) = 0, fine-tuning naively will not have any effect because X₀ completely determines X₁"); its model can still be *sampled* with σ = 0 afterwards (verifier).
- **Sec. 9 path-energy reader.** It is a discretised OC-Flow / HardFlow Problem 2 with a learned, edge-dependent energy.

---

## 4. Errors, overstatements, and gaps

### 4.1 Confirmed errors and misattributions

1. **[15] misattributed (Sec. 9).** "A finite training set gives no calibration guarantee [15]." [15] studies finite-sample *power of evaluation tests* and separates itself from calibration (footnote 1). Reword to a discrimination limitation and cite Gneiting & Raftery 2007 (propriety) and Gneiting, Balabdaoui & Raftery 2007 (calibration).
2. **[13] scope (Sec. 12, App. A).** [13] proves the transport equation only for C²-in-time interpolants plus γ(t)z under Assumption 5. Piecewise-linear teacher paths (second derivatives are Diracs at the knots) and history-dependent paths fall outside it. Cite Liu 2022 / Ambrosio–Gigli–Savaré for the weak form. The phrase "does not require smooth densities [13]" is mis-targeted: [13] obtains smoothness *from* the latent.
3. **[14] scope (Sec. 10).** The chapter's theorem is infinite-horizon and time-invariant, with strictly positive-definite running cost and positive-definite J. The time-varying case is a `<todo>` and there is no terminal cost. Cite a finite-horizon verification theorem and list its conditions: V ∈ C¹, Lipschitz v_θ, admissible controls in L², and a closed-loop solution to ż = v_θ − ∇V. Bounded-below costs are sufficient, not necessary.
4. **Uncredited prior results presented without attribution:**
   - Eq. (27)–(28): Gao Prop. 39(ii) and Albergo–VE Prop. 3.
   - Eq. (33): Hertrich Prop. 13 and [13] Sec. 4.2.
   - App. A marginal preservation: rectified flow Thm 3.3.
   - Eq. (15): HardFlow Problem 2 / OC-Flow.
   - Sec. 10: VGG-Flow.
   - Writer (Secs. 5, 7): [3] and MNM.
   - Centring: PPS.
5. **Internal inconsistency.** Sec. 13 calls EWC "a useful alternative comparison [16]", but the Sec. 17 table has no EWC, online-EWC, P&C or LwF arm.
6. **Sec. 21 characterisations.**
   - [6]: the row omits that RoboTTT is the same FM action-chunk policy class and reads with the noisy candidate at every step. "Persist across decisions" should read "within a rollout; reset to W₀ each rollout".
   - [5]: the row omits the meta-learned W_init/interfaces trained through the write on a robot FM loss, and the joint backbone training.
   - [4]: the "distinction to test" is a disclaimer.
   - [16]: omits the storage asymmetry.
7. **Biology.**
   - [8] is used under a "rapidly learning associative system" framing that it does not support (13-day training; acquisition normal without CA3 NMDA receptors). Nakazawa 2003 does support it.
   - [10] is paired with "replay", which it neither measured nor manipulated.
8. **App. C source citation.** "[1, 2]" for Gaussian-mixture sources: [1] is fine (override), but [2] is Gaussian-source only. Add [13] (Sec. 2.3, App. A: "e.g. a Gaussian or a Gaussian mixture density"), Lipman et al. 2024 (arXiv:2412.06264) and Tong et al. 2023.
9. **Missing venues.** [3] ICML 2025; [4] ICML 2026; [13] JMLR 2025; [17] ICPR 2020; [18] NeurIPS 2021. [1] and [14] need a version or access date.

### 4.2 Technical gaps (confirmed by derivation or numerical check)

1. **The retention bound is vacuous for multimodal targets and loose for the sanity check.**
   - Unimodal σ = 0.15 (Sec. 19 field): L = max|b_t| ≈ 3.41, so C_L ≈ 134. The bound gives about 700‖Δμ‖², while the true W₂² is ‖Δμ‖². A time-dependent |b_t| constant gives about 83; a one-sided constant gives about 1.23.
   - Bimodal ±1, σ = 0.15: the boundary Jacobian is ≈112, giving C_L ≈ e^{224}.
   - At σ = 0.05 even the one-sided constant peaks near 1.5e3 (fm-theory verifier).
   - Remedy: state the time-varying / one-sided forms (Benton; Gao 39(i); Boffi), present the bound as applying to unimodal or near-affine fields, and report empirical W₂ and ∫L_t along trajectories.
   - Benton: "we expect the Lipschitz constant to explode as one approaches the data distribution."
2. **R_old is scale-heterogeneous.** It weighs near-deterministic tasks by about 1/σ (∫κ_t² = π/(4σ)). A single β and tolerance does not "protect each task equally". Normalise per task (P&C normalised Fishers for the same reason).
3. **ConSFT failure mode.** The velocity-MSE-to-reference (LwF) penalty collapsed on FM VLAs (LIBERO-Goal 0.00, RoboTwin-Coord 0.00). R_old differs (old contexts, old ODE trajectories), but this must be shown, with the data-point variant as an ablation.
4. **The endpoint-vs-path comparison is confounded.** Eq. (21) uses the independent coupling; Eq. (23) uses the teacher's noise→action coupling. A same-seed straight-line (reflow) arm is needed. The regularised path option (γ(t)z latent smoothing, or small seed noise per Hertrich Ex. 16) is never considered. Also add the positive condition for per-seed reproduction: ε ↦ Γ_t invertible, e.g. h·Lip(v_θ + r) < 1 for the Euler teacher.
5. **Stochastic teachers.** If any adapted sampler injects noise, knot-grid path supervision recovers the drift, not the marginal-preserving velocity, and under-disperses. Restrict App. A to deterministic teachers, or target the probability-flow velocity.
6. **Eq. (13) is ill-posed as written.**
   - It starts and anchors in query space but is read through a content-space interface.
   - E_{W_k} is undefined.
   - γ has no descent guarantee.
   - The anchor pulls fixed points toward the (possibly partial) cue, which works against completion.
   - Centring requires running the same S-step loop with E_{W₀}.
   - Fix: E_W(m) = −β⁻¹log Σ_j exp(β[(mA+b)_j]) + ½‖m‖² with a unit step gives CCCP updates m ← A softmax(β(Aᵀm + b)); read out once with B at the end ([12] reading).
7. **Value reader.**
   - (a) V is generically not C¹ for multimodal Φ_W, so u* = −∇V jumps at mode boundaries, and a smooth network will blur the jump, leaving ∇V ≈ 0 there.
   - (b) There is no reset construction. Centre the **costs** (C̃ = C_W − C_{W₀}, Φ̃ = Φ_W − Φ_{W₀}, giving V_{W₀} ≡ 0). Centring the value is invalid because the quadratic Hamiltonian couples the terms ([14] reading; algebra verified).
   - (c) Deterministic control does not target p_old·exp(−Φ). Use a memoryless SDE (Adjoint Matching) or a learned source (Uehara).
   - (d) With 16 Euler steps, optimality should be defined through the discrete Bellman recursion; the O(dt) gap is 1.34035 vs 1.33865 in a test.
8. **Sec. 9 ↔ Sec. 10 overlap.** With u_i = (z_{i+1} − z_i)/dt − v_θ(z_i), Eq. (15) is direct transcription of a Sec. 10-type problem. The readers differ mainly in solver. But the edge energy g_φ(z_i, z_{i+1}, …) depends on u, so it is a Lagrangian L(z, ż, t), and Eqs. (19)–(20) do not hold for it. The factorial confounds cost class with solver unless a common state-only cost is used.
9. **Stage B cannot show a compression advantage.** An explicit KV store at k ≤ 16 costs 512 floats < 1,072 and is exact. The crossover is around 33 experiences at equal state size (a reader count). Sweep k beyond it and match **bytes**, not only parameters.
10. **No partial-cue probe.** Sec. 8 makes Eq. (13) conditional on "partial-cue and distractor experiments", but Secs. 16–17 define none. Gate 1 can pass while completion is lost ([8]).
11. **Horizon and extrapolation.** Meta-training uses 1–16 writes, but deployment writes persist far longer. The evidence of mismatch sensitivity ([4] Fig. 1 inner steps; RoboTTT's beyond-window degradation; WAM-TTT's 0.1 vs 0.01 learning rates) means write-count and write-rate extrapolation must be explicit variables.
12. **Delta-rule/decay writers conflict with centring** unless decay targets W₀.
13. **The fast bias d₀ is an interference channel,** but only about 24% of the one-step change at an unrelated query (verifier). Making d₀ slow removes the query-independent "+1" kernel term, but interference remains through non-centred tanh features and the NTK term. Measure drift on unrelated queries after writes.
14. **The stop-gradient in Eq. (7) is inert** with a fixed target encoder. The collapse caveat applies only if the target is trained on l itself. [3] trains θ_V through the outer loss (that it avoids collapse with a frozen backbone is untested).
15. **Loss floor.** Raw L_meta includes the irreducible conditional variance, so report it relative to an estimate of the floor, or compare via an oracle full-history model.
16. **The "final-action residual" baseline** is distinct only if applied once after sampling. A per-step x₁-prediction residual is linearly equivalent to a velocity residual ([1] Remark 16).
17. **Selection bias versus existing practice.** ReGuide retrains with fresh noise on success-filtered endpoints. That is the proposal's option 2, targeting a success-conditioned law, not the noise-corruption case (verifier). Report the selection rule and its induced narrowing.

### 4.3 Judgment calls (defensible, but should be argued)

- **External vs internal reader.** TTT-E2E remarks that interleaved fast weights "prove to be critical" versus end-only placement. That is a related-work remark, not a controlled test (verifier), but together with RoboTTT's strong internal results it makes an external-reader win uncertain. Test centred internal TTT at more than one depth.
- **Nonlinear MLP vs linear memory.** The evidence conflicts (§1.1). This must be ablated.
- **Freezing θ during meta-training.** There is no controlled evidence in either direction.
- **Centring vs capping vs gating.** TraceFlow's cap mattered empirically. The proposal should specify a norm bound or schedule, or show one is unnecessary.
- **Curated association writer vs generic KVB-on-stream writer.** RoboTTT extracted a goal from one video with a generic writer.
- **"No explicit optimality principle" (Sec. 8).** Accurate for control optimality. Adding the Bayes-target characterisation is a framing improvement (verifier).
- **Deviation from [9]'s account.** The transfer-then-reset lifecycle departs from [9]'s "early cortical trace matured with hippocampal support". The proposal labels this an engineering choice. An "early gated slow tag" arm is optional; [10]'s conjunctive, salience-gated tagging argues *against* an ungated slow update on every write.

---

## 5. Missing related work

Legend: **[B]** should be a baseline; **[A]** ablation or protocol source; **[C]** cite only. Threat levels are the gap verifiers'.

**A. Velocity-space steering and residuals on frozen FM or diffusion policies**
- PPS, 2609.09148 (high) **[B, mandatory]**: the centred velocity residual on a frozen FM VLA. Train the proxy on the same k experiences.
- TraceFlow, 2609.20646 (medium; high in one lens) **[B]**: capped velocity guidance from an experience bank. Run a cue-keyed variant over store B.
- FlowCorrect, 2602.22056 (IROS 2026; medium) **[B]**: gated LoRA velocity residual from about 10 corrections, trained along logged ODE paths.
- Retrieve-then-Steer, 2605.10094 (low/medium) **[B, cheap]**: memory injected only at an intermediate state. This is the "t₀-only" control.
- FlowDAgger, 2607.08877 (medium) **[B, Stage E]**: supervised noise-space adaptation from 5–20 interventions; preserves held-out skills.
- Feng et al., On the Guidance of Flow Matching (ICML 2025; low) **[C]**: exact guidance vector field for tilts of the base.
- DSRL (low), Policy Decorator (low; the canonical final-action residual), ResiP (background), DynaGuide (background) **[C]**.
- CFG, Ho & Salimans (cited in [1]) **[B]**: a history-conditioned FM with condition dropout, plus a guidance-scale sweep for r.

**B. Fast-weight and TTT memory in generative and robot transformers**
- DexWorldModel, 2604.16484 (medium) **[C; use to specify the internal-TTT lifecycle]**: writes only from executed history; working copy frozen during the ODE.
- TTT-E2E, 2512.23675 (medium) **[B, internal arm]**: meta-learned initialisation plus test-time FM-loss updates of backbone MLPs. Capacity warning: updating only 1–3 layers "does not scale".
- LaCT (low) and In-Place TTT (ICLR 2026; low) **[C/A]**: clean-write/noisy-read design, Muon writer, task-aligned write objective.
- Metalearned Neural Memory (NeurIPS 2019; medium) **[C]**: direct precedent for the writer.
- Clark et al., Fast Weight Layers (EMNLP 2022; medium) **[C]**: output-side fast weights.
- Titans (NeurIPS 2025; medium) **[A]**: momentum plus decay, re-centred to W₀.
- Gated DeltaNet (ICLR 2025; low) **[B]**: linear delta-rule memory at matched state.
- Nested Learning / CMS (NeurIPS 2025; medium) **[C]**: fast levels reset to a meta-learned initialisation.
- MbPA (ICLR 2018; medium) **[B]**: retrieve-then-local-adapt, discarded after use.
- Elastic TTT (low) **[A]**: anchored writers.
- Miras (low), test-time regression (background), dual form (low), Schlag 2021 and von Oswald 2023 (background), Hinton & Plaut 1987 (background) **[C]**.
- VITA (ICLR 2026; low) **[C]**: never-reset, meta-learned TTT for robot value functions; a reset ablation template.
- SlowFast-VGen (ICLR 2025; medium) **[C/A]**: per-episode fast LoRA written by the denoising loss.

**C. History and in-context memory policies**
- Gated Memory Policy, 2604.18933 (medium; venue per GitHub only) **[B, mandatory]**: gated per-denoising-step history cross-attention. MemMimic "Match Color" ≈ Experiment B (99.0% with a 6,000-frame buffer).
- ContextFlow, 2609.06852 (ECCV 2026; medium) **[B]**: compressed-context in-context FM, which matches π0 fine-tuned on one demo.
- SmoLSTM, 2609.22854 (low) **[B, control]**: persistent matrix memory read once per decision.
- Benchmarks: MemMimic (from GMP); RoboMME (cited in RoboTTT).

**D. Continual learning and retention for FM/diffusion policies**
- "Pretrained VLAs are surprisingly resistant to forgetting", 2603.03818 (medium) **[B, mandatory]**: ER on π0 / GR00T at matched bytes with a buffer sweep.
- ConSFT, 2605.08879 (medium/low) **[B]**: replay-free retention; its LwF velocity-MSE baseline collapsed.
- Memory Anchors, 2608.26545 (low) **[A]**: conflict-region replay selection for D_old and B.
- Masip et al., Generative Distillation (CoLLAs 2024; medium) **[B]**: field-matching retention; endpoint-only replay fails with few-step teachers.
- Reset & Distill (ICLR 2025 retitle; low) **[B]**: lifecycle baseline.
- ReGuide (medium), PLD (medium), PA-RL / MoRE / Q-VGM / FAR (low–background) **[C; FAR as a direct test-time weight-update baseline]**.
- SDFT (low/medium) **[A]**: on-policy consolidation arm.
- Bertrand et al., stability of iterative retraining (ICLR 2024; low) **[C]**: keep a fraction of real data per cycle.
- EWC (Kirkpatrick 2017) and online EWC (Huszár 2017) **[B via P&C arm]**.
- Continual Policy Consolidation, When Robots Sleep, REGEN, TAIL, OMLA, WIZARD, Simple Recipe Works (all low) **[C]**.

**E. FM theory for transfer and retention**
- Gao, Huang & Jiao, Gaussian Interpolation Flows (JMLR 2024; medium) **[C, required]**: the exact source of Eq. (27).
- Benton et al. (TMLR 2024), Albergo–VE (ICLR 2023), ORW-CFM-W2 (ICLR 2025), VGG-Flow Prop. 4 **[C]**.
- Boffi et al. (flow maps; one-sided L) and Bansal et al. (Euler-discretised W₂) (low) **[C]**.
- Rectified flow (ICLR 2023; medium) **[B: reflow arm]**; Liu 2022 **[C]**.
- Hertrich, Chambolle & Delon (NeurIPS 2025; medium) **[C; smoothing arm]**.

**F. Control and steering theory for alternative readers**
- VGG-Flow (NeurIPS 2025; medium) **[B for the value reader]**.
- Adjoint Matching (low) and Uehara et al. (background) **[C, required if Sec. 10 is pursued]**.
- OC-Flow (ICLR 2025; low) **[B for the path reader]**.
- D-Flow (ICML 2024; low) **[A: optimise z₀ vs interior nodes]**.
- QGF (low) **[B if rewards exist]**.

**G. Biology and complementary learning systems**
- McClelland, McNaughton & O'Reilly 1995 and Kumaran, Hassabis & McClelland 2016 (both cited within [9] and [11]) **[C, required]**: the canonical computational frame. It is absent from Sec. 2.
- Nakazawa et al. 2003 (rapid one-trial encoding) and Cravens 2006 **[C]**.
- Primary replay papers (Wilson & McNaughton 1994; Peyrache 2009; Euston 2007) and Girardeau 2009 (ripple necessity) **[C]**.
- Spens & Burgess 2026 follow-up **[C, required]**.
- FearNet (hippocampus / mPFC sleep consolidation with store clearing) **[C]**.

---

## 6. Concrete recommendations

### 6.1 What to borrow

**Protocols**
- [8]'s P3/P4/P5 design:
  - write with the full cue, query with full / partial (graded) / no cue;
  - a paired relative recall index (partial/full, conditioned on passing full-cue recall);
  - restore-and-retest;
  - write-and-query with the reduced cue only (information sufficiency).
- [10]'s calibration: choose k where memory-off-after-reset is at chance, add a positive control (full fine-tune), and add the dose-matched misaligned-consolidation control.
- RoboTTT's protocols: one-shot imitation with an identical prompt, perturbation recovery, rubric scores plus full-success counts, fixed initial placements.
- P&C's evaluation split: teacher for forward transfer, student for overall performance; % single-task; multi-pass revisits; task permutations.
- Lipman Fig. 7 solver-error protocol: fixed seeds against a 1,000-NFE reference at 8/16/32 steps.
- HardFlow's λ sweep, applied to reader scale and energy weight.
- [15]'s power check: before trusting a distributional metric, tune the discrepancies (missing mode, wrong weights, shrunken variance) so that NLL has 80% power at the planned n.

**Diagnostics**
- Writer health ([3] Figs. 4/14): l(W₀; e), l(W_{t−1}; e), l(W_t; e).
- [4] Q/K overlap (queries from z_t vs write keys) and a write-steps sweep.
- Sign-flip control with the **MSE** writer only (under Frobenius loss it is absorbed exactly).
- Information-loss term vs capacity (§1.4).
- Integrated velocity correction *together with* distributional change (non-uniqueness, [2]).
- Estimates of L_t along trajectories next to any bound.
- Per-unit change distribution in θ after consolidation ([10]-style selective reorganisation, loosely).

**Architecture and training choices**
- Loss masking for pure-context writes.
- Sequence action forcing if actions are written.
- TBPTT labelled as a truncation.
- Match the backbone's τ distribution for Stage E (GR00T/π0 use shifted-Beta τ, not uniform).
- Learnable write gate η(e) = η_base·σ(θ·e) ([3]).
- LN+residual wrapper f(q) = q + LN(f_res(q)). Centring stays exact because the W₀ output is subtracted.
- If Eq. (13) is used, define its energy over the fast net's own slots (§4.2).
- Norm cap or schedule on r (TraceFlow; RTC-style clipping is common practice).
- Decay only toward W₀.
- Store B as raw or frozen-encoder records; rebuild W after every θ change.

**Consolidation arms**
- Fresh-noise endpoint FM (Eq. 22).
- Same-seed reflow on unfiltered teacher pairs.
- Smoothed path distillation (x_t = Γ_t + γ(t)z, target V_t + γ̇(t)z, γ(0) = γ(1) = 0; [13] Remark 4 allows Brownian-bridge noise).
- Unsmoothed path (Eq. 23).
- Retention variants: R_old on old trajectories vs Masip-style matching at noised old endpoints vs ER, all per-task normalised; old-teacher-endpoint anchors vs the re-centred anchor.

### 6.2 Mandatory baselines (ranked)

1. **Internal TTT, two forms.**
   - RoboTTT as published (KVB on internal tokens, tanh gate, sequence action forcing).
   - A **centred** internal TTT with the proposal's writer data at 2 or more depths (the output-side limit is the external reader).

   This separates location from centring and from writer.
2. **PPS-style proxy residual** on the same k experiences. This gives centring without memory.
3. **Gated Memory Policy-style** gated history cross-attention at every denoising step (the history-token row), plus **append-only KV/Hopfield** read by the same centred reader (the storage axis), both at matched bytes and latency.
4. **Linear fast memory at matched state:** DeltaNet / Gated DeltaNet (decay recentred to W₀), and last-layer-only writes (which is exact DeltaNet cross-attention by [4]).
5. **Capped retrieval guidance in velocity space** (TraceFlow-style, cue-keyed) over store B.
6. **Direct test-time weight update** on the same experiences: few-step FM-loss LoRA/adapter (FlowCorrect/FAR/TEMP-LoRA style), and MbPA-style retrieve-then-adapt.
7. **Read-location controls:** final-action residual applied once; initialisation-only (Retrieve-then-Steer); read once per decision (SmoLSTM-style); a context/time-only query.
8. **Consolidation and retention:**
   - ER at matched replay bytes with a buffer sweep (2603.03818);
   - a **single-network direct update of θ with the same replay and no fast memory** (P&C's evidence says this can win);
   - a P&C / Reset & Distill instantiation (a full adapter trained on the task, FM-distilled with online EWC or an FM-Fisher penalty, then reset);
   - generative distillation (Masip);
   - ConSFT;
   - the dose-matched decoupled-consolidation control;
   - reflow and smoothed-path arms.
9. **CFG on a history-conditioned FM** with a guidance sweep; a memory-guidance scale w for r.
10. **For optional readers:** VGG-Flow (value), OC-Flow/HardFlow plus D-Flow (path), and HardFlow with the **oracle** constraint or cost (an upper bound when information is given rather than inferred).

### 6.3 Sharpened contribution statement

Suggested wording:

> We study continual adaptation of a frozen flow-matching action-chunk policy through a small persistent fast-weight memory. Building on test-time-training layers [3], internal robot TTT [5, 6] and velocity-space residual steering (PPS), we propose and test a **lifecycle**:
> - causal, meta-trained associative writes from robot experience;
> - a W₀-centred velocity residual read at every solver step, so that the memory-off policy equals the backbone exactly;
> - periodic flow-matching consolidation that preserves the source law, compared across fresh-noise endpoint, same-seed reflow and smoothed path targets;
> - a memory reset behind memory-off transfer, per-task retention, renewed-adaptation and episodic-exemption gates.
>
> The primary contribution is a controlled factorial over storage (parametric fast weights vs append-only KV/retrieval), read location (internal TTT vs external velocity; per-step vs read-once vs initialisation-only) and transfer objective, under matched bytes, information and compute, on a delayed cue-to-action-distribution benchmark with partial-cue probes, and then in robot simulation. The W₂ stability lemma is a known result (Gao et al.; Albergo & Vanden-Eijnden), used here as a qualified certificate for unimodal or near-affine fields.

Also move "reader superiority" to an explicitly falsifiable secondary hypothesis, as Sec. 18 already does, and drop any wording that implies novelty for centring, per-step memory access, the writer or the bound.

---

## 7. Open questions only experiments can answer (Stage B/C)

**Stage B (learned fast memory)**
1. Does a candidate-dependent read at every step beat read-once-per-decision, t₀-only and final-action residuals at matched reader evaluations? Does early-only access re-weight modes but fail to reshape them, as the analysis predicts?
2. At k ∈ {1…16} **and beyond ~33**, which storage wins per byte: tanh-MLP fast weights, DeltaNet, last-layer-only writes, append-only KV/Hopfield, or GMP-style attention? Is the MLP "redundant in short context" ([3]), or does it help as in RoboTTT's long episodes?
3. Does the writer store associations? Answer this with MSE sign-flip, write-steps sweeps, Q/K overlap, and the P3/P4/P5 partial-cue probes, including held-out masks at query time (so completion is not exact-key lookup) and an ablation separating completion by the slow encoder from completion by the fast memory.
4. Under repeated or conflicting cue writes, does the MSE writer average multimodal content toward the conditional mean? Does a softmax-slot or KV memory (explicit β) select instead?
5. Do gains survive write counts and rates beyond the meta-trained range (write-count extrapolation)?
6. Does freezing θ cap the external reader? Compare with a jointly meta-trained variant, reporting memory-off performance of both the original and meta-trained backbones.
7. Is a norm cap on r needed for stability at 8/16 solver steps? Does a memory-guidance scale w > 1 help (reader under-use)?
8. Is the gain memory content or backbone-misfit correction? Use shuffled and wrong-history controls with the **same number of writes**. Report the information-loss term against capacity, and L_meta relative to the floor.
9. Does centred external reading beat PPS trained on the same k experiences, which is centring without persistent memory or meta-learned writes?
10. Is the chosen distributional metric powerful enough at n ≈ 1,000, d = 2? Run a power check, and use NLL where it is defined (direct reader; the discrete-sampler log-determinant in 2-D).

**Stage C (one transfer)**
1. Does fresh-noise endpoint FM recover ≥90% of the teacher gain after reset? Does reflow (same seed) or smoothed path add anything beyond the coupling?
2. For the actual teacher, is ε ↦ Γ_t invertible (h·Lip < 1), and is b* regular in practice? Does averaging arise only where histories are mixed?
3. Does R_old on old ODE trajectories avoid the ConSFT/LwF collapse, and does it beat ER and generative distillation at matched bytes, after per-task normalisation?
4. How loose is Eq. (27) empirically? Estimate L_t along trajectories and compare the bound with the measured same-noise W₂ for unimodal and bimodal targets.
5. Does consolidation distort atypical adapted behaviour toward old modes? Use within-mode variance ratios and an atypicality sweep, especially with β·R_old active.
6. Does the dose-matched but misaligned consolidation control ([10]) remove the gain, showing the gain is transfer of memory content and not extra slow training?
7. Does a stale W under the new θ fail where a rebuilt W succeeds (interface drift)? Does the writer still adapt with the same budget after consolidation (gate 3)?
8. Does the memory-off student keep **partial-cue** robustness, or only full-context performance ([8])?
9. How sensitive is transfer to the teacher-context sampler, and to on-policy (student-rollout, SDFT-style) vs teacher-sampled targets?
10. How many intervening writes can occur before consolidation without losing transfer (cadence)?

---

## Appendix: unresolved or noteworthy reader/verifier disagreements

- **TraceFlow threat level:** medium (three lenses) vs high (the fm-theory lens). All agree it is the nonparametric counterpart of the direct reader and fails when event information is absent from the current state.
- **Benton and Albergo–VE threat level:** low vs medium across lenses. The substance is agreed: the bound is known and needs attribution.
- **Retrieve-then-Steer** (low vs medium), **SDFT** (low vs medium), **ConSFT** (medium vs low). The differences reflect lens emphasis, not facts.
- **Vacuity of the retention bound "for the planned targets":** the [1] verifier says it depends on unspecified mode geometry (m/σ ≈ 2 gives a factor of about e⁴: loose, not vacuous). The [2] verifier says it is vacuous for well-separated multimodal targets whatever σ_min or reformulation is used. Both hold; the proposal should specify the mode geometry.
- **RoboTTT inside-solve dynamics:** whether the update-then-apply order makes the fast state within a solve depend on the candidate is inferred from Eqs. 1–2 and Fig. 2. The paper does not state it.
- **Gated Memory Policy venue:** "CoRL 2026" is supported only by the authors' GitHub; the project page contains a commented-out "RSS 2026".
- **Masip et al. and endpoint vs path:** readers claimed it "already answers" the question; verifiers disagree (a different mechanism, and few-step teachers). I follow the verifiers.
- **P&C retention values** come from pixel-tracing Fig. 2a (approximate), and the relative ordering depends on them.
- **WAM-TTT Table 2** may be mislabelled rather than re-run. Either way its ablation should not be relied on.
- **Frozen θ:** there is no verified evidence either way. The WAM-TTT VLM-freeze ablation does not bear on this (verifier).