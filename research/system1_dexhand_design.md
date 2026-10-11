# System 1 for the dexterous hand beneath Astra: boundary, interface, short memory, optimization

Research note, 2026-10-11. Answers the user's questions after the System 1 pilots (Series C): do we need a System 1,
what should it predict, how should System 2 talk to it (a "System 1.5"), how does it get a short-term memory, and how is
it optimized from very few attempts, as a distribution rather than a point. Focus: the H1 + Inspire dexterous hands in
the existing Isaac benchmark. Evidence tags: [E] measured in our runs, [L] literature (verified references at the end),
[I] my inference.

## 0. The answer in brief

1. **System 1 is needed, for contact, not for spatial reasoning.** Driving the low-level controller (System 0)
   directly, Astra spends most of its effort failing to get fingers around objects [E, Sec. 1]. The boundary is
   time scale plus information: System 1 owns what needs feedback faster than System 2's decision period, or signals
   System 2 cannot see (contact, force, finger configuration, sub-centimetre geometry).
2. **System 1 is a dexterous contact controller conditioned on a desired object motion.** System 2 says how the object
   should move (lift 8 cm, lid to 1.3 rad, into the slot); System 1 produces the hand behaviour that makes it happen.
   It outputs a distribution over action chunks (wrist correction + finger targets) plus a critic / competence estimate.
3. **System 1.5 is a small contract code, not text:** target object, object-relative target motion, tolerance, success
   predicate. It tilts System 1's distribution, π ∝ π_θ · exp(β U), and comes back up as a structured report.
4. **Short-term memory is a small adaptable state on top of a frozen, pretrained System 1**, updated after each attempt
   with expected-value (critic) weights and a KL bound. Never a gradient step toward one lucky action.
5. **Long-term memory is twofold:** Astra stores contracts that worked (retrieval makes the next task zero- or one-shot),
   and System 1's base is consolidated across many tasks with a bounded distribution-level objective.
6. **Start simple:** a scripted closed-loop contact controller with an 8-dimensional parameter z and the per-attempt
   distribution update (no learning), then a small flow policy distilled from it, then the memory ablations.

## 1. What our runs say [E]

| observation | number | what it implies |
|---|---|---|
| Astra driving System 0 on the dex tasks (20 attempts, 441 decisions) | 137 hand closures, 23 (17 %) lifted an object; 322 / 441 decisions (73 %) before the first lift of an attempt | System 2 is doing System 1's job, badly and expensively |
| Insert-And-Unload attempt 5 | 35 of 43 decisions failed grasps before one bimanual grasp worked | grasp acquisition is the bottleneck |
| Decision period | ~20-40 s of model time per decision, ~0.7 s of open-loop execution each | contact events happen inside a decision; System 2 cannot react |
| Local search over Astra's open-loop parameters (Series C, 66 blocks, 4 tasks) | 0 rescues; fresh starts A = B = C (119/180) | perturbing an open-loop program is not a System 1 |
| Same action, same snapshot (Open-Laptop) | final lid angle 1.01-1.59 rad | outcomes are noisy; single rollouts are weak evidence |
| Failures that remain | lid/door stop just short; grasps fail on 8 mm shifts (Unload left can 1/18); a replayed lead-in leaves a can elsewhere | all are missing feedback at contact time |
| Horsea, naive self-write (own predicted chunk as the write target, LIBERO-10) | 2 % vs 36.5 % without the write | point targets from own samples destroy the policy |

## 2. The boundary and how to attribute a failure

**Contract per subgoal** [L: REFLECT, behavior-tree pre/postconditions, layered contracts]. System 2 issues
(object, desired object motion in the object's frame, tolerance, success predicate, budget). Preconditions: target
visible and reachable, command fresh, inside System 1's competence. Postcondition: the predicate.

**Attribution by replay, not by reading logs** [L: Who&When: LLM attributors find the failing step 14 % of the time;
AgenTracer / DoVer use replays with corrected actions]:
1. Localize: the first subgoal whose postcondition fails. All pass but the task fails → **System 2** (wrong or
   incomplete plan) [L: REFLECT].
2. Intent: wrong object, missing or misordered step (checked against privileged sim state, confirmed by an oracle
   executor) → **System 2**.
3. Feasibility: the command is outside System 1's measured competence → **System 2** (it asked for the impossible)
   [L: SayCan / STAP / VINE feasibility scores; HAC penalizes subgoals the low level misses].
4. Interface: same intent re-grounded on the observation at execution time, or re-encoded object-relative; if the
   outcome flips → **interface** [L: HIRO relative goals; ReKep re-grounding; HAMSTER: plans made at t = 0 go stale].
   If erasing the command barely changes behaviour, System 1 ignores the channel → retrain its conditioning
   [L: Fast Plans, Faithful Actions, 2026].
5. Oracle command (fresh, verified) still fails, e.g. an imprecise grasp → **System 1**.
6. Replay N times (System 1 is stochastic: separate skill from luck) [L: CCA].

**Routing the update:** System 2 gets memory (corrected contracts, lessons) and never a penalty when System 1 missed a
feasible, fresh command [L: HAC]. System 1 always learns from what it actually reached (hindsight relabelling), even when
System 2 was wrong [L: HER, SOAR, RoboCat]. The interface is fixed before any weights are touched.

**When to call System 2** [L]: only where task information changes the action ("decision states", where
KL(π(a|s,c) ‖ π0(a|s)) is large; InfoBot), when the value of deliberation exceeds its time cost (Keramati 2011,
Kool 2017), or to interrupt a skill when continuing is worse than switching, Q(s,o) < V(s) (options). Processing moves
to the cheaper channel; the expensive one carries only what the cheap one cannot (Genewein et al. 2015). Measurable:
System 2 calls per success must fall with experience.

## 3. System 1: what it is and what it predicts

- **Inputs (30 Hz):** wrist poses, 12 finger joints per hand, fingertip positions, **fingertip contact forces on every
  finger** (the benchmark only senses the index intermediate link; add sensors through the existing env cfg hook),
  the object pose estimate in the target frame, the contract code c, and the memory state m.
- **Outputs:** a distribution over action chunks (wrist Δpose + finger targets; start with the 6 actuated DoF per hand,
  not the 4 coupled closure groups Astra used) and a **critic** Q(o, c, chunk) or competence p(success | o, c). The
  critic is required: few-attempt updates must use expected outcomes, not realized ones (Sec. 5), and escalation to
  System 2 uses it.
- **Why object motion as the target** [I]: it is where System 2 is reliable (which object, where it should go) and it
  leaves every embodiment-specific decision (contacts, finger order, force, regrasp) to System 1. It is general: lift,
  push a lid, insert, release and in-hand rotation are all "make the object follow this motion"; grasping is a means.
  Goal-channel sufficiency is about whether goals can reproduce the effects of actions, not how rich they are
  (Nachum et al. 2019).
- **Train it so the code matters** [L: Fast Plans; information asymmetry, Galashov 2019]: randomize targets in training
  so the policy cannot succeed without reading c.

## 4. System 1.5: the contract code

- **Content:** object id; target motion as a short sequence of object poses (or a joint target for articulations) in
  the object's own frame; tolerance β (System 2's confidence); success predicate U. A few dozen numbers, regenerated
  only at subgoal boundaries or on escalation.
- **Effect on System 1:** a tilt π(a | o) ∝ π_θ(a | o, c) · exp(β U(a, o)). KL(π ‖ π_θ) is the number of bits System 2
  injected; it should shrink as System 1 improves (Ortega & Braun 2013; Genewein 2015).
- **How to apply the tilt** [L]: sample-and-reweight (draw N chunks, keep in proportion to exp(β U)) stays inside the
  base distribution and is exact as N grows; product sampling (ITPS) beats additive guidance; naive gradient guidance
  approximates a sum, drifts out of distribution and is biased for exponential tilts (QGPO's exact energy guidance and
  energy-weighted flow matching are the correct trained versions).
- **Upward report:** status, failure type (no contact, slip, blocked, timeout), final object pose relative to the
  target, attempts used, competence. This is what makes attribution cheap.
- Keep the code low-dimensional: the per-task sample cost of transfer depends on the complexity of the task-specific
  part only (Tripuraneni et al. 2020). A successor-features "task vector" is the limiting case of such a code
  (Barreto et al. 2017, 2019).

## 5. Short-term memory and few-attempt optimization

**Three tiers on a frozen base** [L: CLS theory; fast weights; RL² / Algorithm Distillation]:
1. *Within an attempt:* closed-loop feedback; later a fast-weight write into the flow (Horsea r_W / TTT), driven by
   surprise against the contract.
2. *Across attempts of one task (the short memory):* a small state that is not reset until the task changes.
3. *Across tasks:* consolidation into the base by interleaved replay, gated on retention of earlier tasks; Astra's
   text memory stores contracts that worked so the next task starts from a retrieved code.

**The update, cheapest first** [L: SfBC, V-GPS, Q-Planning; DSRL; Policy Decorator / ResFiT; MPO / AWR]:
- (a) *Sample and reweight:* N chunks per decision, weights w_i = softmax(Q_i / η) from a small critic trained on all
  rollouts (successes and failures). No policy parameters change.
- (b) *Noise-space policy (DSRL):* a small Gaussian policy over the base flow's input noise, updated per attempt; add
  KL(π_w ‖ N(0, I)) so the action distribution stays near the base (via the data-processing inequality; my derivation).
- (c) *Horsea write with a distributional target:* the write target is the velocity of the tilted distribution
  (energy-weighted flow matching), L = Σ_i w_i ‖v(x_t^i, t) − (a_i − x_0)‖², with an explicit KL / ε bound.
- (d) *Bounded residual* on the base (ResFiT: real dexterous-hand humanoid) when the base lacks a behaviour.

**Point versus distribution** [L + E]: every KL-regularized objective has a distribution as its optimum; a single-action
target is the η → 0 limit and the largest possible step (ESS = 1). Expert Iteration's distribution targets beat
chosen-action targets by about 50 Elo; unregularized iterated reward-weighted flow matching provably collapses to a
point; outcome-filtered or outcome-conditioned learning imitates luck in stochastic environments (Paster 2022,
Brandfonbrener 2022). Our own data agree: identical actions give a 0.6 rad spread, and Horsea's naive self-write fell
from 36.5 % to 2 %. Rules: many samples per decision, expected-value weights from a critic, an explicit KL bound,
monitor ESS = 1 / Σ w².

## 6. Pretraining and what exists

- **Needed:** few-attempt adaptation only works if the base already spans competent behaviours ("fast RL needs slow
  RL", Botvinick 2019). The base is long-term motor memory; the tiers above are short-term.
- **On disk:** the EgoVLA policy (`egovla_release/checkpoints/egovla_policy_ckpt-6720`, wrist + 30 MANO hand
  parameters retargeted to the Inspire hand; about 47 % on Open-Drawer, room 1, per its docs); ACT training code
  (38-D actions: two wrist poses + 24 finger joints, a CVAE latent built in); Open-Drawer data mixes. No dex demos for
  our four tasks, and the ACT base checkpoints were not recovered.
- **Data source for a dexterous System 1:** a scripted closed-loop contact teacher with privileged state generates
  thousands of grasp / lift / push executions across objects, masses, frictions and shifts; RL on object-motion
  targets later adds behaviours scripts cannot (regrasp, in-hand).

## 7. Plan, one step at a time (ablate each step)

| step | what | measures | baseline | Astra calls |
|---|---|---|---|---|
| 1 | **Contact System 1 v0, no learning:** fingertip contact sensors on all fingers; approach to an object-relative pre-grasp, fingers close until contact, grip-force target, lift with slip check, regrasp. z (8-D) = approach offset (3), finger-synergy mix (2), closing speed, force target, lift speed. Short memory = Gaussian q(z) updated after each attempt (expected-value weights, KL bound). Targets come from Astra's recorded intents. | grasp-and-lift success, attempts to first success under the 6 mass / friction conditions and 8 mm shifts; push-until for the lid and door | Astra's 17 % closures; Series C 119/180 fresh starts | none |
| 2 | **System 1 v1, learned:** a small flow policy (or ACT) on v0's rollouts across objects and conditions, observable inputs only, contract code as conditioning, plus a critic | does the learned policy keep v0's success; does the code matter (erase test) | v0 | none |
| 3 | **Memory ablation (your hypothesis):** none vs point update (fine-tune on successes) vs sample-and-reweight vs noise-space policy vs Horsea distributional write | success vs attempt 1 → 5 on held-out objects and conditions; retention of earlier tasks; ESS | step 2 | none |
| 4 | **Astra + System 1:** contract interface, escalation by competence / value of information | Astra calls per success, first-attempt success on new tasks | Astra alone (Insert 0/5, 209 calls) | yes (spend cap) |
| 5 | **Generality:** object-motion targets for articulations and insertion; RL for regrasp / in-hand | success on non-grasp subgoals | step 4 | optional |

## 8. Risks

- Sim contact forces are idealized; keep the contact signal coarse (per-finger normal force) so it transfers.
- Throughput: the harness runs one environment at about 13 control steps per second without rendering; data generation
  for step 2 needs parallel environments.
- The scripted teacher caps step 2 at scripted competence; RL (step 5) is the way past it.
- Astra availability (login revocations, spend cap) gates step 4 only.

## References (verified by literature surveys on 2026-10-11; arXiv IDs unless noted)

Attribution and contracts: REFLECT 2306.15724; Inner Monologue 2207.05608; SayCan 2204.01691; Text2Motion 2303.12153;
STAP 2210.12250; AHA 2410.00371; RoboFAC 2505.12224; RACER 2409.14674; HAMSTER 2502.05485; Fast Plans, Faithful
Actions 2609.30833; YAY Robot 2403.12910; RT-H 2403.01823; VINE 2512.03913; Embodied Agent Interface 2410.07166;
DoReMi 2307.00329; Code-as-Monitor 2412.04455; ReKep 2409.01652; Who&When 2505.00212; AgenTracer 2509.03312;
A2P 2509.10401; DoVer 2512.06749; MAST 2503.13657; Voyager 2305.16291; Reflexion 2303.11366; ExpeL 2308.10144;
SOAR 2407.20635; RoboCat 2306.11706; BOSS 2310.10021; HER 1707.01495; HAC 1712.00948; HIRO 1805.08296; COMA 1705.08926;
CCA 2011.09464; PA-BT 1611.00230; PDDLStream 1802.08705; layered contracts 2409.14902; AutoTAMP 2306.06531.

Theory of hierarchy and memory: Ortega & Braun 2013 (arXiv 1204.6481); Tishby & Polani 2011
(doi:10.1007/978-1-4419-1452-1_19); Genewein, Leibfried, Grau-Moya, Braun 2015 (doi:10.3389/frobt.2015.00027); InfoBot
1901.10902; Galashov et al. 1905.01240; FeUdal Networks 1703.01161; options (doi:10.1016/S0004-3702(99)00052-1);
Daw, Niv, Dayan 2005 (doi:10.1038/nn1560); Keramati et al. 2011 (doi:10.1371/journal.pcbi.1002055); Keramati et al.
2016 (doi:10.1073/pnas.1609094113); Kool, Gershman, Cushman 2017 (doi:10.1177/0956797617708288); Nachum et al.
1909.10618 and 1810.01257; Barreto et al. 1606.05312 and 2106.13105; Tripuraneni et al. 2006.11650; CLS
(doi:10.1037/0033-295X.102.3.419; doi:10.1016/j.tics.2016.05.004); fast weights 1610.06258; Schlag et al. 2102.11174;
TTT 2407.04620; Titans 2501.00663; RL² 1611.02779; MAML 1703.03400; Algorithm Distillation 2210.14215; Botvinick et al.
2019 (doi:10.1016/j.tics.2019.02.006); Expert Iteration 1705.08439; Distral 1707.04175.

Optimizing a distribution with a prior: REPS (AAAI 2010); RWR (doi:10.1145/1273496.1273590); AWR 1910.00177; MPO
1806.06920; AlphaZero 1712.01815; Sampled MuZero 2104.06303; DSRL 2506.15799; CFGRL 2505.23458; RECAP (π*0.6)
2511.14759; ITPS 2411.16627; DynaGuide 2506.13922; V-GPS 2410.13816; SfBC 2209.14548; IDQL 2304.10573; FOREWARN
2502.01828; RoboMonkey 2506.17811; Q-Planning 2608.21204; Policy Decorator 2412.13630; ResiP 2407.16677; ResFiT
2509.19301; energy-weighted flow matching (QIPO) 2503.04975; ORW-CFM-W2 2502.06061; QGPO 2304.12824; Diffuser
2205.09991; DPPO 2409.00588; ReinFlow 2505.22094; Flow-GRPO 2505.05470; FQL 2502.02538; Paster et al. 2205.15967;
Brandfonbrener et al. 2206.01079; Kumar et al. 2204.05618.

Robot System 1 / System 2 interfaces: Helix (figure.ai/news/helix); GR00T N1 2503.14734; Hi Robot 2502.19417; π0.5
2504.16054; LCB 2405.04798; RoboDual 2410.08001; RT-Trajectory 2311.01977; MOKA 2403.03174; VoxPoser 2307.05973;
Language to Rewards 2306.08647; Eureka 2310.12931; Code as Policies 2209.07753; ECoT 2407.08693; FiS-VLA 2506.01953;
Hume 2505.21432; OneTwoVLA 2505.11917; Gemini Robotics 1.5 2510.03342; DexGraspVLA 2502.20900; VLS 2602.03973;
RT-Affordance 2411.02704; KnowNo 2307.01928; 2AM 2609.11308; hierarchical-VLA termination study 2606.10267; 3D HAMSTER
2606.31329; GPT-6 Astra as a policy 2609.24170; System-1.5 Reasoning 2505.18962; Dualformer 2410.09918; SOFAI
2010.06002, 2201.07050.

## 9. What robot System 1 / System 2 systems send between levels [L]

| system | System 2 sends | System 1 | upward |
|---|---|---|---|
| Helix (2025) | one latent vector, 7-9 Hz | 80M policy at 200 Hz, wrist poses + finger flexion / abduction; end-to-end | S1 predicts a "% task completion" action |
| GR00T N1, FiS-VLA, RoboDual, LCB | VLM latents / embeddings | flow or diffusion head, 100+ Hz; trained jointly | none |
| Hi Robot, π0.5, Gemini Robotics 1.5 | language subtasks | VLA conditioned on them | success detection in the orchestrator (Gemini) |
| RT-Trajectory, HAMSTER, RT-Affordance, MOKA | 2D/3D end-effector paths, keypoints, key poses | policy conditioned on the sketch | none |
| ReKep, VoxPoser, Language to Rewards, VLS | cost / reward code over tracked keypoints or value maps | solver or MPC at 5-10 Hz; VLS steers a frozen flow policy (guidance + Feynman-Kac resampling) | violated constraints → backtrack; reward thresholds → stage switch or re-query |
| DexGraspVLA (dexterous) | target box → tracked mask every step | diffusion controller at 20 Hz | planner verifies the grasp, retries |
| Hume | best of 5 chunks chosen by a value head | finishes denoising the chosen chunk at 90 Hz | none |

What the evidence says about the interface:
- **Re-ground at control rate on tracked objects; absolute coordinates go stale.** VoxPoser drops 88 → 70 % under
  disturbances while LLM + primitives drops 24 → 0 %; DexGraspVLA hands System 1 a tracked mask, not coordinates;
  Code-as-Monitor adds +28.7 % under severe disturbance. ReKep's largest error source is the tracker, not the VLM.
- **Geometric codes beat language on the same data.** RT-Trajectory 50-67 % on unseen skills vs 11-26 % for language or
  goal images; RT-Affordance 68-76 % vs 28 % (language); Language to Rewards 90 % vs 50 % for primitive code.
- **Latent channels carry the most information but need joint training and are not checkable**; they suit a System 2
  that can be trained, which Astra is not.
- **Upward channels are rare but matter:** a success detector was the best termination signal in a systematic study of
  hierarchical VLAs; a Gemini 2.5 + π0 dual system issued infeasible and stale commands (OneTwoVLA).
- **No system sends an explicit uncertainty;** uncertainty enters as sampled candidates plus a score (Hume: random
  instead of value-picked chunks costs 13.7 % on LIBERO; V-GPS; FOREWARN; KnowNo's conformal sets for asking for help).
- **Train System 1 to tolerate an imperfect System 2:** condition dropout, spatial noise and temporal jitter (2AM:
  76.3 % vs 14.8 % on LIBERO-Mem).
- **Independent confirmation of our diagnosis:** an evaluation of GPT-6 Astra as a policy found strong semantic
  understanding but poor "precision, dynamic control, or complex bimanual coordination" (2609.24170).
- No robotics paper is called "System 1.5"; no study compares language, latent and geometric interfaces against the same
  System 1, so step 3-4 of our plan would be a direct comparison.

Implications folded into Sections 3-4: the contract's target is anchored to a tracked object and re-grounded every
control tick; System 1 is trained with noisy / dropped / delayed codes; System 2's uncertainty is expressed by sampling
candidate codes and letting System 1's critic score them; System 1 reports completion and failure type upward.
