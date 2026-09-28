# Horsea

**Fast and slow memory for flow-matching robot policies.**

## The problem

A pretrained robot policy is frozen at deployment, but the world it acts in changes. Objects sit
differently, the robot's controller drifts, a gripper behaves unexpectedly, and new tasks appear. We
want a policy that can:

1. **Adapt fast from its own experience.** After a few attempts, including failed ones, it should act
   better, without expert demonstrations, reward signals or retraining at deployment.
2. **Turn that fast adaptation into lasting skill.** It should consolidate what the short memory
   learned into the policy's own weights (long memory), then free the short memory.
3. **Do it repeatedly without forgetting.** After a reset it should learn again, while previously
   learned tasks and old skills keep working.

In short: *learn into short memory → consolidate into long memory → reset → learn again*, and never
forget earlier skills.

The hard part is the first step. A failed attempt contains useful information, e.g. "I commanded a
move to the left but the arm moved right". But the policy must learn *how* such evidence should change
its next decision. Simply imitating its own (failed) actions does not work.

## The approach: energy-based Horsea

- **Long memory:** a frozen flow-matching policy proposes an action chunk.
- **Short memory:** the fast weights of a small **energy network** E_W(action, observation,
  instruction). It is a learned cost over candidate actions. An empty memory has zero energy, i.e. it
  is exactly the base policy.
- **Learned writer:** after each attempt, a meta-trained writer turns the robot's own completed
  interactions (what it saw, what it commanded, what actually happened) into an update of W. No labels,
  rewards or success flags are used at deployment.
- **Acting:** a bounded solver moves the proposed action toward lower energy before it is executed.
- **Offline training:** the robot's exploratory or failed actions are *inputs* to memory. Corrective
  actions at the states it actually visited are *targets*, used only in the training loss.
  Training differentiates through the memory update and the solver.

## Where it stands

Diagnostic testbed: LIBERO-90 tasks with a **hidden rotation of the robot's commands**. The policy
must infer the rotation from its own attempts. The test uses fresh rotations and start states, with the
configuration chosen on dev and frozen in advance. 3 seeds, 210 sequences, mean success over
attempts 2–5 (corrected recorder, 2026-09-29):

| method | success | Horsea minus this [95% CI] |
|---|---|---|
| **Horsea, own experience written to memory** | **36.4%** | – |
| Horsea, memory off | 21.1% | +15.4 [+11.9, +18.9] |
| Horsea, shuffled action–outcome pairing | 23.2% | +13.2 [+9.8, +16.7] |
| Horsea, latest interaction only | 23.3% | +13.1 [+9.6, +16.7] |
| no-history model, same supervision | 22.0% | +14.4 [+11.0, +18.0] |
| TTT2 (RoboTTT-style fast weights), same data | 30.6% | +5.8 [+1.9, +9.9] |

Horsea uses accumulated, correctly paired experience to act better. The advantage over TTT2 is
modest and not significant in every seed.

Scope: the offline targets are oracle corrections, the shift is a hidden rotation, and adaptation
happens between attempts. An external audit found and fixed a recorder bug; the result held on
corrected data. See DEVLOG §7.1–7.2.

Still open:
- confirmation on fresh conditions;
- writer variants;
- consolidation of a persistent change into the long memory, and a second cycle;
- transfer to LIBERO-10.

The full history of designs, experiments, negative results and lessons is in **[DEVLOG.md](DEVLOG.md)**.

## Repository layout

| path | contents |
|---|---|
| `horsea/energy.py`, `energy_eval.py`, `energy_diag.py`, `phi.py` | energy-based Horsea: model, meta-training, closed-loop evaluation, diagnostics, action encoder |
| `horsea/selfplay.py` | hidden-shift self-rollouts with verified corrective targets |
| `horsea/memory.py`, `writer.py`, `r0_eval.py` | fast-memory baselines (TTT2, F-write, residual, KV) and the learned velocity writer |
| `horsea/history2.py`, `cycles_ttt.py` | RoboTTT-style LIBERO-10 experiments and consolidation cycles |
| `horsea/cycles.py`, `lifecycle_v2.py`, `final_table.py` | short → long memory lifecycle with demo-written memory |
| `horsea/base.py`, `features.py`, `rollout.py`, `manifest.py`, `starts.py` | base policy wrapper, feature precomputation, LIBERO rollouts, protocol manifest |
| `scripts/queue.py` | job queue (priorities, dependencies, resource caps) |
| `research/` | related-work syntheses and design notes |
| `DEVLOG.md`, `EXPERIMENTS.md` | development log and the original experiment protocol |

## Requirements

- The `imitation` codebase (base policy `fm_policy_S`, LIBERO runner) and LIBERO, with its datasets.
  Paths are configured in `.libero/config.yaml` and `horsea/paths.py`.
- A base policy trained on 80 LIBERO-90 tasks (`scripts/train_base.py`), then precomputed features
  (`python -m horsea.features --suite libero_90`).
- `experiments/` (features, checkpoints, results) is not tracked; it is regenerated by the scripts.

Example (energy Horsea on the hidden-rotation testbed):

```
python -m horsea.selfplay collect --split writer_train --shifts none rot20 rot-20 rot40 rot-40 rot60 rot-60 --attempts 2
python -m horsea.phi train
python -m horsea.energy train --solve final --n_iter 3 --structured --max_step 0.05 \
    --train_shifts none rot20 rot-20 rot40 rot-40 rot60 rot-60 --out experiments/energy/horsea
python -m horsea.energy_eval --ckpt experiments/energy/horsea/last.pt --cond write --max_step 0.2 \
    --shifts none rot30 rot-30 rot50 rot-50 --out experiments/eval/horsea
```
