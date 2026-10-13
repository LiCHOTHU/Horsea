# Horsea

**Fast and slow memory for robot policies.** Since 2026-10-09 we study this beneath a language-model explorer:
GPT-6 Astra acts as **System 2** (it observes, acts, remembers outcomes and retries), and the question is what a learned
**System 1** with short-term memory should do beneath it.

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

## Phases

| phase | dates (2026) | question | result |
|---|---|---|---|
| Energy-based Horsea on LIBERO | 09-23 → 09-29 | Can a learned writer turn the robot's own attempts into better next actions? | Yes on hidden command rotations (+15.4 points over memory off); not on LIBERO-10. LIBERO was removed on 09-30. |
| RoboTwin 2.0 base policy, internal loops | 09-30 → 10-02 | Do trained internal Transformer loops help a flow-matching policy? | No (confirmation: S3 − N = −0.2 [−4.8, +4.2]). |
| Astra explores scenes directly | 10-09 → 10-10 | How far does an LLM get by observing, acting, remembering and retrying? | LIBERO 2/3; RoboTwin 3/3 single-arm, 1/3 bimanual; dex hand 2/10. |
| Astra → student transfer | 10-10 | Can a policy learn from Astra's interaction data overnight? | No working channel. |
| **System 1 beneath Astra** | 10-10 → now | What should a learned System 1 under Astra do? | See below. |

## Where it stands (2026-10-12)

Platform: the EgoVLA Ego-Humanoid benchmark in Isaac Sim 5.1, a Unitree H1 with two Inspire dexterous hands.

- **Local search around Astra's action program is not a useful System 1.** It rescued 0 of 66 matched blocks on 4
  tasks. What fails needs closed-loop completion ("push until the lid is open") or re-acquisition of the object.
- **Grasp failures:** the robot reaches what Astra commands (median 0.3 cm).
  - On the dex hand, 65 % of failed grasps close on air: Astra aimed wrong.
  - 35 % touch the object and lose it.
- **Cup A → cup B → cylinder C.** Astra learns on cup A; then five methods face a wider, shorter cup B and a narrower,
  taller cylinder C (2 starts each, at most 2 attempts):

| method | cylinder C, first try | cup B, first try | cup B, within 2 tries |
|---|---|---|---|
| Astra with its own history | 2/2 | 1/2 | 2/2 |
| nearest-neighbour replay | 2/2 | 2/2 | 2/2 |
| System 1, empty memory | 2/2 | 0/2 | 0/2 |
| System 1, cup A memory | 2/2 | 1/2 | 2/2 |
| System 1, cup A memory + online updates | 2/2 | 0/2 | 0/2 |

  The memory-based System 1 (experience store, locally weighted linear model, short-horizon planner) kept Astra's
  command on 743 of 745 segments. The differences on cup B are therefore Astra's own variability.
  - Cup B failed in the grip. Astra re-used cup A's grasp and the hand reached it within 0.1 cm. It touched cup B, and
    the cup slipped out when lifted.
  - A 0.2–1.2 cm shift held, which is within System 1's bounds.
  - System 1 had no reason to deviate: its memory held three identical successful grasps, and its goals defended
    Astra's command.

**Conclusion so far:**
- System 1 has a job in contact (grip, slip, regrasp), not in aiming.
- To do that job it needs its own success checks: the object is drawn into the hand while closing, and it follows the
  hand at lift-off.
- It also needs practice data with variation and failures. Copying Astra's successes only teaches it to agree with
  Astra.

**Next (proposed):** a scripted grasp sweep (no Astra calls), then cup B with Astra from at least 5 starts.
Design note: [research/system1_dexhand_design.md](research/system1_dexhand_design.md).

The full history of designs, experiments, negative results and lessons is in **[DEVLOG.md](DEVLOG.md)**
(newest status at the top). Every result also has a page on the results website (see below).

## The LIBERO result: energy-based Horsea

- **Long memory:** a frozen flow-matching policy proposes an action chunk.
- **Short memory:** the fast weights of a small energy network E_W(action, observation, instruction), i.e. a learned
  cost over candidate actions. An empty memory has zero energy, so it is exactly the base policy.
- **Learned writer:** after each attempt, a meta-trained writer turns the robot's own completed interactions (what it
  saw, what it commanded, what actually happened) into an update of W. No labels, rewards or success flags are used at
  deployment.
- **Acting:** a bounded solver moves the proposed action toward lower energy before it is executed.

Testbed: LIBERO-90 tasks with a **hidden rotation of the robot's commands**, on fresh rotations and starts. 3 seeds,
210 sequences, mean success over attempts 2–5:

| method | success | Horsea minus this [95% CI] |
|---|---|---|
| **Horsea, own experience written to memory** | **36.4%** | – |
| Horsea, memory off | 21.1% | +15.4 [+11.9, +18.9] |
| Horsea, shuffled action–outcome pairing | 23.2% | +13.2 [+9.8, +16.7] |
| Horsea, latest interaction only | 23.3% | +13.1 [+9.6, +16.7] |
| no-history model, same supervision | 22.0% | +14.4 [+11.0, +18.0] |
| TTT2 (RoboTTT-style fast weights), same data | 30.6% | +5.8 [+1.9, +9.9] |

The offline targets are oracle corrections and adaptation happens between attempts. The method did not carry over to
LIBERO-10 (DEVLOG §7). LIBERO has since been removed, so this code cannot be rerun as is.

## Repository layout

| path | contents |
|---|---|
| `astra_explore/` | Astra harness: Codex CLI wrapper (`codex_astra.py`), Landlock sandbox, environments (`libero_env.py`, `robotwin_env.py`, `isaac_env.py`), per-platform runners, schemas and reports, `tests/` |
| `astra_explore/s1s2/` | overnight Astra → student transfer study |
| `astra_explore/s1local/` | System 1 local-adaptation pilots, grasp-failure diagnosis, controlled grasp test |
| `astra_explore/cup_transfer/` | cup A → B → C study: assets, task environment, System 1, prompt, runner, analysis, per-grasp check |
| `astra_explore/viz/` | results website: static site builder and server launcher |
| `horsea/rt/` | RoboTwin port: flow-matching policy, data, training, internal-loop graphs and analyses, deploy wrapper |
| `horsea/energy.py`, `energy_eval.py`, `energy_diag.py`, `phi.py`, `selfplay.py` | energy-based Horsea (LIBERO era) |
| `horsea/memory.py`, `writer.py`, `history2.py`, `cycles*.py`, `lifecycle_v2.py` | fast-memory baselines, learned writers, RoboTTT-style comparison, short → long lifecycle (LIBERO era) |
| `scripts/queue.py` | job queue (priorities, dependencies, resource caps) |
| `research/` | related-work syntheses and design notes, including the System 1 design |
| `DEVLOG.md`, `EXPERIMENTS.md` | development log and the original LIBERO protocol |

`experiments/` (data, checkpoints, runs) is not tracked, except for a few reports.

## Requirements

- **Astra:** the Codex CLI with access to `gpt-6-astra` (saved login). Each call runs in a Landlock sandbox with tools
  disabled.
- **Dex hand:** the EgoVLA release (`~/workspace/egovla_release`) with Isaac Sim 5.1 / IsaacLab 2.3, conda env `hongyi`
  (`source ~/workspace/hongyi_deps/activate_hongyi.sh`). One simulator needs about 10.5 GB of GPU memory.
- **RoboTwin 2.0:** `~/workspace/RoboTwin`, conda env `robotwin`. SAPIEN rendering needs the desktop's X display
  (`DISPLAY=:0`).
- **LIBERO-era code:** needs LIBERO and the `imitation` codebase, which were removed on 2026-09-30.

## Running

Cup transfer study (Isaac; results under `experiments/cup_transfer_2026-10-12/`):

```
source ~/workspace/hongyi_deps/activate_hongyi.sh
cd ~/workspace/egovla_release/egovla
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=~/workspace/Horsea/astra_explore:$PYTHONPATH
python -m cup_transfer.run_cup stage0 --obj cup_b          # reset, limits and feasibility checks per object
python -m cup_transfer.run_cup source --obj cup_a          # three Astra attempts on cup A
python -m cup_transfer.run_cup target --obj cup_b          # starts x methods x at most 2 attempts (resumable)
cd ~/workspace/Horsea/astra_explore
python -m cup_transfer.analyze_cup --run run               # results.md, S(b) curves, per-case chains
python -m cup_transfer.grasp_check --run run               # per-grasp geometry and outcome
```

`--policy smoke` replaces Astra with a scripted grasp for pipeline tests. `experiments/cup_transfer_2026-10-12/cup_job.sh`
wraps a stage and waits for free GPU memory first, because the GPU is shared.

Results website, on the workstation:

```
cd astra_explore/viz && python3 launcher.py restart    # rebuild the pages and serve them on 127.0.0.1:8765
```

From a laptop: `ssh -L 8765:127.0.0.1:8765 licho@130.207.121.116`, then open http://localhost:8765/. Series A covers
Astra's runs, B the transfer study, and C the System 1 work. See [astra_explore/viz/README.md](astra_explore/viz/README.md).
