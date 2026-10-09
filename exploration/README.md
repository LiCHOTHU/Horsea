# Exploration

An independent RoboTwin project on branch `exploration`, implementing the supplied
prior-guided exploration plan. The motor repertoire is fixed; a small categorical
memory chooses which grasp and manipulation option to try. Horsea's neural motor
policy, energy model, and neural memory are not part of this pilot.

## Current experiment

This is an **instrumented-scene diagnostic**, using the visible markers approved
by the user. It varies the placement of one microwave asset (`044_microwave/7167`),
so it does not test unseen objects or mechanisms. Colored, non-colliding render
markers are attached during scene construction. Calibrated RGB-D cameras are
shared by every method. The latest scene has the head camera, two side views,
and one lower front view. The four-view training recording passed its camera
check; staged execution is undergoing the next batch smoke check.

Online feedback uses color/depth tracking and measured robot proprioception.
RoboTwin's cached gripper command is deliberately avoided: finger joint positions
provide the measured opening. The reported EEF pose is transformed to the tool
center with the fixed robot calibration. No object joint, contact, segmentation,
or simulator success flag enters the feedback extractor, memory, or selector.
Privileged scene setup and evaluation are explicitly isolated.

The low-level executor replays robot waypoints recorded by the task expert on a
training scene, translated by the observed housing markers. Grasp 0 uses the
recorded approach; grasp 1 tilts it 30 degrees around the robot tool point. Modes
are the demonstrated opening arc, a tangent pull, and a vertical lift. These are
explicit executable options, not untrained neural mode tokens. A shared 2.5 cm
verification pull follows the grasp. A tail runs only after an observed `ready`.
Every physical attempt constructs a fresh simulator world at the canonical scene
seed and uses fresh controller/planner randomness paired across comparison arms.

Local H200 checks on allocation `13797429` established successful staged execution
on training scene 210000 and development scene 220000. The latter physically
opened the door but returned `unknown` when the three-view tracker lost a marker.
This motivated the fourth view; its result is pending, not assumed successful.
On the saved three-view training sequence, opening-angle MAE was 0.0126 radians
over 23 frames. The four-view training recording reduced this to 0.0068 radians
with no missing frames. These training checks are not independent feedback validation.
The first four-view staged smoke exposed delayed physical finger closure during
verification. Readiness now uses measured closure at the end of that motion;
the saved observation trace and a regression test validate the correction.

Allocation `13797429` subsequently expired. Submitted jobs:

| Job | Purpose | Submission state |
| --- | --- | --- |
| 13798384 | Corrected staged smoke using the preserved four-view training reference | Submitted |
| 13798392 | 10-scene development array, six options × three repeats = 180 attempts | Depends on smoke passing; at most two GPUs |

Both use the frozen source snapshot and manifest in
`experiments/exploration/pilot_20261001_markers_v4/`. Job state is a submission-time
record; use Slurm for live status. The original three-view observations, failed
approach tests, and evaluation labels remain under `experiments/exploration/`.
The existing `run_13795925` results use the **synthetic backend**, not robots.
The earlier smoke 13798174 and its observations are preserved; its blocked array
13798191 was canceled and replaced by the jobs above.

## Components and invariants

| File | Responsibility |
| --- | --- |
| `events.py`, `feedback.py` | Four prefix events, six tail events, goal-first classification, explicit unknowns |
| `memory.py` | 44 structured Dirichlet parameters; scalar marginals matched to smoothed structured priors |
| `selectors.py` | Greedy, shared-prefix Thompson sampling, exact nine-branch one-observation VOI |
| `vision.py`, `reference.py` | Camera-only marker tracking; actual finger and tool proprioception; training skill recording |
| `instrumentation.py` | Shared scene cameras and visible markers; fixed asset construction |
| `robotwin_backend.py` | Common option executor; prefix has no mode argument |
| `trials.py` | Fresh resets, paired streams, three probes plus separate exploitation, shared-history readers |
| `evaluation.py`, `development_audit.py` | Segregated physical truth and capability/perception audit |
| `metrics.py` | Actual exploitation success, observed goal proxy, paired configuration bootstrap, costs and calibration |
| `gates.py` | Require a matching frozen development audit before real confirmation runs |
| `synthetic.py`, `calibrate.py`, `traces.py` | Explicitly synthetic mechanism checks, not robotic evidence |

VOI enumerates hypothetical observations on copies and never executes them.
Prefix failures update only the relevant grasp row. Ready tails update both the
ready count and the selected mode's tail row. Unknowns have their own categories.
All-zero VOI falls back deterministically to posterior-mean choice. Repeated
one-step VOI is a myopic policy, not optimal multi-step planning.

`observations.jsonl` holds decisions, events, memory snapshots, commands, tracked
frames, and paths to raw RGB-D/proprioceptive `.npz` files. `evaluation_only.jsonl`
holds actual success, object joints, and audit contacts. Contact presence is an
imperfect audit proxy for readiness. Metrics report simulator success separately
from the visual goal; simulated failures and missing measurements are retained.

Physical steps are counted separately from waypoint-command budgets (16 prefix,
24 tail). Capture occurs at command boundaries, so brief slips between captures
can be missed. Tracker confidence is a deterministic validity indicator, not a
calibrated probability. Marker dimensions and temporal consistency reject false
color correspondences; missing reset visibility never establishes a new closed
reference midway through an attempt.

## Run and inspect

From the repository root, with the provided RoboTwin environment:

```bash
bash scripts/exploration_gpu.sh -m unittest discover -s tests/exploration -v
bash scripts/exploration_gpu.sh -m exploration.traces --out experiments/exploration/mechanism_traces
```

The four JSON traces expose prefix-only failure updates, an unhelpful ready tail,
an unknown observation, and an informative lower-immediate-reward VOI choice
with zero VOI for a single final option. There are 25 automated tests.

For a local GPU development sweep with a successful training reference:

```bash
bash scripts/exploration_gpu.sh -m exploration.run sweep \
  --backend robotwin --split dev --scenes 220000 --replicates 0 \
  --reference /absolute/path/reference_retracked.json --out /absolute/path/sweep
bash scripts/exploration_gpu.sh -m exploration.development_audit --root /absolute/path/sweep
```

The queued development array writes separate scene directories under the pilot's
`development/` directory. After all shards finish, aggregate with the **same
frozen snapshot**:

```bash
EXPLORATION_SOURCE_ROOT="$PWD/experiments/exploration/pilot_20261001_markers_v4/snapshot" \
  bash scripts/exploration_gpu.sh -m exploration.development_audit \
  --root "$PWD/experiments/exploration/pilot_20261001_markers_v4/development"
```

For the active or common-history experiments, replace `sweep` with `active` or
`common`, and supply `--prior TRAIN_OBSERVATIONS.jsonl` when available. Without a
training prior the driver uses uniform concentration-one priors. Prior fitting
rejects development and confirmation scene IDs. Scalar concentration supports
the declared development search `{1,3,6}`. The generic batch driver runs one
explicit stage; it does not automatically collect the full training budget.

```bash
sbatch scripts/exploration_run.sbatch robotwin dev active REFERENCE_JSON TRAIN_JSONL
```

Real confirmation additionally requires `--gate development_audit.json` matching
the reference hash, implementation hash, thresholds, and all ten development
scenes. Exhaustive confirmation sweeps are forbidden. Freeze the scalar strength
after its development search before confirmation; do not select it on final results.

## Remaining research stages

The predeclared splits are 30 training scenes (210000–210029), 10 development
scenes (220000–220009), and 30 confirmation scenes (230000–230029). The proposed
full budgets are 540 training attempts, 1,800 active-history confirmation attempts,
and 180 separately budgeted common-history attempts. **These full budgets have
not been launched.** Capability and independent feedback checks come first.

The five active methods are prior-only, scalar VOI, structured greedy, structured
Thompson, and structured VOI. All receive identical sensors, controller options,
reset rules, and attempt budgets. Confirmation scenes cannot be discarded after
seeing results. Structured VOI must improve actual fresh exploitation success over
both prior-only and structured greedy; a common-history benefit alone establishes
evidence use, not better active exploration. No exploration advantage has yet been
demonstrated by the robot experiments.
