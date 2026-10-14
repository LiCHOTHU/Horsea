# Control diagnostics: data tables (generated 2026-10-14 01:13; 96 episodes)

## Success per executor

| executor | all | nominal grasps | perturbed (pose errors, imperfect contacts) | safety violations |
|---|---|---|---|---|
| C0 | 14/24 | 6/12 | 8/12 | 0 |
| C1 | 14/24 | 6/12 | 8/12 | 0 |
| C2 | 0/24 | 0/12 | 0/12 | 0 |
| C2b | 13/24 | 5/12 | 8/12 | 0 |

## Per condition (2 repeats each): successes, diagnosed category of every failure

| condition | object | kind | C0 | C1 | C2 (regrasps) | C2b (regrasps) |
|---|---|---|---|---|---|---|
| A_nom | cup_a | nominal grasp | 2/2 | 2/2 | 0/2 - grasp unsuitable, contact control ([2, 2]) | 2/2 ([0, 0]) |
| C_nom | cyl_c | nominal grasp | 2/2 | 2/2 | 0/2 - contact control, contact control ([2, 2]) | 1/2 - grasp unsuitable ([0, 2]) |
| B_nom | cup_b | nominal grasp | 0/2 - grasp unsuitable, grasp unsuitable | 0/2 - grasp unsuitable, grasp unsuitable | 0/2 - grasp unsuitable, grasp unsuitable ([2, 2]) | 0/2 - grasp unsuitable, grasp unsuitable ([2, 2]) |
| g1_nom | dev_g1 | nominal grasp | 0/2 - grasp unsuitable, grasp unsuitable | 0/2 - grasp unsuitable, grasp unsuitable | 0/2 - grasp unsuitable, grasp unsuitable ([2, 2]) | 0/2 - grasp unsuitable, grasp unsuitable ([2, 2]) |
| g4_nom | dev_g4 | nominal grasp | 0/2 - grasp unsuitable, grasp unsuitable | 0/2 - grasp unsuitable, grasp unsuitable | 0/2 - grasp unsuitable, grasp unsuitable ([2, 2]) | 0/2 - grasp unsuitable, grasp unsuitable ([2, 2]) |
| g2_nom | dev_g2 | nominal grasp | 2/2 | 2/2 | 0/2 - contact control, contact control ([2, 2]) | 2/2 ([0, 0]) |
| A_x+1 | cup_a | pose error | 0/2 - grasp unsuitable, grasp unsuitable | 0/2 - grasp unsuitable, grasp unsuitable | 0/2 - grasp unsuitable, grasp unsuitable ([2, 2]) | 0/2 - grasp unsuitable, grasp unsuitable ([2, 2]) |
| A_y-1 | cup_a | pose error | 0/2 - grasp unsuitable, grasp unsuitable | 0/2 - grasp unsuitable, grasp unsuitable | 0/2 - grasp unsuitable, grasp unsuitable ([2, 2]) | 0/2 - grasp unsuitable, grasp unsuitable ([2, 2]) |
| C_z+1 | cyl_c | pose error | 2/2 | 2/2 | 0/2 - contact control, contact control ([2, 2]) | 2/2 ([0, 0]) |
| g3_yaw12 | dev_g3 | pose error (orientation) | 2/2 | 2/2 | 0/2 - contact control, contact control ([2, 2]) | 2/2 ([0, 0]) |
| A_touch | cup_a | imperfect initial contact | 2/2 | 2/2 | 0/2 - contact control, contact control ([2, 1]) | 2/2 ([0, 0]) |
| C_touchlow | cyl_c | imperfect initial contact | 2/2 | 2/2 | 0/2 - contact control, contact control ([2, 2]) | 2/2 ([0, 0]) |

## Failure categories and object behaviour

- C0: {'success': 14, 'grasp unsuitable': 10}; object: {'followed the hand': 14, 'slipped': 4, 'remained supported (never left the table)': 6}
- C1: {'success': 14, 'grasp unsuitable': 10}; object: {'followed the hand': 14, 'slipped': 5, 'remained supported (never left the table)': 5}
- C2: {'grasp unsuitable': 11, 'contact control': 13}; object: {'slipped': 22, 'remained supported (never left the table)': 2}
- C2b: {'success': 13, 'grasp unsuitable': 11}; object: {'followed the hand': 13, 'slipped': 10, 'remained supported (never left the table)': 1}

## Tracking (wrist distance to the commanded reference at decision ends) and closures

| executor | end of approach: median / max cm | end of closing (in contact): median / max cm | measured closures at the end of closing (mean) |
|---|---|---|---|
| C0 | 0.102 / 0.54 | 0.517 / 1.956 | [0.274, 0.438, 0.502, 0.202] |
| C1 | 0.05 / 0.536 | 0.489 / 1.971 | [0.273, 0.44, 0.504, 0.207] |
| C2 | 0.102 / 0.536 | 0.62 / 1.964 | [0.273, 0.44, 0.502, 0.202] |
| C2b | 0.102 / 0.54 | 0.647 / 1.956 | [0.278, 0.439, 0.501, 0.195] |

## C2b's loop (the recalibrated trigger): {"episodes_with_regrasp": 11, "regrasps": 22, "rescued": 0, "rules_fired": {"thumb above the top / over the opening": 20, "default": 1, "fingers 1.39 cm from the wall": 1}, "statuses": {"program": 13, "slip after the last regrasp": 11}}

## Paired (same condition and start): {"C1-C0": {"a_better": 0, "b_better": 0, "pairs": 24}, "C2-C0": {"a_better": 0, "b_better": 14, "pairs": 24}, "C2b-C0": {"a_better": 0, "b_better": 1, "pairs": 24}, "C2b-C1": {"a_better": 0, "b_better": 1, "pairs": 24}}

## Gate: {"c0_failures": 10, "c0_failure_categories": {"grasp unsuitable": 10}, "fixed_by_C1": 0, "fixed_by_C2": 0, "fixed_by_C2b": 0, "nominal_success": {"C0": "6/12", "C1": "6/12", "C2": "0/12", "C2b": "5/12"}, "decision": "the chosen grasp itself is unsuitable in most failures: improve proposal selection before motor-policy training"}
