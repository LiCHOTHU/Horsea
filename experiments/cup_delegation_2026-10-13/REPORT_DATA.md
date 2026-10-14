# Delegation study: data tables (generated 2026-10-13 21:15)

## Experiment A, REPLAY, pre-registered settings (h = 1, geometry representation), 64 source interactions

first / within 2 / within 3 attempts, scenes x 3 selections (A0: scenes)

| method | primary (A, B, C) | additional (dev family, own records removed) | nominal-failing only, primary | nominal-failing only, additional |
|---|---|---|---|---|
| A0 nominal | 6 / 6 / 6 of 16 | 8 / 8 / 8 of 16 | 0 / 0 / 0 of 10 | 0 / 0 / 0 of 8 |
| A1 one correction | 43 / 43 / 43 of 48 | 34 / 34 / 34 of 48 | 30 / 30 / 30 of 30 | 17 / 17 / 17 of 24 |
| A2 ranked search | 43 / 48 / 48 of 48 | 34 / 40 / 43 of 48 | 30 / 30 / 30 of 30 | 17 / 20 / 20 of 24 |
| A3 context-free updating | 43 / 47 / 48 of 48 | 34 / 41 / 45 of 48 | 30 / 30 / 30 of 30 | 17 / 19 / 23 of 24 |
| A4 contextual frozen | 45 / 48 / 48 of 48 | 28 / 33 / 36 of 48 | 30 / 30 / 30 of 30 | 12 / 14 / 14 of 24 |
| A5 contextual updating | 45 / 48 / 48 of 48 | 28 / 43 / 45 of 48 | 30 / 30 / 30 of 30 | 12 / 20 / 21 of 24 |

Paired (scene x selection; a better / b better / same):

| comparison | question | primary, first | primary, within 3 | additional, first | additional, within 3 |
|---|---|---|---|---|---|
| A1-A0 | Was the original grasp simply poorly calibrated? | 30 / 5 / 13 | 30 / 5 / 13 | 17 / 7 / 24 | 17 / 7 / 24 |
| A2-A1 | Does trying a short list provide sufficient adaptation? | 0 / 0 / 48 | 5 / 0 / 43 | 0 / 0 / 48 | 9 / 0 / 39 |
| A4-A2 | Does conditioning on the current state improve selection? | 3 / 1 / 44 | 0 / 0 / 48 | 5 / 11 / 32 | 2 / 9 / 37 |
| A3-A2 | Does feedback-driven updating help beyond moving down a list? | 0 / 0 / 48 | 0 / 0 / 48 | 0 / 0 / 48 | 3 / 1 / 44 |
| A5-A4 | Does updating the contextual model help? | 0 / 0 / 48 | 0 / 0 / 48 | 0 / 0 / 48 | 9 / 0 / 39 |
| A5-A3 | Does context improve how feedback is generalized? | 3 / 1 / 44 | 0 / 0 / 48 | 5 / 11 / 32 | 2 / 2 / 44 |

## Settings grid, REPLAY (all 32 development scenes x 3 selections = 96): first / within 3

| method | representation | h | 16 | 64 | 256 |
|---|---|---|---|---|---|
| A1 | none | 0.5 | 34 / 34 | 60 / 60 | 63 / 63 |
| A1 | none | 1.0 | 44 / 44 | 77 / 77 | 67 / 67 |
| A1 | none | 2.0 | 71 / 71 | 84 / 84 | 84 / 84 |
| A1 | none | 4.0 | 74 / 74 | 83 / 83 | 84 / 84 |
| A2 | none | 0.5 | 34 / 70 | 60 / 87 | 63 / 87 |
| A2 | none | 1.0 | 44 / 75 | 77 / 91 | 67 / 91 |
| A2 | none | 2.0 | 71 / 86 | 84 / 93 | 84 / 93 |
| A2 | none | 4.0 | 74 / 91 | 83 / 93 | 84 / 92 |
| A3 | none | 0.5 | 34 / 71 | 60 / 87 | 63 / 87 |
| A3 | none | 1.0 | 44 / 78 | 77 / 93 | 67 / 92 |
| A3 | none | 2.0 | 71 / 87 | 84 / 92 | 84 / 93 |
| A3 | none | 4.0 | 74 / 85 | 83 / 92 | 84 / 92 |
| A4 | current | 0.5 | 58 / 68 | 57 / 78 | 79 / 85 |
| A4 | current | 1.0 | 57 / 75 | 83 / 92 | 89 / 93 |
| A4 | current | 2.0 | 71 / 86 | 86 / 94 | 88 / 91 |
| A4 | current | 4.0 | 74 / 93 | 85 / 92 | 87 / 91 |
| A4 | geometry | 0.5 | 40 / 45 | 41 / 45 | 75 / 77 |
| A4 | geometry | 1.0 | 51 / 77 | 73 / 84 | 91 / 96 |
| A4 | geometry | 2.0 | 66 / 87 | 81 / 93 | 89 / 95 |
| A4 | geometry | 4.0 | 74 / 94 | 84 / 92 | 87 / 92 |
| A4 | post_geometry | 0.5 | 42 / 46 | 42 / 46 | 74 / 80 |
| A4 | post_geometry | 1.0 | 50 / 71 | 72 / 91 | 90 / 95 |
| A4 | post_geometry | 2.0 | 62 / 85 | 83 / 94 | 89 / 95 |
| A4 | post_geometry | 4.0 | 73 / 93 | 84 / 92 | 84 / 94 |
| A5 | current | 0.5 | 58 / 76 | 57 / 86 | 79 / 90 |
| A5 | current | 1.0 | 57 / 88 | 83 / 93 | 89 / 95 |
| A5 | current | 2.0 | 71 / 85 | 86 / 93 | 88 / 92 |
| A5 | current | 4.0 | 74 / 85 | 85 / 93 | 87 / 92 |
| A5 | geometry | 0.5 | 40 / 84 | 41 / 84 | 75 / 84 |
| A5 | geometry | 1.0 | 51 / 73 | 73 / 93 | 91 / 96 |
| A5 | geometry | 2.0 | 66 / 83 | 81 / 91 | 89 / 95 |
| A5 | geometry | 4.0 | 74 / 87 | 84 / 93 | 87 / 94 |
| A5 | post_geometry | 0.5 | 42 / 56 | 42 / 57 | 74 / 80 |
| A5 | post_geometry | 1.0 | 50 / 73 | 72 / 92 | 90 / 94 |
| A5 | post_geometry | 2.0 | 62 / 86 | 83 / 94 | 89 / 95 |
| A5 | post_geometry | 4.0 | 73 / 86 | 84 / 92 | 84 / 95 |

## Experiment B, REPLAY at the chosen settings (context-free h = 2.0; contextual current, h = 2.0): first / within 2 / within 3 of 96

| method | 0 | 16 | 64 | 256 |
|---|---|---|---|---|
| A0 nominal | 14 / 14 / 14 of 32 | - | - | - |
| A1 one correction | 42 / 42 / 42 of 96 | 71 / 71 / 71 of 96 | 84 / 84 / 84 of 96 | 84 / 84 / 84 of 96 |
| A2 ranked search | 42 / 45 / 45 of 96 | 71 / 80 / 86 of 96 | 84 / 87 / 93 of 96 | 84 / 88 / 93 of 96 |
| A3 context-free updating | 42 / 45 / 48 of 96 | 71 / 83 / 87 of 96 | 84 / 87 / 92 of 96 | 84 / 88 / 93 of 96 |
| A4 contextual frozen | 42 / 45 / 45 of 96 | 71 / 79 / 86 of 96 | 86 / 90 / 94 of 96 | 88 / 90 / 91 of 96 |
| A5 contextual updating | 42 / 45 / 48 of 96 | 71 / 80 / 85 of 96 | 86 / 91 / 93 of 96 | 88 / 90 / 92 of 96 |

Per selection (within 3 of 32), to show the spread:

| method | budget | selection 0 | selection 1 | selection 2 |
|---|---|---|---|---|
| A2 | 16 | 15 / 23 | 27 / 31 | 29 / 32 |
| A2 | 64 | 28 / 32 | 28 / 30 | 28 / 31 |
| A2 | 256 | 28 / 31 | 28 / 31 | 28 / 31 |
| A4 | 16 | 16 / 23 | 26 / 31 | 29 / 32 |
| A4 | 64 | 28 / 32 | 30 / 30 | 28 / 32 |
| A4 | 256 | 31 / 31 | 29 / 30 | 28 / 30 |
| A5 | 16 | 16 / 23 | 26 / 30 | 29 / 32 |
| A5 | 64 | 28 / 31 | 30 / 30 | 28 / 32 |
| A5 | 256 | 31 / 32 | 29 / 30 | 28 / 30 |

Budget rule: {"E2_family": {"within_3": {"0": 45, "16": 86, "64": 93}, "chosen_budget": 64}, "E3": {"within_3": {"0": 48, "16": 85, "64": 93}, "chosen_budget": 64}}

First picks at 64 interactions (count over scenes x selections) and decision latency:

| method | first picks | median / max decision ms |
|---|---|---|
| A1 | h25 62, h19 34 | 1.09 / 1.69 |
| A2 | h25 62, h19 34 | 1.10 / 1.68 |
| A3 | h25 62, h19 34 | 1.10 / 1.61 |
| A4 | h25 52, h19 37, h07 5, h04 2 | 1.45 / 2.34 |
| A5 | h25 52, h19 37, h07 5, h04 2 | 1.49 / 2.45 |

## Experiment D, REPLAY, representations at the pre-registered h = 1, 64 interactions

| method | representation | primary | additional |
|---|---|---|---|
| A4 | current | 45 / 48 / 48 of 48 | 38 / 43 / 44 of 48 |
| A4 | geometry | 45 / 48 / 48 of 48 | 28 / 33 / 36 of 48 |
| A4 | post_geometry | 44 / 46 / 48 of 48 | 28 / 40 / 43 of 48 |
| A5 | current | 45 / 48 / 48 of 48 | 38 / 44 / 45 of 48 |
| A5 | geometry | 45 / 48 / 48 of 48 | 28 / 43 / 45 of 48 |
| A5 | post_geometry | 44 / 45 / 48 of 48 | 28 / 41 / 44 of 48 |

## Experiment C, conservative weight from development data: NOISY REPLAY (10 draws per scene and selection)

| update | retry rule | first | within 2 | within 3 | of |
|---|---|---|---|---|---|
| frozen | confirm_once | 730 | 730 | 810 | 960 |
| frozen | no_repeat | 730 | 810 | 840 | 960 |
| w0.1 | confirm_once | 730 | 850 | 890 | 960 |
| w0.1 | no_repeat | 730 | 850 | 900 | 960 |
| w0.25 | confirm_once | 730 | 900 | 930 | 960 |
| w0.25 | no_repeat | 730 | 900 | 930 | 960 |
| w0.5 | confirm_once | 730 | 900 | 920 | 960 |
| w0.5 | no_repeat | 730 | 900 | 920 | 960 |
| w1.0 | confirm_once | 730 | 910 | 930 | 960 |
| w1.0 | no_repeat | 730 | 910 | 930 | 960 |

Chosen conservative weight: 0.25 (conservative weight = the value in {0.1, 0.25, 0.5} with the most successes within 3 attempts under confirm_once, pooled over development scenes, selections and draws; ties (within 0.5 % of the scenes) go to 0.25)

## Experiment A, FRESH EXECUTIONS at the chosen settings, 64 interactions (22 scenes, 352 runs, 412 rollouts)

| method | all scenes x selections | nominal-failing scenes only |
|---|---|---|
| A0 nominal | 10 / 12 / 12 of 22 | 0 / 2 / 2 of 12 |
| A1 one correction | 58 / 63 / 64 of 66 | 31 / 34 / 35 of 36 |
| A2 ranked search | 63 / 65 / 65 of 66 | 34 / 35 / 35 of 36 |
| A3 context-free updating | 62 / 65 / 66 of 66 | 32 / 35 / 36 of 36 |
| A4 contextual frozen | 58 / 65 / 65 of 66 | 28 / 35 / 35 of 36 |
| A5 contextual updating | 59 / 64 / 66 of 66 | 30 / 34 / 36 of 36 |

| comparison | first: a better / b better / same | within 3 |
|---|---|---|
| A1-A0 | 31 / 3 / 32 | 29 / 1 / 36 |
| A2-A1 | 6 / 1 / 59 | 1 / 0 / 65 |
| A4-A2 | 1 / 6 / 59 | 1 / 1 / 64 |
| A3-A2 | 1 / 2 / 63 | 1 / 0 / 65 |
| A5-A4 | 3 / 2 / 61 | 1 / 0 / 65 |
| A5-A3 | 1 / 4 / 61 | 0 / 0 / 66 |

Per object (first / within 3 of runs):

| method | cup_a | cup_b | cyl_c | dev_f1 | dev_f2 | dev_f3 | dev_f4 | dev_g1 | dev_g2 | dev_g3 | dev_g4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 2/2 of 2 | 0/0 of 2 | 1/2 of 2 | 0/0 of 2 | 2/2 of 2 | 0/0 of 2 | 1/2 of 2 | 0/0 of 2 | 2/2 of 2 | 2/2 of 2 | 0/0 of 2 |
| A1 | 6/6 of 6 | 6/6 of 6 | 3/5 of 6 | 3/5 of 6 | 5/6 of 6 | 6/6 of 6 | 6/6 of 6 | 6/6 of 6 | 5/6 of 6 | 6/6 of 6 | 6/6 of 6 |
| A2 | 6/6 of 6 | 6/6 of 6 | 6/6 of 6 | 4/5 of 6 | 6/6 of 6 | 6/6 of 6 | 6/6 of 6 | 6/6 of 6 | 5/6 of 6 | 6/6 of 6 | 6/6 of 6 |
| A3 | 6/6 of 6 | 6/6 of 6 | 6/6 of 6 | 2/6 of 6 | 6/6 of 6 | 6/6 of 6 | 6/6 of 6 | 6/6 of 6 | 6/6 of 6 | 6/6 of 6 | 6/6 of 6 |
| A4 | 6/6 of 6 | 6/6 of 6 | 5/6 of 6 | 0/5 of 6 | 6/6 of 6 | 6/6 of 6 | 5/6 of 6 | 6/6 of 6 | 6/6 of 6 | 6/6 of 6 | 6/6 of 6 |
| A5 | 6/6 of 6 | 6/6 of 6 | 5/6 of 6 | 1/6 of 6 | 6/6 of 6 | 6/6 of 6 | 5/6 of 6 | 6/6 of 6 | 6/6 of 6 | 6/6 of 6 | 6/6 of 6 |

## Experiment C, FRESH EXECUTIONS (240 runs, 282 rollouts)

| method / update / retry | scenes | first | within 3 | retries (succeeded) | changed after a failure (then held) | abandoned a candidate that held elsewhere at this start |
|---|---|---|---|---|---|---|
| A2|frozen|confirm_once | 24 | 21 | 24 | 3 (2) | 1 (1) | 1 |
| A2|frozen|no_repeat | 24 | 21 | 24 | 0 (0) | 3 (3) | 3 |
| A3|w1.0|confirm_once | 24 | 20 | 24 | 3 (2) | 2 (2) | 2 |
| A3|w1.0|no_repeat | 24 | 19 | 24 | 0 (0) | 5 (5) | 5 |
| A4|frozen|confirm_once | 24 | 21 | 23 | 3 (2) | 1 (0) | 1 |
| A4|frozen|no_repeat | 24 | 19 | 24 | 0 (0) | 5 (5) | 5 |
| A5|w0.25|confirm_once | 24 | 21 | 23 | 1 (1) | 3 (1) | 2 |
| A5|w0.25|no_repeat | 24 | 22 | 23 | 0 (0) | 3 (1) | 3 |
| A5|w1.0|confirm_once | 24 | 21 | 23 | 1 (0) | 3 (2) | 3 |
| A5|w1.0|no_repeat | 24 | 20 | 24 | 0 (0) | 5 (4) | 4 |

Candidates with both outcomes at one start (fresh executions): {"dev_f1|v03|h19": [false, true, true, true, true, true, false, false, false, true, true, false, true, false, false, false, false, true, true, true, true, false, false], "dev_f1|v03|h25": [true, true, true, true, true, false, true, true, true, true, false, true, true], "dev_f1|v04|h19": [false, false, false, false, false, false, true, true, true, true, false, false, false, true, false, false, false, false, false, false, false, false, true, false, true], "dev_f1|v04|h25": [true, true, true, true, true, false, true, true, true, true, true, true, true, false, true, false, true], "dev_f1|v04|h09": [true, false, false], "dev_f4|v03|h25": [true, true, true, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true], "dev_f4|v03|h19": [true, true, true, false, true, false, true, false, true, true, true, true], "dev_f4|v04|h19": [true, true, true, true, true, true, true, true, false, false, true]}

## Experiment D, interface pilot: REPLAY on banks executed at Astra's actual proposals (6 scenes)

| method | first / within 2 / within 3 |
|---|---|
| A0 nominal | 2 / 2 / 2 of 6 |
| A1 one correction | 17 / 17 / 17 of 18 |
| A2 ranked search | 17 / 18 / 18 of 18 |
| A3 context-free updating | 17 / 18 / 18 of 18 |
| A4 contextual frozen | 17 / 18 / 18 of 18 |
| A5 contextual updating | 17 / 18 / 18 of 18 |

Holding corrections per proposal: {"cup_a|v08": ["h00", "h01", "h04", "h05", "h06", "h07", "h09", "h10", "h13", "h14", "h15", "h16", "h19", "h21", "h22", "h24", "h25", "nominal"], "cup_b|v05": ["h01", "h04", "h09", "h10", "h15", "h19", "h24", "h25"], "cup_b|v06": ["h01", "h04", "h10", "h15", "h19", "h24", "h25"], "cyl_c|v07": ["h00", "h01", "h02", "h04", "h05", "h06", "h07", "h11", "h13", "h15", "h16", "h19", "h20", "h21", "h25", "h26", "h27", "h28", "h29", "nominal"], "dev_f1|v09": ["h04", "h09", "h16", "h19", "h24"], "dev_g4|v10": ["h01", "h04", "h07", "h15", "h16", "h19", "h21", "h25", "h29"]}

Replay drift: {"cup_a|v08": {"wrist_cm": 0.0, "object_cm": 0.001, "steps": [55, 55]}, "cup_b|v05": {"wrist_cm": 0.001, "object_cm": 0.0, "steps": [55, 55]}, "cup_b|v06": {"wrist_cm": 0.0, "object_cm": 0.0, "steps": [55, 55]}, "cyl_c|v07": {"wrist_cm": 0.001, "object_cm": 0.0, "steps": [55, 55]}, "dev_f1|v09": {"wrist_cm": 0.001, "object_cm": 0.0, "steps": [55, 55]}, "dev_g4|v10": {"wrist_cm": 0.001, "object_cm": 0.001, "steps": [55, 55]}}

## Extension: changed nominal grasp (reference grasp from above), REPLAY on 11 cups x 3 selections

| method | cup A's program records (mismatched) | same-program records from other cups (matched) |
|---|---|---|
| A0 nominal | 10 / 10 / 10 of 11 | - |
| A1 one correction | 31 / 31 / 31 of 33 | 33 / 33 / 33 of 33 |
| A2 ranked search | 31 / 33 / 33 of 33 | 33 / 33 / 33 of 33 |
| A3 context-free updating | 31 / 33 / 33 of 33 | 33 / 33 / 33 of 33 |
| A4 contextual frozen | 28 / 30 / 30 of 33 | 33 / 33 / 33 of 33 |
| A5 contextual updating | 28 / 33 / 33 of 33 | 33 / 33 / 33 of 33 |

Nominal held per cup: {"cup_a|v11": true, "cup_b|v11": true, "cyl_c|v11": true, "dev_f1|v11": true, "dev_f2|v11": true, "dev_f3|v11": true, "dev_f4|v11": false, "dev_g1|v11": true, "dev_g2|v11": true, "dev_g3|v11": true, "dev_g4|v11": true}

Holding corrections per cup: {"cup_a|v11": 31, "cup_b|v11": 27, "cyl_c|v11": 30, "dev_f1|v11": 30, "dev_f2|v11": 30, "dev_f3|v11": 21, "dev_f4|v11": 17, "dev_g1|v11": 31, "dev_g2|v11": 31, "dev_g3|v11": 30, "dev_g4|v11": 21}

## Extension: initialization curve in FRESH executions (22 fresh development scenes x 3 selections = 66)

| method | 16 | 64 | 256 |
|---|---|---|---|
| A2 ranked search | 44 / 54 / 58 of 66 | 63 / 65 / 65 of 66 | 61 / 64 / 65 of 66 |
| A5 contextual updating | 47 / 58 / 60 of 66 | 59 / 64 / 66 of 66 | 64 / 66 / 66 of 66 |

Per selection (first / within 3 of 22):

| method | budget | selection 0 | selection 1 | selection 2 |
|---|---|---|---|---|
| A2 | 16 | 10 / 15 | 15 / 22 | 19 / 21 |
| A2 | 64 | 21 / 22 | 20 / 21 | 22 / 22 |
| A2 | 256 | 21 / 22 | 19 / 21 | 21 / 22 |
| A5 | 16 | 11 / 17 | 18 / 21 | 18 / 22 |
| A5 | 64 | 19 / 22 | 19 / 22 | 21 / 22 |
| A5 | 256 | 22 / 22 | 21 / 22 | 21 / 22 |

## Development pilot of the compared systems (Astra proposals on development cups): 4 scenes

| system | first | within 2 | within 3 | wall clock: within 120 s / 240 s, mean scene s | robot time: within 60 s / 120 s, mean scene s | Astra calls / scene | decision s / scene | proposal failures |
|---|---|---|---|---|---|---|---|---|
| E0 | 0 | 0 | 1 | 1 / 1, 137.4 | 0 / 1, 65.6 | 5.0 | 22.398 | 0 |
| E1 | 3 | 4 | 4 | 3 / 4, 89.8 | 4 / 4, 42.1 | 3.25 | 7.824 | 0 |
| E2 | 3 | 4 | 4 | 4 / 4, 81.4 | 4 / 4, 34.0 | 2.0 | 0.003 | 0 |
| E3 | 4 | 4 | 4 | 4 / 4, 69.7 | 4 / 4, 33.1 | 2.0 | 0.002 | 0 |

| pair | key | a better | b better | sign test p |
|---|---|---|---|---|
| E0-E1 | first | 0 | 3 | 0.25 |
| E0-E1 | within3 | 0 | 3 | 0.25 |
| E0-E2 | first | 0 | 3 | 0.25 |
| E0-E2 | within3 | 0 | 3 | 0.25 |
| E0-E3 | first | 0 | 4 | 0.125 |
| E0-E3 | within3 | 0 | 3 | 0.25 |
| E1-E2 | first | 0 | 0 | 1.0 |
| E1-E2 | within3 | 0 | 0 | 1.0 |
| E1-E3 | first | 0 | 1 | 1.0 |
| E1-E3 | within3 | 0 | 0 | 1.0 |
| E2-E3 | first | 0 | 1 | 1.0 |
| E2-E3 | within3 | 0 | 0 | 1.0 |

## Experiment E, PROSPECTIVE (locked cups): 20 scenes

| system | first | within 2 | within 3 | wall clock: within 120 s / 240 s, mean scene s | robot time: within 60 s / 120 s, mean scene s | Astra calls / scene | decision s / scene | proposal failures |
|---|---|---|---|---|---|---|---|---|
| E0 | 0 | 1 | 7 | 4 / 7, 122.1 | 2 / 7, 65.4 | 4.95 | 22.993 | 0 |
| E1 | 20 | 20 | 20 | 20 / 20, 61.9 | 20 / 20, 39.8 | 3.0 | 7.128 | 0 |
| E2 | 20 | 20 | 20 | 20 / 20, 54.5 | 20 / 20, 32.7 | 2.0 | 0.002 | 0 |
| E3 | 20 | 20 | 20 | 20 / 20, 54.2 | 20 / 20, 32.7 | 2.0 | 0.002 | 0 |

| pair | key | a better | b better | sign test p |
|---|---|---|---|---|
| E0-E1 | first | 0 | 20 | 0.0 |
| E0-E1 | within3 | 0 | 13 | 0.0002 |
| E0-E2 | first | 0 | 20 | 0.0 |
| E0-E2 | within3 | 0 | 13 | 0.0002 |
| E0-E3 | first | 0 | 20 | 0.0 |
| E0-E3 | within3 | 0 | 13 | 0.0002 |
| E1-E2 | first | 0 | 0 | 1.0 |
| E1-E2 | within3 | 0 | 0 | 1.0 |
| E1-E3 | first | 0 | 0 | 1.0 |
| E1-E3 | within3 | 0 | 0 | 1.0 |
| E2-E3 | first | 0 | 0 | 1.0 |
| E2-E3 | within3 | 0 | 0 | 1.0 |
