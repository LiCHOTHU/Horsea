# Overnight System-1 / System-2 transfer study: status report

Study root: `/home/licho/workspace/Horsea/experiments/s1s2_overnight_2026-10-10`. Created 2026-10-10 06:16:37. Report regenerated after every activity.

## Budget

```
{
 "global_used": 454,
 "global_cap": 600,
 "per_platform": {
  "robotwin_bimanual": 117,
  "dexhand": 118,
  "libero": 106,
  "robotwin_single": 113
 },
 "per_platform_cap": 150,
 "per_cell": {
  "dexhand/SRC": 118,
  "libero/E01": 106,
  "robotwin_bimanual/SRC": 117,
  "robotwin_single/E01": 27,
  "robotwin_single/SRC": 86
 },
 "tokens": {
  "input_tokens": 7485641,
  "cached_input_tokens": 2516096,
  "cache_write_input_tokens": 0,
  "output_tokens": 264835,
  "reasoning_output_tokens": 63171
 },
 "seconds_left": 591,
 "window_start": "2026-10-10 06:16:37",
 "window_hours": 8.0
}
```

## 12 x 4 matrix

| cell | configuration | libero | robotwin_single | robotwin_bimanual | dexhand |
|---|---|---|---|---|---|
| E01 | Astra-only (teacher capability and interaction / computation cost) | COMPLETE 1/3 dev | PENDING_BUDGET 1/1 of 3 requested dev | PENDING_BUDGET | PENDING_BUDGET |
| E02 | Original pretrained S1, unchanged | NOT_AVAILABLE | COMPLETE 2/5 dev; test 3/10 | COMPLETE 3/3 dev; test 2/10 | NOT_AVAILABLE |
| E03 | S1 trained on D1 (first successful episode) | BLOCKED_INTERFACE | COMPLETE 1/3 dev | BLOCKED_DATA | BLOCKED_DATA |
| E04 | S1 trained on D3 (first three successful episodes) | BLOCKED_INTERFACE | BLOCKED_DATA | BLOCKED_DATA | BLOCKED_DATA |
| E05 | S1 trained on DN (first N = min(10, available) successful episodes) - main anchor | BLOCKED_INTERFACE | COMPLETE 0/5 dev; test 1/10 | BLOCKED_DATA | BLOCKED_DATA |
| E06 | E05 with a shorter executed action chunk (about half, >= 1) | BLOCKED_INTERFACE | COMPLETE 0/3 dev | BLOCKED_DATA | BLOCKED_DATA |
| E07 | S1 trained on all valid teacher trajectories up to the N-th success (successes + failures) | BLOCKED_INTERFACE | COMPLETE 0/3 dev | BLOCKED_DATA | BLOCKED_DATA |
| E08 | E05 then one scheduled Astra takeover at half the native step budget, current observation only | BLOCKED_INTERFACE | NO_STUDENT_SIGNAL | BLOCKED_DATA | BLOCKED_DATA |
| E09 | E08 plus the last three student interaction records before handover | BLOCKED_INTERFACE | NO_STUDENT_SIGNAL | BLOCKED_DATA | BLOCKED_DATA |
| E10 | E05 continued for dU updates on its original data | BLOCKED_INTERFACE | COMPLETE 0/3 dev; test 3/10 | BLOCKED_DATA | BLOCKED_DATA |
| E11 | E05 continued for dU updates on original data + Astra corrections (half / half batches) | BLOCKED_INTERFACE | NO_STUDENT_SIGNAL | BLOCKED_DATA | BLOCKED_DATA |
| E12 | E05 continued for dU updates on original data + additional Astra-from-reset demos (half / half) | BLOCKED_INTERFACE | NO_STUDENT_SIGNAL | BLOCKED_DATA | BLOCKED_DATA |

Status vocabulary: COMPLETE, RUNNING, PENDING_BUDGET, BLOCKED_INTERFACE, BLOCKED_DATA, NO_STUDENT_SIGNAL, NOT_APPLICABLE, DUPLICATE, FAILED (plus BLOCKED_NO_CORRECTIONS for E11 and PENDING for not yet reached). Numbers are successes / evaluated episodes on the stated pool (dev unless marked test).

## libero

Cohort: none (student blocked). Student: none available.
Audit: student cells blocked: no compatible pretrained student (Horsea base80 deleted 2026-09-30; it held task 48 out); the surviving from-scratch FM pipeline (imitation FlowMatchingPolicy) needs processed observation keys (hand_mat, camera extrinsics/intrinsics) from deleted LIBERO-90 data, and a bridge from Astra-recorded episodes was not built within the overnight setup window

1. **Could Astra's behavior be converted into valid student training data?** student cells blocked: no compatible pretrained student (Horsea base80 deleted 2026-09-30; it held task 48 out); the surviving from-scratch FM pipeline (imitation FlowMatchingPolicy) needs processed observation keys (hand_mat, camera extrinsics/intrinsics) from deleted LIBERO-90 data, and a bridge from Astra-recorded episodes was not built within the overnight setup window; source episodes 0 (0 valid, 0 successful); N = n/a
2. **Did the standalone student improve?** no pretrained student (E02 NOT_AVAILABLE); anchor (DN) vs D1 student: not answered (E05 BLOCKED_INTERFACE, E03 BLOCKED_INTERFACE)
3. **Did additional successful demonstrations improve stability?** D3 vs D1: not answered (E04 BLOCKED_INTERFACE, E03 BLOCKED_INTERFACE); DN vs D1: not answered (E05 BLOCKED_INTERFACE, E03 BLOCKED_INTERFACE)
4. **Did more frequent observation help?** half executed chunk vs anchor: not answered (E06 BLOCKED_INTERFACE, E05 BLOCKED_INTERFACE)
5. **Did naive imitation of failed attempts help or hurt?** successes+failures vs successes only: not answered (E07 BLOCKED_INTERFACE, E05 BLOCKED_INTERFACE)
6. **Did Astra takeover help the same student?** anchor + one takeover vs anchor alone (assistance, not learning): not answered (E08 BLOCKED_INTERFACE, E05 BLOCKED_INTERFACE)
7. **Did recent history improve takeover?** takeover with three student records vs current observation only: not answered (E09 BLOCKED_INTERFACE, E08 BLOCKED_INTERFACE)
8. **After corrective training, did the student improve with Astra disabled?** after corrective training vs anchor: not answered (E11 BLOCKED_INTERFACE, E05 BLOCKED_INTERFACE)
9. **Was that improvement greater than extra training or additional reset demonstrations?** corrections vs extra training: not answered (E11 BLOCKED_INTERFACE, E10 BLOCKED_INTERFACE); corrections vs extra reset demos: not answered (E11 BLOCKED_INTERFACE, E12 BLOCKED_INTERFACE)
10. **What is the smallest next experiment justified by the evidence?** to be written from the completed cells

Cell status reasons:

- E02 NOT_AVAILABLE: no compatible pretrained student exists for this platform
- E03 BLOCKED_INTERFACE: student cells blocked: no compatible pretrained student (Horsea base80 deleted 2026-09-30; it held task 48 out); the surviving from-scratch FM pipeline (imitation FlowMatchingPolicy) needs processed observation keys (hand_mat, camera extrinsics/intrinsics) from deleted LIBERO-90 data, and a bridge from Astra-recorded episodes was not built within the overnight setup window
- E04 BLOCKED_INTERFACE: student cells blocked: no compatible pretrained student (Horsea base80 deleted 2026-09-30; it held task 48 out); the surviving from-scratch FM pipeline (imitation FlowMatchingPolicy) needs processed observation keys (hand_mat, camera extrinsics/intrinsics) from deleted LIBERO-90 data, and a bridge from Astra-recorded episodes was not built within the overnight setup window
- E05 BLOCKED_INTERFACE: student cells blocked: no compatible pretrained student (Horsea base80 deleted 2026-09-30; it held task 48 out); the surviving from-scratch FM pipeline (imitation FlowMatchingPolicy) needs processed observation keys (hand_mat, camera extrinsics/intrinsics) from deleted LIBERO-90 data, and a bridge from Astra-recorded episodes was not built within the overnight setup window
- E06 BLOCKED_INTERFACE: student cells blocked: no compatible pretrained student (Horsea base80 deleted 2026-09-30; it held task 48 out); the surviving from-scratch FM pipeline (imitation FlowMatchingPolicy) needs processed observation keys (hand_mat, camera extrinsics/intrinsics) from deleted LIBERO-90 data, and a bridge from Astra-recorded episodes was not built within the overnight setup window
- E07 BLOCKED_INTERFACE: student cells blocked: no compatible pretrained student (Horsea base80 deleted 2026-09-30; it held task 48 out); the surviving from-scratch FM pipeline (imitation FlowMatchingPolicy) needs processed observation keys (hand_mat, camera extrinsics/intrinsics) from deleted LIBERO-90 data, and a bridge from Astra-recorded episodes was not built within the overnight setup window
- E08 BLOCKED_INTERFACE: student cells blocked: no compatible pretrained student (Horsea base80 deleted 2026-09-30; it held task 48 out); the surviving from-scratch FM pipeline (imitation FlowMatchingPolicy) needs processed observation keys (hand_mat, camera extrinsics/intrinsics) from deleted LIBERO-90 data, and a bridge from Astra-recorded episodes was not built within the overnight setup window
- E09 BLOCKED_INTERFACE: student cells blocked: no compatible pretrained student (Horsea base80 deleted 2026-09-30; it held task 48 out); the surviving from-scratch FM pipeline (imitation FlowMatchingPolicy) needs processed observation keys (hand_mat, camera extrinsics/intrinsics) from deleted LIBERO-90 data, and a bridge from Astra-recorded episodes was not built within the overnight setup window
- E10 BLOCKED_INTERFACE: student cells blocked: no compatible pretrained student (Horsea base80 deleted 2026-09-30; it held task 48 out); the surviving from-scratch FM pipeline (imitation FlowMatchingPolicy) needs processed observation keys (hand_mat, camera extrinsics/intrinsics) from deleted LIBERO-90 data, and a bridge from Astra-recorded episodes was not built within the overnight setup window
- E11 BLOCKED_INTERFACE: student cells blocked: no compatible pretrained student (Horsea base80 deleted 2026-09-30; it held task 48 out); the surviving from-scratch FM pipeline (imitation FlowMatchingPolicy) needs processed observation keys (hand_mat, camera extrinsics/intrinsics) from deleted LIBERO-90 data, and a bridge from Astra-recorded episodes was not built within the overnight setup window
- E12 BLOCKED_INTERFACE: student cells blocked: no compatible pretrained student (Horsea base80 deleted 2026-09-30; it held task 48 out); the surviving from-scratch FM pipeline (imitation FlowMatchingPolicy) needs processed observation keys (hand_mat, camera extrinsics/intrinsics) from deleted LIBERO-90 data, and a bridge from Astra-recorded episodes was not built within the overnight setup window

Astra calls charged on this platform: 106 (E01 106).

## robotwin_single

Cohort: pretrained-policy adaptation (seen task). Student: Horsea FM base step300000 (8e7500ea54404ce2).
Audit: pools from T_place_container_plate.json; camera parity 0.25 px; vector inside stats True

1. **Could Astra's behavior be converted into valid student training data?** replay of SRC_1500001_1791627424: ee deviation median 0.0 cm, p90 0.1 cm; success recorded True / replayed True; source episodes 4 (4 valid, 2 successful); N = 2
2. **Did the standalone student improve?** anchor vs original pretrained student: E05 0/5 vs E02 2/5; per start (E05/E02) 1500017: F/F, 1500018: F/S, 1500019: F/F, 1500020: F/S, 1500021: F/F (exploratory)
3. **Did additional successful demonstrations improve stability?** D3 vs D1: not answered (E04 BLOCKED_DATA); DN vs D1: E05 0/5 vs E03 1/3; per start (E05/E03) 1500017: F/S, 1500018: F/F, 1500019: F/F (exploratory)
4. **Did more frequent observation help?** half executed chunk vs anchor: E06 0/3 vs E05 0/5; per start (E06/E05) 1500017: F/F, 1500018: F/F, 1500019: F/F (exploratory)
5. **Did naive imitation of failed attempts help or hurt?** successes+failures vs successes only: E07 0/3 vs E05 0/5; per start (E07/E05) 1500017: F/F, 1500018: F/F, 1500019: F/F (exploratory)
6. **Did Astra takeover help the same student?** anchor + one takeover vs anchor alone (assistance, not learning): not answered (E08 NO_STUDENT_SIGNAL)
7. **Did recent history improve takeover?** takeover with three student records vs current observation only: not answered (E09 NO_STUDENT_SIGNAL, E08 NO_STUDENT_SIGNAL)
8. **After corrective training, did the student improve with Astra disabled?** after corrective training vs anchor: not answered (E11 NO_STUDENT_SIGNAL)
9. **Was that improvement greater than extra training or additional reset demonstrations?** corrections vs extra training: not answered (E11 NO_STUDENT_SIGNAL); corrections vs extra reset demos: not answered (E11 NO_STUDENT_SIGNAL, E12 NO_STUDENT_SIGNAL)
10. **What is the smallest next experiment justified by the evidence?** to be written from the completed cells

Data/interface diagnosis (outside the matrix; E02, E03, E05 on the source starts of DN (their own training starts); no Astra calls): starts [1500001, 1500003]: E02 0/2 1500001:F, 1500003:F; E03 0/2 1500001:F, 1500003:F; E05 0/2 1500001:F, 1500003:F

Cell status reasons:

- E01 PENDING_BUDGET: PENDING_BUDGET: remaining study calls below this episode's call ceiling
- E04 BLOCKED_DATA: D3 needs three successful source episodes; only 2 were obtained within the source allotment
- E08 NO_STUDENT_SIGNAL: anchor (E05) had no success in 5 dev evaluations
- E09 NO_STUDENT_SIGNAL: anchor (E05) had no success in 5 dev evaluations
- E11 NO_STUDENT_SIGNAL: anchor (E05) had no success in 5 dev evaluations
- E12 NO_STUDENT_SIGNAL: anchor (E05) had no success in 5 dev evaluations

Astra calls charged on this platform: 113 (E01 27, SRC 86).

## robotwin_bimanual

Cohort: pretrained-policy adaptation (seen task). Student: Horsea FM base step300000 (8e7500ea54404ce2).
Audit: pools from T_lift_pot.json; camera parity 0.25 px; vector inside stats True

1. **Could Astra's behavior be converted into valid student training data?** replay of SRC_1500001_1791629609: ee deviation median 0.1 cm, p90 0.4 cm; success recorded False / replayed False; source episodes 4 (4 valid, 0 successful); N = 0
2. **Did the standalone student improve?** anchor vs original pretrained student: not answered (E05 BLOCKED_DATA)
3. **Did additional successful demonstrations improve stability?** D3 vs D1: not answered (E04 BLOCKED_DATA, E03 BLOCKED_DATA); DN vs D1: not answered (E05 BLOCKED_DATA, E03 BLOCKED_DATA)
4. **Did more frequent observation help?** half executed chunk vs anchor: not answered (E06 BLOCKED_DATA, E05 BLOCKED_DATA)
5. **Did naive imitation of failed attempts help or hurt?** successes+failures vs successes only: not answered (E07 BLOCKED_DATA, E05 BLOCKED_DATA)
6. **Did Astra takeover help the same student?** anchor + one takeover vs anchor alone (assistance, not learning): not answered (E08 BLOCKED_DATA, E05 BLOCKED_DATA)
7. **Did recent history improve takeover?** takeover with three student records vs current observation only: not answered (E09 BLOCKED_DATA, E08 BLOCKED_DATA)
8. **After corrective training, did the student improve with Astra disabled?** after corrective training vs anchor: not answered (E11 BLOCKED_DATA, E05 BLOCKED_DATA)
9. **Was that improvement greater than extra training or additional reset demonstrations?** corrections vs extra training: not answered (E11 BLOCKED_DATA, E10 BLOCKED_DATA); corrections vs extra reset demos: not answered (E11 BLOCKED_DATA, E12 BLOCKED_DATA)
10. **What is the smallest next experiment justified by the evidence?** to be written from the completed cells

Cell status reasons:

- E01 PENDING_BUDGET: 33 platform calls left, below one episode's 40-call ceiling
- E03 BLOCKED_DATA: no successful valid source episode
- E04 BLOCKED_DATA: no successful valid source episode
- E05 BLOCKED_DATA: no successful valid source episode
- E06 BLOCKED_DATA: no successful valid source episode
- E07 BLOCKED_DATA: no successful valid source episode
- E08 BLOCKED_DATA: no successful valid source episode
- E09 BLOCKED_DATA: no successful valid source episode
- E10 BLOCKED_DATA: no successful valid source episode
- E11 BLOCKED_DATA: no successful valid source episode
- E12 BLOCKED_DATA: no successful valid source episode

Astra calls charged on this platform: 117 (SRC 117).

## dexhand

Cohort: distillation from scratch (ACT, stateless). Student: ACT trained from scratch on Astra episodes.
Audit: pools from the protocol shuffle (room 1, table 2); student ACT from scratch

1. **Could Astra's behavior be converted into valid student training data?** replay of SRC_5788_1791631900: wrist tracking median 1.3 cm; final can error 0.0 m; success recorded False / replayed False; source episodes 6 (6 valid, 0 successful); N = 0
2. **Did the standalone student improve?** no pretrained student (E02 NOT_AVAILABLE); anchor (DN) vs D1 student: not answered (E05 BLOCKED_DATA, E03 BLOCKED_DATA)
3. **Did additional successful demonstrations improve stability?** D3 vs D1: not answered (E04 BLOCKED_DATA, E03 BLOCKED_DATA); DN vs D1: not answered (E05 BLOCKED_DATA, E03 BLOCKED_DATA)
4. **Did more frequent observation help?** half executed chunk vs anchor: not answered (E06 BLOCKED_DATA, E05 BLOCKED_DATA)
5. **Did naive imitation of failed attempts help or hurt?** successes+failures vs successes only: not answered (E07 BLOCKED_DATA, E05 BLOCKED_DATA)
6. **Did Astra takeover help the same student?** anchor + one takeover vs anchor alone (assistance, not learning): not answered (E08 BLOCKED_DATA, E05 BLOCKED_DATA)
7. **Did recent history improve takeover?** takeover with three student records vs current observation only: not answered (E09 BLOCKED_DATA, E08 BLOCKED_DATA)
8. **After corrective training, did the student improve with Astra disabled?** after corrective training vs anchor: not answered (E11 BLOCKED_DATA, E05 BLOCKED_DATA)
9. **Was that improvement greater than extra training or additional reset demonstrations?** corrections vs extra training: not answered (E11 BLOCKED_DATA, E10 BLOCKED_DATA); corrections vs extra reset demos: not answered (E11 BLOCKED_DATA, E12 BLOCKED_DATA)
10. **What is the smallest next experiment justified by the evidence?** to be written from the completed cells

Cell status reasons:

- E01 PENDING_BUDGET: 32 platform calls left, below one episode's 40-call ceiling
- E02 NOT_AVAILABLE: no compatible pretrained student exists for this platform
- E03 BLOCKED_DATA: no successful valid source episode
- E04 BLOCKED_DATA: no successful valid source episode
- E05 BLOCKED_DATA: no successful valid source episode
- E06 BLOCKED_DATA: no successful valid source episode
- E07 BLOCKED_DATA: no successful valid source episode
- E08 BLOCKED_DATA: no successful valid source episode
- E09 BLOCKED_DATA: no successful valid source episode
- E10 BLOCKED_DATA: no successful valid source episode
- E11 BLOCKED_DATA: no successful valid source episode
- E12 BLOCKED_DATA: no successful valid source episode

Astra calls charged on this platform: 118 (SRC 118).

## Per-cell metrics (cells with evaluated episodes or training)

| platform | cell | status | succ/n | starts | native steps | policy calls | Astra calls | Astra tokens (in/out) | wall s | updates | dataset | retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| libero | E01 | COMPLETE | 1/3 | [] | 885 | 0 | 106 | 1489588/50433 | 0 | - | - | 0 |
| robotwin_single | E01 | PENDING_BUDGET | 1/1 | [] | 348 | 0 | 27 | 414222/16736 | 0 | - | - | 0 |
| robotwin_single | E02 | COMPLETE | 2/5 | [1500017, 1500018, 1500019, 1500020, 1500021] | 1518 | 191 | 0 | - | 976.5 | - | - | 0 |
| robotwin_single | E03 | COMPLETE | 1/3 | [1500017, 1500018, 1500019] | 973 | 122 | 0 | - | 668.5 | 1000 | D1: 1 ep / 200 anchors | 0 |
| robotwin_single | E05 | COMPLETE | 0/5 | [1500017, 1500018, 1500019, 1500020, 1500021] | 2000 | 250 | 0 | - | 1313.2 | 1000 | DN: 2 ep / 467 anchors | 0 |
| robotwin_single | E06 | COMPLETE | 0/3 | [1500017, 1500018, 1500019] | 1200 | 300 | 0 | - | 890.5 | - | - | 0 |
| robotwin_single | E07 | COMPLETE | 0/3 | [1500017, 1500018, 1500019] | 1200 | 150 | 0 | - | 988.5 | 1000 | Dfail: 3 ep / 902 anchors | 0 |
| robotwin_single | E10 | COMPLETE | 0/3 | [1500017, 1500018, 1500019] | 1200 | 150 | 0 | - | 777.0 | 500 | DN: 2 ep / 467 anchors | 0 |
| robotwin_bimanual | E02 | COMPLETE | 3/3 | [1500044, 1500048, 1500049] | 338 | 44 | 0 | - | 266.4 | - | - | 0 |

Machine-readable: `metrics.json` (per platform and cell, including failure stages, hybrid handovers and test results).

## Orchestrator log (latest 60 lines)

```
2026-10-10 07:31:10 START dexhand__teacher__SRC__min3 (try 1/3)
2026-10-10 08:03:52 DONE  dexhand__teacher__SRC__min3 in 1961 s: SRC: 4 new episodes, stop: call allotment spent; stream totals {'episodes': 4, 'successes': 0, 'valid_episodes': 4, 'astra_calls': 76, 'native_steps': 1600}
2026-10-10 08:03:52 START dexhand__replay____first (try 1/3)
2026-10-10 08:05:18 DONE  dexhand__replay____first in 86 s: replay of SRC_5788_1791631900: wrist tracking median 1.3 cm; final can error 0.0 m; success recorded False / replayed False
2026-10-10 08:05:18 START robotwin_bimanual__teacher__SRC__unlock (try 1/3)
2026-10-10 08:16:33 DONE  robotwin_bimanual__teacher__SRC__unlock in 675 s: SRC: 1 new episodes (0 successes), stop: PENDING_BUDGET: remaining study calls below this episode's call ceiling; stream totals {'episodes': 4, 'successes': 0, 'valid_episodes': 4, 'astra_calls': 117, 'native_steps': 1638}
2026-10-10 08:16:33 START dexhand__teacher__SRC__unlock (try 1/3)
2026-10-10 08:33:54 DONE  dexhand__teacher__SRC__unlock in 1041 s: SRC: 2 new episodes, stop: PENDING_BUDGET: remaining study calls below this episode's call ceiling; stream totals {'episodes': 6, 'successes': 0, 'valid_episodes': 6, 'astra_calls': 118, 'native_steps': 2400}
2026-10-10 08:33:55 START robotwin_single__eval__E02__dev3 (try 1/3)
2026-10-10 08:44:25 DONE  robotwin_single__eval__E02__dev3 in 631 s: E02 1/3 (exec_chunk 8)
2026-10-10 08:44:25 START robotwin_single__train__E03__s0 (try 1/3)
2026-10-10 08:44:44 DONE  robotwin_single__train__E03__s0 in 19 s: E03: 1000 updates on ['D1'] (200 anchors); final loss 0.0081
2026-10-10 08:44:44 START robotwin_single__eval__E03__dev3 (try 1/3)
2026-10-10 08:56:11 DONE  robotwin_single__eval__E03__dev3 in 687 s: E03 1/3 (exec_chunk 8)
2026-10-10 08:56:11 START robotwin_single__train__E05__s0 (try 1/3)
2026-10-10 08:56:32 DONE  robotwin_single__train__E05__s0 in 20 s: E05: 1000 updates on ['DN'] (467 anchors); final loss 0.0233
2026-10-10 08:56:32 START robotwin_single__eval__E05__dev3 (try 1/3)
2026-10-10 09:09:54 DONE  robotwin_single__eval__E05__dev3 in 803 s: E05 0/3 (exec_chunk 8)
2026-10-10 09:09:54 orchestrator up (pid 1600702); window ends 2026-10-10 14:16:37
2026-10-10 09:09:54 START robotwin_bimanual__eval__E02__dev3 (try 1/3)
2026-10-10 09:14:40 DONE  robotwin_bimanual__eval__E02__dev3 in 285 s: E02 3/3 (exec_chunk 8)
2026-10-10 09:14:40 START robotwin_single__eval__E06__dev3 (try 1/3)
2026-10-10 09:29:49 DONE  robotwin_single__eval__E06__dev3 in 909 s: E06 0/3 (exec_chunk 4)
2026-10-10 09:29:49 START robotwin_single__eval__E05__dev_expand5 (try 1/3)
2026-10-10 09:38:55 DONE  robotwin_single__eval__E05__dev_expand5 in 546 s: E05 0/2 (exec_chunk 8)
2026-10-10 09:38:55 START robotwin_single__train__E10__s0 (try 1/3)
2026-10-10 09:39:12 DONE  robotwin_single__train__E10__s0 in 17 s: E10: 500 updates on ['DN'] (467 anchors); final loss 0.0163
2026-10-10 09:39:12 START robotwin_single__eval__E10__dev3 (try 1/3)
2026-10-10 09:52:28 DONE  robotwin_single__eval__E10__dev3 in 796 s: E10 0/3 (exec_chunk 8)
2026-10-10 09:52:28 orchestrator up (pid 1600702); window ends 2026-10-10 14:16:37
2026-10-10 09:52:28 START robotwin_single__eval__DIAG_E02__train_starts (try 1/3)
2026-10-10 10:01:05 DONE  robotwin_single__eval__DIAG_E02__train_starts in 517 s: DIAG_E02 0/2 (exec_chunk 8)
2026-10-10 10:01:05 START robotwin_single__eval__DIAG_E03__train_starts (try 1/3)
2026-10-10 10:09:40 DONE  robotwin_single__eval__DIAG_E03__train_starts in 515 s: DIAG_E03 0/2 (exec_chunk 8)
2026-10-10 10:09:40 START robotwin_single__eval__DIAG_E05__train_starts (try 1/3)
2026-10-10 10:19:00 DONE  robotwin_single__eval__DIAG_E05__train_starts in 560 s: DIAG_E05 0/2 (exec_chunk 8)
2026-10-10 10:19:00 START robotwin_single__train__E07__s0 (try 1/3)
2026-10-10 10:19:27 DONE  robotwin_single__train__E07__s0 in 27 s: E07: 1000 updates on ['Dfail'] (902 anchors); final loss 0.0305
2026-10-10 10:19:27 START robotwin_single__eval__E07__dev3 (try 1/3)
2026-10-10 10:36:15 DONE  robotwin_single__eval__E07__dev3 in 1007 s: E07 0/3 (exec_chunk 8)
2026-10-10 10:36:15 START robotwin_single__teacher__E01__e01 (try 1/3)
2026-10-10 10:46:29 DONE  robotwin_single__teacher__E01__e01 in 615 s: E01: 1 new episodes (1 successes), stop: PENDING_BUDGET: remaining study calls below this episode's call ceiling; stream totals {'episodes': 1, 'successes': 1, 'valid_episodes': 1, 'astra_calls': 27, 'native_steps': 348}
2026-10-10 10:46:29 START libero__teacher__E01__e01 (try 1/3)
2026-10-10 11:18:10 DONE  libero__teacher__E01__e01 in 1901 s: E01: {'episodes': 3, 'successes': 1, 'astra_calls': 106, 'native_steps': 885}
2026-10-10 11:18:10 orchestrator up (pid 1600702); window ends 2026-10-10 14:16:37
2026-10-10 11:18:10 START robotwin_single__eval__E02__test10 (try 1/3)
2026-10-10 11:52:31 DONE  robotwin_single__eval__E02__test10 in 2060 s: E02 3/10 (exec_chunk 8)
2026-10-10 11:52:31 START robotwin_single__eval__E05__test10 (try 1/3)
2026-10-10 12:35:53 DONE  robotwin_single__eval__E05__test10 in 2602 s: E05 1/10 (exec_chunk 8)
2026-10-10 12:35:53 START robotwin_single__eval__E10__test10 (try 1/3)
2026-10-10 13:11:01 DONE  robotwin_single__eval__E10__test10 in 2108 s: E10 3/10 (exec_chunk 8)
2026-10-10 13:11:01 START robotwin_bimanual__eval__E02__test10 (try 1/3)
2026-10-10 14:00:01 DONE  robotwin_bimanual__eval__E02__test10 in 2940 s: E02 2/10 (exec_chunk 8)
2026-10-10 14:00:01 START robotwin_single__eval__E02__dev_expand5 (try 1/3)
2026-10-10 14:06:23 DONE  robotwin_single__eval__E02__dev_expand5 in 381 s: E02 1/2 (exec_chunk 8)
2026-10-10 14:06:23 SKIP robotwin_single__eval__E03__dev_expand5: estimated 672 s > 614 s left in the window
2026-10-10 14:06:23 SKIP robotwin_single__eval__E06__dev_expand5: estimated 672 s > 614 s left in the window
2026-10-10 14:06:23 SKIP robotwin_single__eval__E07__dev_expand5: estimated 672 s > 614 s left in the window
2026-10-10 14:06:23 SKIP robotwin_single__eval__E10__dev_expand5: estimated 672 s > 614 s left in the window
2026-10-10 14:06:23 SKIP robotwin_bimanual__eval__E02__dev_expand5: estimated 588 s > 614 s left in the window
```
