# Does the System 1 conclusion scale to more dex-hand tasks? (status 2026-10-10 19:56)

**Conclusion under test** (first pilot, Stack-Can-Into-Drawer): bounded local search around Astra's program does not
beat repeating it (H1), history-adaptive proposals do not beat a fixed distribution (H2), and what fails is either a
near-threshold shortfall that needs closed-loop feedback (door) or brittleness on fresh starts that a single adaptation
start with a saturating score cannot detect (grasp).

**Plan:** four more tasks of the same benchmark, hand and harness: the three bimanual ones (Unload-Cans,
Insert-And-Unload-Cans, Sort-Cans) and Open-Laptop (single-arm by design; the articulated analogue of the door). Per task:
Astra alone with the existing no-guidance harness until its first success (at most 10 attempts, 40 calls per 400 steps),
then the same pilot protocol on its program (task-generic code, regression-checked against the first pilot).

## Status

| task | Astra (System 2) | pilot |
|---|---|---|
| Stack-Can-Into-Drawer (first pilot) | success at attempt 10 of the earlier 10-attempt run | done: 3 operations x 6 conditions |
| Unload-Cans | **success at attempt 2** (20 decisions, step 432; right hand only, the left hand never moved) | **done**: 3 operations x 6 conditions, 26 min |
| Open-Laptop | **blocked** | - |
| Insert-And-Unload-Cans | **blocked** | - |
| Sort-Cans | **blocked** | - |

**Blocker (external, since 19:53):** every Astra call returns HTTP 400 "The 'gpt-6-astra' model is not supported when
using Codex with a ChatGPT account" (Codex CLI unchanged, 0.162.0; the same setup worked 19:02-19:17). Each blocked run
stopped after two failed calls through the harness's infrastructure guard; no model output was produced. A watcher
(`experiments/dexqueue_2026-10-10/access_watch.py`, systemd unit `astra-access-watch`) re-checks every 30 min for up to
12 h and re-queues the three Astra runs automatically when the model is reachable again.

**Update 21:27:** between 20:55 and 21:25 the Codex CLI lost its ChatGPT login (`codex login status`: "Not logged in"; every call now fails with 401 "Missing bearer or basic authentication"). The first watcher treated that new error as reachable and restarted the queue at 21:25; the restarted Open-Laptop and Insert-And-Unload-Cans runs (`*_r2`) each stopped after two failed calls and the queue was stopped again. The watcher now treats every CLI failure as blocked (unit `astra-access-watch-r3`, 30-min checks, 12 h) and re-queues the runs as `*_r3` once a login is restored (`codex login`, interactive, by the user).

## Evidence so far (2 tasks, 6 operations, 36 matched blocks)

| task | operation | conditions | ceiling (original succeeds first try) | rescued by local search | succeeded only by repeating the original | no success | fresh-start success A / B / C |
|---|---|---|---|---|---|---|---|
| Stack-Can-Into-Drawer | grasp | 6 | 6 | 0 | 0 | 0 | 14/18 / 14/18 / 14/18 |
| Stack-Can-Into-Drawer | transport | 6 | 6 | 0 | 0 | 0 | 18/18 / 18/18 / 18/18 |
| Stack-Can-Into-Drawer | door | 6 | 3 | 0 | 2 | 1 | 10/18 / 9/18 / 9/18 |
| Unload-Cans | grasp right can | 6 | 6 | 0 | 0 | 0 | 18/18 / 18/18 / 18/18 |
| Unload-Cans | unload right can | 6 | 6 | 0 | 0 | 0 | 18/18 / 18/18 / 18/18 |
| Unload-Cans | grasp left can | 6 | 6 | 0 | 0 | 0 | 1/18 / 1/18 / 1/18 |

Pooled over the six operation strata (paired, operation-stratified bootstrap):

| contrast | fresh-start success | 95 % CI | attempts saved to first success | 95 % CI |
|---|---|---|---|---|
| H1: B - A | -0.9 pp | [-2.8, 0.0] | -0.4 | [-0.8, 0.0] |
| H2: C - B | 0.0 pp | [0.0, 0.0] | 0.0 | [0.0, 0.0] |

**Reading.** [E] On the second task the pattern repeats: Astra's program succeeded on the first attempt in all 18
Unload-Cans blocks (most perturbed variants also succeeded: 51/52, 48/48, 54/64), so there was nothing for local search to
repair and B, C and A are identical. [E] The left-can grasp succeeded on its adaptation start in 6/6 conditions but on only
1/18 fresh starts (tray shifted 8 mm): the brittleness seen in the first pilot's grasp, stronger. [I] So far the conclusion
scales: local search and short-term memory add nothing where Astra's program works, and the failures that matter (fresh
starts) are invisible to single-start adaptation with a saturating score. Still untested beyond one task: the
near-threshold push failure (the door) - Open-Laptop was meant to test it.

Files: `crosstask.md` / `crosstask.json` (this table), `../s1local_unload_cans_v2_2026-10-10/` (Unload-Cans pilot),
`../s1local_generic_regression/REGRESSION.md` (generic code vs the first pilot), `../dexqueue_2026-10-10/` (queue, logs,
access watcher).
