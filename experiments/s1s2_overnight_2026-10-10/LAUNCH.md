# Launch record
- Unit: s1s2-overnight (systemd --user transient service, same convention as the project's horsea-queue service)
- Command: /usr/bin/python3 -u -m s1s2.orchestrator --root /home/licho/workspace/Horsea/experiments/s1s2_overnight_2026-10-10 --hours 8 --platforms robotwin_single,robotwin_bimanual
  (dexhand and libero join through platforms_enabled.json when their adapters pass their smoke tests)
- Code revision at launch: 86fe0ab
- Preflight cost outside this study's ledger: 26 Astra calls spent by an orchestrator dry run (test root, scratchpad) before it
  was stopped; it produced one complete RoboTwin place_container_plate teacher episode (seed 1500001, 22 calls, 409 native
  steps, no success) and one partial episode. That data is NOT used by this study.
- Started: 2026-10-10 06:16:37
- 06:22:15 dexhand enabled after its smoke test passed (code 16d2f8e)
06:49:49 boundary reached; restarting orchestrator with new code
06:49:52 restarted (code 656ec74); libero enabled (E01 only)
- 06:51 code update (deployed between activities; orchestrator re-executes at the next stage boundary via RELOAD). Declared
  before any hybrid, correction or extra-demo episode ran:
  - Takeover call ceiling 20 for E08, E09 and correction collection (the pilots' 40 calls per 400 native steps, scaled to the
    remaining half of the physical budget); identical across the three.
  - An episode starts only if the study budget (per-platform 150 / global 600, unchanged) still covers its full call ceiling,
    so no evaluation or collection episode is ever cut short by the study budget; starts not reached are PENDING_BUDGET.
  - E12 extra demos: exactly the Astra calls and native steps that correction collection used (the last episode is cut to the
    remainder and counts as an acquisition cost); a correction collection with zero takeovers leaves no matched budget.
  - Allotments are checked between episodes, so a collection can exceed its allotment by at most one episode
    (RoboTwin single-arm source collection: allotment 60, actual 86 calls; the fourth episode started at 58 calls).
  - Final stages split: E01 Astra-only reference on the dev starts for every platform first, then the locked test-pool
    evaluation (10 test starts) of the standalone students E02, E05, E10, E11, E12.
  - A cell with fewer evaluated starts than requested is PENDING_BUDGET with its exact partial denominator.
- 06:53 code update (before any E03-E12 activity ran):
  - D3 exists only with three successful source episodes. With fewer, E04 is BLOCKED_DATA (it is not trained on a smaller
    set under the D3 name); E05 trains on DN (N = 2) or, with N = 1, is a DUPLICATE of E03; with N = 3 it is a DUPLICATE of E04.
  - E08 and E09 run start by start (E08 then E09 on the same development start). A start begins only if the remaining
    platform budget covers both takeover ceilings (2 x 20 calls), so a binding budget cannot leave an unpaired start.
  - Report: partial cells keep exact denominators; comparisons list raw per-start paired outcomes; test-pool results and
    cell status reasons are shown; Astra calls per cell per platform.
- Disclosure (observed 06:58): RoboTwin executes an Astra planner motion as a whole path, so the last Astra motion of an
  episode can end past the 400-step limit (source episodes 2 and 4 of place_container_plate ended at 436 and 426 native
  steps, both failures). The student cannot overrun (one native step per qpos action). The controller is left unchanged;
  any success reached after step 400 will be flagged in the report.
- 07:30 rule added before dexhand source collection started (applies to every platform): if a platform's source allotment
  ends without any successful episode, source collection continues one full-ceiling episode at a time, within the
  platform's 150-call reservation, until the first success (stage source_unlock). Reason: robotwin_bimanual ended its
  60-call allotment with 0 successes in 3 episodes (86 calls), which blocks every student-learning cell there; the spec
  ranks "unlock minimal valid source data" (step 2) above the Astra-only reference (step 8), which competes for the same
  remaining calls.
- 08:38 stage dev_expand added as the LAST stage (spec section 10: "expand important comparisons to five development starts
  if time permits"): standalone-student cells that are COMPLETE on 3 dev starts get the remaining 2 dev starts, after the
  locked test-pool evaluation (which the spec prefers). Hybrids are not expanded (they would spend Astra calls).
- Source collection outcome: robotwin_single 2 successes / 4 episodes (86 calls); robotwin_bimanual 0 / 4 (117 calls,
  incl. 1 unlock episode); dexhand 0 / 6 (118 calls, incl. 2 unlock episodes). Bimanual and dex student-learning cells
  are therefore BLOCKED_DATA; their remaining calls (33, 32) are below one E01 episode's 40-call ceiling.
- 09:42 robotwin_single anchor E05 = 0/5 dev -> NO_STUDENT_SIGNAL for E08, E09, E11, E12 (spec gate). Per the spec
  ("prioritize data/interface diagnosis"), stage `diagnosis` added: E02, E03 and E05 are evaluated on the two source starts
  whose Astra episodes form DN (their own training starts). No Astra calls, no new configuration, kept outside the matrix.
