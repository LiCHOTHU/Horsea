# Feasibility (admission) check before the Astra dex-hand run — 2026-10-09/10

Privileged, scripted probes (object poses known to the author; nothing here was shown to Astra). Purpose: establish whether
`Humanoid-Stack-Can-Into-Drawer-v0` (EgoVLA Ego-Humanoid benchmark, Isaac Sim 5.1, H1 + two Inspire hands) is solvable
through the interface offered to Astra (absolute wrist targets tracked by the benchmark's differential IK + hand closure
fractions). RoboTwin had a scripted expert for this; this benchmark has none, so the author probed by hand.
Scripts: `c_*.py` (run inside `isaac_shell.py`, a long-lived Isaac process); outputs of the last shell session: `out_*.txt`;
snapshots: `snapshots/`. Interface probe: `../isaac_stack_can_into_drawer/` (`probe_isaac_dex.py`).

## Task facts established

- Success (env code): can upright on the saucer (horizontal distance < 9 cm, vertical < 2 cm between their roots) AND the
  cabinet's door (`bottom_joint`, revolute 0..95 deg, starts open at 0) closed to >= 80 % (>= 1.327 rad).
- The open door is a vertical panel hinged at the cabinet's front-left corner, swung toward the robot on its left. The
  saucer is inside the cabinet (z 1.17, x 0.65). The can stands on the table in front of the cabinet.
- **The task is bimanual by construction**: the left arm starts outside the open door (the env sets shoulder yaw 1.29 "to
  clear the drawer door"); the door's sweep passes through the can's position, so the can must be on the saucer before the
  door can close (probe: the left hand pushed the door to 1.22 rad, where it hit the can and tipped it).
- Scene: the project's evaluation protocol skips table 1 of rooms 1-3 (seen in training), so the first evaluation scene is
  room 1, table 2, episode 0 = randomize_idx 5788 (can at (0.372, -0.077), 7.6 cm in front of the cabinet base). The first
  scene probed (room 1 table 1, idx 542) had the can 2.3 cm from the cabinet; it was dropped for that reason and because it
  is a training combination.

## Interface facts (disclosed to Astra in generic form)

- 0.226 s wall per 30 Hz control step (5 cameras rendered); Isaac + env ~9-12.5 GB GPU.
- A far wrist target given to the IK in one jump makes the arm swing wildly (one probe: hand flew to z 1.70); the project only
  ever feeds small per-step targets, so the harness interpolates the commanded wrist pose (<= 1 cm and 3 deg per step).
- Hand closure rate-limited to 1/30 per step (full close in 1 s); a 10-step ramp flung the can off the table (V9).
- Reachable right-hand orientations near the can: fingers forward with palm left (err ~1 cm), palm-down tilted 20 deg left
  (0.12 cm / 0.2 deg); exact palm-down 2.6 cm / 19 deg; fingers-down and palm-forward unreachable (40-120 deg residual).
- Straight-line + slerp paths that combine a large reorientation with translation can sweep the fingers through objects
  (it displaced the can in several probes; X1-X4 and Q1 were confounded by this).

## Grasp attempts (right hand, about 40 trials)

| family | result |
|---|---|
| side wrap, palm facing the can, fingers forward, 2-scalar closure (S1-S2, V1-V12, W1-W4, X1-X4, Y1-Y4) | no lift; fist forms beside the can, or curling fingers push / spin / fling the can |
| palm forward, fingers sideways (G4, G5) | unreachable |
| fingers-down pinch (P1, P3) | unreachable |
| palm-down lateral thumb-index pinch (P5-P7) | IK stalls 4-5 cm high; thumb knocks the can |
| two-palm squeeze, left hand routed around the door (Q1, S1, S2) | left hand gets around the door cleanly, but at the can the wrists stall at y >= +0.02 (left) and <= -0.08 (right); palms stay ~1.6 cm too far apart; pushing the can toward the left palm moved it 2.9 cm; no lift |

Diagnostic (`c_diag.py`): during closure the fingertips reach at most ~5.6 cm in front of the palm and never pass the can's
centre line (radial gap 0.2-4 cm outside the can), so a palm-side wrap of the 6.4 cm can is geometrically impossible with
coupled finger joints. With separate knuckle / tip / thumb-rotation control the thumb can stand ~10 cm out from the palm, but
the index-to-thumb lateral gap (~6.6 cm) leaves ~1 mm clearance around the can for a lateral pinch.

## Conclusion

Door closing through the interface: demonstrated (blocked only by the can). Can lift through the interface: **not
demonstrated** in about 40 scripted trials (~2.5 h). The most plausible route is a two-handed squeeze with the hands angled
inward, or moving the can to where both arms reach before squeezing; neither was completed. The live Astra run was launched
with the 4-group hand interface (finger_base, finger_tip, thumb_rotate, thumb_bend) on scene 5788 with this caveat: success
may not be reachable through this interface, and the report states it.

## Postscript (after the live run)

In the live run Astra lifted the can 14.5 cm (attempt 3, decision 13: index contact 13.7 N, knuckles blocked at 0.37), using a
grasp the author had not tried: a near-palm-down hand (palm (-0.09, 0.23, -0.97), fingers (0.88, 0.47, 0.03)) over the can's
upper body with knuckles 0.60, tips 0.65, thumb rotation 0.55, thumb bend 0.30, wrist at (0.30, -0.15, 1.15). So lifting the can
IS achievable through this interface; the full task (transport, placement, door) remains undemonstrated.
