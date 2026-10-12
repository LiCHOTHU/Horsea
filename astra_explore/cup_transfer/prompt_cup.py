"""Astra's prompt and response schema for the cup-transfer experiment (identical text in every method arm).

Differences from the dex pilot prompt (run_dex.build_prompt), all interface facts, no strategies or hints:
  * the object's tracked pose (base centre, up axis, velocity) and observable size are measured and reported (the brief:
    every method may receive the simulator-derived object pose and dimensions);
  * an execution layer runs each decision in 0.1 s segments and may adjust the active hand's wrist translation / closures
    within stated bounds toward the declared interaction mode; its adjustments are reported (zero in the baseline arm);
  * two extra required fields: interaction_mode and active_hand.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_dex  # noqa: E402
from schema_dex import Invalid, response_schema, validate  # noqa: E402

MODES = ["approach", "close", "lift", "hold", "transport", "release", "reposition", "other"]
HANDS_OR_NONE = ["left", "right", "none"]


def response_schema_cup():
    s = response_schema()
    s["properties"]["interaction_mode"] = {"type": "string", "enum": MODES}
    s["properties"]["active_hand"] = {"type": "string", "enum": HANDS_OR_NONE}
    s["required"] = s["required"] + ["interaction_mode", "active_hand"]
    return s


def validate_cup(resp, known_ids):
    if not isinstance(resp, dict):
        raise Invalid("response is not an object")
    extra = {k: resp.get(k) for k in ("interaction_mode", "active_hand")}
    if extra["interaction_mode"] not in MODES:
        raise Invalid(f"interaction_mode must be one of {MODES}")
    if extra["active_hand"] not in HANDS_OR_NONE:
        raise Invalid(f"active_hand must be one of {HANDS_OR_NONE}")
    clean, warnings = validate({k: v for k, v in resp.items() if k not in extra}, known_ids)
    clean.update(extra)
    return clean, warnings


LAYER = (
    "EXECUTION LAYER: each decision is executed in segments of 3 control steps (0.1 s). Before each segment an execution layer "
    "may adjust the ACTIVE hand's commanded wrist translation by at most 1.5 cm per axis and each of its four closure fractions by at "
    "most 0.15 relative to your command, aiming at the local goal of your declared interaction_mode: approach / reposition = reach "
    "your commanded wrist position relative to the object without pushing or tipping it; close = close the hand on the object "
    "without pushing or tipping it; lift = raise the object together with the hand; hold = keep the object lifted and still; "
    "transport = carry the object with the hand; release = open the hand without disturbing the object; other = no adjustment. "
    "Wrist orientation is never adjusted; the other hand always executes your command unchanged. Every result reports the layer's "
    "adjustments (execution_layer) and how the object moved.")


def build_prompt_cup(task, obs, history_view, budgets, note):
    p = run_dex.build_prompt(task, obs, history_view, budgets, note)
    p = p.replace("There is no task policy, grasp planner, collision avoidance or object-pose sensor acting for you: "
                  "your only information is the attached camera images, the robot's own measurements below, and your recorded interaction "
                  "history.",
                  "There is no task policy, grasp planner or collision avoidance acting for you. Your information is the attached camera "
                  "images, the robot's own measurements below, the tracked pose and size of the object (OBJECT below), and your recorded "
                  "interaction history.")
    obj = obs["object"]
    insert = ("OBJECT (tracked by the harness): " + json.dumps(obj) + " - base_centre = the centre of the object's bottom face; "
              "up_axis = the object's axis direction (z near 1 = upright); rise_above_start_cm = height of the base above where it rested "
              "at the start of this attempt.\n" + LAYER + "\n")
    p = p.replace("CURRENT ROBOT MEASUREMENTS: ", insert + "CURRENT ROBOT MEASUREMENTS: ", 1)
    p = p.replace("RESPONSE FIELDS: assessment_of_previous_action",
                  "RESPONSE FIELDS: interaction_mode (one of " + ", ".join(MODES) + ": what this decision does, see EXECUTION LAYER); "
                  "active_hand (left, right or none: the hand this decision manipulates with); assessment_of_previous_action", 1)
    return p
