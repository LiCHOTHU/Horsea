"""Astra as the online decision-maker through the installed Codex CLI (saved ChatGPT login, no API key).

One fresh `codex exec` process per decision, run under a Landlock filesystem sandbox (landlock_exec.py): the model
process can read only system libraries, its own binary and the decision's input folder; it cannot list or read
the Codex home, the simulator, the evaluator or any other experiment file. Tools, plugins, MCP servers, project
docs, web search and session persistence are disabled. The prompt and images are the ONLY information channel.
"""
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from PIL import Image

from schema import ACTION_KEYS, MAX_REPEAT, response_schema

DISABLED_FEATURES = ["shell_tool", "unified_exec", "unified_exec_tty", "apps", "plugins", "memories", "multi_agent",
                     "multi_agent_v2", "browser_use", "browser_use_external", "computer_use", "image_generation",
                     "view_image", "skill_search", "skill_mcp_dependency_install", "hooks", "code_mode",
                     "in_app_browser", "sleep_tool", "worktrees", "goals",
                     "tool_search", "js_repl", "search_tool", "mentions_v2", "collaboration_modes",
                     "realtime_conversation", "remote_plugin", "plugin_sharing", "shell_snapshot"]
ALLOWED_ITEM_TYPES = {"reasoning", "agent_message"}
HERE = Path(__file__).resolve().parent


def codex_binary():
    p = shutil.which("codex")
    if p is None:
        raise RuntimeError("codex CLI not found on PATH")
    return os.path.realpath(p)


def landlock_spec(decision_dir, codex_home):
    bin_dir = str(Path(codex_binary()).parent.parent)       # the release directory (bin/, codex-resources/, ...)
    rules = [{"path": p, "access": "ro"} for p in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/proc",
                                                   "/run/systemd/resolve", "/run/resolvconf", bin_dir)]
    rules += [{"path": "/dev", "access": "rw"}, {"path": str(decision_dir), "access": "rw"},
              {"path": str(codex_home), "access": "files_rw_nolist"}]
    env = {"HOME": os.path.expanduser("~"), "CODEX_HOME": str(codex_home), "PATH": "/usr/bin:/bin",
           "TMPDIR": str(decision_dir), "LANG": "C.UTF-8", "TERM": "dumb", "NO_COLOR": "1"}
    return {"rules": rules, "env": env}


def build_prompt(task, obs, history_view, budgets, previous_image_note):
    axes = obs["camera_axes"]
    ax = lambda cam: "; ".join(f"{k}: {v}" for k, v in axes[cam].items())
    lines = [
        "You directly control a simulated Franka Panda arm (LIBERO / robosuite). There is no motion planner, expert policy, "
        "grasp detector or object-pose sensor acting for you: your only information is the attached camera images, the "
        "robot's own measurements below, and your recorded interaction history. Reply with ONE JSON object matching the "
        "provided schema. Use no tools.",
        "",
        f"TASK: \"{task['instruction']}\"",
        "The attempt ends when the environment itself detects task completion, when the step horizon is reached, when you "
        "set stop=true, or when the model-call budget is exhausted. Only the environment judges completion; stop=true is "
        "NOT success. Previous attempts (if any) ended without completion unless stated otherwise.",
        "",
        "ACTION INTERFACE (verified on this exact environment):",
        f"- {len(ACTION_KEYS)} numbers in [-1, 1]: dx, dy, dz = end-effector translation deltas in the WORLD frame; drx, dry, "
        "drz = rotation deltas about the world x, y, z axes; gripper.",
        f"- One control step = {task['control_dt_s']} s. Your command is applied `repeat` times (1-{MAX_REPEAT}); each "
        "repetition moves the robot FURTHER (it is not holding a pose). Measured: at |d|=1 the hand moves about 0.8-1.1 cm "
        "per step (6 steps of dx=+1 -> +5.0 cm x; 6 steps of dy=+1 -> +6.8 cm y; 6 steps of dz=+1 -> +5.9 cm z). Rotation: "
        "about 3 degrees per step at |dr|=1, with ~0.5 cm/step of position drift, so re-check position after rotating.",
        f"- gripper: +1 drives the fingers toward closed, -1 toward open, 0 holds the current opening ({task['gripper_note']}). "
        "Closing from the start opening takes about 8-10 steps. Finger opening width is reported in cm: fully open ~7.8, "
        "start ~4.2, fully closed on nothing ~0.2; it stays larger when an object is between the fingers. Keep the fingers "
        "closed (gripper 0 or +1) while carrying.",
        "- To wait and observe, send all zeros with a small repeat.",
        "- Start orientation: quaternion (x,y,z,w) ~ (1,0,0,0) means the gripper points straight down with the fingers "
        "opening along the world y axis.",
        "",
        "IMAGES: image 1 = 'agentview', a fixed camera in front of the table facing the robot (robot base at the top of the "
        "image, near table edge at the bottom). image 2 = 'wrist', the camera on the gripper looking along the fingers; the "
        "two dark shapes at the bottom are the fingertips." + previous_image_note,
        "Where a +5 cm hand move along each world axis appears RIGHT NOW (from camera calibration):",
        f"  agentview: {ax('agentview')}",
        f"  wrist:     {ax('wrist')}",
        "",
        "CURRENT ROBOT MEASUREMENTS: " + json.dumps(obs["proprio"]),
        f"env_step {obs['env_step']} of {task['attempt_horizon_steps']} this attempt; task_complete: {obs['task_complete']}.",
        "BUDGET: " + json.dumps(budgets),
        "",
        "INTERACTION HISTORY (recorded by the harness; `result` fields are measured, not your guesses):",
        json.dumps(history_view),
        "",
        "RESPONSE FIELDS: assessment_of_previous_action (first: compare the measured/visible effect of your previous action "
        "with what you expected and say what it implies; null only if there was no previous action in this attempt); action "
        "(7 numbers in [-1,1]); repeat (integer 1-20; capped by the remaining steps); stop (true only to give up this attempt); "
        "intent (1-2 sentences: what this action is for now); hypothesis (what you are testing with it, or null); "
        "expected_change (what should visibly/measurably change if it works); references (ids of past interactions you "
        "relied on); memory_update (REPLACES your running memory: 'observations' = established facts you saw or measured, "
        "citing interaction ids, max 8 items; 'hypotheses' = unconfirmed explanations, max 4; 'summary' <= 1200 chars, carry "
        "forward older facts with their ids). Distinguish observations from hypotheses. Be concise.",
    ]
    return "\n".join(lines)


class CodexAstra:
    def __init__(self, model="gpt-6-astra", reasoning="medium", timeout=180.0, codex_home=None, deny_net=False):
        self.model, self.reasoning, self.timeout, self.deny_net = model, reasoning, float(timeout), deny_net
        self.codex_home = Path(codex_home or os.environ.get("CODEX_HOME", os.path.expanduser("~/.codex")))
        self.binary = codex_binary()
        self.python = os.path.realpath(shutil.which("python3") or "python3")

    def command(self, decision_dir, schema_path, out_path, images):
        cmd = [self.binary, "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
               "--json", "--color", "never", "-m", self.model, "-C", str(decision_dir), "--output-schema", str(schema_path),
               "-o", str(out_path), "-c", 'approval_policy="never"', "-c", f'model_reasoning_effort="{self.reasoning}"',
               "-c", 'web_search="disabled"', "-c", "project_doc_max_bytes=0", "-c", "mcp_servers={}",
               "-c", "features.skip_host_skill_discovery=true", "-c", "suppress_unstable_features_warning=true"]
        for f in DISABLED_FEATURES:
            cmd += ["-c", f"features.{f}=false"]
        for p in images:
            cmd += ["-i", str(p)]
        cmd.append("-")
        return cmd

    def act(self, decision_dir, images, prompt):
        """images: dict name -> uint8 array (attached in dict order). Returns (parsed JSON or None, metadata)."""
        d = Path(decision_dir)
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, (name, px) in enumerate(images.items(), 1):
            p = d / f"image{i}_{name}.png"
            Image.fromarray(px).save(p)
            paths.append(p)
        schema_path, out_path = d / "response_schema.json", d / "response.json"
        schema_path.write_text(json.dumps(response_schema()))
        (d / "prompt.txt").write_text(prompt)
        spec_path = d / "landlock_spec.json"
        spec_path.write_text(json.dumps(dict(landlock_spec(d, self.codex_home), deny_net=self.deny_net), indent=1))
        cmd = self.command(d, schema_path, out_path, paths)
        launcher = [self.python, str(HERE / "landlock_exec.py"), str(spec_path), "--"] + cmd
        (d / "command.json").write_text(json.dumps({"launcher": launcher, "stdin": "prompt.txt"}, indent=1))
        meta = {"requested_model": self.model, "reasoning": self.reasoning, "timeout_s": self.timeout}
        t0 = time.monotonic()
        proc = subprocess.Popen(launcher, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True, env={"PATH": "/usr/bin:/bin"})
        try:
            stdout, stderr = proc.communicate(prompt, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            meta.update(latency_s=round(time.monotonic() - t0, 2), outcome="timeout", returncode=None)
            (d / "stdout.jsonl").write_text(stdout or "")
            (d / "stderr.txt").write_text(stderr or "")
            return None, meta
        meta.update(latency_s=round(time.monotonic() - t0, 2), returncode=proc.returncode)
        (d / "stdout.jsonl").write_text(stdout or "")
        (d / "stderr.txt").write_text(stderr or "")
        events = []
        for line in (stdout or "").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        meta["n_events"] = len(events)
        meta["item_types"] = sorted({e.get("item", {}).get("type") for e in events if e.get("item")} - {None})
        meta["model_identity_in_events"] = sorted({str(v) for e in events for k, v in _walk(e) if k == "model"})
        meta["usage"] = next((e.get("usage") for e in reversed(events) if e.get("type") == "turn.completed"), None)
        meta["reasoning_summaries"] = [v for e in events if e.get("item", {}).get("type") == "reasoning"
                                       for k, v in _walk(e["item"]) if k in ("text", "summary") and isinstance(v, str)]
        violations = [t for t in meta["item_types"] if t not in ALLOWED_ITEM_TYPES | {"error"}]
        meta["error_items"] = [e["item"].get("message") for e in events if e.get("item", {}).get("type") == "error"][:6]
        meta["error_events"] = [e.get("message") or e.get("error") for e in events if e.get("type") == "error"][:6]
        failed = [e for e in events if e.get("type") == "turn.failed"]
        completed = any(e.get("type") == "turn.completed" for e in events)
        if proc.returncode != 0:
            meta.update(outcome="cli_error", stderr_tail=(stderr or "")[-1500:])
            return None, meta
        if violations:
            meta.update(outcome="isolation_violation", violations=violations)
            return None, meta
        if failed or not completed:
            meta.update(outcome="model_error", errors=[e.get("error") for e in failed][:3] or meta["error_events"])
            return None, meta
        if not out_path.exists():
            meta.update(outcome="no_structured_output")
            return None, meta
        try:
            parsed = json.loads(out_path.read_text())
        except json.JSONDecodeError as ex:
            meta.update(outcome="unparseable_output", error=str(ex))
            return None, meta
        meta["outcome"] = "response"
        return parsed, meta


def _walk(x, key=None):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _walk(v, k)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v, key)
    else:
        yield key, x
