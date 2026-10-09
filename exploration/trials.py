"""Shared trial manager; physical attempts use fresh streams paired across arms.

Backends receive a prefix request with no mode argument. Mode-specific code is
called only after READY. A backend must keep its evaluation truth out of the
returned observation record and expose it separately through evaluation().
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

from .events import Event, Option, Prefix
from .memory import ScalarMemory
from .protocol import METHODS, atomic_json, rng_for, seed_for
from .selectors import select


@dataclass
class PrefixResult:
    event: Prefix
    control_steps: int
    fingerprint: str
    observations: dict


@dataclass
class TailResult:
    event: object
    control_steps: int
    observations: dict


def new_memory(method, prior, scalar_concentration):
    if method not in METHODS:
        raise ValueError(method)
    return ScalarMemory(prior, scalar_concentration) if method == "scalar_voi" else prior.copy()


def probe_strategy(method):
    return {"prior": "greedy", "scalar_voi": "voi", "structured_greedy": "greedy",
            "structured_thompson": "thompson", "structured_voi": "voi"}[method]


class TrialManager:
    def __init__(self, backend, protocol):
        self.backend, self.protocol = backend, protocol

    def attempt(self, scene, replicate, attempt, option, namespace="active"):
        start = time.perf_counter()
        # Reset RNG determines scene only; controllers get fresh explicit streams.
        key = (namespace, scene, replicate, attempt)
        try:
            self.backend.reset(scene, seed_for(*key, "physics"))
            prefix = self.backend.execute_prefix(option.grasp, rng_for(*key, "prefix", option.grasp))
            if not 0 <= prefix.control_steps <= self.protocol.prefix_control_budget:
                raise ValueError("Prefix exceeded the common control budget")
            tail = None
            if prefix.event == Prefix.READY:
                tail = self.backend.execute_mode(option, rng_for(*key, "tail", option.grasp, option.mode))
                if not 0 < tail.control_steps <= self.protocol.tail_control_budget:
                    raise ValueError("Tail must execute within the common control budget")
            event = Event(option, prefix.event, None if tail is None else tail.event)
            online = {"scene": scene, "replicate": replicate, "attempt": attempt,
                      "event": event.record(), "prefix_fingerprint": prefix.fingerprint,
                      "prefix_steps": prefix.control_steps, "tail_steps": 0 if tail is None else tail.control_steps,
                      "observations": {"prefix": prefix.observations,
                                       "tail": None if tail is None else tail.observations},
                      "seconds": time.perf_counter() - start,
                      "execution_seed": seed_for(*key, "physics")}
            # This dict is written to a different file and is NEVER passed to memories/selectors.
            truth = dict(self.backend.evaluation(), scene=scene, replicate=replicate, attempt=attempt,
                         option=asdict(option))
            return event, online, truth
        finally:
            self.backend.close()

    def active(self, method, prior, scene, replicate, out):
        out = Path(out)
        memory = new_memory(method, prior, self.protocol.scalar_concentration)
        online, evaluation = [], []
        for attempt in range(self.protocol.probes + 1):
            final = attempt == self.protocol.probes
            choice = select(memory, "greedy" if final else probe_strategy(method),
                            rng_for("selector", scene, replicate, attempt, method))
            c = memory.options[choice.index]
            before = memory.snapshot()
            event, record, truth = self.attempt(scene, replicate, attempt, c)
            if method != "prior" and not final:
                memory.update(event)
            record.update(method=method, phase="exploit" if final else "probe", decision=asdict(choice),
                          memory_before=before, memory_after=memory.snapshot())
            truth.update(method=method, phase=record["phase"])
            online.append(record)
            evaluation.append(truth)
        self.save(out, online, evaluation)
        return online, evaluation

    def common_history(self, prior, scene, out):
        records, truths, history = [], [], []
        for attempt, (r, m) in enumerate(self.protocol.common_history_options):
            event, record, truth = self.attempt(scene, 0, attempt, Option(r, m), "common")
            record.update(method="shared_collection", phase="probe")
            truth.update(method="shared_collection", phase="probe")
            records.append(record)
            truths.append(truth)
            history.append(event)
        for method in ("prior", "scalar_voi", "structured_voi"):
            memory = new_memory(method, prior, self.protocol.scalar_concentration)
            before = memory.snapshot()
            if method != "prior":
                for event in history:
                    memory.update(event)
            choice = select(memory, "greedy")
            event, record, truth = self.attempt(scene, 0, self.protocol.probes,
                                               memory.options[choice.index], "common")
            record.update(method=method, phase="exploit", decision=asdict(choice),
                          memory_before=before, memory_after=memory.snapshot())
            truth.update(method=method, phase="exploit")
            records.append(record)
            truths.append(truth)
        self.save(Path(out), records, truths)
        return records, truths

    @staticmethod
    def save(out, online, evaluation, complete=True):
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in (("observations.jsonl", online), ("evaluation_only.jsonl", evaluation)):
            path = out / name
            temp = path.with_suffix(".tmp")
            temp.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows))
            temp.replace(path)
        if complete:
            atomic_json(out / "complete.json", {"attempts": len(online)})
