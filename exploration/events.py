"""The online writer's complete input contract (no simulator labels)."""
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Prefix(IntEnum):
    GOAL = 0
    READY = 1
    NOT_READY = 2
    UNKNOWN = 3


class Tail(IntEnum):
    SLIP = 0
    NO_RESPONSE = 1
    AWAY = 2
    TOWARD = 3
    GOAL = 4
    UNKNOWN = 5


@dataclass(frozen=True, order=True)
class Option:
    grasp: int
    mode: int

    def __post_init__(self):
        if type(self.grasp) is not int or type(self.mode) is not int:
            raise TypeError("Option identities must be integer indices")
        if self.grasp < 0 or self.mode < 0:
            raise ValueError("Negative option identity")


@dataclass(frozen=True)
class Event:
    option: Option
    prefix: Prefix
    tail: Optional[Tail] = None

    def __post_init__(self):
        if not isinstance(self.prefix, Prefix):
            raise TypeError("prefix must be a Prefix event")
        if self.tail is not None and not isinstance(self.tail, Tail):
            raise TypeError("tail must be a Tail event")
        if (self.prefix == Prefix.READY) != (self.tail is not None):
            raise ValueError("Exactly READY prefixes execute and record a tail")

    def utility(self, success_only=False):
        if self.prefix == Prefix.GOAL or self.tail == Tail.GOAL:
            return 1.0
        if self.tail == Tail.TOWARD and not success_only:
            return 0.5
        return 0.0

    def record(self):
        return {"grasp": self.option.grasp, "mode": self.option.mode,
                "prefix": self.prefix.name.lower(),
                "tail": None if self.tail is None else self.tail.name.lower()}

    @classmethod
    def from_record(cls, record):
        expected = {"grasp", "mode", "prefix", "tail"}
        if set(record) != expected:
            raise ValueError("Writer records must contain only observable event fields")
        return cls(Option(record["grasp"], record["mode"]),
                   Prefix[record["prefix"].upper()],
                   None if record["tail"] is None else Tail[record["tail"].upper()])
