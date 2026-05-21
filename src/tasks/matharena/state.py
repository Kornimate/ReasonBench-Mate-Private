from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ...typedefs import State
from .parser import parse_grading


@dataclass(frozen=True)
class StateMathArena(State):
    """State representation for MathArena task."""
    
    problem_idx: str
    problem: str
    answer: str
    parsed_problem: Optional[dict] = None
    steps: List[str] = field(default_factory=list)
    current_state: str = ""
    randomness: int = 0
    step_n: int = 0
    values: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.parsed_problem is None:
            object.__setattr__(self, "parsed_problem", parse_grading(self.problem))
        if not self.current_state:
            object.__setattr__(self, "current_state", self.problem)

    def copy(self) -> "StateMathArena":
        return StateMathArena(
            problem_idx=self.problem_idx,
            problem=self.problem,
            answer=self.answer,
            parsed_problem=self.parsed_problem,
            steps=self.steps.copy(),
            current_state=self.current_state,
            randomness=self.randomness,
            step_n=self.step_n,
            values=self.values.copy(),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "StateMathArena":
        parsed_problem = parse_grading(data["problem"])
        return cls(
            problem_idx=data["problem_idx"],
            problem=data["problem"],
            answer=str(data["answer"]),
            parsed_problem=parsed_problem,
        )

    def to_dict(self) -> dict:
        return {
            "problem_idx": self.problem_idx,
            "problem": self.problem,
            "answer": self.answer,
            "parsed_problem": self.parsed_problem,
            "steps": self.steps,
            "current_state": self.current_state,
            "randomness": self.randomness,
            "step_n": self.step_n,
        }

    def clone(self, randomness: int = None) -> "StateMathArena":
        return StateMathArena(
            problem_idx=self.problem_idx,
            problem=self.problem,
            answer=self.answer,
            parsed_problem=self.parsed_problem,
            steps=self.steps.copy(),
            current_state=self.current_state,
            randomness=randomness if randomness is not None else self.randomness,
            step_n=self.step_n,
            values=self.values.copy(),
        )

    def serialize(self) -> dict:
        return {
            "problem": self.problem,
            "current_state": self.current_state,
            "steps": " -> ".join(self.steps),
        }

    def get_seed(self) -> int:
        return self.randomness

    def __hash__(self) -> int:
        return hash(str(self.serialize()))
