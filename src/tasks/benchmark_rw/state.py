from dataclasses import dataclass, field
from typing import Dict, List

from ...typedefs import State


@dataclass(frozen=True)
class StateBenchmarkRW(State):
    puzzle: str
    current_state: str
    steps: List[str]
    source_id: str
    question: str
    randomness: int
    step_n: int = 0
    values: Dict = field(default_factory=dict)

    def serialize(self) -> dict:
        return {
            "source_id": self.source_id,
            "question": self.question,
            "current_state": self.current_state,
            "steps": " -> ".join(self.steps),
        }

    def clone(self, randomness: int = None) -> "StateBenchmarkRW":
        return StateBenchmarkRW(
            puzzle=self.puzzle,
            current_state=self.current_state,
            steps=self.steps,
            source_id=self.source_id,
            question=self.question,
            randomness=randomness or self.randomness,
            step_n=self.step_n,
            values=self.values,
        )

    def get_seed(self) -> int:
        return self.randomness

    def __hash__(self) -> int:
        return hash(str(self.serialize()))
