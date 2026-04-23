from dataclasses import dataclass, field
from typing import Dict, List

from ...typedefs import State


@dataclass(frozen=True)
class StatePubMedQA(State):
    prompt_text: str
    instance_input: str
    question: str
    current_state: str
    steps: List[str]
    answer: str
    source_id: str
    randomness: int
    step_n: int = 0
    values: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)

    def serialize(self) -> dict:
        return {
            "question": self.question,
            "current_state": self.current_state,
            "steps": " -> ".join(self.steps),
            "metrics": dict(self.metrics),
        }

    def clone(self, randomness: int = None) -> "StatePubMedQA":
        return StatePubMedQA(
            prompt_text=self.prompt_text,
            instance_input=self.instance_input,
            question=self.question,
            current_state=self.current_state,
            steps=self.steps,
            answer=self.answer,
            source_id=self.source_id,
            randomness=randomness or self.randomness,
            step_n=self.step_n,
            values=self.values,
            metrics=self.metrics,
        )

    def get_seed(self) -> int:
        return self.randomness

    def __hash__(self) -> int:
        return hash(str(self.serialize()))
