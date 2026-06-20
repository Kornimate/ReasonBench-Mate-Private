from typing import Tuple

from .prompts import io as TASK_PROMPT
from .state import StateBenchmarkRW
from ... import BenchmarkFactory
from ...typedefs import Benchmark


@BenchmarkFactory.register
class BenchmarkBenchmarkRW(Benchmark):
    def __init__(self, path: str = "", split: str = "single", max_len: int = None, question: str | None = None):
        self.name = "benchmark_rw"
        self.question = question or "What is the best way to solve this real-world problem?"
        self.data = [
            {
                "id": "user_input_0",
                "question": self.question,
                "puzzle": TASK_PROMPT.format(question=self.question),
            }
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, StateBenchmarkRW]:
        sample = self.data[idx]
        return idx, StateBenchmarkRW(
            puzzle=sample["puzzle"],
            current_state="",
            steps=[],
            source_id=sample["id"],
            question=sample["question"],
            randomness=None,
        )


BenchmarkBenchmark_RW = BenchmarkBenchmarkRW
