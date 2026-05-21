from typing import Tuple

import pandas as pd

from .state import StateMathArena
from ... import BenchmarkFactory
from ...typedefs import Benchmark
from ...utils import deterministic_shuffle, parse_n_split

from .parser import parse_grading


def get_split_sizes(total_size, proportions=(0.1, 0.4, 0.25, 0.25)):
    sizes = [max(1, int(total_size * p)) for p in proportions]
    return sizes


@BenchmarkFactory.register
class BenchmarkMathArena(Benchmark):
    def __init__(self, path: str, split: str = "mini", max_len: int = None):
        self.name = "matharena"
        
        df = pd.read_json(path.replace("csv", "jsonl"), lines=True,
                          compression='gzip')
        df.reset_index(inplace=True)
        
        # Parse the problems using the MathArena parser
        df['parsed_problem'] = df['problem'].apply(parse_grading)

        # Prepare the dataset
        data = list(zip(df['problem_idx'], df['problem'], df['parsed_problem'], df['answer']))

        # Compute the idxs for each subset
        shuffled_idxs = deterministic_shuffle(list(range(len(data))))
        mini, train, val, test = get_split_sizes(len(data))

        mini_set_idxs = shuffled_idxs[:mini]
        train_set_idxs = shuffled_idxs[mini:mini + train]
        validation_set_idxs = shuffled_idxs[mini + train:mini + train + val]
        test_set_idxs = shuffled_idxs[mini + train + val:mini + train + val + test]

        if split == "full":
            self.data = data
        elif split == "single":
            self.data = data[:1]
        elif split == "mini":
            self.data = [data[i] for i in mini_set_idxs] or data[:1]
        elif split.startswith("n["):
            self.data = deterministic_shuffle(data)[:parse_n_split(split)]
        elif split == "train":
            self.data = [data[i] for i in train_set_idxs]
        elif split == "validation":
            self.data = [data[i] for i in validation_set_idxs]
        elif split == "test":
            self.data = [data[i] for i in test_set_idxs]
        else:
            raise ValueError(f"Invalid set name: {split}")

        if max_len:
            self.data = self.data[:max_len]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[int, StateMathArena]:
        index = self.data[idx][0]
        problem = self.data[idx][1]
        parsed_problem = self.data[idx][2]
        answer = self.data[idx][3]

        # Create a state object
        state = StateMathArena(
            problem_idx=index,
            problem=problem,
            current_state=problem,
            steps=[],
            answer=str(answer),
            parsed_problem=parsed_problem,
            randomness=0,
        )
        return index, state

