"""Small bandit allocators for ReAgents allocation variants."""

import math
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ThompsonArm:
    """Beta-style arm state for bounded fractional rewards."""

    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0
    reward_sum: float = 0.0
    last_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.pulls if self.pulls else 0.0

    def sample(self) -> float:
        return random.betavariate(max(self.alpha, 1e-9), max(self.beta, 1e-9))

    def update(self, reward: float, discount: float = 1.0) -> None:
        reward = float(max(0.0, min(1.0, reward)))
        discount = float(max(0.0, min(1.0, discount)))

        if discount < 1.0:
            self.alpha = 1.0 + discount * (self.alpha - 1.0)
            self.beta = 1.0 + discount * (self.beta - 1.0)
            self.reward_sum *= discount
            self.pulls = int(round(self.pulls * discount))

        self.alpha += reward
        self.beta += 1.0 - reward
        self.pulls += 1
        self.reward_sum += reward
        self.last_reward = reward

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "pulls": self.pulls,
            "reward_sum": self.reward_sum,
            "mean_reward": self.mean_reward,
            "last_reward": self.last_reward,
        }


@dataclass
class ThompsonBandit:
    """Thompson sampling over bounded rewards in [0, 1]."""

    n_arms: int
    discount: float = 1.0
    arms: list[ThompsonArm] = field(init=False)

    def __post_init__(self) -> None:
        if self.n_arms < 1:
            raise ValueError("ThompsonBandit requires at least one arm")
        self.arms = [ThompsonArm() for _ in range(self.n_arms)]

    def select(self, active_indices: list[int] | None = None) -> int:
        indices = list(range(self.n_arms)) if active_indices is None else list(active_indices)
        if not indices:
            return -1

        unpulled = [i for i in indices if self.arms[i].pulls == 0]
        if unpulled:
            return int(random.choice(unpulled))

        samples = [(self.arms[i].sample(), i) for i in indices]
        max_sample = max(sample for sample, _ in samples)
        tied = [i for sample, i in samples if sample == max_sample]
        return int(random.choice(tied))

    def update(self, arm_index: int, reward: float) -> None:
        self.arms[arm_index].update(reward=reward, discount=self.discount)

    def probabilities(self) -> list[float]:
        draws = 128
        wins = np.zeros(self.n_arms, dtype=float)
        for _ in range(draws):
            samples = [arm.sample() for arm in self.arms]
            wins[int(np.argmax(samples))] += 1.0
        probs = wins / wins.sum() if wins.sum() else np.ones(self.n_arms) / self.n_arms
        return probs.tolist()

    def snapshot(self) -> list[dict]:
        return [arm.as_dict() | {"arm": i} for i, arm in enumerate(self.arms)]


@dataclass
class UCBBandit:
    """UCB1 allocator for trajectory or operator selection."""

    n_arms: int
    c: float = math.sqrt(2.0)
    pulls: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.n_arms < 1:
            raise ValueError("UCBBandit requires at least one arm")
        self.pulls = np.zeros(self.n_arms, dtype=int)
        self.rewards = np.zeros(self.n_arms, dtype=float)

    def select(self, active_indices: list[int] | None = None) -> int:
        indices = list(range(self.n_arms)) if active_indices is None else list(active_indices)
        if not indices:
            return -1

        unpulled = [i for i in indices if self.pulls[i] == 0]
        if unpulled:
            return int(random.choice(unpulled))

        total = max(1, int(self.pulls.sum()))
        scores = []
        for i in indices:
            mean = self.rewards[i] / max(1, self.pulls[i])
            bonus = self.c * math.sqrt(math.log(total + 1.0) / max(1, self.pulls[i]))
            scores.append((mean + bonus, i))

        max_score = max(score for score, _ in scores)
        tied = [i for score, i in scores if score == max_score]
        return int(random.choice(tied))

    def update(self, arm_index: int, reward: float) -> None:
        reward = float(max(0.0, min(1.0, reward)))
        self.pulls[arm_index] += 1
        self.rewards[arm_index] += reward

    def snapshot(self) -> list[dict]:
        return [
            {
                "arm": i,
                "pulls": int(self.pulls[i]),
                "reward_sum": float(self.rewards[i]),
                "mean_reward": float(self.rewards[i] / self.pulls[i]) if self.pulls[i] else 0.0,
            }
            for i in range(self.n_arms)
        ]
