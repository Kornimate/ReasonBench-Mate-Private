import math
import random
import re
from typing import List, Tuple
from dataclasses import dataclass

from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr

from ...typedefs import Environment, MAX_SEED
from ... import EnvironmentFactory
from .state import StateMathArena


@dataclass
@EnvironmentFactory.register
class EnvironmentMathArena(Environment):
    """Environment for MathArena task."""

    @staticmethod
    def step(state: StateMathArena, action: str) -> StateMathArena:
        action = normalize_action(action)
        steps = state.steps + [action]
        current_state = "\n".join(steps)

        random.seed(state.randomness)
        randomness = random.randint(0, MAX_SEED)

        return StateMathArena(
            problem_idx=state.problem_idx,
            problem=state.problem,
            answer=state.answer,
            parsed_problem=state.parsed_problem,
            steps=steps,
            current_state=current_state,
            randomness=randomness,
            step_n=state.step_n + 1,
            values=state.values.copy(),
        )

    @staticmethod
    def is_final(state: StateMathArena) -> bool:
        return bool(state.steps) and extract_final_answer(state.steps[-1]) is not None

    @staticmethod
    def evaluate(state: StateMathArena) -> Tuple[bool, float]:
        if not EnvironmentMathArena.is_final(state):
            return False, 0.0

        predicted_answer = extract_final_answer(state.steps[-1])
        is_correct = answers_match(predicted_answer, state.answer)
        return True, float(is_correct)

    @staticmethod
    def is_valid(state: StateMathArena, action: str = None) -> bool:
        if not state.problem or state.answer is None:
            return False

        steps = state.steps if action is None else state.steps + [action]
        return all(isinstance(step, str) and bool(step.strip()) for step in steps)

    @staticmethod
    def apply_action(state: StateMathArena, action: str) -> StateMathArena:
        return EnvironmentMathArena.step(state, action)

    @staticmethod
    def get_valid_actions(state: StateMathArena) -> List[str]:
        actions = [
            "Analyze[Identify the relevant quantities and constraints.]",
            "Explain[Derive the next mathematical relation.]",
        ]
        if state.steps:
            actions.append("Finish[answer]")

        return actions


def normalize_action(action: str) -> str:
    action = str(action or "").strip()
    if not action:
        return "Analyze[No useful step was produced.]"
    if extract_final_answer(action) is not None:
        return f"Finish[{extract_final_answer(action)}]"
    return action


def extract_final_answer(text: str) -> str | None:
    text = str(text or "").strip()
    patterns = [
        r"Finish\[(.*?)\]\s*$",
        r"Final Answer\s*:\s*(.*)$",
        r"final answer is\s*(.*)$",
        r"answer is\s*(.*)$",
        r"Answer\s*:\s*(.*)$",
        r"\\boxed\{([^{}]+)\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return clean_answer(match.group(1))
    return None


def clean_answer(answer: str) -> str:
    answer = str(answer or "").strip()
    answer = re.sub(r"^\$+|\$+$", "", answer)
    answer = answer.strip(" .,\n\t")
    return answer


def normalize_answer(answer: str) -> str:
    answer = clean_answer(answer).lower()
    answer = answer.replace("\\,", "")
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = re.sub(r"\\(?:dfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", answer)
    answer = re.sub(r"\\text\{([^{}]*)\}", r"\1", answer)
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("{", "(").replace("}", ")")
    answer = re.sub(r"\s+", "", answer)
    return answer


def answers_match(predicted: str, expected: str) -> bool:
    predicted_norm = normalize_answer(predicted)
    expected_norm = normalize_answer(expected)
    if predicted_norm == expected_norm:
        return True

    predicted_expr = _to_expr(predicted_norm)
    expected_expr = _to_expr(expected_norm)
    if predicted_expr is None or expected_expr is None:
        return False

    try:
        difference = simplify(predicted_expr - expected_expr)
        if difference == 0:
            return True
        return bool(math.isclose(float(difference), 0.0, rel_tol=1e-9, abs_tol=1e-9))
    except Exception:
        return False


def _to_expr(answer: str):
    if not re.fullmatch(r"[0-9a-zA-Z_+\-*/().^]+", answer):
        return None
    try:
        return parse_expr(answer.replace("^", "**"), evaluate=True)
    except Exception:
        return None
