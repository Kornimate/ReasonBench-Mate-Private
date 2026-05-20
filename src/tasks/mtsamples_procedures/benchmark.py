import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from helm.benchmark.scenarios.mtsamples_procedures_scenario import (
    MTSamplesProceduresScenario,
)

from .state import StateMTSamplesProcedures
from ... import BenchmarkFactory
from ...typedefs import Benchmark


def _get_nested_attr(obj: Any, attr_path: str, default: Any = None) -> Any:
    current = obj
    for attr in attr_path.split("."):
        current = getattr(current, attr, default)
        if current is default:
            return default
    return current


def _coerce_reference_text(instance: Any) -> str:
    references = _get_nested_attr(instance, "references", None)
    if references:
        parts = []
        for reference in references:
            output = getattr(reference, "output", None)
            text = getattr(output, "text", None) if output is not None else getattr(reference, "text", "")
            if text:
                parts.append(str(text).strip())
        if parts:
            return "\n".join(parts).strip()

    output_text = _get_nested_attr(instance, "output.text", None)
    if output_text:
        return str(output_text).strip()

    first_ref = _get_nested_attr(instance, "first_correct_reference.output.text", None)
    if first_ref:
        return str(first_ref).strip()

    return ""


def load_instances(dataset_dir: Path) -> List[Dict[str, str]]:
    scenario = MTSamplesProceduresScenario()
    helm_instances = scenario.get_instances(str(dataset_dir))

    instances: List[Dict[str, str]] = []
    for idx, instance in enumerate(helm_instances):
        prompt_text = _get_nested_attr(instance, "input.text", None)
        answer = _coerce_reference_text(instance)
        if not prompt_text or not answer:
            continue

        prompt_text = str(prompt_text).strip()
        title = (
            _get_nested_attr(instance, "id", None)
            or _get_nested_attr(instance, "split", None)
            or f"mtsamples_procedures_{idx}"
        )

        note_marker = "\nNote:"
        note_index = prompt_text.lower().find(note_marker.lower())
        extracted_note_text = (
            prompt_text[note_index + len(note_marker):].strip()
            if note_index != -1
            else prompt_text
        )

        instances.append(
            {
                "id": str(title),
                "title": str(title),
                "note_text": extracted_note_text,
                "answer": answer,
                "puzzle": prompt_text,
            }
        )

    return instances


def split_instances(instances: List[Dict[str, str]], split: str) -> List[Dict[str, str]]:
    if split == "single":
        return instances[:1]
    if split == "mini":
        return instances[:10]

    shuffled = list(instances)
    random.Random(0).shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * 0.6)
    n_validation = int(n_total * 0.2)

    train = shuffled[:n_train]
    validation = shuffled[n_train:n_train + n_validation]
    test = shuffled[n_train + n_validation:]

    if split == "train":
        return train
    if split == "validation":
        return validation
    if split == "test":
        return test
    raise ValueError(f"Invalid split name: {split}")


@BenchmarkFactory.register
class BenchmarkMTSamplesProcedures(Benchmark):
    def __init__(self, path: str, split: str = "mini", max_len: int = None):
        self.name = "mtsamples_procedures"
        dataset_dir = Path("datasets/medical/mtsamples_procedures")
        instances = load_instances(dataset_dir)
        self.data = split_instances(instances, split)

        if max_len is not None:
            self.data = self.data[:max_len]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, StateMTSamplesProcedures]:
        sample = self.data[idx]
        state = StateMTSamplesProcedures(
            puzzle=sample["puzzle"],
            current_state="",
            steps=[],
            answer=sample["answer"],
            title=sample["title"],
            source_id=sample["id"],
            note_text=sample["note_text"],
            randomness=None,
        )
        return idx, state
