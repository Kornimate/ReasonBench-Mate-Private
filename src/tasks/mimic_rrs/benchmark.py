import importlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .prompts import io as TASK_PROMPT
from .state import StateMimicRRS
from ... import BenchmarkFactory
from ...typedefs import Benchmark
from ...utils import parse_n_split

MEDHELM_TASKS_PATH = Path("datasets/medical/medhelm_tasks.json")


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


def _load_medhelm_task_spec(task_name: str) -> Dict[str, Any]:
    task_specs = json.loads(MEDHELM_TASKS_PATH.read_text(encoding="utf-8"))
    if task_name not in task_specs:
        raise KeyError(f"Task '{task_name}' is not defined in {MEDHELM_TASKS_PATH}.")
    return task_specs[task_name]


def _instantiate_medhelm_scenario(task_name: str, dataset_dir: Path) -> Any:
    task_spec = _load_medhelm_task_spec(task_name)
    class_path = task_spec["class_name"]
    module_name, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            "The MIMIC-RRS benchmark requires the optional HELM dependency "
            f"providing `{class_path}`."
        ) from exc
    scenario_cls = getattr(module, class_name)
    args = dict(task_spec.get("args", {}))
    args.setdefault("data_path", str(dataset_dir))
    return scenario_cls(**args)


def _extract_findings(prompt_text: str) -> str:
    text = prompt_text.strip()
    match = re.search(r"Findings:\s*(.*?)\s*Impression:\s*$", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"Findings:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return text


def load_instances(dataset_dir: Path) -> List[Dict[str, str]]:
    scenario = _instantiate_medhelm_scenario("mimic_rrs", dataset_dir)
    helm_instances = scenario.get_instances(str(dataset_dir))

    instances: List[Dict[str, str]] = []
    for idx, instance in enumerate(helm_instances):
        prompt_text = _get_nested_attr(instance, "input.text", None)
        answer = _coerce_reference_text(instance)
        if not prompt_text or not answer:
            continue

        prompt_text = str(prompt_text).strip()
        findings = _extract_findings(prompt_text)
        source_id = (
            _get_nested_attr(instance, "id", None)
            or _get_nested_attr(instance, "split", None)
            or f"mimic_rrs_{idx}"
        )

        instances.append(
            {
                "id": str(source_id),
                "findings": findings,
                "answer": answer,
                "puzzle": TASK_PROMPT.format(findings=findings),
            }
        )

    return instances


def split_instances(instances: List[Dict[str, str]], split: str) -> List[Dict[str, str]]:
    if split == "single":
        return instances[:1]
    if split == "mini":
        return instances[:10]
    if split.startswith("n["):
        return instances[:parse_n_split(split)]

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
class BenchmarkMimic_RRS(Benchmark):
    def __init__(self, path: str, split: str = "mini", max_len: int = None):
        self.name = "mimic_rrs"
        dataset_dir = Path("datasets/medical/mimic_rrs")
        instances = load_instances(dataset_dir)
        self.data = split_instances(instances, split)

        if max_len is not None:
            self.data = self.data[:max_len]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, StateMimicRRS]:
        sample = self.data[idx]
        state = StateMimicRRS(
            puzzle=sample["puzzle"],
            current_state="",
            steps=[],
            answer=sample["answer"],
            source_id=sample["id"],
            findings=sample["findings"],
            randomness=None,
        )
        return idx, state


BenchmarkMIMICRRS = BenchmarkMimic_RRS
