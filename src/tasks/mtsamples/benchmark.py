import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .state import StateMTSamples
from ... import BenchmarkFactory
from ...typedefs import Benchmark


SECTION_PRIORITY = ("PLAN", "SUMMARY", "FINDINGS")
SECTION_ALIASES = {
    "PLAN": "PLAN",
    "POST-OP PLAN": "PLAN",
    "ASSESSMENT AND PLAN": "PLAN",
    "SUMMARY": "SUMMARY",
    "FINDINGS": "FINDINGS",
    "OPERATIVE FINDINGS": "FINDINGS",
    "MAJOR FINDINGS": "FINDINGS",
    "GROSS FINDINGS": "FINDINGS",
    "INTRAOPERATIVE FINDINGS": "FINDINGS",
}
SECTION_PATTERN = re.compile(
    r"(?P<label>ASSESSMENT AND PLAN|POST-OP PLAN|OPERATIVE FINDINGS|INTRAOPERATIVE FINDINGS|MAJOR FINDINGS|GROSS FINDINGS|SUMMARY|FINDINGS|PLAN)\s*:\s*",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ").replace("Â", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_reference_section(text: str) -> Optional[Dict[str, str]]:
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return None

    candidates = []
    for idx, match in enumerate(matches):
        raw_label = re.sub(r"\s+", " ", match.group("label").upper()).strip()
        canonical = SECTION_ALIASES.get(raw_label)
        if canonical is None:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = normalize_whitespace(text[start:end])
        if not content:
            continue
        candidates.append(
            {
                "canonical_label": canonical,
                "raw_label": raw_label,
                "content": content,
                "span_start": match.start(),
                "span_end": end,
            }
        )

    if not candidates:
        return None

    for section_name in SECTION_PRIORITY:
        for candidate in candidates:
            if candidate["canonical_label"] == section_name:
                return candidate
    return None


def remove_span(text: str, start: int, end: int) -> str:
    return normalize_whitespace(text[:start] + "\n\n" + text[end:])


def build_task_prompt(title: str, section_name: str, note_text: str) -> str:
    return (
        "You are given a procedure-related medical note with one section removed.\n"
        f"Write the missing {section_name} section only.\n"
        "Do not add a heading, bullet list, disclaimer, or extra commentary.\n"
        "Base the section only on the note content.\n\n"
        f"Procedure note title: {title}\n"
        f"Missing section: {section_name}\n\n"
        "Note:\n"
        f"{note_text}"
    )


def load_instances(dataset_dir: Path) -> List[Dict[str, str]]:
    instances: List[Dict[str, str]] = []
    for path in sorted(dataset_dir.glob("*.txt")):
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        parsed = extract_reference_section(raw_text)
        if parsed is None:
            continue

        source_text = remove_span(raw_text, parsed["span_start"], parsed["span_end"])
        if not source_text:
            continue

        title = path.stem
        section_name = parsed["canonical_label"]
        answer = parsed["content"]
        instances.append(
            {
                "id": str(path.name),
                "title": title,
                "section_name": section_name,
                "note_text": source_text,
                "answer": answer,
                "puzzle": build_task_prompt(title, section_name, source_text),
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
class BenchmarkMTSamples(Benchmark):
    def __init__(self, path: str, split: str = "mini", max_len: int = None):
        self.name = "mtsamples"
        dataset_dir = Path("datasets/medical/mtsamples_procedures")
        instances = load_instances(dataset_dir)
        self.data = split_instances(instances, split)

        if max_len is not None:
            self.data = self.data[:max_len]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, StateMTSamples]:
        sample = self.data[idx]
        state = StateMTSamples(
            puzzle=sample["puzzle"],
            current_state="",
            steps=[],
            answer=sample["answer"],
            section_name=sample["section_name"],
            title=sample["title"],
            source_id=sample["id"],
            note_text=sample["note_text"],
            randomness=None,
        )
        return idx, state
