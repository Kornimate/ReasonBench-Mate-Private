import re

try:
    from .matharena_ethz.src.matharena.parser import parse_grading as _parse_grading
except ModuleNotFoundError:
    _parse_grading = None


def parse_grading(text: str) -> dict:
    if _parse_grading is not None:
        return _parse_grading(text)

    pattern = re.compile(
        r"""Category:\s*(.*?)(?:\s|\n)*Points\s+awarded:\s*(.*?)(?:\s|\n)*Description:\s*(.*?)(?:\s|\n)*(?=Category:|$)""",
        re.DOTALL | re.VERBOSE,
    )
    matches = pattern.findall(text)
    
    result = {"points": sum(int(points) for _, points, _ in matches), "details": []}
    
    for title, points, desc in matches:
        result["details"].append({
            "title": title,
            "points": int(points),
            "desc": desc
        })
    
    if not matches:
        result["problem"] = text
        result["boxed_answers"] = re.findall(r"\\boxed\{([^{}]+)\}", text)

    return result
