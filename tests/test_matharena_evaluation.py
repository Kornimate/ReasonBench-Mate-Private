from src.tasks.matharena.agents import parse_actions, parse_single_action
from src.tasks.matharena.environment import answers_match, extract_final_answer, normalize_action


def test_extract_final_answer_prefers_boxed_inside_finish_blob():
    response = r"""Finish[**

\[
n = \frac{625}{256}
\]

The problem states that \(n = \frac{j}{k}\), with \(j\) and \(k\) coprime.

## Answer:

\(\boxed{881}\)]"""

    assert extract_final_answer(response) == "881"
    assert answers_match(extract_final_answer(response), "881")


def test_extract_final_answer_handles_nested_boxed_fraction():
    response = r"""
### Answer:
\[
\boxed{\boxed{\frac{4}{3}}}
\]
"""

    assert extract_final_answer(response) == r"\frac{4}{3}"
    assert answers_match(extract_final_answer(response), "4/3")


def test_extract_final_answer_handles_reagents_conclusive_analyze_action():
    response = (
        "Analyze[The total number of pairs of lines is C(40, 2) = 780. "
        "The higher-multiplicity points account for 173 pairs. Since total pairs is 780, "
        "the remaining pairs are 780 - 173 = 607. These remaining pairs correspond to "
        "intersection points where exactly 2 lines meet. Each such point accounts for 1 pair, "
        "so the number of these points equals 607.]"
    )

    assert extract_final_answer(response) == "607"


def test_extract_final_answer_handles_markdown_label_with_next_line():
    response = """
### **Final answer:**

881
"""

    assert extract_final_answer(response) == "881"


def test_extract_final_answer_handles_final_result_label():
    response = r"## **Final Result:** \(j + k = 881\)"

    assert extract_final_answer(response) == "881"


def test_extract_final_answer_does_not_finish_setup_step():
    response = (
        "Analyze[The problem asks for the number of points where exactly 2 lines intersect. "
        "Since there are 40 lines, each pair of lines intersects at exactly one point.]"
    )

    assert extract_final_answer(response) is None


def test_normalize_action_converts_conclusive_analyze_to_finish():
    response = (
        "Analyze[Since total pairs is 780, the remaining pairs are 780 - 173 = 607. "
        "Each such point accounts for 1 pair, so the number of these points equals 607.]"
    )

    assert normalize_action(response) == "Finish[607]"


def test_parse_single_action_keeps_content_after_generic_action_header():
    response = """Analyze[problem]

The problem asks for the area of a circle given its radius of 5cm."""

    assert parse_single_action(response) == (
        "Analyze[The problem asks for the area of a circle given its radius of 5cm]"
    )


def test_parse_actions_keeps_inline_content_after_generic_headers():
    response = """Analyze[problem] Focus on identifying function components
Explain[math concepts] Review power rule and derivative rules"""

    assert parse_actions(response) == [
        "Analyze[Focus on identifying function components]",
        "Explain[Review power rule and derivative rules]",
    ]
