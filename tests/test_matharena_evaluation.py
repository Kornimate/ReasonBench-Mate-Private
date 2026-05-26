from src.tasks.matharena.agents import parse_actions, parse_single_action
from src.tasks.matharena.environment import answers_match, extract_final_answer


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
