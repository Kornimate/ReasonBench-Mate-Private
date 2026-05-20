import argparse
import json
import re
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Optional, Set


PROMPT_TEMPLATE = """You are tasked with evaluating the quality of the generated impression section
of a radiology report based on the provided findings.
Your goal is to assess how well the impression section captures the all the clinical findings and
how it compares to the gold response in terms of accuracy, completeness, and clarity.

The user's request will be provided in these tags:
<user_request>
{QUESTION}
</user_request>

The response will be provided in these tags:
<response>
{RESPONSE}
</response>

Some potential correct responses will be provided in these tags:
<gold_response>
{GOLD_RESPONSE}
</gold_response>

Carefully analyze the <response>.
For each of the following categories, rate the Response on a scale of 1 to 5 (1 = very poor, 5 = excellent),
and provide a short justification for your score.

Your evaluation should focus on the following criteria:

Evaluation Criteria:
Accuracy (1-5)
- Does the impression correctly reflect the key findings from the radiology report?

Completeness (1-5)
- Does the impression include all important findings and address the clinical question?

Clarity (1-5)
- Is the impression easy for referring clinicians to understand?

Output Format:
Output the evaluation as a single valid JSON object matching the following structure:
{
    "accuracy": {
        "score": 0,
        "explanation": "Explain why this score was given."
    },
    "completeness": {
        "score": 0,
        "explanation": "Explain why this score was given."
    },
    "clarity": {
        "score": 0,
        "explanation": "Explain why this score was given."
    }
}

Ensure the output is valid JSON:
- Use **double quotes** (") for all keys and string values.
- When quoting text or sections inside the explanations, use escaped double quotes (\") to
  maintain valid JSON formatting.
- Do not include any additional information in the output.
"""

ANNOTATION_CRITERIA: Dict[str, Set[str]] = {
    "accuracy": {"score", "explanation"},
    "completeness": {"score", "explanation"},
    "clarity": {"score", "explanation"},
}


MIMIC_RRS_INSTRUCTIONS = (
    "Generate the impression section of the radiology report based on its findings. "
    "This will not be used to diagnose nor treat any patients. Be as concise as possible."
)
MIMIC_RRS_INPUT_NOUN = "Findings"
MIMIC_RRS_OUTPUT_NOUN = "Impression"
MIMIC_RRS_MAX_TOKENS = 128
MIMIC_RRS_STOP_SEQUENCES: List[str] = []
MIMIC_RRS_DEFAULT_JURY_SCORE = 1.0


class JuryOutputParseError(ValueError):
    """Raised when an LLM-as-judge output cannot be parsed as a valid MIMIC-RRS annotation."""


def construct_mimic_rrs_llm_prompt(findings: str) -> str:
    """
    Reconstruct the exact zero-shot prompt sent to the evaluated LLM for MIMIC-RRS.

    This mirrors get_generation_adapter_spec(...) in medhelm_run_specs.py for mimic_rrs:
    - instructions are passed through format_instructions(), adding one trailing newline
    - input_prefix is "Findings:\\n"
    - input_suffix is "\\n"
    - output_prefix is "Impression:\\n", then rstrip() is applied when the gold output is omitted
    - max_train_instances is 0, so no in-context examples are inserted
    - Prompt.text joins the instruction block and eval instance block with AdapterSpec.instance_prefix, "\\n"
    """
    instructions_block = f"{MIMIC_RRS_INSTRUCTIONS}\n"
    eval_instance_block = f"{MIMIC_RRS_INPUT_NOUN}:\n{findings}\n{MIMIC_RRS_OUTPUT_NOUN}:"
    return "\n".join([instructions_block, eval_instance_block])


def construct_mimic_rrs_judge_prompt(findings: str, response: str, gold_response: str) -> str:
    """
    Reconstruct the exact prompt sent to the LLM-as-jury annotator for MIMIC-RRS.

    This mirrors LLMAsJuryAnnotator._interpolate_prompt() with the PROMPT_TEMPLATE from
    mimic_rrs_annotator.py.
    """
    tmpl_text = (
        PROMPT_TEMPLATE.replace("{QUESTION}", "$QUESTION")
        .replace("{RESPONSE}", "$RESPONSE")
        .replace("{GOLD_RESPONSE}", "$GOLD_RESPONSE")
    )
    return Template(tmpl_text).substitute(
        {
            "QUESTION": findings,
            "RESPONSE": response,
            "GOLD_RESPONSE": gold_response,
        }
    )


def sanitize_judge_output(model_response: str) -> str:
    """Mirror LLMAsJuryAnnotator._sanitize_model_response() by extracting the outer JSON object."""
    json_match = re.search(r"\{.*\}", model_response, re.DOTALL)
    return json_match.group(0) if json_match else model_response


def parse_mimic_rrs_judge_output(model_response: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse and validate one raw LLM-as-judge response.

    The expected JSON object has three criteria: accuracy, completeness, and clarity. Each criterion must contain
    a score and explanation. If the response has the same incomplete-JSON failure mode handled by HELM
    ("Expecting ',' delimiter"), this function retries with one appended closing brace.
    """
    sanitized_output = sanitize_judge_output(model_response)
    try:
        annotation = json.loads(sanitized_output)
    except json.JSONDecodeError as error:
        if error.msg != "Expecting ',' delimiter":
            raise JuryOutputParseError(f"Could not parse judge output as JSON: {error}") from error
        try:
            annotation = json.loads(f"{sanitized_output}}}")
        except json.JSONDecodeError as retry_error:
            raise JuryOutputParseError(f"Could not parse judge output after appending a closing brace: {retry_error}")

    validate_mimic_rrs_annotation(annotation)
    return annotation


def validate_mimic_rrs_annotation(annotation: Dict[str, Any]) -> None:
    """Validate the MIMIC-RRS annotation structure used by LLMAsJuryAnnotator._validate_annotation()."""
    for criterion, expected_subkeys in ANNOTATION_CRITERIA.items():
        if criterion not in annotation:
            raise JuryOutputParseError(f"Judge output is missing criterion {criterion!r}")
        for subkey in expected_subkeys:
            if subkey not in annotation[criterion]:
                raise JuryOutputParseError(f"Judge output criterion {criterion!r} is missing subkey {subkey!r}")


def calculate_mimic_rrs_jury_score(
    annotations: Iterable[Optional[Dict[str, Dict[str, Any]]]],
    default_score: float = MIMIC_RRS_DEFAULT_JURY_SCORE,
) -> float:
    """
    Calculate mimic_rrs_accuracy from parsed judge annotations.

    This mirrors LLMJuryMetric.evaluate_generation(): for each successful judge model output, collect every
    criterion's integer score, then return their arithmetic mean. For MIMIC-RRS, that means averaging
    accuracy.score, completeness.score, and clarity.score across all configured judge models. If no scores are
    available, HELM returns the run spec's default_score, which is 1.0 for mimic_rrs.
    """
    scores: List[int] = []
    for annotation in annotations:
        if annotation is None:
            continue
        for criterion_output in annotation.values():
            scores.append(int(criterion_output["score"]))
    return sum(scores) / len(scores) if scores else default_score


def calculate_mimic_rrs_jury_score_from_outputs(
    model_responses: Iterable[str],
    default_score: float = MIMIC_RRS_DEFAULT_JURY_SCORE,
) -> float:
    """Parse raw judge outputs and calculate the final MIMIC-RRS jury score."""
    annotations = [parse_mimic_rrs_judge_output(model_response) for model_response in model_responses]
    return calculate_mimic_rrs_jury_score(annotations=annotations, default_score=default_score)
