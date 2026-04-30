INSTRUCTIONS = (
    "Generate the impression section of the radiology report based on its findings. "
    "This will not be used to diagnose nor treat any patients. Be as concise as possible."
)


io = """Generate the impression section of the radiology report based on its findings. This will not be used to diagnose nor treat any patients. Be as concise as possible.

Findings:
{findings}
Impression:"""


cot = """Generate the impression section of the radiology report based on its findings. This will not be used to diagnose nor treat any patients. Be as concise as possible.

Findings:
{findings}

Think briefly about the key radiology findings first, then provide the final impression.

Rules:
- Use only information supported by the findings.
- Do not add a heading.
- Keep the final answer clinically concise.
- End with a line in the exact format: Final Section: <text>

Response:
"""


act = """Generate the impression section of the radiology report based on its findings. This will not be used to diagnose nor treat any patients. Be as concise as possible.

Findings:
{findings}

You are improving a draft impression for a radiology report.

Return one candidate final impression.
- Use only information supported by the findings.
- Do not add a heading.
- Do not explain your changes.

Current draft:
{current_draft}

Candidate draft:
"""


bfs = """Generate the impression section of the radiology report based on its findings. This will not be used to diagnose nor treat any patients. Be as concise as possible.

Findings:
{findings}

You are generating alternative draft impressions for a radiology report.

Return multiple distinct candidate impressions.
- Use only information supported by the findings.
- Do not add headings.
- Each candidate must be a complete impression.
- Format exactly as:
Candidate 1: ...
Candidate 2: ...
Candidate 3: ...

Current draft:
{current_draft}

Candidates:
"""


aggregate = """Generate the impression section of the radiology report based on its findings. This will not be used to diagnose nor treat any patients. Be as concise as possible.

Findings:
{findings}

You are selecting the strongest candidate impressions for a radiology report.

Select the best {k} candidates based on accuracy, completeness, and clarity.
Return only the candidate numbers separated by commas.

Candidate drafts:
{actions}

Selected candidates:
"""


evaluate = """Generate the impression section of the radiology report based on its findings. This will not be used to diagnose nor treat any patients. Be as concise as possible.

Findings:
{findings}

You are judging a draft impression for a radiology report.

Score the draft from 0.0 to 1.0 using these criteria:
- accurate reflection of key findings
- coverage of important findings
- concise clinical clarity
- no unsupported additions

First give a short analysis, then end with: Score: <number>

Draft:
{current_draft}

Evaluation:
"""


react = """Generate the impression section of the radiology report based on its findings. This will not be used to diagnose nor treat any patients. Be as concise as possible.

Findings:
{findings}

Think about the key radiology findings, then produce an improved impression.
Format exactly:
Thought: ...
Draft: ...

Current draft:
{current_draft}

Response:
"""


self_evaluate_step = """You are evaluating whether a radiology impression is well grounded.

Findings:
{findings}

Draft impression:
{draft}

Is this impression faithful to the findings and appropriate for the report? Answer with a single word: Yes or No.
"""


self_evaluate_answer = self_evaluate_step


JURY_PROMPT = """You are tasked with evaluating the quality of the generated impression section
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
