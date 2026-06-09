# Sources:
# - 

io = """Here are information about a patient, return a reasonable treatment plan for the patient.
Patient Notes: {cleaned_text}
Answer: """

cot = """Here are information about a patient, return a reasonable treatment plan for the patient.
Patient Notes: {cleaned_text}

Think briefly about the salient details first, then provide the final missing text.

Rules:
- Use only information supported by the note.
- Do not add a heading.
- Keep the final answer clinically concise.
- End with a line in the exact format: Final Section: <text>

Response:
"""

act = """Here are information about a patient, return a reasonable treatment plan for the patient.
Patient Notes: {cleaned_text}

You are improving a draft for missing text from a procedure-related medical note.

Return one candidate final draft for the missing note text.
- Use only information supported by the note.
- Do not add a heading.
- Do not explain your changes.

Current draft:
{current_draft}

Candidate draft:
"""

bfs = """Here are information about a patient, return a reasonable treatment plan for the patient.
Patient Notes: {cleaned_text}

You are generating alternative drafts for missing text from a procedure-related medical note.

Return multiple distinct candidate drafts for the missing note text.
- Use only information supported by the note.
- Do not add headings.
- Each candidate must be a complete draft of the missing text.
- Format exactly as:
Candidate 1: ...
Candidate 2: ...
Candidate 3: ...

Current draft:
{current_draft}

Candidates:
"""

aggregate = """Here are information about a patient, return a reasonable treatment plan for the patient.
Patient Notes: {cleaned_text}

You are selecting the strongest candidate drafts for missing text from a procedure-related medical note.

Select the best {k} candidates based on faithfulness to the note, clinical usefulness, and completeness of the missing text.
Return only the candidate numbers separated by commas.

Candidate drafts:
{actions}

Selected candidates:
"""

evaluate = """Here are information about a patient, return a reasonable treatment plan for the patient.
Patient Notes: {cleaned_text}

You are judging a draft for missing text from a procedure-related medical note.

Score the draft from 0.0 to 1.0 using these criteria:
- factual support from the note
- fit as the missing text
- coverage of the most important information
- no unsupported additions

First give a short analysis, then end with: Score: <number>

Draft:
{current_draft}

Evaluation:
"""

react = """Here are information about a patient, return a reasonable treatment plan for the patient.
Patient Notes: {cleaned_text}

You are completing missing text from a procedure-related medical note.

Think about what information is missing from the note, then produce an improved draft.
Format exactly:
Thought: ...
Draft: ...

Current draft:
{current_draft}

Response:
"""

critic = """Here are information about a patient, return a reasonable treatment plan for the patient.
Patient Notes: {cleaned_text}

Act as a verifier. Check the current draft for unsupported medical details, missing salient information, and poor fit as the missing text. Return one revised final draft only.
- Use only information supported by the note.
- Do not add a heading.
- Do not explain your changes.

Current draft:
{current_draft}

Draft:
"""

corrector = """Here are information about a patient, return a reasonable treatment plan for the patient.
Patient Notes: {cleaned_text}

Act as a corrector. Repair the current draft so it is faithful to the note, clinically useful, concise, and complete as the missing text. Return one corrected draft only.
- Use only information supported by the note.
- Do not add a heading.
- Do not explain your changes.

Current draft:
{current_draft}

Draft:
"""

planner = """Here are information about a patient, return a reasonable treatment plan for the patient.
Patient Notes: {cleaned_text}

Act as a planner. Internally identify what the missing text should cover, then return one final draft that follows that plan.
- Use only information supported by the note.
- Do not add a heading.
- Do not explain your changes.

Current draft:
{current_draft}

Draft:
"""

self_evaluate_step = """You are evaluating whether a draft for missing medical-note text is well grounded.

Note:
{note_text}

Draft:
{draft}

Is this draft faithful to the note and appropriate as the missing text? Answer with a single word: Yes or No.
"""

self_evaluate_answer = self_evaluate_step

JURY_PROMPT = """You are a medical expert tasked with evaluating the quality of a
generated response of a clinical scenario.
Your goal is to assess how well the response addresses the patient case, follows clinical best practices,
and compares to the gold response in terms of accuracy, completeness, and clarity.

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

Evaluation Criteria:
Accuracy (1-5)
- Does the response provide correct medical advice based on clinical guidelines and the patient case?

Completeness (1-5)
- Does the response cover all important aspects of care for this clinical scenario?

Clarity (1-5)
- Is the response written clearly and organized in a way that clinicians can easily follow?

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
