io = """You are completing missing text from a procedure-related medical note.

Task:
- Write only the missing note text.
- Use only information supported by the note.
- Do not add a heading.
- Do not add warnings, disclaimers, or commentary.

Procedure note title: {title}

Note:
{note_text}

Output:
"""

cot = """You are completing missing text from a procedure-related medical note.

Think briefly about the salient details first, then provide the final missing text.
Rules:
- Use only information supported by the note.
- Do not add a heading.
- Keep the final answer clinically concise.
- End with a line in the exact format: Final Section: <text>

Procedure note title: {title}

Note:
{note_text}

Response:
"""

act = """You are improving a draft for missing text from a procedure-related medical note.

Return one candidate final draft for the missing note text.
- Use only information supported by the note.
- Do not add a heading.
- Do not explain your changes.

Procedure note title: {title}

Note:
{note_text}

Current draft:
{current_draft}

Candidate draft:
"""

bfs = """You are generating alternative drafts for missing text from a procedure-related medical note.

Return multiple distinct candidate drafts for the missing note text.
- Use only information supported by the note.
- Do not add headings.
- Each candidate must be a complete draft of the missing text.
- Format exactly as:
Candidate 1: ...
Candidate 2: ...
Candidate 3: ...

Procedure note title: {title}

Note:
{note_text}

Current draft:
{current_draft}

Candidates:
"""

aggregate = """You are selecting the strongest candidate drafts for missing text from a procedure-related medical note.

Select the best {k} candidates based on faithfulness to the note, clinical usefulness, and completeness of the missing text.
Return only the candidate numbers separated by commas.

Procedure note title: {title}

Note:
{note_text}

Candidate drafts:
{actions}

Selected candidates:
"""

evaluate = """You are judging a draft for missing text from a procedure-related medical note.

Score the draft from 0.0 to 1.0 using these criteria:
- factual support from the note
- fit as the missing text
- coverage of the most important information
- no unsupported additions

First give a short analysis, then end with: Score: <number>

Procedure note title: {title}

Note:
{note_text}

Draft:
{current_draft}

Evaluation:
"""

react = """You are completing missing text from a procedure-related medical note.

Think about what information is missing from the note, then produce an improved draft.
Format exactly:
Thought: ...
Draft: ...

Procedure note title: {title}

Note:
{note_text}

Current draft:
{current_draft}

Response:
"""

self_evaluate_step = """You are evaluating whether a draft for missing medical-note text is well grounded.

Note:
{note_text}

Draft:
{draft}

Is this draft faithful to the note and appropriate as the missing text? Answer with a single word: Yes or No.
"""

self_evaluate_answer = self_evaluate_step

# based on https://arxiv.org/pdf/2505.23802 -> open-ended evaluation with a jury prompt
JURY_PROMPT = """You are a medical expert tasked with evaluating the quality of generated missing text for a procedure-related medical note.
Your goal is to assess how well the generated text captures the clinical information from the note and compare it to the reference text (gold standard) only when needed.

The procedure note title will be provided in these tags:
<title>{title}</title>

The source note will be provided in these tags:
<note>{note_text}</note>

The generated missing text will be provided in these tags:
<response>{response}</response>

The reference missing text will be provided in these tags:
<gold_response>{gold_response}</gold_response>

Carefully review the <response> based on the <note> and compare it to the <gold_response> when needed.
For each of the following criteria, rate the response on a scale of 1 to 5 (1 = very poor, 5 = excellent), and provide a short justification for your score.

Evaluation Criteria:
Accuracy (1-5) - Does the response provide correct clinical information based on the note?
Completeness (1-5) - Does the response include all important medical details needed for the missing text?
Clarity (1-5) - Is the response written clearly and organized in an appropriate clinical style?

Output Format:
Output the evaluation as a single valid JSON object matching the following structure:
{{"accuracy":{{"score":1,"explanation":"Explain why this score was given."}},"completeness":{{"score":1,"explanation":"Explain why this score was given."}},"clarity":{{"score":1,"explanation":"Explain why this score was given."}}}}

Ensure the output is valid JSON:
- Use double quotes for all keys and string values.
- When quoting text or sections inside the explanations, use escaped double quotes to maintain valid JSON formatting.
- Do not include any additional information in the output.
"""
