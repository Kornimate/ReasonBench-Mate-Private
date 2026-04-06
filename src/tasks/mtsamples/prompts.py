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
