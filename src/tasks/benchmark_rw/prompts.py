INSTRUCTIONS = (
    "Solve the user's real-world problem. Make reasonable assumptions explicit, "
    "prioritize practical correctness, and give a useful final answer."
)


io = """Solve the user's problem as well as possible.

Problem:
{question}

Final Answer:"""


cot = """Solve the user's problem as well as possible.

Problem:
{question}

Think through the problem carefully, then provide the final answer.

Rules:
- State important assumptions when they matter.
- Prefer concrete, actionable reasoning over vague advice.
- End with a line in the exact format: Final Answer: <answer>

Response:
"""


act = """You are improving a candidate solution to a real-world problem.

Problem:
{question}

Current solution:
{current_draft}

Return one improved candidate solution.
- Preserve useful work from the current solution.
- Correct mistakes and fill important gaps.
- Make the answer practical and directly responsive.
- Do not include meta-commentary.

Candidate draft:
"""


bfs = """You are generating alternative candidate solutions to a real-world problem.

Problem:
{question}

Current solution:
{current_draft}

Return multiple distinct candidate solutions.
Format exactly as:
Candidate 1: ...
Candidate 2: ...
Candidate 3: ...

Candidates:
"""


aggregate = """You are selecting the strongest candidate solutions to a real-world problem.

Problem:
{question}

Select the best {k} candidates based on correctness, completeness, feasibility, and clarity.
Return only the candidate numbers separated by commas.

Candidate solutions:
{actions}

Selected candidates:
"""


evaluate = """You are judging a candidate solution to a real-world problem.
Be strict and use the full scoring range. Generic, plausible advice should not receive a high score.

Problem:
{question}

Score the solution from 0.0 to 1.0 using these criteria:
- correctness and sound reasoning
- completeness relative to the user's request
- practical usefulness and feasibility
- clarity and appropriate caveats
- specificity to this exact problem
- clear sequencing, tradeoffs, and failure modes when relevant

Scoring guide:
- 0.0-0.2: unusable or mostly irrelevant
- 0.3-0.5: partially useful but vague, generic, or missing major parts
- 0.6-0.75: solid but still incomplete or not well tailored
- 0.8-0.9: strong, actionable, and mostly complete
- 0.95-1.0: exceptional; hard to substantially improve

First give a short analysis, then end with: Score: <number>

Solution:
{current_draft}

Evaluation:
"""


react = """Think about the user's problem, then produce an improved solution.

Problem:
{question}

Current solution:
{current_draft}

Format exactly:
Thought: ...
Draft: ...

Response:
"""


self_evaluate_step = """You are evaluating whether a solution adequately answers a real-world problem.

Problem:
{question}

Solution:
{draft}

Is this solution correct, complete, useful, and clear enough to be considered solved? Answer with a single word: Yes or No.
"""


self_evaluate_answer = self_evaluate_step


JURY_PROMPT = """You are tasked with evaluating the quality of a generated solution to a real-world problem.
Your goal is to assess how well the response solves the user's request.
Be a strict judge. A response should only receive very high scores if it is tailored to the specific request,
practically actionable, complete enough to execute, and clear about important tradeoffs or assumptions.
Generic but plausible advice should receive middling scores, even if it is not wrong.

The user's request will be provided in these tags:
<user_request>
{QUESTION}
</user_request>

The response will be provided in these tags:
<response>
{RESPONSE}
</response>

Carefully analyze the <response>. Use the full 1-5 range:
- 1 = unusable, mostly wrong, or does not answer the request
- 2 = weak, vague, missing major parts, or likely to mislead
- 3 = acceptable but generic, incomplete, or only partly actionable
- 4 = strong and mostly complete, but still has noticeable gaps or weak tailoring
- 5 = excellent, specific, actionable, well scoped, and robust to likely edge cases

Do not give a 5 unless the response would be hard to substantially improve.
For each category, rate the Response on a scale of 1 to 5 (1 = very poor, 5 = excellent),
and provide a short justification for your score.

Evaluation Criteria:
Accuracy (1-5)
- Is the reasoning sound, factual, and consistent with the user's request?
- Does it avoid unsupported assumptions or oversimplified claims?

Completeness (1-5)
- Does the response address the important parts of the problem and include necessary caveats or assumptions?
- Does it include enough concrete steps, priorities, and failure modes for the user to act on it?

Clarity (1-5)
- Is the response understandable, well organized, and directly useful?
- Is it specific rather than boilerplate?

Important scoring guidance:
- A first-pass list of reasonable best practices is usually a 3, not a 5.
- A good but generic answer is usually a 3 or 4.
- Reserve 5 for answers that are highly tailored to the exact problem and include practical sequencing,
  tradeoffs, risks, and what to do if assumptions fail.

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
- Use double quotes for all keys and string values.
- Do not include markdown fences or any additional information.
"""
