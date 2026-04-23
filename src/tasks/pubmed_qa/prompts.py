INSTRUCTIONS = (
    "Answer A for yes, B for no or C for maybe. "
    "Do not include any explanation or additional text. "
    "Output only the letter on a single line."
)


io = """{prompt_text}"""


cot = """{instance_input}

Answer A for yes, B for no or C for maybe.
Think briefly about the evidence first, then end with a final answer.

Format exactly:
Thought: ...
Final Answer: <A/B/C>
"""


act = """{instance_input}

Answer A for yes, B for no or C for maybe.
You are improving the current answer choice based on the evidence.
Return only one improved answer choice.

Current answer:
{current_answer}

Answer:
"""


bfs = """{instance_input}

Answer A for yes, B for no or C for maybe.
Generate multiple distinct candidate answer choices.

Format exactly:
Candidate 1: <A/B/C>
Candidate 2: <A/B/C>
Candidate 3: <A/B/C>

Current answer:
{current_answer}

Candidates:
"""


aggregate = """{instance_input}

Select the best {k} candidates based on how well they are supported by the evidence.
Return only the candidate numbers separated by commas.

Candidate answers:
{actions}

Selected candidates:
"""


evaluate = """{instance_input}

You are judging whether the candidate answer is well supported by the evidence.
Score the candidate from 0.0 to 1.0.
Use higher scores for answers that are clearly grounded in the provided evidence.

Candidate answer:
{current_answer}

First give a short analysis, then end with: Score: <number>
"""


react = """{instance_input}

Answer A for yes, B for no or C for maybe.
Think about the evidence, then produce an improved answer.

Format exactly:
Thought: ...
Answer: <A/B/C>

Current answer:
{current_answer}
"""


self_evaluate_step = """{instance_input}

Candidate answer: {answer}

Is this answer directly supported by the provided evidence? Answer with a single word: Yes or No.
"""


self_evaluate_answer = self_evaluate_step
