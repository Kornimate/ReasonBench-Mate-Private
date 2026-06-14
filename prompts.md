# Prompts used for our experiments

**Tasks**
- [Game of 24](#game-of-24)
- [HLE](#hle)
- [HotpotQA](#hotpotqa)
- [HumanEval](#humaneval)
- [LogiQA](#logiqa)
- [MathArena](#matharena)
- [MIMIC-RRS](#mimic-rrs)
- [MTSamples Procedures](#mtsamples-procedures)
- [PubMedQA](#pubmedqa)
- [SciBench](#scibench)
- [Sonnet Writing](#sonnet-writing)

## Game of 24
```python
# Updated
act = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Do not explain simply list one possible next step, as well as all the remaining numbers and nothing else.

Example: 2 8 8 14
Possible next step:
14 + 2 = 16 (left: 8 8 16)

Example: 1 4 6
Possible next step:
1 * 4 = 4 (left: 4 6)

Example: 1 3
Possible next step:
1 * 3 = 3 (left: 3)

Input: {input}
Possible next step:
'''

bfs = '''Use numbers and basic arithmetic operations (+ - * /). Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.  Follow the format of the following examples. Do not explain simply list possible next steps as well as all the remaining numbers and nothing else.

Example: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)

Example: 1 3
Possible next steps:
1 + 3 = 4 (left: 4)
1 * 3 = 3 (left: 3)
3 - 1 = 2 (left: 2)
3 / 1 = 3 (left: 3)
1 - 3 = -2 (left: -2)

Input: {input}
Possible next steps:
'''

aggregate = '''
Please select {n_select_sample} step from the proposed step list, which you believe can reach 24. Each of the proposed steps, uses two of the input numbers to obtain a new number. Do not explain, just list the selected steps as well as all the remaining numbers and nothing else. See the examples for how you are expected to respond.

Things you should consider:
- Do not change at all a step, simply select it.
- You can only select steps from proposed next steps and you cannot change or propose new steps.
- The selected steps should be able to reach 24.
- The selected step must be a valid step, following the format of the possible next steps listed in the examples.

Example: 4 8 8
Number of steps to select: 3
Proposed next steps:
(1) 4 * 8 = 32 (left: 8 32)
(2) 8 * 8 = 64 (left: 4 64)
(3) 4 - 8 = -4 (left: -4 8)
(4) 8 - 8 = 0 (left: 0 4)
(5) 8 / 4 = 2 (left: 2 8)
Selected Next Step Set:
1, 2, 5

Remember, your task is to select {n_select_sample} steps from the proposed next steps. Do not change the steps, just select them. Return only the indexes of the selected steps. Do not include any other information, explanations, comments or conclusions.

Input: {state}
Number of steps to select: {n_select_sample}
Proposed next steps:
{proposal}
Selected Next Step Set:
'''

cot = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Return only the complete answer. If the steps are already given, just return the final expression following the given steps. Do not make any simplifications.

Example: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24

Example: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24

Example: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24

Example: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24

Example: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24

Input: {input}
'''

# Updated
evaluate = '''Evaluate if given numbers can reach 24 by responding with the following: "sure", "likely" or "impossible". Follow the format of the following examples. Try to be brief.

Example: 10 14
10 + 14 = 24
sure

Example: 11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible

Example: 4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure

Example: 4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure

Example: 5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely

Example: 5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely

Example: 10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible

Example: 1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible

Input: {input}
'''

# Taken from Tree of Thoughts paper
evaluate_answer = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24. Do not explain simply list the judgement.
Example: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge:
sure
Example: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge:
sure
Example: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge:
sure
Example: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge:
impossible
Example: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge:
impossible
Example: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge:
impossible
Input: {input}
Answer: {answer}
Judge:'''

react = """Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Do not explain simply list one possible next step, as well as all the remaining numbers and nothing else. Before providing the next step, provide a short thought on the problem and the existing steps.

Use the following format:
"Thought: $...$"
Possible next step:
$...$".

Example: 2 8 8 14
Thought: 14 is big, maybe I can subtract something from it to get a number I can multiply up to 24.
Possible next step:
14 - 8 = 6 (left: 2 8 6)

Input: {input}
"""

# RAP prompt for Game24
rap = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Think step by step and plan your moves carefully.

Example: 2 8 8 14
Thought: I need to get to 24. Let me analyze the numbers and plan my moves.
Action: 14 + 2 = 16 (left: 8 8 16)
Thought: Now I have 8, 8, and 16. I can multiply 8 and 8 to get 64, but that's too big. Let me try another approach.
Action: 8 * 8 = 64 (left: 16 64)
Thought: 64 is too big. Let me try a different combination.
Action: 16 - 8 = 8 (left: 8 8)
Thought: Now I have two 8s. I can multiply them to get 64, but that's too big. Let me try another approach.
Action: 8 + 8 = 16 (left: 16)
Thought: I'm stuck. Let me try a different initial move.
Action: 8 / 2 = 4 (left: 4 8 14)
Thought: This looks promising. Now I have 4, 8, and 14.
Action: 14 - 4 = 10 (left: 8 10)
Thought: Now I have 8 and 10. I can multiply them to get 80, but that's too big. Let me try another approach.
Action: 10 - 8 = 2 (left: 2)
Thought: I'm stuck again. Let me try one more initial move.
Action: 2 * 8 = 16 (left: 8 14 16)
Thought: Now I have 8, 14, and 16. I can add 8 and 16 to get 24!
Action: 8 + 16 = 24 (left: 24)

Input: {input}
'''

# Self-evaluation prompts for Game24
self_evaluate_step = '''You are evaluating a reasoning step in the Game of 24. Given the current numbers and the proposed step, determine if this step is correct and logical. Consider:
1. Does the step use valid arithmetic operations?
2. Is the step a logical move towards reaching 24?
3. Does it follow the rules of using exactly two numbers at a time?

Previous steps:
{previous_steps}

Current numbers: {input}
Proposed step: {step}

Is this reasoning step correct? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete solution to the Game of 24. Given the input numbers, the steps taken, and the final answer, determine if the solution is correct. Consider:
1. Does it use each input number exactly once?
2. Are all arithmetic operations valid?
3. Does it correctly reach 24?
4. Are the steps logically connected?

Input numbers: {input}

Steps taken:
{steps}

Final answer: {answer}

Is this solution correct? Answer with a single word: Yes or No.
'''
```

## HotpotQA
```python
###################
###---Prompts---###
###################
act = """Solve a question answering task with sequential Action steps. Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to find the immediate next action. Answer in the format given by the examples and mention nothing more.

Question: {question}
{current_state}"""

react = """Solve a question answering task with interleaving Thought and Action steps. Thought can reason about the current situation, and Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to find the immediate next thought and action. Answer them in the format given by the examples and mention nothing more.

Question: {question}
{current_state}"""

bfs = """We're solving a question answering task with sequential Action steps. Your task is to propose multiple possible next actions given the current trajectory. Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task. When you provide your answer, only state the essential information, without full sentences or explanations.


You may take as many steps as necessary.

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to propose multiple immediate next actions. Answer in the format given by the examples and mention nothing more.

Question: {question}
{current_state}

Possible Actions:
"""

evaluate = '''Analyze the trajectories of a solution to a question answering
task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types:
(1) Search[entity]: In this case, your evaluation should be influenced based on whether useful information is found in the resulting observation.
(2) Lookup[keyword]: ]: In this case, your evaluation should be influenced based on whether useful information is found in the resulting observation.
(3) Finish[answer]: In this case, your evaluation should be influenced based on whether the answer is correct or not which will be presented in the resulting observation.

Given a question and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest available thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude with your value estimation which can be an integer number from 1 to 10.

Below some examples are give.

{examples}

(END OF EXAMPLES)

Remember, your task is to evaluate the correctness of the latest available thought (if available), action, and observation based on your reasoning analysis. Answer in the format given by the examples and mention nothing more. Make sure to indicate the correctness score at the end of your answer in the following format: "Correctness score : <score>".

Question: {question}
{current_state}

Evaluation:
'''

aggregate = '''Analyze the trajectories of a solution to a question answering
task. The trajectories are labeled by environmental observations about the situation and actions that can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Given a question, trajectories and possible actions, select {k} actions that you believe are the best and most relevant to the question. Focus on the latest available action and observation, where you should only select actions from the possible actions. Do not generate additional thoughts or actions. Return only the selected actions in the format given by the examples.

Below some examples are given.

{examples}

(END OF EXAMPLES)

Remember, your task is to select the {k} best actions from the possible actions. Answer in the format given by the examples and mention nothing more.

Question: {question}
{current_state}
possible actions:
{actions}

Selected actions:
'''

################################
###---Examples for fewshot---###
################################
examples_bfs = [
"""Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?

Possible Actions:
Search[Adam Clayton Powell (film)]
Search[The Saimaa Gesture (film)]
Search[Finish rock music]
Search[Finish documentaries]
Search[Juice Leskinen]
Search[Documentary film]
""",

"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.

Possible Actions:
Lookup[named after]
Lookup[Allie Goertz]
Lookup[Matt Groening]
Lookup[name]
Search[Allie Goertz]
Search[The Simpsons]
Search[Allie Goertz Simspons]
""",

"""Question: What profession does Nicholas Ray and Elia Kazan have in common?
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.

Possible Actions:
Finish[director, screenwriter, actor]
Finish[film director, screenwriter, actor]
Finish[director, screenwriter and actor]
Lookup[Nicholas Ray]
Lookup[profession]
Lookup[producer]""",
]


examples_act = [
"""Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions:
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Action 5: Finish[1,800 to 7,000 ft]""",

"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.
Action 3: Finish[Richard Nixon]"""

"""Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Action 3: Finish[The Saimaa Gesture]""",

"""Question: What profession does Nicholas Ray and Elia Kazan have in common?
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Action 3: Finish[director, screenwriter, actor]""",

"""Question: Which magazine was started first Arthur's Magazine or First for Women?
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.
Action 3: Finish[Arthur's Magazine]""",

"""Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.
Action 3: Finish[yes]"""]

examples_react = [
"""Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions:
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]""",

"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[Richard Nixon]"""

"""Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Thought 2: To find the documentary, I can search Adam Clayton Powell (film).
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Thought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action 3: Finish[The Saimaa Gesture]""",

"""Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action 3: Finish[director, screenwriter, actor]""",

"""Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.
Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action 3: Finish[Arthur's Magazine]""",

"""Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.
Action 3: Finish[yes]"""]

examples_evaluate = [
"""Question: Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?
Thought 1: I need to search for a list of Mexican Formula One race car drivers, then eliminate the one who drove for Force India and was born in 1990.
Action 1: Search[Formula One drivers from Mexico]
Observation 1: There have been six Formula One drivers from Mexico who have taken part in races since the championship began in 1950. Sergio P\u00e9rez is the most successful, having won six Grands Prix. Since 2021, P\u00e9rez drives for the Red Bull team.. P\u00e9rez is the only Mexican F1 driver to achieve a pole position. He took his first pole position at the 2022 Saudi Arabian Grand Prix, in his 219th Grand Prix meeting (a record for the most Grands Prix before a pole).
Thought 2: Sergio P\u00e9rez, born in 1990 and drove for Force India, is not the Mexican driver we are looking for. The passage does not give me enough information about other Mexican drivers who have achieved a podium finish. I need to look up more information.
Action 2: Lookup[Formula One drivers from Mexico]
Observation 2: No more results.
Thought 3: The available information doesn't allow me to determine which other Mexican Formula One race car driver, besides Sergio P\u00e9rez, has held the podium. With the search function failing to provide a full list of Mexican Formula One drivers, I need to return different search results. Let's try searching for Mexican Grand Prix winners, as they would have held the podium.
Action 3: Search[Mexican Grand Prix winners]
Observation 3: Could not find Mexican Grand Prix winners. Similar: ['Mexican Grand Prix', 'List of Formula One Grand Prix winners', '1990 Mexican Grand Prix', '2018 Mexican Grand Prix', '2019 Mexican Grand Prix'].
Thought 4: Given the difficulties in finding a direct list of Mexican Grand Prix winners, I should search for the Mexican Grand Prix to get a broader picture of the race's history. This might include winners.
Action 4: Search[Mexican Grand Prix]

Evaluation:
This trajectory is incorrect as my search should be related to Mexican Formula One race car drivers, not winners of the Mexican Grand Prix, a seperate event. A better search would have been for the List of Formula One Grand Prix winners, as suggested.
Thus the correctness score is 3""",

"""Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844\u00e2\u0080\u00931846) was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.[1][2] In May 1846 it was merged into Godey's Lady's Book.[3]

Evaluation:
This trajectory is correct as it is reasonable to search for the first magazine provided in the question. It is also better to have simple searches corresponding to a single entity, making this the best action.
Thus the correctness score is 10""",

"""Question: The football manager who recruited David Beckham managed Manchester United during what timeframe?Question: The football manager who recruited David Beckham managed Manchester United during what timeframe?
Thought 1: I need to search for the football manager who recruited David Beckham, and then find out the timeframe in which he managed Manchester United.
Action 1: Search[David Beckham's football manager]
Observation 1: Could not find David Beckham's football manager. Similar: ['David Beckham', 'Inter Miami CF (redirect from Miami Beckham United)', 'Victoria Beckham', 'The Beckham Experiment', 'Miami Freedom Park'].
Thought 2: I couldn't find the information on David Beckham's football manager. I need to search Manchester United and find out the timeframe of the manager.
Action 2: Search[Manchester United]
Observation 2: Manchester United Football Club, commonly referred to as Man United (often stylised as Man Utd), or simply United, is a professional football club based in Old Trafford, Greater Manchester, England. The club competes in the Premier League, the top division in the English football league system. Nicknamed the Red Devils, they were founded as Newton Heath LYR Football Club in 1878, but changed their name to Manchester United in 1902. After a spell playing in Clayton, Manchester, the club moved to their current stadium, Old Trafford, in 1910.. Domestically, Manchester United have won a record 20 league titles, 12 FA Cups, six League Cups and a record 21 FA Community Shields.
Thought 3: The information about David Beckham's football manager is not available. I will search for Manchester United's managers to find the timeframe.
Action 3: Search[Manchester United managers]
Observation 3: Manchester United Football Club is a professional football club based in Old Trafford, Greater Manchester, England. The club was formed in Newton Heath in 1878 as Newton Heath LYR F.C., and played their first competitive match in October 1886, when they entered the First Round of the 1886–87 FA Cup. The club was renamed Manchester United F.C. in 1902, and moved to Old Trafford in 1910.. From the beginning of the club's official managerial records in 1892 to the start of the 2022–23 season, Manchester United have had 23 full-time managers.
Thought 4: The information about David Beckham's football manager is not available. The question asks for the timeframe when the manager managed Manchester United, so I need to find a list of Manchester United managers.
Action 4: Search[list of Manchester United managers]
Observation 4: Manchester United Football Club is a professional football club based in Old Trafford, Greater Manchester, England. The club was formed in Newton Heath in 1878 as Newton Heath LYR F.C., and played their first competitive match in October 1886, when they entered the First Round of the 1886–87 FA Cup. The club was renamed Manchester United F.C. in 1902, and moved to Old Trafford in 1910.. From the beginning of the club's official managerial records in 1892 to the start of the 2022–23 season, Manchester United have had 23 full-time managers.

Evaluation:
This trajectory is mostly incorrect as I was not able to retrieve the relevant information to answer the question. The searches are too broad and not relevant to the specific player recruited, David Beckham. The last two actions are also too similar. It would be better to first search for David Beckham and lookup information related to his manager.
Thus the correctness score is 4""",

"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[President Richard Nixon]

Evaluation:
This trajectory is correct as all of my thoughts and actions are correct. It makes sense to search for Milhouse first as it is the central subject of the question. It is also correct to directly look up the relevant information in the article, instead of trying another search.
Thus the correctness score is 10"""
]

examples_aggregate = [
"""
Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?

Possible Actions:
Search[Adam Clayton Powell (film)]
Search[The Saimaa Gesture (film)]
Search[Finish rock music]
Search[Finish documentaries]
Search[Juice Leskinen]
Search[Documentary film]

Selected actions:
Search[Adam Clayton Powell (film)]
Search[The Saimaa Gesture (film)]
Search[Finish documentaries]
""",

"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.

Possible Actions:
Lookup[named after]
Lookup[Allie Goertz]
Lookup[Matt Groening]
Lookup[name]
Search[Allie Goertz]
Search[The Simpsons]
Search[Allie Goertz Simspons]

Selected actions:
Lookup[named after]
Lookup[name]
Search[The Simpsons]
""",

"""Question: What profession does Nicholas Ray and Elia Kazan have in common?
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.

Possible Actions:
Finish[director, screenwriter, actor]
Finish[film director, screenwriter, actor]
Finish[director, screenwriter and actor]
Lookup[Nicholas Ray]
Lookup[profession]
Lookup[producer]

Selected actions:
Finish[director, screenwriter, actor]
Finish[film director, screenwriter, actor]
Finish[director, screenwriter and actor]""",
]

self_evaluate_step = '''You are evaluating a reasoning step in a question answering task. Given the current state and the proposed step, determine if this step is correct and logical. Consider:
1. Is the search/lookup action relevant to finding the answer?
2. Is the thought process logical and focused on the question?
3. Does it follow the rules of using Search, Lookup, and Finish actions appropriately?

Current state: {current_state}
Proposed step: {step}

Is this reasoning step correct? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete solution to a question answering task. Given the question, the steps taken, and the final answer, determine if the solution is correct. Consider:
1. Does it use appropriate search and lookup actions to find relevant information?
2. Are all actions logically connected and relevant to the question?
3. Does it correctly answer the question based on the information found?
4. Are the steps taken efficient and focused?

Question: {question}

Steps taken:
{steps}

Final answer: {answer}

Is this solution correct? Answer with a single word: Yes or No.
'''
```

## HumanEval
```python
SIMPLE_CHAT_INSTRUCTION = "You are a programming assistant. You will be given a function signature and docstring. You should fill in the following text of the missing function body. For example, the first line of the completion should have 4 spaces for the indendation so that it fits syntactically with the preceding signature."
SIMPLE_CHAT_INSTRUCTION_V2 = """You are an AI that only responds with only {lang} code. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature)."""

aggregate_prompt = """You are a programming assistant, who is helping user to write efficient and correct codes. You will be given multiple implementations of the same function. You should choose the {k} best implementation based on the following criterias:
1. Correctness: The implementation should return the correct output.
2. Efficiency: The implementation should be efficient in terms of time and space complexity.
3. Readability: The implementation should be readable and understandable.
4. Style: The implementation should follow the style guide of the language.
5. Testability: The implementation should be testable.

Remember your task is to choose the {k} best implementation based on the above criterias. Make sure to return only the indexes of the selected implementations, separated by commas. Do not include any other explanations, introduction, conclusions or thoughts. Just return the indexes of the selected implementations.
Function signature and docstring:
{prompt}
Implementations:
{implementations}
Chosen implementation:
"""

evaluation_prompt = """You are a programming assistant, who is helping the user to evaluate a generated code. You will be given a single implementation of a function, and you should evaluate it based on the following criteria:

1. **Correctness**: Does the implementation return the correct output for different inputs?
2. **Efficiency**: Is the implementation efficient in terms of time and space complexity?
3. **Readability**: Is the code readable and understandable? Is it easy to follow?
4. **Style**: Does the implementation follow the style guide of the language (naming conventions, indentation, etc.)?
5. **Testability**: Is the implementation testable? Can it be easily tested with unit tests?

Evaluate the code on each criterion with a score from 1 to 10 (integers only, no fractions). Then give an overall score as the sum of all scores.

Function signature and docstring:
{prompt}

Implementation:
{implementation}

Evaluation scores:
- Correctness: <score>
- Efficiency: <score>
- Readability: <score>
- Style: <score>
- Testability: <score>

Overall Score: <final score>

Do not include any further thoughts or reasoning, just the evaluation scores and the final overall score."""


SIMPLE_CHAT_INSTRUCTION_BFS = """
You are an AI that only responds with {lang} code. You will be given a function signature and its docstring by the user.
Write multiple full implementations (at least two), each restating the function signature. Use a different approach for each.
Mark the start and end of each implementation using triple backticks, like this:
\`\`\`
<Code implementation here>
\`\`\`
Each implementation should be fully contained within its own set of backticks, without any additional markers.
"""

react = """You are a programming assistant solving a coding task. Think step by step and plan your implementation carefully. You will be given a function signature and docstring, and you need to implement the function.

For each step:
1. Think about what needs to be done
2. Write code to implement that step
3. Consider edge cases and error handling

Example:
Function signature and docstring:
def add_numbers(a: int, b: int) -> int:
    '''Add two numbers and return the result.'''

Thought: I need to implement a simple addition function. The function takes two integers and returns their sum. I should handle basic input validation.

Action: python
def add_numbers(a: int, b: int) -> int:
    '''Add two numbers and return the result.'''
    # Input validation
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Both arguments must be integers")
    return a + b


Thought: The implementation looks good. It includes:
1. Type hints for parameters and return value
2. Input validation to ensure both arguments are integers
3. Simple and efficient addition operation
4. Proper docstring preservation

Action: Finish[The implementation is complete and correct]

Function signature and docstring:
{prompt}

Current implementation:
{current_state}

Remember to think step by step and write clear, efficient code."""

self_evaluate_step = '''You are evaluating a reasoning step in a code generation task. Given the function signature, current implementation, and the proposed step, determine if this step is correct and logical. Consider:
1. Is the code syntactically correct?
2. Does it follow the function's requirements?
3. Is it a logical next step in the implementation?
4. Does it handle edge cases appropriately?

Function signature and docstring:
{prompt}

Current implementation:
{current_state}

Proposed step:
{step}

Is this reasoning step correct? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete solution to a code generation task. Given the function signature, the implementation steps, and the final code, determine if the solution is correct. Consider:
1. Does the implementation match the function signature and docstring?
2. Is the code syntactically correct and follows language style guidelines?
3. Does it handle all edge cases and error conditions?
4. Is it efficient and readable?
5. Does it include appropriate tests or validation?

Function signature and docstring:
{prompt}

Implementation steps:
{steps}

Final code:
{answer}

Is this solution correct? Answer with a single word: Yes or No.
'''
```

## SciBench
```python
###################
###---Prompts---###
###################

io = '''
Given a science problem, your task is to answer the question step-by-step in a clear and specific manner.
The format of the solution is limited to: "Solution: ...\nSummary: The final answer is $...$"
Please complete the answer step-by-step, and finally outline the final answer.
Problem: {problem}
Solution:'''


react = '''Given a science problem, you need to answer the problem based on your existing knowledge. The input may include some existing steps to solve the question and you should continue to complete the solution based on these existing steps.

If the input does not provide any existing steps, you need to analyze the problem and then give the first step in solving or calculating the problem. If partial solution steps are provided, you need to output the next step along the lines of the existing steps.
The output format is limited to: "Next step: ..." where ... indicates omitted output information, which is the next step in the answer that you should give. Your output must be a complete reasoning step, which should include detailed calculations, reasoning, choosing answers, etc.

If the existing steps are already sufficient, you can output "The final answer is: $...$" where ... indicates the final answer to the question.

Before providing the next step, provide a short thought on the problem and the existing steps. Use the following format:
"Thought: $...$"
Next step: $...$".

Below is the input, please follow the specified format for your output.

Problem: {problem}
Existing steps:
{existing_steps}
Output:'''

# # ReST-MCTS*
act = '''Given a science problem, you need to answer the problem based on your existing knowledge. The input may include some existing steps to solve the question and you should continue to complete the solution based on these existing steps.

If the input does not provide any existing steps, you need give the first step in solving or calculating the problem. If partial solution steps are provided, you need to output the next step along the lines of the existing steps.
The output format is limited to: "Next step: ..." where ... indicates omitted output information, which is the next step in the answer that you should give. Your output must be a complete step, which may include detailed calculations, reasoning, choosing answers, etc. but no reasoning.

If the existing steps are already sufficient, you can output "The final answer is: $...$" where ... indicates the final answer to the question.

Below is the input, please follow the specified format for your output.

Problem: {problem}
Existing steps:
{existing_steps}
Output:'''

aggregate = '''Given a science proplem, you need to answer the problem based on your existing knowledge. The input may include some existing steps to solve the question and you should choose from the given steps, which best helps you get towards a solution to the question.

From the partial or fully solutions, your task is to select {k} partial or full solutions that best solves or calculates the problem. Your output must be the numbers of the selected partial or full solutions, without any explanation, reasoning, introduction, conclusion or modifucations.

Below is the input, please only output the {k} indexes of your choices.

Problem: {problem}
Solutions:
{steps}
Output:'''



bfs = '''Given a science problem, you need to answer the problem based on your existing knowledge. The input may include some existing steps to solve the question and you should continue to complete the solution based on these existing steps.

If the input does not provide any existing steps, you need give the first step in solving or calculating the problem. If partial solution steps are provided, you need to output the next step along the lines of the existing steps.
The output format is limited to: "Next step: ..." where ... indicates omitted output information, which is the next step in the answer that you should give. Your output must be a complete step, which may include detailed calculations, reasoning, choosing answers, etc. but no reasoning.

If the existing steps are already sufficient, you can output "The final answer is: $...$" where ... indicates the final answer to the question.

Please provide MULTIPLE alternative next steps. Use the following format:
"Next step: $...$
Next step: $...$
Next step: $...$".

Below is the input, please follow the specified format for your output.

Problem: {problem}
Existing steps:
{existing_steps}
Output:'''

# Summary prompt (ReST-MCTS*)
summary = '''
Given a math problem and its corresponding solution, your task is to extract the final answer obtained in the solution.
You should summarize the answer using the format: "The final answer is $...$". Replace "..." with the answer obtained in the solution.
Problem: {problem}
Solution: {existing_steps}
Extracted answer:'''

# ReST-MCTS* (Translated using google translate)
evaluate = '''Your task is to assess whether the provided solution steps can successfully solve the given science/mathematics problem and output a score.
The score should be a decimal between 0 and 1. If all the provided steps are incorrect (every step is wrong), the score should be 0. If all steps are correct and the final answer is successfully calculated, the score should be 1. The more errors there are in the steps, the closer the score should be to 0. The closer the steps are to the final correct answer, the closer the score should be to 1.
Steps that only contain verbal descriptions without any mathematical expressions should generally receive a low score. A score equal to or greater than 0.9 can only be given if the answer has already been calculated to a specific numerical value. If the thought process is complete but the answer is not computed, or only the mathematical expression is written without solving it, the score must be below 0.9.

First provide an analysis, then the score. Your analysis and scoring should be entirely based on the given steps. Do not continue solving the problem. Please study the following examples.

{examples}

Below is a problem and the existing steps, with analysis and scoring. Be careful not to output the next steps in the analysis, and the scoring should be based entirely on the steps given in the input.
The output format is limited to: "Analysis:...\nScore:...", where ... indicates omitted output content, which is the part you need to fill in.

Input:
Problem: {problem}
Existing steps:
{existing_steps}
Output:'''

self_evaluate_step = '''You are evaluating a step in a scientific procedure. Given the task requirements and the current step, determine if this step is correct and contributes meaningfully to the solution. Consider:
1. Is the step scientifically valid and logically sound?
2. Does it follow from the previous steps?
3. Does it help progress toward solving the task?
4. Does it avoid major scientific misconceptions?

Task: {input}

Previous steps:
{previous_steps}

Current step: {step}

Is this step correct and well-reasoned? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete scientific solution. Given the task requirements and the full answer, determine if it meets all criteria. Consider:
1. Is the answer scientifically accurate?
2. Does it clearly and correctly address the main question?
3. Are all reasoning steps logically consistent and well-justified?
4. Does it use appropriate scientific terminology and avoid misconceptions?
5. Is it complete and concise?

Task: {input}

Answer:
{answer}

Is this scientific answer correct and well-reasoned? Answer with a single word: Yes or No.
'''

################################
###---Examples for fewshot---###
################################
examples_evaluate = [
"""Question: Discuss for what value of p, the generalized integral $\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx$ converges.
Existing steps:
Step 1: To illustrate the convergence of the integral, consider splitting the integral into two parts: $$ \\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2} dx = \\int_0^1 \\frac{x^p \\ln x}{(1+x^2)^2} dx + \\int_1^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2} dx $$
Step 2: For the first part, $0 \\leq \\frac{x^p \\ln x}{(1+x^2)^2} \\leq x^p$, so it converges if and only if $p>-2$.
Output:
Analysis: Step 1 correctly gets the idea of ​​splitting the integral, but the derivation of step 2 is wrong, and there are problems in judging the convergence. $0 \\leq \\frac{x^p \\ln x}{(1+x^2)^2} \\leq x^p$, according to \\int_0^1 x^p dx converges if and only if $p>-1$, so the original integral converges if and only if $p>-1$, not $p>-2$.
Score: 0.1""",

"""Question: Find the value of the largest term of the sequence ${n^{1/n}}$ (n=1, 2, 3... are positive integers).
Existing steps:
Step 1: Consider taking the derivative: We can regard the sequence $n^{1/n}$ as a function $f(x) = x^{1/x}$, and then find the derivative $f'(x)$ of the function. By taking the derivative, we can find the increase and decrease of the function, and then determine the positive integer $n$ value corresponding to the maximum value of the sequence.
Output:
Analysis: Step 1 of the existing steps is correct. It establishes the basic idea of ​​solving the problem, that is, treating the sequence as a function and analyzing the increase and decrease of the function by taking the derivative. However, this is only part of the solution. Further steps are required to find the positive integer $n$ corresponding to the maximum value and to obtain the maximum value. Therefore, the existing steps have not yet inferred the answer.
Score: 0.2""",

"""Question: Find the average value of the function $f(x)=1+x^2$ on the interval $[-1,2]$.
Existing steps:
Step 1: Use definite integral to solve the average value: We can use definite integral to solve the average value of the function on the interval $[-1,2]$.
Step 2: First, we need to calculate the definite integral $\\int_{-1}^{2} (1+x^2) dx=6$.
Step 3: Then, we can use the properties of the definite integral to divide the result of the definite integral by the length of the interval, that is, $\\frac{\\int_{-1}^{2} (1+x^2) dx}{3}$, which should be the average value of the function on the interval.
Step 4: Calculate the above formula and get the result of $\\frac{\\int_{-1}^{2} (1+x^2) dx}{3}=\\frac{6}{3}=2$, so the average value of the function is 2.
Output:
Analysis: All steps are derived correctly, and the existing steps have calculated the answer to $2$, which can get a full score of 1.
Score: 1""",

"""Question: Find the value of the largest term of the sequence ${n^{1/n}}$ (n=1, 2, 3... are positive integers).
Existing steps:
Step 1: Consider the derivative: We can regard the sequence $n^{1/n}$ as the function $f(x) = x^{1/x}$, and then find the derivative $f'(x)$ of the function. By finding the derivative, we can find the increase and decrease of the function, and then determine the positive integer $n$ value corresponding to the maximum value of the sequence.
Step 2: Based on the idea of ​​the previous step, for the function $f(x) = x^{1/x}$, we can take the natural logarithm to simplify the derivation process, and get $g(x)=\\ln(f(x)) = \\frac{1}{x}\\ln(x)$, and then find the derivative of g(x).
Step 3: We perform the derivation, $$\\frac{d}{dx}\\left(\\ln(f(x))\\right) = -\\frac{1}{x^2} \\ln(x) - \\frac{1}{x^2} + \\frac{1}{x^2} \\ln(x) = -\\frac{1}{x^2}$$. This derivative is always negative, indicating that $f(x)$ is decreasing within the range of positive integers. ,
Output:
Analysis: The first two steps correctly analyzed the idea of ​​​​derivation, but the specific derivation process in step 3 was wrong. The correct procedure for the derivation is: $$\\frac{d}{dx}\\left(\\ln(f(x))\\right) = -\\frac{1}{x^2} \\ln(x) + \\frac{1}{x^2}$$, not $-\\frac{1}{x^2}$.
Score: 0.2""",

"""Question: Find the average value of the function $f(x)=1+x^2$ on the interval $[-1,2]$.
Existing steps:
Step 1: Consider the value of the function at the endpoints of the interval: We can calculate the value of the function at the endpoints of the interval $x=-1$ and $x=2$, that is, $f(-1)=1+(-1)^2=2$ and $f(2)=1+2^2=5$.
Step 2: Then we can calculate the average value of the function at these two endpoints, that is, $\\frac{2+5}{2}=3.5$. This is the average value of the function on the interval $[-1,2]$.
Output:
Analysis: All derivation steps are wrong, and 0 points should be given. The average value of a function on an interval should be equal to the integral of the function on the interval divided by the length of the interval, that is, $\\frac{\\int_{-1}^{2} (1+x^2) dx}{3}=2$. It cannot be simply considered to be equal to the average value of the function value at the endpoints of the interval.
Score: 0""",

"""Question: Find the value of the largest term of the sequence ${n^{1/n}}$ (n=1, 2, 3... are positive integers).
Existing steps:
Step 1: Consider taking the derivative: We can regard the sequence $n^{1/n}$ as the function $f(x) = x^{1/x}$, and then find the derivative $f'(x)$ of the function. By taking the derivative, we can find the increase and decrease of the function, and then determine the positive integer $n$ value corresponding to the maximum value of the sequence.
Step 2: Based on the idea of ​​the previous step, for the function $f(x) = x^{1/x}$, we can take the natural logarithm to simplify the derivation process, and get $g(x)=\\ln(f(x)) = \\frac{1}{x}\\ln(x)$, and then find the derivative of g(x).
Step 3: After calculation, we get the derivative of $g(x)$ to be $$-\\frac{1}{x^2}\\ln(x) + \\frac{1}{x^2}$$
Output:
Analysis: All the steps are derived correctly, but the value of the maximum term has not been specifically calculated, that is, the answer has not been calculated. It is also necessary to analyze the positive and negative nature of the derivative to understand the increase and decrease of $f(x)$.
Score: 0.6""",

"""Question: Discuss for what value of p, the generalized integral $\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx$ converges.
Existing steps:
Step 1: Let $J=\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_1=\\int_0^{1} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_2=\\int_1^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, then the generalized integral $J$ converges if and only if $J_1, J_2$ both converge.
Step 2: When $x \\rightarrow 0^+$, $\\frac{x^p \\ln x}{(1+x^2)^2} \\sim x^p \\ln x$, so $J_1$ converges if and only if $p > -1$.
Step 3: When $x \\rightarrow +\\infty$, $\\frac{x^p \\ln x}{(1+x^2)^2} \\sim \\frac{\\ln x}{x^{4-p}}$, so $J_2$ converges if and only if $p < 4$.
Output:
Analysis: The first two steps are correct, but the deduction of step 3 is wrong. When $x \\rightarrow +\\infty$, $\\frac{x^p \\ln x}{(1+x^2)^2} \\sim \\frac{\\ln x}{x^{4-p}}$, according to \\int_0^{+\\infty} x^m dx converges if and only if $m<-1$, so the original integral converges if and only if $p-4 < -1$, that is, $p < 3$, not $p < 4$.
Score: 0.2""",

"""Question: Find the maximum value of the function $f(x)=-\\frac{1}{2}*(x^2)+2*x-1$ on R.
Existing steps:
Step 1: Find the derivative: We can find the derivative $f'(x)$ of the function $f(x)$, that is, $f'(x)=-x+2$. By taking the derivative, we can find the increase or decrease of the function, and then determine the $x$ value corresponding to the maximum value of the function on R.
Step 2: We can calculate the value of $f'(x)$ when $x=1$, that is, $f'(1)=1$. From this, we can see that at $x=1$, the function $f(x)$ reaches its maximum value, that is, its maximum value.
Output:
Analysis: The first step is correct, but the second step is wrong. Calculating the value of $f'(x)$ at $x=1$ does not tell us the increase or decrease of the function as a whole, and is meaningless. From $f'(1)=1$, it cannot be deduced that the function $f(x)$ reaches its maximum value at $x=1$, and the maximum value should satisfy the derivative of 0.
Score: 0.1""",

"""Question: Find the value of the largest term of the sequence ${n^{1/n}}$ (n=1, 2, 3... are positive integers).
Existing steps:
Step 1: Consider the derivative: We can regard the sequence $n^{1/n}$ as the function $f(x) = x^{1/x}$, and then find the derivative $f'(x)$ of the function. By finding the derivative, we can find the increase and decrease of the function, and then determine the positive integer $n$ value corresponding to the maximum value of the sequence.
Step 2: Based on the idea of ​​the previous step, for the function $f(x) = x^{1/x}$, we can take the natural logarithm to simplify the derivation process, and get $g(x)=\\ln(f(x)) = \\frac{1}{x}\\ln(x)$, and then find the derivative of g(x).
Step 3: After calculation, we get the derivative of $g(x)$ is $$-\\frac{1}{x^2}\\ln(x) + \\frac{1}{x^2}$$
Step 4: Next, we can analyze the positive and negative value of the derivative. This derivative is negative when $x > e$ and positive when $x < e$. This means that the function $f(n)$ is decreasing when $n > e$ and increasing when $n < e$.
Output:
Analysis: All the existing steps are correctly derived, and the function's increase and decrease are analyzed, but the value of the maximum term has not been specifically calculated, that is, the answer has not been calculated, so a score greater than or equal to 0.9 cannot be given. However, since the existing steps are very close to calculating the answer, the score should be close to 0.9.
Score: 0.8""",

"""Question: Find the area of ​​the figure enclosed by the function $f(x)=x+1$ and the straight lines $x=0$, $x=1$ and the x-axis.
Existing steps:
Step 1: According to the geometric meaning of definite integrals, solving the definite integral of the function is the area of ​​the required figure, and the calculation result can be directly used as the final answer.
Output:
Analysis: The analysis in step 1 is correct, but the expression is vague and it is of little help in solving the problem, and the answer is not actually calculated, so only a small score can be given. A more appropriate statement is: According to the geometric meaning of the definite integral, the area to be sought should be the definite integral of $f(x)=x+1$ on the interval $[0,1]$.
Score: 0.1"""
]
```

## Sonnet Writing
```python
act = '''You are fluent in sonnet writing and can only respond with your sonnet and nothing else.
Below, you will be given your task and keywords to include in your sonnet writing. Remember to return only your sonnet writing.

{input}'''

aggregate = '''There have been created some sonnet writings with respect to the following task:
{task}

Select the {k} best sonnet writings with respect to the task, without any modification to the sonnet writings. Do not add any explanation, comments, introduction or conclusion, your should only return the sonnets writings.
You select a sonnet writing by returning the number of that sonnet writing.

{examples}

(End of examples)

Remember, your task is to select the {k} best sonnet writings. Do not add any explanation, comments, introduction, conclusions or modifications.
You should only return the numbers of the selected sonnet writings.

{sonnets}'''

evaluate = '''You are given a sonnet writing below. Your task is to evaluete the sonnet based on the criterias:
    1. Follows the rhyme scheme: {rhyme_scheme}
    2. Contains these exact words: {words}

{examples}

(END OF EXAMPLES)

Remember, your task is to evaluate the given sonnet between 0 and 10. Do not add any explanations, comments, introduction or conclusions.

{sonnet}
'''

examples_evaluate = [
    '''Required Rhyme Scheme: ABAB CDCD EFEF GG
Required Words: grass, value, jail

The river bends beside the morning grass (A)
A shimmer dances on the silver tide (B)
The hours like golden moments swiftly pass (A)
And fortune's value flows with gentle pride (B)

The jail of fear breaks open with the breeze (C)
The open fields forgive the bitter rain (D)
The grass revives beneath the waking trees (C)
And songs of hope replace the cries of pain (D)

Bright grasslands whisper secrets to the skies (E)
The jail of winter thaws beneath the light (F)
The value found in spring will never die (E)
But bloom in endless fields beyond our sight (F)

The grass will grow where broken dreams once fall (G)
And value shines within the hearts of all. (G)

---END-OF-SONNET---

evaluation: 10
''',
'''Required Rhyme Scheme: ABAB CDCD EFEF GG
Required Words: grass, value, jail

The mountain sighs beneath the winter snow
The river carves its path with solemn might
The flowers sleep beneath the frozen glow
Awaiting touch of spring’s returning light

No jail confines the wild and restless breeze
It tumbles past the valley’s sleeping door
The grass will rise when winter grants release
Yet value fades along the rocky shore

The colors blend beneath a paling sky
The stars retreat behind a drifting veil
Though grass returns, some dreams must say goodbye
And hearts once brave grow weary and grow pale

The seasons turn, and leave us wondering still
What value clings to hope, and what to will.

---END-OF-SONNET---

evaluation: 10
''',
'''Required Rhyme Scheme: ABAB CDCD EFEF GG
Required Words: grass, value, jail

Upon the cliffs the storm begins to roar
The candles flicker in the empty hall
The silver mist creeps underneath the door
A solemn hush descends upon the wall

No grassy fields are near this barren land
No value glimmers in the heavy rain
No jail can hold the fury in its hand
The waves collapse against the rocks in vain

The sailors cry into the endless dark
No flame to guide them through the cruel, wild gale
No harvest waits beyond the fading spark
Just shattered hopes imprisoned without bail

Their names are lost beneath the mourning wave
No grass, no value, only storms to brave.

---END-OF-SONNET---

evaluation: 10
''',
'''Required Rhyme Scheme: ABAB CDCD EFEF GG
Required Words: grass, value, jail

The twilight burns across the shattered shore (A)
Old lanterns flicker in the dying mist (B)
The crows descend where empty houses mourn (C)
A bitter memory clenched within a fist (B)

The river moans beneath a sky of ash (D)
Its course forgotten by the sleeping stone (E)
The mountains tremble under thunder’s crash (D)
The plains are silent, aching and alone (F)

No voices rise above the broken field (G)
No banners fly beneath the heavy rain (H)
The meadow bows beneath the turning wheel (I)
While dreams decay and vanish in their pain (H)

Night falls without a sound, a hollow art (J)
And leaves the world with one forsaken heart. (J)

---END-OF-SONNET---

evaluation: 0
'''
]

aggregate_examples = [
    '''Beneath the whisper of the waving grass (A)
The river sings a song of ancient lore (B)
Each golden moment holds a priceless mass (A)
Of value set beyond the richest store. (B)

The sunlight weaves a pattern soft and bright (C)
Across the meadows where the daisies sail, (D)
Yet even beauty feels a distant blight (C)
When love is locked within a silent jail. (D)

O time, who takes but seldom grants us grace, (E)
Within your tides do all our dreams entwine, (F)
Yet still the grass shall whisper of this place, (E)
A testament to moments lost in time. (F)

So hold the value of each fleeting breath, (G)
For life escapes more swiftly than its death. (G)

---END-OF-SONNET---

The jail of sorrow keeps my heart in thrall (A)
While grass grows wild beyond the broken gate. (B)
I find no value in these chains at all, (A)
Yet still I bear the burden of my fate. (B)

A lonesome wind creeps through the ruined hall (C)
And sings of dreams that once could not prevail, (D)
Now dust and silence answer every call, (C)
Their memories locked in an endless jail. (D)

But hope, like grass, persists beyond the stone (E)
And value shines where shadows used to tread; (F)
A broken heart can yet become its own, (E)
A phoenix rising from the dreams long dead. (F)

O jail, O grass, O value lost and won, (G)
Your cycle ends anew with each day's sun. (G)

---END-OF-SONNET---

I saw a meadow bright with golden flame (A)
And heard the jailer laughing in his cell; (B)
The value of the fields he could not name, (A)
For grass and sunlight meant no tales to tell. (B)

The sky was pale and heavy with the rain, (C)
While iron gates were rusted in the gale; (D)
A robin sang a melancholy strain (C)
Outside the damp and sorrow-scented jail. (D)

The grass grew thick and wove a velvet sheet (E)
Where value lay in every drop of dew, (F)
And in the mud the jailer dragged his feet, (E)
Unknowing that the grass would soon renew. (F)

The jail will fall, the grass will rise instead, (G)
For nature reclaims even iron and dread. (G)

---END-OF-SONNET---

Selected 2 Best sonnets:
1
2''',
'''Upon the hill where once the grass was green
The winds now whisper secrets of the vale,
A place where none but memories have been
And sorrow weeps behind a broken jail.

The stars above reflect in pools below
While tender dreams dissolve without a trace,
Yet hope still carves its message in the snow
And leaves the value written on its face.

The seasons turn with neither grief nor song
And grass will rise to meet the morning tide,
The heart must travel even when it's wrong,
And seek the dreams that time cannot deride.

We hold the broken past within our hands,
And plant new hopes upon the barren lands.

---END-OF-SONNET---

In fields of gold where bending grasses sway
I found a road that led beyond the stream,
A shining path that twisted night to day,
And led me farther from the jail of dream.

The value of a single step was small
Yet each one built a bridge of lasting stone;
The stars above seemed waiting for the call
To guide me far from everything I'd known.

The blades of grass were wet with morning's grace
And songs of sparrows filled the misty air;
They sang of freedom no jail could replace
Of journeys shaped by hope beyond despair.

I walked the road and felt the future start,
The grass and stars still singing in my heart.

---END-OF-SONNET---

The jail was hidden deep beneath the trees (A)
Where no soft grass could ever hope to grow. (B)
The shadows whispered warnings on the breeze, (A)
A chilling song the heart would never know. (B)

No value marked the time that passed away, (C)
The walls were cold and silent in their might, (D)
And grassless stones forever bleak and gray (C)
Reflected only sorrow in their sight. (D)

Yet through the cracks a single flower rose (E)
Defiant in the darkness and the pain, (F)
Its fragile beauty trembling as it grows, (E)
A symbol that the spirit will remain. (F)

Where grass may fail, and even value die, (G)
Still hope will blossom if it dares to try. (G)

---END-OF-SONNET---

Selected 2 best sonnets:
2
3'''
]

# React prompt for SonnetWriting
react = '''You are fluent in sonnet writing. You will be given a task that specifies the rhyme scheme and words that must be included. Your task is to write a sonnet that follows these requirements while maintaining poetic quality.

You should think step by step about the logical reasoning needed to write the sonnet:

1. First, analyze the rhyme scheme and required words
2. Then, plan how to incorporate the required words naturally
3. Next, develop a theme that connects these words meaningfully
4. Finally, write the sonnet following the required format

(Example)
Task: Write a sonnet with rhyme scheme ABAB CDCD EFEF GG and include the words: grass, value, jail

Thought: Let me analyze the requirements step by step:
1. Need to follow ABAB CDCD EFEF GG rhyme scheme
2. Must include "grass", "value", and "jail" naturally
3. Should create a theme connecting these words
4. Need to write in iambic pentameter

Thought: Let me plan the sonnet structure:
1. First quatrain: Introduce "grass" and set the scene
2. Second quatrain: Incorporate "jail" and develop theme
3. Third quatrain: Use "value" and reinforce theme
4. Couplet: Conclude with a meaningful message

Thought: Now I'll write the sonnet following this plan:

The river bends beside the morning grass (A)
A shimmer dances on the silver tide (B)
The hours like golden moments swiftly pass (A)
And fortune's value flows with gentle pride (B)

The jail of fear breaks open with the breeze (C)
The open fields forgive the bitter rain (D)
The grass revives beneath the waking trees (C)
And songs of hope replace the cries of pain (D)

Bright grasslands whisper secrets to the skies (E)
The jail of winter thaws beneath the light (F)
The value found in spring will never die (E)
But bloom in endless fields beyond our sight (F)

The grass will grow where broken dreams once fall (G)
And value shines within the hearts of all. (G)

---
The river bends beside the morning grass (A)
A shimmer dances on the silver tide (B)
The hours like golden moments swiftly pass (A)
And fortune's value flows with gentle pride (B)

The jail of fear breaks open with the breeze (C)
The open fields forgive the bitter rain (D)
The grass revives beneath the waking trees (C)
And songs of hope replace the cries of pain (D)

Bright grasslands whisper secrets to the skies (E)
The jail of winter thaws beneath the light (F)
The value found in spring will never die (E)
But bloom in endless fields beyond our sight (F)

The grass will grow where broken dreams once fall (G)
And value shines within the hearts of all. (G)

---END-OF-SONNET---

(End of example)

Remember, your task is to think step by step and write a sonnet that follows the required rhyme scheme and includes all specified words naturally. Do not add any explanation, comments, introduction or conclusion, you shall only return your sonnet followed by ---END-OF-SONNET---.

Task: {input}

Current reasoning:
{current_state}

---
'''

# Self-evaluation prompts for SonnetWriting
self_evaluate_step = '''You are evaluating a step in sonnet writing. Given the task requirements and the current step, determine if this step is correct and contributes to a good sonnet. Consider:
1. Does it follow the required rhyme scheme?
2. Does it maintain iambic pentameter?
3. Does it contribute to the overall theme?
4. Does it use the required words naturally?

Task: {input}

Previous steps:
{previous_steps}

Current step: {step}

Is this step correct and well-written? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete sonnet. Given the task requirements and the sonnet, determine if it meets all criteria. Consider:
1. Does it follow the required rhyme scheme?
2. Does it include all required words naturally?
3. Does it maintain iambic pentameter throughout?
4. Does it have a coherent theme and imagery?
5. Is it grammatically correct and poetic?

Task: {input}

Sonnet:
{sonnet}

Is this sonnet correct and well-written? Answer with a single word: Yes or No.
'''
```


## HLE
```python
###################
###---Prompts---###
###################

io = """You will be given a question. Simply provide the final answer. Do not provide any explanations or intermediate steps.  The format to respond is "Final Answer: ..."""

cot = """You will be given a question. Think step by step and provide your reasoning before giving the final answer. The format to respond is "Final Answer: ...", where "..." is the final answer."""


act = '''Given a problem, you need to answer based on your existing knowledge. The input may include some existing steps to solve the question and you should continue to complete the solution based on these existing steps.

If the input does not provide any existing steps, you need to give the first step in solving or calculating the problem. If partial solution steps are provided, you need to output only the next step along the lines of the existing steps.

The output format is limited to: "Next step: ..." where ... indicates omitted output information, which is the next step in the answer that you should give. Your output must be a single complete step, which may include detailed calculations, one node of reasoning (eg. a sentence), choosing answers, etc.

If the existing steps are already sufficient, you can output "The final answer is: $...$" where ... indicates the final answer to the question.

Below is the input, please follow the specified format for your output.

Problem: {problem}
Existing steps:
{existing_steps}
Output:'''

bfs = '''Given a problem, you need to answer on your existing knowledge. The input may include some existing steps to solve the question and you should continue to complete the solution based on these existing steps.

If the input does not provide any existing steps, you need give the first step in solving or calculating the problem. If partial solution steps are provided, you need to output the next step along the lines of the existing steps.

The output format is limited to: "Next step: ..." where ... indicates omitted output information, which is the next step in the answer that you should give. Your output must be a single complete step, which may include detailed calculations, one node of reasoning (eg. a sentence), choosing answers, etc.

If the existing steps are already sufficient, you can output "The final answer is: $...$" where ... indicates the final answer to the question.

Please provide MULTIPLE alternative next steps. Use the following format:
"Next step: $...$
Next step: $...$
Next step: $...$".

Below is the input, please follow the specified format for your output.

Problem: {problem}
Existing steps:
{existing_steps}
Output:'''

aggregate = '''Given a  proplem, you need to answer based on your existing knowledge. The input may include some existing steps to solve the question and you should choose from the given steps, which best helps you get towards a solution to the question.

From the partial or fully solutions, your task is to select {k} partial or full solutions that best solves or calculates the problem. Your output must be the numbers of the selected partial or full solutions, without any explanation, reasoning, introduction, conclusion or modifucations.

Below is the input, please only output the {k} indexes of your choices.

Problem: {problem}
Solutions:
{steps}
Output:'''

react = """Solve a human-labeled explanation task with interleaving Thought and Action steps. Thought can reason about the current situation, and Action can be three types:

(1) Analyze[topic], which analyzes the given topic in the context of the question and image.
(2) Explain[aspect], which provides explanations about specific aspects of the topic.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to find the immediate next thought and action. Answer them in the format given by the examples and mention nothing more.

Question: {question}
{current_state}"""

summary = '''
Given a math problem and its corresponding solution, your task is to extract the final answer obtained in the solution.
You should summarize the answer using the format: "The final answer is $...$". Replace "..." with the answer obtained in the solution.
Problem: {problem}
Solution: {existing_steps}
Extracted answer:'''

evaluate = '''Your task is to assess whether the provided solution steps can successfully solve the given science/mathematics problem and output a score.
The score should be a decimal between 0 and 1. If all the provided steps are incorrect (every step is wrong), the score should be 0. If all steps are correct and the final answer is successfully calculated, the score should be 1. The more errors there are in the steps, the closer the score should be to 0. The closer the steps are to the final correct answer, the closer the score should be to 1.
A score equal to or greater than 0.9 can only be given if the answer has already been calculated to a specific value. If the thought process is complete but the answer is not computed, or only the mathematical expression is written without solving it, the score must be below 0.9.

First provide an analysis, then the score. Your analysis and scoring should be entirely based on the given steps. Do not continue solving the problem.

Below is a problem and the existing steps, with analysis and scoring. Be careful not to output the next steps in the analysis, and the scoring should be based entirely on the steps given in the input.
The output format is limited to: "Analysis:...\nScore:...", where ... indicates omitted output content, which is the part you need to fill in.

Input:
Problem: {problem}
Existing steps:
{existing_steps}
Output:'''

self_evaluate_step = '''You are evaluating the next reasoning/action step for a question-answering task.
Decide whether the proposed step is useful, relevant, and logically consistent with the question and prior steps.
Answer with only "Yes" or "No".

Question: {question}
Previous steps:
{previous_steps}

Proposed step:
{step}

Answer:'''

self_evaluate_answer = '''You are evaluating whether a proposed final answer correctly answers the question.
Answer with only "Yes" or "No".

Question: {question}
Reasoning steps:
{steps}

Proposed final answer:
{answer}

Reference answer:
{correct_answer}

Answer:'''

### Judge prompt to evaluate the final answer correctness based on the ground truth answer.
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""



################################
###---Examples for fewshot---###
################################
examples_bfs = [
"""
What is the purpose of the illustrated UI component?

Possible Actions:
Analyze[UI component layout]
Analyze[visual elements]
Analyze[interactive features]
Analyze[user flow]
Explain[component functionality]
Explain[design patterns]
""",

"""Question: How does this visualization represent the data hierarchy?
Action 1: Analyze[visualization structure]
Observation 1: The visualization uses a tree-like structure with connected nodes and branches to show relationships between different data elements.

Possible Actions:
Explain[node relationships]
Explain[visual hierarchy]
Explain[data flow]
Analyze[node types]
Analyze[connection patterns]
Finish[hierarchical tree structure]
"""]

examples_act = [
"""Question: What type of information architecture pattern is shown in the image?
Action 1: Analyze[layout structure]
Observation 1: The interface shows a nested hierarchy with main categories and subcategories organized in a tree-like structure.
Action 2: Explain[navigation pattern]
Observation 2: Users can navigate through different levels of content using expandable/collapsible sections.
Action 3: Finish[hierarchical navigation pattern]""",

"""Question: How does the color scheme contribute to the user experience?
Action 1: Analyze[color palette]
Observation 1: The interface uses a combination of primary colors for main actions and muted tones for secondary elements.
Action 2: Explain[color hierarchy]
Observation 2: The color scheme helps establish visual hierarchy and guides user attention to important elements.
Action 3: Finish[visual hierarchy and attention guidance]"""]

examples_react = [
"""Question: What type of information architecture pattern is shown in the image?
Thought 1: I need to analyze the overall layout and structure of the interface first.
Action 1: Analyze[layout structure]
Observation 1: The interface shows a nested hierarchy with main categories and subcategories organized in a tree-like structure.
Thought 2: I should examine how users navigate through this structure.
Action 2: Explain[navigation pattern]
Observation 2: Users can navigate through different levels of content using expandable/collapsible sections.
Thought 3: Based on the layout and navigation, this is clearly a hierarchical navigation pattern.
Action 3: Finish[hierarchical navigation pattern]""",

"""Question: How does the color scheme contribute to the user experience?
Thought 1: I should first analyze the colors used in the interface.
Action 1: Analyze[color palette]
Observation 1: The interface uses a combination of primary colors for main actions and muted tones for secondary elements.
Thought 2: Now I need to understand how these colors affect the interface organization.
Action 2: Explain[color hierarchy]
Observation 2: The color scheme helps establish visual hierarchy and guides user attention to important elements.
Thought 3: The color scheme primarily serves to create visual hierarchy and guide attention.
Action 3: Finish[visual hierarchy and attention guidance]"""]

examples_evaluate = [
"""Question: What type of information architecture pattern is shown in the image?
Thought 1: I need to analyze the overall layout and structure of the interface first.
Action 1: Analyze[layout structure]
Observation 1: The interface shows a nested hierarchy with main categories and subcategories organized in a tree-like structure.

Evaluation:
The trajectory is correct as it starts with analyzing the fundamental layout structure, which is essential for identifying the information architecture pattern. The observation accurately describes the hierarchical nature of the interface.
Thus the correctness score is 10""",

"""Question: How does the color scheme contribute to the user experience?
Thought 1: I should first analyze the colors used in the interface.
Action 1: Analyze[color palette]
Observation 1: The interface uses a combination of primary colors for main actions and muted tones for secondary elements.

Evaluation:
The trajectory is correct as it begins with a systematic analysis of the color palette, which is fundamental to understanding its impact on user experience. The observation provides specific details about the color usage and its purpose.
Thus the correctness score is 10"""]

```

## LogiQA
```python
io = '''You are participating in a multiple-choice logic quiz. You will be given a paragraph, a question, and four choices.
Select the one choice that follows best from the paragraph.
Return only the letter: a, b, c, or d.

{paragraph}

{question}
{choices}'''

cot = '''You are participating in a multiple-choice logic quiz. You will be given a paragraph, a question, and four choices.
Think briefly about the logical relationship between the paragraph and the choices, then end with exactly "Answer: X" where X is a, b, c, or d.

{paragraph}

{question}
{choices}'''

bfs = '''You are participating in a multiple-choice logic quiz. You will be given a paragraph, a question, and four choices.
Generate multiple candidate answer choices. Each candidate must be only a, b, c, or d.

Format exactly:
Candidate 1: <a/b/c/d>
Candidate 2: <a/b/c/d>
Candidate 3: <a/b/c/d>

Current answer:
{current_answer}

{paragraph}

{question}
{choices}

Candidates:'''

act = '''You are participating in a multiple-choice quiz. You will be given a paragraph, which  contains the information needed to answer the question.
After the paragraph you will be given your question together with the four choices.

(Example)

In the planning of a new district in a township, it was decided to build a special community in the southeast, northwest, centered on the citizen park.These four communities are designated as cultural area, leisure area, commercial area and administrative service area.It is known that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area.

Based on the above statement, which of the following can be derived?
A.Civic Park is north of the administrative service area
B.The leisure area is southwest of the cultural area
C.The cultural district is in the northeast of the business district
D.The business district is southeast of the leisure area

---
Answer: a

(End of example)

Remember, your task is to select only one choise, which you think answers the question with respect to the paragraph given. Do not add any explanation, comments, introduction or conclusion, you shall only return your answer.

{paragraph}

{question}
{choises}'''

aggregate = '''You are participating in a multiple-choice quiz. You will be given a paragraph, which contains the information needed to answer the question.
After the paragraph you will be given your question together with the four choices.

You will be given possible answers to the given quiz, and it is your task to select the {k} best choices.

(Example)
In the planning of a new district in a township, it was decided to build a special community in the southeast, northwest, centered on the citizen park.These four communities are designated as cultural area, leisure area, commercial area and administrative service area.It is known that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area.

Based on the above statement, which of the following can be derived?
A.Civic Park is north of the administrative service area
B.The leisure area is southwest of the cultural area
C.The cultural district is in the northeast of the business district
D.The business district is southeast of the leisure area

---
Possible answers:
(1) Answer: a
(2) Answer: a
(3) Answer: b
(4) Answer: c
(5) Answer: a

Selected answers:
(1) Answer: a
(2) Answer: a
(5) Answer: a

(End of example)

Remember, your task is to select exactly {k} answers, which you think is correct with respect to the question and paragraph. Do not add any explanation, comments, introduction, conclusion, or modification, you shall only return the indexes of your {k} selected answers.

{paragraph}

{question}
{choices}

---
Possible answers:
{actions}

Selected answers:'''

evaluate = '''You are the judge of a multiple-choice quiz. You will be given the paragraph, which contains the information needed to answer the question.
After the paragraph you will be given the question together with choices and a proposed answer. It is then your task to judge the answer by correctness ("incorrect", "plausible", or "correct").

{examples}

(End of examples)

Remember, your task is to judge the chosen answer by its correctness ("incorrect", "plausible", or "correct"). Do not add any explanation, comments, introduction or conclusion, you shall only return your judgement.

{paragraph}

{question}
{choices}
---
Answer: {answer}
---'''

evaluate_examples = [
    '''In the planning of a new district in a township, it was decided to build a special community in the southeast, northwest, centered on the citizen park.These four communities are designated as cultural area, leisure area, commercial area and administrative service area.It is known that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area.

Based on the above statement, which of the following can be derived?
A.Civic Park is north of the administrative service area
B.The leisure area is southwest of the cultural area
C.The cultural district is in the northeast of the business district
D.The business district is southeast of the leisure area
---
Answer: a
---
correct
''',
'''In recent years, graduate entrance examinations have continued to heat up.Correspondingly, a variety of postgraduate counseling classes have emerged, especially English and political counseling classes are almost a must for the postgraduates.Xiaozhuang, who has just started working, also intends to take the postgraduate entrance exam, so Xiaozhuang must take English tutoring classes
Which of the following can best strengthen the above argument
A.If you take an English tutoring class, you can pass the graduate entrance exam
B.Only those who intend to take the graduate entrance exam will participate in the English tutoring class
C.Even if you take an English tutoring class, you may not be able to pass the graduate entrance exam
D.If you do not participate in the English tutoring class, you cannot pass the graduate entrance exam
---
Answer: D
---
correct''',
'''Compared with small and medium-sized cities, especially small cities and towns, large cities have higher living costs, which inevitably limits the entry of rural population.Therefore, the development of large cities alone cannot actually achieve urbanization
Which of the following is the conclusion must be assumed
A.Urbanization is the only way for China's development
B.Simple development of large cities is not conducive to the promotion of urbanization
C.To achieve urbanization, the city must fully absorb the rural population
D.The attractiveness of large cities to the rural population in the outside world is significantly lower than that of small and medium-sized cities
---
Answer: a
---
incorrect'''
]

# React prompt for LogiQA
react = '''You are participating in a multiple-choice quiz. You will be given a paragraph, which contains the information needed to answer the question.
After the paragraph you will be given your question together with the four choices.

You should think step by step about the logical reasoning needed to answer the question:

1. First, analyze what information is given in the paragraph
2. Then, understand how this information relates to the question
3. Next, evaluate each choice against the given information
4. Finally, make a logical conclusion about which choice is correct

(Example)

In the planning of a new district in a township, it was decided to build a special community in the southeast, northwest, centered on the citizen park. These four communities are designated as cultural area, leisure area, commercial area and administrative service area. It is known that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area.

Based on the above statement, which of the following can be derived?
A. Civic Park is north of the administrative service area
B. The leisure area is southwest of the cultural area
C. The cultural district is in the northeast of the business district
D. The business district is southeast of the leisure area

Thought: Let me analyze the given information step by step:
1. The paragraph describes relative positions of areas
2. We know: administrative service area is southwest of cultural area
3. We know: cultural area is southeast of leisure area
4. These areas are centered on the citizen park

Thought: Let me evaluate each choice:
A. Cannot be determined - Civic Park's position relative to administrative service area is not stated
B. Incorrect - This contradicts given info (cultural area is southeast of leisure area)
C. Cannot be determined - Business district's position relative to cultural district is not stated
D. Cannot be determined - Business district's position relative to leisure area is not stated

Thought: Based on the given information, none of the choices can be definitively derived. The paragraph only gives relative positions between administrative service area, cultural area, and leisure area.

---
Answer: a

(End of example)

Remember, your task is to think step by step and select only one choice that you think answers the question with respect to the paragraph given. Do not add any explanation, comments, introduction or conclusion, you shall only return your answer in the format "Answer: X" where X is one of a, b, c, or d.

{paragraph}

{question}
{choices}

Current reasoning:
{current_state}

---
Answer: '''

# Self-evaluation prompts for LogiQA
self_evaluate_step = '''You are evaluating a reasoning step in a logical reasoning task. Given the paragraph, question, choices, and the proposed reasoning step, determine if this step is correct and logical. Consider:
1. Does the reasoning follow from the given information?
2. Is the analysis of the choices accurate?
3. Are the logical connections valid?
4. Does it avoid making unsupported assumptions?

Paragraph: {paragraph}

Question: {question}
Choices: {choices}

Previous steps:
{previous_steps}

Current step: {step}

Is this reasoning step correct? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete solution to a logical reasoning task. Given the paragraph, question, choices, and the reasoning process, determine if the solution is correct. Consider:
1. Does the reasoning process follow logically from the given information?
2. Is the analysis of each choice thorough and accurate?
3. Are all logical connections valid and well-supported?
4. Does the final answer follow from the reasoning?
5. Are there any unsupported assumptions?

Paragraph: {paragraph}

Question: {question}
Choices: {choices}

Reasoning steps:
{steps}

Final answer: {answer}

Is this solution correct? Answer with a single word: Yes or No.
'''

```

## MathArena
```python
# Step-by-step analysis prompt
act = '''Given a math problem, analyze it step by step. Each step should either analyze the problem, explain concepts, or provide the final answer.
Return exactly one action. Put the full useful content inside the brackets.

Example Problem: Find the area of a circle with radius 5cm.
Step: Analyze[The problem asks for the area of a circle given its radius of 5cm.]

Example Problem: Solve the equation 2x + 4 = 12
Step: Analyze[We need to isolate x by subtracting 4 from both sides and then dividing by 2.]

Input Problem: {input}
Next step:
'''

# Brainstorming multiple approaches prompt
bfs = '''Given a math problem, list multiple possible approaches to solve it.

Example Problem: Find the derivative of y = x² + 3x
Possible approaches:
Analyze[problem] Focus on identifying function components
Explain[math concepts] Review power rule and derivative rules
Analyze[solution approach] Apply derivative rules step by step
Explain[solution steps] Show each term's derivative
Finish[y' = 2x + 3]

Input Problem: {input}
Possible approaches:
'''

# Complete solution chain-of-thought prompt
cot = '''Solve the given math problem step by step, showing your work and reasoning.

Example Problem: Solve 3x + 6 = 15
Steps:
Analyze[problem] Linear equation with one variable x
Explain[math concepts] Need to isolate x by using inverse operations
Analyze[solution approach] Subtract 6 from both sides, then divide by 3
Explain[solution steps] 3x = 9, then x = 3
Finish[3]

Input Problem: {input}
'''

evaluate = '''Evaluate the current partial solution to the math problem. Respond with a numeric score from 0 to 1, where 0 means the partial solution is useless or repetitive, 0.5 means it is somewhat relevant but incomplete, and 1 means it makes strong progress toward the final answer.

Example Problem: Find x if 2x = 10
Partial solution:
Analyze[Subtracting 2 from both sides solves it.]
Score: 0

Example Problem: Find x if 2x = 10
Partial solution:
Explain[Divide both sides by 2 to get x = 5.]
Score: 1

Input Problem: {input}
Partial solution:
{steps}
Score:
'''

# Solution verification prompt
evaluate_answer = '''Given a math problem and a proposed solution, determine if the solution is correct (correct/incorrect).

Example Problem: Solve 2x = 8
Proposed solution: x = 4
Judge:
correct

Example Problem: Find the area of a circle with radius 3
Proposed solution: A = 28.27
Judge:
incorrect

Input Problem: {input}
Proposed solution: {answer}
Judge:
'''

# Method compatibility prompts for roles that were not present in the original task.
io = cot
react = act

aggregate = '''Select the best {k} actions to continue solving the math problem. Return only the numbers of the selected actions.

Input Problem: {input}
Candidate actions:
{actions}
'''

self_evaluate_step = '''Given a math problem and a proposed reasoning step, determine if the step is useful (yes/no).

Input Problem: {input}
Previous steps:
{previous_steps}
Proposed step:
{step}
Judge:
'''

self_evaluate_answer = '''Given a math problem and a proposed solution, determine if the solution is correct (yes/no).

Input Problem: {input}
Proposed solution: {answer}
Judge:
'''

```

## MIMIC-RRS
```python
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

```

## MTSamples Procedures
```python
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

```

## PubMedQA
```python
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

```
