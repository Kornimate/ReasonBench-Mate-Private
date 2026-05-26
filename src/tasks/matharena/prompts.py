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
