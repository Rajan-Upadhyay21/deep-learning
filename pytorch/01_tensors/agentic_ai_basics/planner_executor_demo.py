
Code files:

## `planner_executor_demo.py`

```python
def planner(user_goal):
    if "summary" in user_goal.lower():
        return ["read input", "extract key points", "write summary"]
    if "search" in user_goal.lower():
        return ["identify topic", "retrieve documents", "draft answer"]
    return ["understand request", "prepare response"]

def executor(plan):
    completed_steps = []
    for step in plan:
        completed_steps.append(f"Completed: {step}")
    return completed_steps

goal = "Create a summary of the given document"

plan = planner(goal)
results = executor(plan)

print("Goal:")
print(goal)

print("\nPlan:")
print(plan)

print("\nExecution:")
for item in results:
    print(item)
