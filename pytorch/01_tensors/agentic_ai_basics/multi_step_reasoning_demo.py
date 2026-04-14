def solve_task(task):
    steps = [
        "Step 1: Understand the task",
        "Step 2: Identify needed information",
        "Step 3: Produce intermediate reasoning",
        "Step 4: Generate final answer"
    ]

    return {
        "task": task,
        "steps": steps,
        "final_answer": f"Completed multi-step reasoning for: {task}"
    }

result = solve_task("Compare two recommendation strategies")

print("Task:")
print(result["task"])

print("\nReasoning Steps:")
for step in result["steps"]:
    print(step)

print("\nFinal Answer:")
print(result["final_answer"])
