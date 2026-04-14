def select_tool(task):
    task = task.lower()

    if "calculate" in task or "math" in task:
        return "calculator_tool"
    if "search" in task or "find" in task:
        return "retrieval_tool"
    if "email" in task or "message" in task:
        return "communication_tool"
    return "general_reasoning_tool"

tasks = [
    "Calculate the monthly average",
    "Search company policy documents",
    "Draft an email reply",
    "Explain this concept"
]

for task in tasks:
    print(f"Task: {task}")
    print(f"Selected Tool: {select_tool(task)}")
    print("-" * 40)
