agent_runs = [
    {"task": "summarize report", "success": True, "steps_used": 3, "tool_errors": 0},
    {"task": "search policy", "success": True, "steps_used": 4, "tool_errors": 1},
    {"task": "draft email", "success": False, "steps_used": 2, "tool_errors": 1},
]

success_rate = sum(run["success"] for run in agent_runs) / len(agent_runs)
average_steps = sum(run["steps_used"] for run in agent_runs) / len(agent_runs)
total_tool_errors = sum(run["tool_errors"] for run in agent_runs)

print("Agent Evaluation Summary:")
print("Success Rate:", success_rate)
print("Average Steps Used:", average_steps)
print("Total Tool Errors:", total_tool_errors)

print("\nDetailed Runs:")
for run in agent_runs:
    print(run)
