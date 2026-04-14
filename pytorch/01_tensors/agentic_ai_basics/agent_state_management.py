agent_state = {
    "task": "Analyze customer feedback",
    "current_step": "retrieve comments",
    "completed_steps": [],
    "status": "running"
}

def update_state(state, finished_step, next_step):
    state["completed_steps"].append(finished_step)
    state["current_step"] = next_step
    return state

print("Initial State:")
print(agent_state)

agent_state = update_state(agent_state, "retrieve comments", "summarize themes")
agent_state = update_state(agent_state, "summarize themes", "prepare final report")

print("\nUpdated State:")
print(agent_state)
