def think(question):
    return f"Thought: I should identify whether this needs retrieval or direct reasoning for '{question}'."

def act(question):
    if "latest" in question.lower() or "document" in question.lower():
        return "Action: use retrieval tool"
    return "Action: use direct reasoning"

def observe(action):
    if "retrieval" in action:
        return "Observation: relevant documents were found."
    return "Observation: enough information available internally."

def answer(question, observation):
    return f"Final answer for '{question}' based on: {observation}"

question = "Find the latest company policy update"

thought = think(question)
action = act(question)
observation = observe(action)
final = answer(question, observation)

print(thought)
print(action)
print(observation)
print(final)
