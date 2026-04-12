def build_prompt(role, task, input_text):
    prompt = f"""
Role: {role}

Task: {task}

Input:
{input_text}

Response:
"""
    return prompt.strip()

prompt = build_prompt(
    role="Helpful AI Assistant",
    task="Summarize the input in one sentence",
    input_text="PyTorch is a popular deep learning framework used for neural networks and AI research."
)

print("Generated Prompt:")
print(prompt)
