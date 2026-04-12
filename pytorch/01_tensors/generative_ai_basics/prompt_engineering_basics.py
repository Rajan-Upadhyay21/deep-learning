def create_prompt(role, instruction, context, user_input):
    return f"""
Role: {role}

Instruction: {instruction}

Context: {context}

User Input: {user_input}

Answer:
""".strip()

prompt = create_prompt(
    role="Helpful AI assistant",
    instruction="Answer clearly in two sentences.",
    context="The user is learning generative AI basics.",
    user_input="What is prompt engineering?"
)

print("Prompt Template:")
print(prompt)
