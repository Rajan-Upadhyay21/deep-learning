def generate_answer(question):
    return f"Initial answer for: {question}"

def reflect_on_answer(answer):
    return "Reflection: improve clarity and add one more useful detail."

def revise_answer(answer, reflection):
    return answer + " | Revised with better clarity and added detail."

question = "How does RAG improve LLM responses?"

initial_answer = generate_answer(question)
reflection = reflect_on_answer(initial_answer)
final_answer = revise_answer(initial_answer, reflection)

print("Question:")
print(question)

print("\nInitial Answer:")
print(initial_answer)

print("\nReflection:")
print(reflection)

print("\nFinal Answer:")
print(final_answer)
