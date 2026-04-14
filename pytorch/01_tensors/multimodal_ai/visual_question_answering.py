image_context = {
    "objects": ["cat", "chair", "window"],
    "colors": ["white", "brown"],
    "location": "living room"
}

question = "What animal is in the image?"

def answer_question(context, question_text):
    if "animal" in question_text.lower():
        return context["objects"][0]
    if "where" in question_text.lower():
        return context["location"]
    return "Answer not found."

answer = answer_question(image_context, question)

print("Question:")
print(question)

print("\nAnswer:")
print(answer)
