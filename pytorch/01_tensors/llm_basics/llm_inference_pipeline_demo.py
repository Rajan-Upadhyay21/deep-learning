def tokenize(text):
    return text.lower().replace(".", "").split()

def build_prompt(user_input):
    return f"Answer the following question clearly:\n{user_input}"

def mock_generate(prompt):
    return f"Generated response for prompt: {prompt}"

def inference_pipeline(user_input):
    tokens = tokenize(user_input)
    prompt = build_prompt(user_input)
    response = mock_generate(prompt)

    return {
        "tokens": tokens,
        "prompt": prompt,
        "response": response
    }

result = inference_pipeline("How do transformers work in AI?")

print("Tokens:")
print(result["tokens"])

print("\nPrompt:")
print(result["prompt"])

print("\nResponse:")
print(result["response"])
