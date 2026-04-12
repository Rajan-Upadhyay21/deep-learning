prompt = "A futuristic city skyline at sunset with flying cars"

generation_steps = [
    "Receive text prompt",
    "Convert prompt into internal text representation",
    "Map prompt to visual concept space",
    "Generate image representation step by step",
    "Decode representation into final image"
]

print("Prompt:")
print(prompt)

print("\nConceptual Image Generation Steps:")
for step_number, step in enumerate(generation_steps, start=1):
    print(f"{step_number}. {step}")
