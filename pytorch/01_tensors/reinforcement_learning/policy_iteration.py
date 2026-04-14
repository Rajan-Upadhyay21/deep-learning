states = ["S1", "S2", "S3"]
policy = {
    "S1": "right",
    "S2": "up",
    "S3": "left"
}

print("Initial Policy:")
for state, action in policy.items():
    print(f"{state} -> {action}")

policy["S2"] = "right"

print("\nImproved Policy:")
for state, action in policy.items():
    print(f"{state} -> {action}")
