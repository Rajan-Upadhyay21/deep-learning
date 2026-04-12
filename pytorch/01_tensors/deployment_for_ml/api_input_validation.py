def validate_input(data):
    required_fields = ["age", "salary"]

    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    if data["age"] < 0:
        return False, "Age cannot be negative"

    if data["salary"] < 0:
        return False, "Salary cannot be negative"

    return True, "Input is valid"

sample_data = {"age": 30, "salary": 60000}

is_valid, message = validate_input(sample_data)

print("Validation Result:", is_valid)
print("Message:", message)
