def planner(user_request):
    if "summarize" in user_request.lower():
        return "summarization"
    if "search" in user_request.lower():
        return "retrieval"
    return "general_response"

def executor(task_type, user_request):
    if task_type == "summarization":
        return f"Summary task selected for: {user_request}"
    if task_type == "retrieval":
        return f"Retrieval task selected for: {user_request}"
    return f"General response task selected for: {user_request}"

user_request = "Search information about transformers and summarize it"

task_type = planner(user_request)
result = executor(task_type, user_request)

print("User Request:")
print(user_request)

print("\nPlanned Task:")
print(task_type)

print("\nExecution Result:")
print(result)
