def route_request(user_request):
    request = user_request.lower()

    if "summarize" in request:
        return "summarization_workflow"
    if "search" in request or "retrieve" in request:
        return "retrieval_workflow"
    if "email" in request:
        return "communication_workflow"
    if "analyze" in request:
        return "analysis_workflow"
    return "general_workflow"

requests = [
    "Summarize this article",
    "Search internal notes for the answer",
    "Analyze customer reviews",
    "Write an email reply"
]

for request in requests:
    print(f"Request: {request}")
    print(f"Routed To: {route_request(request)}")
    print("-" * 40)
