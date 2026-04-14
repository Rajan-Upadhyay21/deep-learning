def decompose_task(task):
    if "build ai assistant" in task.lower():
        return [
            "define user requirement",
            "choose model workflow",
            "add retrieval if needed",
            "add tool use",
            "test outputs"
        ]
    return [
        "understand goal",
        "split into subtasks",
        "complete subtasks",
        "combine final result"
    ]

task = "Build AI assistant for document Q&A"
subtasks = decompose_task(task)

print("Task:")
print(task)

print("\nSubtasks:")
for subtask in subtasks:
    print("-", subtask)
