memory = []

def store_memory(item):
    memory.append(item)

def retrieve_memory():
    return memory

store_memory("User prefers concise responses")
store_memory("User is working on AI portfolio folders")
store_memory("Current topic is agentic AI basics")

print("Stored Memory:")
for item in retrieve_memory():
    print(item)
