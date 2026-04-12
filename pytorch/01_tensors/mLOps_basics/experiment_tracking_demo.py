experiments = [
    {"run_id": 1, "lr": 0.01, "batch_size": 32, "accuracy": 0.87},
    {"run_id": 2, "lr": 0.001, "batch_size": 32, "accuracy": 0.90},
    {"run_id": 3, "lr": 0.001, "batch_size": 64, "accuracy": 0.92}
]

print("Experiment Tracking Logs:")
for experiment in experiments:
    print(experiment)

best_run = max(experiments, key=lambda x: x["accuracy"])

print("\nBest Run:")
print(best_run)
