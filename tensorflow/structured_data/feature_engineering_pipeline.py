import numpy as np


def build_features(raw_data):
    age = raw_data[:, 0]
    income = raw_data[:, 1]
    spending = raw_data[:, 2]

    income_per_age = income / (age + 1e-6)
    spending_ratio = spending / (income + 1e-6)
    high_income_flag = (income > 0.7).astype("float32")

    engineered = np.column_stack([
        age,
        income,
        spending,
        income_per_age,
        spending_ratio,
        high_income_flag,
    ]).astype("float32")
    return engineered


def main():
    raw_data = np.random.rand(6, 3).astype("float32")
    features = build_features(raw_data)

    print("Raw shape:", raw_data.shape)
    print("Engineered shape:", features.shape)
    print(features)


if __name__ == "__main__":
    main()
