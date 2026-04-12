import pandas as pd

data = {
    "age": [25, 30, -5, 40],
    "salary": [50000, 60000, 55000, None],
    "department": ["HR", "IT", "Finance", "IT"]
}

df = pd.DataFrame(data)

print("Input Data:")
print(df)

invalid_age_rows = df[df["age"] < 0]
missing_salary_rows = df[df["salary"].isnull()]

print("\nRows with Invalid Age:")
print(invalid_age_rows)

print("\nRows with Missing Salary:")
print(missing_salary_rows)
