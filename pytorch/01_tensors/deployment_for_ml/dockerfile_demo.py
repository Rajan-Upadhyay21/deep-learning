dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "flask_model_api.py"]
""".strip()

print("Dockerfile Example:")
print(dockerfile_content)
