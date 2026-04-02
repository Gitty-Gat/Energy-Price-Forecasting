FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
COPY conf ./conf

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python", "src/pipelines/forecast.py", "--help"]
