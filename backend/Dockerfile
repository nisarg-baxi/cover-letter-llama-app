FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG  HUGGINGFACE_TOKEN
RUN pip install huggingface_hub
RUN huggingface-cli login --token ${HUGGINGFACE_TOKEN}

COPY . .

# Expose the port the app runs on
EXPOSE 5000

CMD ["python3", "app.py"]