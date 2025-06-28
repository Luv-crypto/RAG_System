# ——————————————————————————————————————
# Stage 1: install dependencies
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ——————————————————————————————————————
# Stage 2: copy code and run
FROM python:3.10-slim
WORKDIR /app

# Silence ChromaDB telemetry inside the container
ENV ANONYMIZED_TELEMETRY=False \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5000 \
    FLASK_DEBUG=False \
    PYTHONUNBUFFERED=1

# Copy deps from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your app code
COPY .env .
COPY . .


# Expose the port your Flask app listens on
EXPOSE 5000

# Launch your app
CMD ["python", "app.py"]
