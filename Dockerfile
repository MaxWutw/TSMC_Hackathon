# Vision Assistant
FROM python:3.9-slim AS visionassistant

EXPOSE 8001
WORKDIR /app
COPY va/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY va/ /app/va/
WORKDIR /app/va
CMD ["gunicorn", "-c", "service/vaGunicornConfig.py", "service.vaApp:app"]


# Function Calling
FROM pytorch/pytorch:latest AS functioncalling

EXPOSE 8002
WORKDIR /workspace
COPY tool/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY tool/ /workspace/tool/
WORKDIR /workspace/tool
RUN mkdir -p grounding-dino-base
CMD ["python3", "api_service.py"]
