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


# Function Calling: grounding-dino
FROM pytorch/pytorch:latest AS groundingdino

EXPOSE 8002
WORKDIR /workspace
COPY tool/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY tool/ /workspace/tool/
WORKDIR /workspace/tool
RUN mkdir -p grounding-dino-base
CMD ["python3", "dino_service.py"]


# Function Calling: inpainting
FROM python:3.9-slim AS inpainting

EXPOSE 8003
WORKDIR /workspace
COPY va/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY tool/ /app/tool/
WORKDIR /app/tool
RUN mkdir -p grounding-dino-base
CMD ["python3", "inpainting_service.py"]


# Function Calling: imagen
FROM python:3.9-slim AS imagen

EXPOSE 8004
WORKDIR /app
COPY va/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY tool/ /app/tool/
WORKDIR /app/tool
RUN mkdir -p grounding-dino-base
CMD ["python3", "imagen_service.py"]

