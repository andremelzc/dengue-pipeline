# Imagen para reproducir el pipeline con Python 3.13.3 y las versiones fijas de requirements.txt.
# Construir:  docker build -t dengue-pipeline .
# Pruebas:    docker run --rm dengue-pipeline pytest -q
# Pipeline:   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" dengue-pipeline python -m dengue_pipeline --help
FROM python:3.13.3-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .
COPY config ./config
COPY tests ./tests
COPY notebooks ./notebooks
CMD ["python", "-m", "dengue_pipeline", "--help"]
