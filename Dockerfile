# Imagen para reproducir el pipeline. Verificado el 22-nov-2026: la imagen se construye y las pruebas pasan dentro.
# Construir:  docker build -t dengue-pipeline .
# Pruebas:    docker run --rm dengue-pipeline pytest -q
# Pipeline:   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" dengue-pipeline python -m dengue_pipeline --help
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"
COPY config ./config
COPY tests ./tests
COPY notebooks ./notebooks
CMD ["python", "-m", "dengue_pipeline", "--help"]
