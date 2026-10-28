# Universal Dockerfile for Hugging Face Spaces, Render, and Container Platforms
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

WORKDIR /app

# Install minimal system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt requirements-streamlit.txt* ./
RUN pip install --no-cache-dir -r requirements.txt && \
    (test -f requirements-streamlit.txt && pip install --no-cache-dir -r requirements-streamlit.txt || true)

# Copy application files
COPY . .

# Create non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Default port: 7860 for Hugging Face Spaces, or dynamic $PORT for Render / Cloud Run
EXPOSE 7860

ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0 --server.headless=true"]
