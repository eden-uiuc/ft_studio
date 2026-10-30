# Use the Python 3.12 base
FROM python:3.12-slim

# Grab uv directly from the official astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Hugging Face requires running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy project files
COPY --chown=user:user . .

# Use uv to install directly from pyproject.toml into the system environment
RUN uv sync --frozen --system

# Expose the port Hugging Face expects
EXPOSE 7860

ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Start the NiceGUI app
CMD ["python", "main.py"]