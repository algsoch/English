FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p /app/english_talker_app/uploads /app/english_talker_app/static/audio

# Set the default port for the application
ENV PORT=3500
ENV HOST=0.0.0.0

# Command to run the application
CMD ["python", "-m", "english_talker_app.app.main"]