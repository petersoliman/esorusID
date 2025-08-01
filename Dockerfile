# Base image
FROM python:3.11-slim

# Install system dependencies (libgl1 is required for OpenCV/Pillow)
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies (using simplified requirements)
COPY requirements_simple.txt .
RUN pip install --no-cache-dir -r requirements_simple.txt

# Copy app code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/static/recommendations /app/static/uploads

# Expose port
EXPOSE 8000

# Start the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
