# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port (e.g., FastAPI defaults to 8000)
EXPOSE 8000

# Start the app (edit this based on your app)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]