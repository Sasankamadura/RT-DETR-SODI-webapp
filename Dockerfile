# Use an official Python runtime as a parent image
# Slim version for smaller footprint, 3.10 is stable
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt .

# Install Python dependencies
# Note: Installing CPU version of onnxruntime for the docker image to keep size down and compatibility up
# If you have a GPU-enabled server, change this to onnxruntime-gpu
RUN pip install --no-cache-dir -r requirements.txt
# Uninstall GPU version if present from requirements and forcefuly install CPU version
RUN pip uninstall -y onnxruntime-gpu && pip install onnxruntime

# Copy the entire application
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the application
# We use the python -m uvicorn approach
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
