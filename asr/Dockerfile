# Use an official Python image as the base
FROM python:3.9.6

# Set the working directory inside the container
WORKDIR /asr

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app into the container
COPY asr_api.py asr_api.py

# Expose port 8001 for the FastAPI server
EXPOSE 8001

# Command to run the FastAPI application
CMD ["uvicorn", "asr_api:app", "--host", "0.0.0.0", "--port", "8001"]