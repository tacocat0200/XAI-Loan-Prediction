# Use the official Python image from the Docker Hub as a base image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing pyc files and to enable unbuffered mode
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create a working directory for the application
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application directory into the container at /app
COPY . /app/

# Expose the port the app runs on
EXPOSE 5000

# Specify the command to run the application
CMD ["python", "app.py"]
