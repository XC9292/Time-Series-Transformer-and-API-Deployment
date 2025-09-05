# Stage 1: Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set environment variables
# Prevents Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE 1 
# Ensures that Python output is sent straight to terminal (e.g., for logging)
ENV PYTHONUNBUFFERED 1  

# Set the working directory in the container
WORKDIR /code

# Install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code into the container
COPY ./app /code/app
COPY ./scripts /code/scripts
COPY ./data /code/data
COPY ./model /code/model

# Expose the port the app runs on
EXPOSE 8000

# The command to run the FastAPI application using Uvicorn
# --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]