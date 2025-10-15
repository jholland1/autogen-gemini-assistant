#Build with: 
    # docker build -t autogen-executor:latest .\docker_executor

# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container to /workspace,
# which is where AutoGen mounts the host's work_dir.
WORKDIR /workspace

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Setup an app user so the container doesn't run as the root user
RUN useradd app
USER app

CMD /bin/bash