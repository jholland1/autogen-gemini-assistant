#Build with: 
    # docker build -t autogen-executor:latest .\docker_executor

# Use an official Python runtime as a parent image
# FROM python:3.12-slim
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install Python - not necesssary if using python base image
RUN apt-get update && \
    apt-get install -y \
        python3-pip \
        python3-dev && \
    ln -s /usr/bin/python3 /usr/bin/python
   
# Copy the requirements file into the container
COPY requirements.txt .

# Activate the virtual environment and install dependencies
RUN pip install --no-cache-dir -U -r requirements.txt --break-system-packages

# Upgrade pip
#RUN pip install --upgrade pip --break-system-packages

# Set the working directory in the container to /workspace,
#  which is where AutoGen mounts the host's work_dir.
WORKDIR /workspace

# Setup an app user so the container doesn't run as the root user
RUN useradd app
USER app

ENTRYPOINT ["/bin/bash"]