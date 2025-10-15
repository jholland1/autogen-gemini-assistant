# AutoGen Gemini Assistant

**Note: These instructions are tailored for Windows OS.**

This repository contains an AutoGen-based assistant that leverages the Gemini model for various tasks. It includes configurations for both local execution and Dockerized environments.

## Table of Contents
- [Local Environment Setup](#local-environment-setup)
- [Docker Desktop Setup](#docker-desktop-setup)
- [Docker Container Setup](#docker-container-setup)
- [Running the Code](#running-the-code)
  - [Running `main.py`](#running-mainpy)
  - [Running `gemini_mwe_test.py`](#running-gemini_mwe_testpy)

## Local Environment Setup

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jholland1/autogen-gemini-assistant.git
    cd autogen-gemini-assistant
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Create a `.env` file:**
    Create a file named `.env` in the root directory of the project and add your API keys and other environment variables. For example:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    GOOGLE_SEARCH_ENGINE_ID="YOUR_GOOGLE_SEARCH_ENGINE_ID"
    ```
    Replace `"YOUR_GEMINI_API_KEY"`, `"YOUR_GOOGLE_API_KEY"`, and `"YOUR_GOOGLE_SEARCH_ENGINE_ID"` with your actual keys.

## Docker Desktop Setup

If you plan to run the project using Docker, ensure you have Docker Desktop installed and running.

1.  **Download Docker Desktop:**
    Visit the official Docker website and download Docker Desktop for your operating system: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

2.  **Install and Configure:**
    Follow the installation instructions for your OS. After installation, make sure Docker Desktop is running.

## Docker Container Setup

To build and prepare the Docker container for the project:

1.  **Navigate to the `docker_executor` directory:**
    ```bash
    cd docker_executor
    ```

2.  **Build the Docker image:**
    ```bash
    docker build -t autogen-executor:latest .
    ```
    This command builds a Docker image named `autogen-executor` with the tag `latest` using the `Dockerfile` in the current directory.

## Running the Code

### Running `main.py`

The `main.py` script demonstrates the core functionality of the AutoGen assistant.

1.  **Ensure your `.env` file is configured** with `GEMINI_API_KEY`, `GOOGLE_API_KEY`, and `GOOGLE_SEARCH_ENGINE_ID`.

2.  **Run the script:**
    ```bash
    python app/main.py
    ```

### Running `gemini_mwe_test.py`

The `gemini_mwe_test.py` script is a minimal working example to test the Gemini API integration.

1.  **Ensure your `.env` file is configured** with `GEMINI_API_KEY`.

2.  **Run the script:**
    ```bash
    python app/gemini_mwe_test.py
    ```
