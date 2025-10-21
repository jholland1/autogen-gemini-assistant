
#Swarm documentation - https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/swarm.html

import os
import asyncio
from dotenv import load_dotenv

# Import the specific client from the extensions module
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent

# Import AutoGen AgentChat high-level API components
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_core.models import ModelInfo
from autogen_core.tools import FunctionTool
from docker.types import DeviceRequest


import time
import requests
from bs4 import BeautifulSoup
import glob

# --- 1. CONFIGURATION ---
# Load environment variables from.env file
load_dotenv(".env")

# --- 2. TOOL DEFINITION ---
def get_page_content(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        words = text.split()
        content = ""
        max_chars = 50000
        for word in words:
            if len(content) + len(word) + 1 > max_chars:
                break
            content += " " + word
        return content.strip()
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return ""

def google_search(query: str, num_results: int = 15, max_chars: int = 15000) -> list:  # type: ignore[type-arg]
    # num_results = number of web pages to get content from from the search results
    # max_chars = number of characters to scrape from the web pages

    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        raise ValueError("API key or Search Engine ID not found in environment variables")

    url = "https://customsearch.googleapis.com/customsearch/v1"
    params = {"key": str(api_key), "cx": str(search_engine_id), "q": str(query), "num": str(num_results)}

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(response.json())
        raise Exception(f"Error in API request: {response.status_code}")

    results = response.json().get("items", [])

    enriched_results = []
    for item in results:
        body = get_page_content(item["link"])
        enriched_results.append(
            {"title": item["title"], "link": item["link"], "snippet": item["snippet"], "body": body}
        )
        time.sleep(1)  # Be respectful to the servers

    return enriched_results

def read_python_files_from_folder() -> str:
    """
    Reads all Python files from a specified folder and returns their content as a single,
    properly formatted Python code string.
    If no Python files are found or the folder does not exist, it returns "No existing python files found".
    """
    folder_path = "existing_code"  # Folder containing existing python files to give to the requirements analyst
    if not os.path.isdir(folder_path):
        return "No existing python files found - Path not found"

    python_files = glob.glob(os.path.join(folder_path, "*.py"))
    if not python_files:
        return "No existing python files found - No .py files in directory"

    all_code = []
    for file_path in python_files:
        with open(file_path, "r", encoding="utf-8") as f:
            all_code.append(f"# --- Start of file: {os.path.basename(file_path)} ---")
            all_code.append(f"\n'''python\n")
            all_code.append(f.read())
            all_code.append(f"\n'''\n")
            all_code.append(f"\n# --- End of file: {os.path.basename(file_path)} ---\n\n")
    return "\n".join(all_code)

def read_output_files() -> str:
    output_folder = "coding_output"  # Folder where the python script saves its output files
    if not os.path.isdir(output_folder):
        return "No output files found - Path not found"

    output_files = glob.glob(os.path.join(output_folder, "*"))
    if not output_files:
        return "No output files found - No files in directory"

    all_output = []
    for file_path in output_files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            all_output.append(f"{os.path.basename(file_path)}")
    return "\n".join(all_output)

def read_installed_packages() -> str:
    with open("docker_executor/requirements.txt", "r", encoding="utf-8") as f:
        return f.read()

from pydantic import BaseModel
class ScriptOutput(BaseModel):
    topic: str
    takeaway: str
    captions: list[str]

async def main():
    run_local = False
    if run_local :
        # Create a client with your local Ollama model
        model_client = OllamaChatCompletionClient(
            model="huihui_ai/gpt-oss-abliterated:20b",
            model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="openai", structured_output=True),
            parallel_tool_calls=False,
            response_format=ScriptOutput,
            # max_retries=150,
        )
    else :
    # --- MODEL CLIENT SETUP ---
        model_client = OpenAIChatCompletionClient(
            model="gemini-2.5-flash-lite",
            # model="gemini-2.5-pro",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.environ.get("GEMINI_API_KEY"),
            # model_info={"family": "gemini", "vision": True, "function_calling": True, "json_output": True},
            model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="gemini", structured_output=True),
            parallel_tool_calls=False,
            max_retries=150,
            # api_rate_limit=0.05,
        )

    # --- AGENT AND EXECUTOR SETUP ---
    executor = DockerCommandLineCodeExecutor(
        image="autogen-executor:latest",
        timeout=3000,
        work_dir="coding_output",
        stop_container=False,
        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])]
    )
    await executor.start()

    google_search_tool = FunctionTool(
        google_search, description="Search Google for information, returns results with a snippet and body content"
    )

    fetch_url_content_tool = FunctionTool(
        get_page_content, description="Fetch content from a web page URL and return content"
    )

    read_python_files_tool = FunctionTool(
        read_python_files_from_folder, description="Reads all Python files from a specified folder and returns their content."
    )

    installed_packages_tool = FunctionTool(
        read_installed_packages, description="Reads the currently installed python packages and their versions from the requirements.txt file."
    )

    get_output_filenames_tool = FunctionTool(
        read_output_files, description="Reads the names of all output files generated by the python script"
    )

    requirements_analyst = AssistantAgent(
        name="requirements_analyst",
        system_message="""You are a requirements analyst expert. Your job is to write detailed python coding requirements. DO NOT write any code. Execute the following
        steps to complete this task:
        1. Check to see if there is existing code to modify by calling the 'read_python_files_tool'
        2. Assess whether the user is requesting additional requirements or bug fixes to existing code.
        3. If the user provided you with a URL, use the 'fetch_url_content_tool' tool to retreive content from that URL
        4. Research solutions for the additional requirements or bug fixes using the 'google_search_tool'
        5. Check existing dependencies usingt the 'installed_packages_tool'. If additional dependencies are absolutely necessary,
        list them in your message and why they are needed and then handoff to user. 
        6. Write detailed coding requirements and a step-by-step algorithm specification. 
        7. Return your message, and handoff to the python_coder.
        """,
        handoffs=["python_coder","user"],
        tools=[fetch_url_content_tool,google_search_tool,read_python_files_tool,installed_packages_tool],
        model_client=model_client,
        reflect_on_tool_use=True,
    )

    python_code_execution_tool = PythonCodeExecutionTool(executor)

    python_coder = AssistantAgent(
        name="python_coder",
        system_message=""""You are an expert Python programmer. You will receive detailed requirements, existing code (if any), and research
         to aid in writing and executing a python script. Perform the following steps BEFORE handing off to the user:
        1. Search the web using the 'google_search_tool' for any additional information you need to write the code.
        2. Write a complete, executable Python script based on the requirements and research. Save all output to files in the working directory,
        it is very important that final results are saved to files in the working directory (no subfolders, save in root of working directory),
        plots and/or animations should be saved as image files, data should be saved in csv files, and enough information should be written
        to the console to catch any errors and undersand the output of the script.
        3. Execute the code using the 'python_code_execution_tool' tool. Always execute the code after writing or modifying it in any way.
        4. Use the 'get_output_filenames_tool' tool to get a list of all output files generated by the script to make sure expected files were created during code execution.
        5. (if necessary) Debug the code and try again until it runs successfully, be sure to search the web for solutions to any difficult bugs or errors.
        6. When the code runs successfully, handoff to the user using the 'transfer_to_user' tool.
        """,
        model_client=model_client,
        handoffs=["user"],
        tools=[python_code_execution_tool, google_search_tool, get_output_filenames_tool,read_python_files_tool],
        reflect_on_tool_use=True,
    )

    # --- TEAM SETUP (RoundRobinGroupChat) ---
    # termination_condition = MaxMessageTermination(max_messages=20) | TextMentionTermination("APPROVE")
    termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE")

    agent_team = [requirements_analyst, python_coder] 

    team=Swarm(agent_team, termination_condition=termination)

    # --- INITIATE CHAT AND STREAM RESPONSES ---

    async def run_team_stream() -> None:
        task = input("What is your task? \nUser: ")
        task_result = await Console(team.run_stream(task=task))
        last_message = task_result.messages[-1]

        while isinstance(last_message, HandoffMessage) and last_message.target == "user":
            user_message = input("User: ")

            task_result = await Console(
                team.run_stream(task=HandoffMessage(source="user", target=last_message.source, content=user_message))
            )
            last_message = task_result.messages[-1]

    await run_team_stream()
    await model_client.close()
    await executor.stop()

if __name__ == "__main__":
    asyncio.run(main())
