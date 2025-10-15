
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

import time
import requests
from bs4 import BeautifulSoup

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
        timeout=300,
        work_dir="coding_output",
        stop_container=False
    )
    await executor.start()

    google_search_tool = FunctionTool(
        google_search, description="Search Google for information, returns results with a snippet and body content"
    )

    fetch_url_content_tool = FunctionTool(
        get_page_content, description="Fetch content from a web page URL and return content"
    )

    requirements_analyst = AssistantAgent(
        name="requirements_analyst",
        system_message="""The python_coder is in charge of writing and executing code.
        You are a requirements analyst. Your goal is to gather information and write a detailed requirements list and algorithm specification
        to pass to a python coder. If the user provided you with a URL, use the 'fetch_url_content_tool' tool to retreive content from that
        link to improve your requirements specification. If the user requested you to search for solutions, use the google_search_tool
        to search for relevant examples, algorithms, existing python repositories, etc. to improve your requirements specification. 
        Do not write any code. If you need information from the user, you must first send your message, then you can handoff to the user.
        State the final specification and detailed requirements and then handoff to the python_coder. After sending your message handoff 
        to python_coder.
        """,
        handoffs=["python_coder","user"],
        tools=[fetch_url_content_tool,google_search_tool],
        model_client=model_client,
        reflect_on_tool_use=True,
    )

    python_code_execution_tool = PythonCodeExecutionTool(executor)

    python_coder = AssistantAgent(
        name="python_coder",
        system_message=""""You are an expert Python programmer. Write a complete, executable Python script based on the 
        requirements and research. Ensure your code is well-commented. If necessary, perform a Google search using the 'google_search_tool'
        to retreive relevant code examples to use in the script to meet specified requirements.
        Ensure that the code you write will terminate when complete and save require output. Do not expect user interaction to view results 
        (i.e. save plots as png, don't terminate with plt.show()). 
        Execute the code using the 'python_code_execution_tool' tool, if it runs successfully return the std_out messages and handoff to the results_evaluator, 
        if it fails return std_out and std_err and handoff to the code_debugger. Only handoff to results_evaluator or code_debugger, DO NOT handoff to user.
        """,
        model_client=model_client,
        handoffs=["results_evaluator","code_debugger"],
        tools=[python_code_execution_tool, google_search_tool],
        reflect_on_tool_use=True,
    )

    code_debugger = AssistantAgent( #python_coder seems to debug itself when reflect_on_tool_use=True, not sure this is necessary.
        name="code_debugger",
        system_message="""You are a debugging expert. Analyze code output and execution errors (if any) and provide specific, 
        targeted fixes (if needed). Do not rewrite the entire script. Direct your fix to the python_coder. If requirements clarification
        is required to resolve an error, request requirements clarification by sending your message and calling the handoff to the requirements_analyst.
        If you encounter an error that is difficult to resolve, use the 'google_search_tool' tool to search the web for solutions to the
        error.
        """,
        handoffs=["python_coder","requirements_analyst","google_search_tool"],
        model_client=model_client,
    )

    results_evaluator = AssistantAgent(
        name="results_evaluator",
        system_message="""You are a helpful critical evaluator of python script performance. You will receive output from a Python script
        and you will critically evaluate it against its specified requirements. If the script meets requirements, you will pass control and hand off to 
        the user. If the code does not meet requirements please explain the requirement that is not met and why it was not met, and 
        handoff to the requirements_analyst for revision of specifications.
        """,
        handoffs=["user","requirements_analyst"],
        model_client=model_client,
    )


    # --- TEAM SETUP (RoundRobinGroupChat) ---
    # termination_condition = MaxMessageTermination(max_messages=20) | TextMentionTermination("APPROVE")
    termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE")

    agent_team = [requirements_analyst, python_coder, code_debugger, results_evaluator] 

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
