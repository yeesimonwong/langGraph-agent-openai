import os
from dotenv import load_dotenv
import logging
import paramiko
import httpx
from httpx_socks import SyncProxyTransport
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated, Sequence
import operator
import uuid
from pydantic import BaseModel, Field
import tempfile
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info(f"Environment variables loaded: SSH_HOST={os.getenv('SSH_HOST')}, REMOTE_PATH={os.getenv('REMOTE_PATH')}, REQ_PATH={os.getenv('REQ_PATH')}")

# Set up SOCKS5 proxy
def setup_proxy():
    try:
        proxy_host = os.getenv("SOCKS5_HOST")
        proxy_port = os.getenv("SOCKS5_PORT")
        proxy_username = os.getenv("SOCKS5_USERNAME")
        proxy_password = os.getenv("SOCKS5_PASSWORD")
        if not all([proxy_host, proxy_port, proxy_username, proxy_password]):
            raise ValueError("Missing SOCKS5 proxy environment variables")
        proxy_url = f"socks5://{proxy_username}:{proxy_password}@{proxy_host}:{proxy_port}"
        proxy_transport = SyncProxyTransport.from_url(proxy_url)
        return httpx.Client(transport=proxy_transport, timeout=30.0)
    except Exception as e:
        logger.error(f"Failed to set up proxy: {str(e)}")
        raise

try:
    http_client = setup_proxy()
    logger.info("Proxy setup successful")
except Exception as e:
    logger.error(f"Failed to initialize proxy: {str(e)}")
    raise

# Initialize LLM
try:
    llm = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        http_client=http_client
    )
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {str(e)}")
    raise

# Define state
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage], operator.add]
    setup_success: bool
    folder_created: bool
    venv_created: bool
    file_copied: bool
    pip_installed: bool
    error_message: str
    user_input_needed: bool
    pip_retry_count: int
    command_outputs: list

# Define Pydantic model for SSH credentials
class Credentials(BaseModel):
    hostname: str = Field(description="Hostname of the SSH server")
    username: str = Field(description="Username for SSH login")
    password: str = Field(description="Password for SSH login")
    port: int = Field(description="Port for SSH connection")

# SSH connection helper
def get_ssh_client():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=os.getenv("SSH_HOST", "localhost"),
        port=int(os.getenv("SSH_PORT", 22)),
        username=os.getenv("SSH_USERNAME", ""),
        password=os.getenv("SSH_PASSWORD", ""),
        timeout=10
    )
    return ssh

# Define tools
@tool
def perform_login(credentials: Credentials) -> dict:
    """Establish an SSH connection to a remote server."""
    logger.info(f"Attempting SSH login to {credentials.username}@{credentials.hostname}:{credentials.port}")
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=credentials.hostname,
            port=credentials.port,
            username=credentials.username,
            password=credentials.password,
            timeout=10
        )
        ssh.close()
        logger.info(f"SSH login successful for {credentials.username}@{credentials.hostname}")
        return {
            "setup_success": True,
            "error_message": "",
            "command_outputs": [f"SSH login successful for {credentials.username}@{credentials.hostname}"]
        }
    except Exception as e:
        logger.error(f"SSH login failed: {str(e)}")
        return {
            "setup_success": False,
            "error_message": f"SSH login failed: {str(e)}",
            "command_outputs": [f"SSH login failed: {str(e)}"]
        }

@tool
def create_folder() -> dict:
    """Create a folder on the remote server."""
    remote_path = os.getenv("REMOTE_PATH", "test01")
    logger.info(f"Creating folder at {remote_path} on remote server")
    try:
        ssh = get_ssh_client()
        stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_path}")
        stderr_output = stderr.read().decode()
        stdout_output = stdout.read().decode()
        ssh.close()
        if stderr_output:
            logger.error(f"Failed to create folder {remote_path}: {stderr_output}")
            return {
                "folder_created": False,
                "error_message": f"Failed to create folder: {stderr_output}",
                "command_outputs": [f"mkdir -p {remote_path}\nError: {stderr_output}"]
            }
        logger.info(f"Folder created at {remote_path} on remote server")
        return {
            "folder_created": True,
            "error_message": "",
            "command_outputs": [f"mkdir -p {remote_path}\nOutput: {stdout_output}"]
        }
    except Exception as e:
        logger.error(f"Failed to create folder: {str(e)}")
        return {
            "folder_created": False,
            "error_message": f"Failed to create folder: {str(e)}",
            "command_outputs": [f"mkdir -p {remote_path}\nError: {str(e)}"]
        }

@tool
def create_venv() -> dict:
    """Create a virtual environment on the remote server."""
    remote_path = os.getenv("REMOTE_PATH", "test01")
    venv_path = f"{remote_path}/venv"
    logger.info(f"Creating venv at {venv_path} on remote server")
    try:
        ssh = get_ssh_client()
        python_cmd = os.getenv("PYTHON_VERSION", "python3")
        stdin, stdout, stderr = ssh.exec_command(f"{python_cmd} --version")
        stderr_output = stderr.read().decode()
        stdout_output = stdout.read().decode()
        if stderr_output:
            logger.error(f"Python {python_cmd} not available on remote server: {stderr_output}")
            ssh.close()
            return {
                "venv_created": False,
                "error_message": f"Python {python_cmd} not available: {stderr_output}",
                "command_outputs": [f"{python_cmd} --version\nError: {stderr_output}"]
            }
        stdin, stdout, stderr = ssh.exec_command(f"{python_cmd} -m venv {venv_path}")
        stderr_output = stderr.read().decode()
        stdout_output = stdout.read().decode()
        ssh.close()
        if stderr_output:
            logger.error(f"Failed to create venv at {venv_path}: {stderr_output}")
            return {
                "venv_created": False,
                "error_message": f"Failed to create venv: {stderr_output}",
                "command_outputs": [f"{python_cmd} -m venv {venv_path}\nError: {stderr_output}"]
            }
        logger.info(f"Virtual environment created at {venv_path} on remote server using {python_cmd}")
        return {
            "venv_created": True,
            "error_message": "",
            "command_outputs": [f"{python_cmd} -m venv {venv_path}\nOutput: {stdout_output}"]
        }
    except Exception as e:
        logger.error(f"Failed to create venv: {str(e)}")
        return {
            "venv_created": False,
            "error_message": f"Failed to create venv: {str(e)}",
            "command_outputs": [f"{python_cmd} -m venv {venv_path}\nError: {str(e)}"]
        }

@tool
def copy_requirements() -> dict:
    """Copy requirements.txt to the remote server."""
    remote_path = os.getenv("REMOTE_PATH", "test01")
    req_path = os.getenv("REQ_PATH", "requirements.txt")
    remote_requirements_path = f"{remote_path}/requirements.txt"
    logger.info(f"Copying requirements.txt from {req_path} to {remote_requirements_path} on remote server")
    try:
        with open(req_path, 'r') as f:
            local_requirements = f.read()
            logger.info(f"Content of {req_path}:\n{local_requirements}")
        ssh = get_ssh_client()
        sftp = ssh.open_sftp()
        sftp.put(req_path, remote_requirements_path)
        sftp.close()
        ssh.close()
        logger.info(f"Successfully copied requirements.txt to {remote_requirements_path}")
        return {
            "file_copied": True,
            "error_message": "",
            "command_outputs": [f"Copied requirements.txt from {req_path} to {remote_requirements_path}\nContent:\n{local_requirements}"]
        }
    except Exception as e:
        logger.error(f"Failed to copy requirements.txt: {str(e)}")
        return {
            "file_copied": False,
            "error_message": f"Failed to copy requirements.txt: {str(e)}",
            "command_outputs": [f"Copy requirements.txt to {remote_requirements_path}\nError: {str(e)}"]
        }

@tool
def install_requirements() -> dict:
    """Run pip install -r requirements.txt in the remote virtual environment."""
    remote_path = os.getenv("REMOTE_PATH", "test01")
    venv_path = f"{remote_path}/venv"
    requirements_path = f"{remote_path}/requirements.txt"
    logger.info(f"Running pip install -r {requirements_path} in venv at {venv_path}")
    try:
        ssh = get_ssh_client()
        command = f"source {venv_path}/bin/activate && pip install -r {requirements_path} --verbose"
        stdin, stdout, stderr = ssh.exec_command(command)
        stderr_output = stderr.read().decode()
        stdout_output = stdout.read().decode()
        ssh.close()
        logger.info(f"pip install stdout:\n{stdout_output}")
        logger.info(f"pip install stderr:\n{stderr_output}")
        if "ERROR: Cannot install" in stderr_output and "because these package versions have conflicting dependencies" in stderr_output:
            logger.error(f"Dependency conflict in pip install: {stderr_output}")
            return {
                "pip_installed": False,
                "error_message": f"Dependency conflict: {stderr_output}",
                "command_outputs": [f"pip install -r {requirements_path}\nStdout: {stdout_output}\nError: {stderr_output}"]
            }
        if stderr_output and "ERROR" in stderr_output:
            logger.error(f"Failed to install requirements: {stderr_output}")
            return {
                "pip_installed": False,
                "error_message": f"Failed to install requirements: {stderr_output}",
                "command_outputs": [f"pip install -r {requirements_path}\nStdout: {stdout_output}\nError: {stderr_output}"]
            }
        logger.info(f"Successfully installed requirements from {requirements_path}")
        return {
            "pip_installed": True,
            "error_message": "",
            "command_outputs": [f"pip install -r {requirements_path}\nStdout: {stdout_output}"]
        }
    except Exception as e:
        logger.error(f"Failed to run pip install: {str(e)}")
        return {
            "pip_installed": False,
            "error_message": f"Failed to run pip install: {str(e)}",
            "command_outputs": [f"pip install -r {requirements_path}\nError: {str(e)}"]
        }

@tool
def resolve_conflicts(error_message: str) -> dict:
    """Use LLM to resolve dependency conflicts by generating a new requirements.txt."""
    remote_path = os.getenv("REMOTE_PATH", "test01")
    req_path = os.getenv("REQ_PATH", "requirements.txt")
    requirements_path = f"{remote_path}/requirements.txt"
    logger.info(f"Resolving dependency conflicts for {requirements_path}")
    try:
        # Read current requirements.txt from remote server
        ssh = get_ssh_client()
        sftp = ssh.open_sftp()
        with sftp.file(requirements_path, 'r') as f:
            current_requirements = f.read().decode()
        sftp.close()
        logger.info(f"Current remote requirements.txt:\n{current_requirements}")
        ssh.close()

        # Read local requirements.txt for comparison
        try:
            with open(req_path, 'r') as f:
                local_requirements = f.read()
            logger.info(f"Local requirements.txt at {req_path}:\n{local_requirements}")
        except FileNotFoundError:
            logger.warning(f"Local requirements.txt at {req_path} not found, using remote content")
            local_requirements = current_requirements

        # Prepare prompt for LLM
        prompt = f"""
        You are an expert in resolving Python package dependency conflicts. The following `pip install -r requirements.txt` command failed with this error:

        {error_message}

        The current `requirements.txt` contains:
        ```
        {current_requirements}
        ```

        Analyze the error and generate a new `requirements.txt` with compatible package versions to resolve the conflict. Include all packages from the original list (langchain-core, langchain-openai, paramiko, pydantic) with mutually compatible versions. Use the latest compatible versions available as of July 2025 unless specific versions are required to resolve the conflict.

        **Important**: Return only the new `requirements.txt` content as plain text, with one package per line in the format `package==version`. For example:
        langchain-core==0.3.68
        langchain-openai==0.3.27
        paramiko==3.5.1
        pydantic==2.11.7

        Do not include any explanatory text, comments, Markdown formatting (such as triple backticks ```), or any other content. If no resolution is possible, return an empty string.
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        new_requirements = response.content.strip()
        # Clean LLM output to remove triple backticks and extra whitespace
        new_requirements = re.sub(r'^```.*?\n|```$', '', new_requirements, flags=re.MULTILINE).strip()
        logger.info(f"LLM generated requirements.txt (after cleaning):\n{new_requirements}")

        # Validate the LLM output
        valid_format = True
        if not new_requirements:
            valid_format = False
            logger.error("LLM returned empty requirements.txt")
        else:
            # Check if each line matches the expected package==version format
            lines = new_requirements.splitlines()
            for line in lines:
                if not re.match(r"^[a-zA-Z0-9_-]+==[0-9]+\.[0-9]+\.[0-9]+(?:\.[0-9]+|(?:\.post[0-9]+)?)?$", line.strip()):
                    valid_format = False
                    logger.error(f"Invalid line in LLM output: {line}")
                    break

        if not valid_format:
            logger.error("LLM failed to provide a valid requirements.txt format")
            return {
                "file_copied": False,
                "error_message": "LLM failed to provide a valid requirements.txt format",
                "command_outputs": [f"Resolve conflicts for {requirements_path}\nLLM output (after cleaning):\n{new_requirements}\nError: Invalid requirements.txt format"]
            }

        # Write new requirements.txt locally and upload to remote server
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            temp_file.write(new_requirements)
            temp_file_path = temp_file.name
        ssh = get_ssh_client()
        sftp = ssh.open_sftp()
        sftp.put(temp_file_path, requirements_path)
        sftp.close()
        ssh.close()
        os.unlink(temp_file_path)
        logger.info(f"Uploaded new requirements.txt to {requirements_path}")
        return {
            "file_copied": True,
            "error_message": "",
            "command_outputs": [f"Generated and uploaded new requirements.txt:\n{new_requirements}"]
        }
    except Exception as e:
        logger.error(f"Failed to resolve conflicts: {str(e)}")
        return {
            "file_copied": False,
            "error_message": f"Failed to resolve conflicts: {str(e)}",
            "command_outputs": [f"Resolve conflicts for {requirements_path}\nError: {str(e)}"]
        }

# Define nodes
def ssh_login_node(state: AgentState):
    logger.info("Entering ssh_login_node")
    credentials = {
        "hostname": os.getenv("SSH_HOST", "localhost"),
        "username": os.getenv("SSH_USERNAME", ""),
        "password": os.getenv("SSH_PASSWORD", ""),
        "port": int(os.getenv("SSH_PORT", 22))
    }
    result = perform_login.invoke({"credentials": credentials})
    state["messages"].append(HumanMessage(content=f"SSH login result: {result}"))
    state.update(result)
    state["command_outputs"].extend(result["command_outputs"])
    return state

def create_folder_node(state: AgentState):
    logger.info("Entering create_folder_node")
    result = create_folder.invoke({})
    state["messages"].append(HumanMessage(content=f"Create folder result: {result}"))
    state.update(result)
    state["command_outputs"].extend(result["command_outputs"])
    return state

def create_venv_node(state: AgentState):
    logger.info("Entering create_venv_node")
    result = create_venv.invoke({})
    state["messages"].append(HumanMessage(content=f"Venv creation result: {result}"))
    state.update(result)
    state["command_outputs"].extend(result["command_outputs"])
    return state

def copy_requirements_node(state: AgentState):
    logger.info("Entering copy_requirements_node")
    result = copy_requirements.invoke({})
    state["messages"].append(HumanMessage(content=f"Copy requirements result: {result}"))
    state.update(result)
    state["command_outputs"].extend(result["command_outputs"])
    return state

def install_requirements_node(state: AgentState):
    logger.info("Entering install_requirements_node")
    state["pip_retry_count"] = state.get("pip_retry_count", 0) + 1
    if state["pip_retry_count"] > 5:
        logger.error("Max retries (5) reached for pip install")
        state["error_message"] = "Max retries (5) reached for pip install"
        state["command_outputs"].append("Max retries (5) reached for pip install")
        return state
    result = install_requirements.invoke({})
    state["messages"].append(HumanMessage(content=f"Pip install result: {result}"))
    state.update(result)
    state["command_outputs"].extend(result["command_outputs"])
    return state

def resolve_conflicts_node(state: AgentState):
    logger.info("Entering resolve_conflicts_node")
    result = resolve_conflicts.invoke({"error_message": state["error_message"]})
    state["messages"].append(HumanMessage(content=f"Resolve conflicts result: {result}"))
    state.update(result)
    state["command_outputs"].extend(result["command_outputs"])
    return state

# Define routing
def route_flow(state: AgentState) -> str:
    logger.info(f"Routing state: setup_success={state.get('setup_success', False)}, "
                f"folder_created={state.get('folder_created', False)}, "
                f"venv_created={state.get('venv_created', False)}, "
                f"file_copied={state.get('file_copied', False)}, "
                f"pip_installed={state.get('pip_installed', False)}, "
                f"pip_retry_count={state.get('pip_retry_count', 0)}")
    if state.get("pip_retry_count", 0) > 5:
        logger.error("Exiting due to max retry limit exceeded")
        return "end"
    if state.get("error_message", ""):
        if "Dependency conflict" in state["error_message"]:
            return "resolve_conflicts"
        if "Invalid requirement" in state["error_message"] or "Failed to install requirements" in state["error_message"]:
            return "end"  # Stop if LLM output is invalid or other pip errors persist
        return "install_requirements"  # Retry for other pip errors
    if not state.get("setup_success", False):
        return "ssh_login"
    if not state.get("folder_created", False):
        return "create_folder"
    if not state.get("venv_created", False):
        return "create_venv"
    if not state.get("file_copied", False):
        return "copy_requirements"
    if not state.get("pip_installed", False):
        return "install_requirements"
    return "end"

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("ssh_login", ssh_login_node)
workflow.add_node("create_folder", create_folder_node)
workflow.add_node("create_venv", create_venv_node)
workflow.add_node("copy_requirements", copy_requirements_node)
workflow.add_node("install_requirements", install_requirements_node)
workflow.add_node("resolve_conflicts", resolve_conflicts_node)
workflow.set_entry_point("ssh_login")
workflow.add_conditional_edges(
    "ssh_login",
    route_flow,
    {"ssh_login": "ssh_login", "create_folder": "create_folder", "install_requirements": "install_requirements", "end": END}
)
workflow.add_conditional_edges(
    "create_folder",
    route_flow,
    {"create_folder": "create_folder", "create_venv": "create_venv", "install_requirements": "install_requirements", "end": END}
)
workflow.add_conditional_edges(
    "create_venv",
    route_flow,
    {"create_venv": "create_venv", "copy_requirements": "copy_requirements", "install_requirements": "install_requirements", "end": END}
)
workflow.add_conditional_edges(
    "copy_requirements",
    route_flow,
    {"copy_requirements": "copy_requirements", "install_requirements": "install_requirements", "end": END}
)
workflow.add_conditional_edges(
    "install_requirements",
    route_flow,
    {"install_requirements": "install_requirements", "resolve_conflicts": "resolve_conflicts", "end": END}
)
workflow.add_conditional_edges(
    "resolve_conflicts",
    route_flow,
    {"install_requirements": "install_requirements", "end": END}
)
graph = workflow.compile()

# Run workflow
initial_state = {
    "messages": [HumanMessage(content="Start setup with SSH login to sodaray.com, create folder, create venv, copy requirements, install requirements, and resolve conflicts")],
    "setup_success": False,
    "folder_created": False,
    "venv_created": False,
    "file_copied": False,
    "pip_installed": False,
    "error_message": "",
    "user_input_needed": False,
    "pip_retry_count": 0,
    "command_outputs": []
}
try:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}
    logger.info(f"Invoking graph with config: {config}")
    result = graph.invoke(initial_state, config)
    if result.get("pip_installed", False):
        logger.info("Workflow completed successfully")
        print("ok")
    else:
        logger.error(f"Workflow failed: {result['error_message']}")
        print(f"Failed: {result['error_message']}")
    print("\nFull command outputs:")
    for output in result["command_outputs"]:
        print(output)
except Exception as e:
    logger.error(f"Workflow failed: {str(e)}")
    print(f"Failed: {str(e)}")
    print("\nFull command outputs:")
    for output in result["command_outputs"]:
        print(output)
finally:
    http_client.close()
