#!/usr/bin/env python3
"""
SSH RAG Agent: Deploys a retrieval-only RAG API to a remote server and tests it locally with OpenAI.
Requirements:
- Environment variables: SSH_HOST, SSH_PORT, SSH_USERNAME, SSH_PASSWORD, REMOTE_PATH, REQ_PATH, PDF_PATH, RAG_API_URL, PYTHON_VERSION, OPENAI_API_KEY, SOCKS5_HOST, SOCKS5_PORT, SOCKS5_USERNAME, SOCKS5_PASSWORD
- Files: requirements.txt (at REQ_PATH for both local and remote), PDF (at PDF_PATH)
- Remote server: Python 3 (e.g., python3), SSH server, curl, lsof, write/execute permissions for REMOTE_PATH, direct internet access for pip install
- Local machine: Internet access for OpenAI API via SOCKS5 proxy or direct
- Remote server: Accessible via local network for SSH
"""

import os
import stat
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from tools.common import AgentState, logger
from tools.login import perform_login, ssh_login_node
from tools.create_folder import create_folder, create_folder_node
from tools.create_venv import create_venv, create_venv_node
from tools.copy_requirements import copy_requirements, copy_requirements_node
from tools.install_requirements import install_requirements, install_requirements_node
from tools.resolve_conflicts import resolve_conflicts, resolve_conflicts_node
from tools.create_rag import create_rag, create_rag_node
from tools.test_rag_api import test_rag_api, test_rag_api_node
import uuid

# Set up logging before loading .env
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables explicitly
env_path = os.path.join(os.path.dirname(__file__), ".env")
logger.info(f"Attempting to load .env file from {env_path}")

# Check if .env file exists
if not os.path.exists(env_path):
    logger.error(f".env file not found at {env_path}. Please create it with required variables.")
    raise FileNotFoundError(f".env file not found at {env_path}. Create it with SSH_HOST, SSH_PORT, SSH_USERNAME, SSH_PASSWORD, REMOTE_PATH, REQ_PATH, PDF_PATH, RAG_API_URL, PYTHON_VERSION, OPENAI_API_KEY.")

# Check .env file permissions
try:
    file_stat = os.stat(env_path)
    if not (file_stat.st_mode & (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)):
        logger.error(f".env file at {env_path} is not readable. Fix permissions with: chmod 600 {env_path}")
        raise PermissionError(f".env file at {env_path} is not readable.")
except PermissionError as e:
    logger.error(str(e))
    raise

# Read .env file to check for syntax errors and log contents
try:
    with open(env_path, 'r') as f:
        env_content = f.read().strip()
        if not env_content:
            logger.error(f".env file at {env_path} is empty.")
            raise ValueError(f".env file at {env_path} is empty.")
        # Mask sensitive values
        masked_content = env_content
        for key in ["OPENAI_API_KEY", "SSH_PASSWORD", "SOCKS5_PASSWORD"]:
            value = os.getenv(key, "")
            if value:
                masked_content = masked_content.replace(value, "*****")
        logger.info(f".env file content (sensitive values masked):\n{masked_content}")
except Exception as e:
    logger.error(f"Failed to read .env file at {env_path}: {str(e)}. Check for syntax errors (e.g., missing '=' or invalid characters).")
    raise

# Load .env file
try:
    load_dotenv(env_path, override=True)
    logger.info(f"Loaded .env file from {env_path}")
except Exception as e:
    logger.error(f"Failed to load .env file from {env_path}: {str(e)}. Check for syntax errors (e.g., missing '=' or invalid characters).")
    raise

# Validate environment variables immediately
required_env_vars = [
    "SSH_HOST", "SSH_PORT", "SSH_USERNAME", "SSH_PASSWORD",
    "REMOTE_PATH", "REQ_PATH", "PDF_PATH", "RAG_API_URL",
    "PYTHON_VERSION", "OPENAI_API_KEY"
]
optional_env_vars = ["SOCKS5_HOST", "SOCKS5_PORT", "SOCKS5_USERNAME", "SOCKS5_PASSWORD", "BYPASS_PROXY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}. Ensure they are set in {env_path}.")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}. Ensure they are set in {env_path}.")

# Log loaded variables for debugging
logger.info(f"Loaded environment variables: SSH_HOST={os.getenv('SSH_HOST')}, SSH_PORT={os.getenv('SSH_PORT')}, REMOTE_PATH={os.getenv('REMOTE_PATH')}, PDF_PATH={os.getenv('PDF_PATH')}, RAG_API_URL={os.getenv('RAG_API_URL')}, OPENAI_API_KEY=***, PYTHON_VERSION={os.getenv('PYTHON_VERSION')}, BYPASS_PROXY={os.getenv('BYPASS_PROXY', 'false')}")

# Define routing
def route_flow(state: AgentState) -> str:
    logger.info(f"Routing state: setup_success={state.get('setup_success', False)}, "
                f"folder_created={state.get('folder_created', False)}, "
                f"venv_created={state.get('venv_created', False)}, "
                f"file_copied={state.get('file_copied', False)}, "
                f"pip_installed={state.get('pip_installed', False)}, "
                f"rag_created={state.get('rag_created', False)}, "
                f"rag_tested={state.get('rag_tested', False)}, "
                f"pip_retry_count={state.get('pip_retry_count', 0)}, "
                f"rag_retry_count={state.get('rag_retry_count', 0)}, "
                f"sftp_retry_count={state.get('sftp_retry_count', 0)}")
    result = None
    if state.get("pip_retry_count", 0) > 5:
        result = "end"
    elif state.get("rag_retry_count", 0) > 3:
        result = "end"
    elif state.get("sftp_retry_count", 0) > 3:
        result = "end"
    elif state.get("error_message", ""):
        if "Dependency conflict" in state["error_message"]:
            result = "resolve_conflicts"
        elif "Invalid requirement" in state["error_message"]:
            result = "end"
        elif "Failed to install requirements" in state["error_message"] or "failed to provide" in state["error_message"].lower():
            if not state.get("file_copied", False):
                result = "copy_requirements"
            else:
                result = "install_requirements"
        elif "pypdf" in state["error_message"].lower():
            result = "end"
        elif "upload" in state["error_message"].lower():
            state["sftp_retry_count"] = state.get("sftp_retry_count", 0) + 1
            if state["sftp_retry_count"] <= 3:
                result = "create_rag"
            else:
                result = "end"
        else:
            result = "test_rag_api" if state.get('rag_created', False) else "end"
    elif not state.get("setup_success", False):
        result = "ssh_login"
    elif not state.get("folder_created", False):
        result = "create_folder"
    elif not state.get("venv_created", False):
        result = "create_venv"
    elif not state.get("file_copied", False):
        result = "copy_requirements"
    elif not state.get("pip_installed", False):
        result = "install_requirements"
    elif not state.get("rag_created", False):
        result = "create_rag"
    elif not state.get("rag_tested", False):
        result = "test_rag_api"
    else:
        result = "end"
    logger.info(f"route_flow returning: {result}")
    return result

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("ssh_login", ssh_login_node)
workflow.add_node("create_folder", create_folder_node)
workflow.add_node("create_venv", create_venv_node)
workflow.add_node("copy_requirements", copy_requirements_node)
workflow.add_node("install_requirements", install_requirements_node)
workflow.add_node("create_rag", create_rag_node)
workflow.add_node("resolve_conflicts", resolve_conflicts_node)
workflow.add_node("test_rag_api", test_rag_api_node)

workflow.set_entry_point("ssh_login")

# Define conditional edges
workflow.add_conditional_edges(
    "ssh_login",
    route_flow,
    {
        "ssh_login": "ssh_login",
        "create_folder": "create_folder",
        "create_venv": "create_venv",
        "copy_requirements": "copy_requirements",
        "install_requirements": "install_requirements",
        "create_rag": "create_rag",
        "test_rag_api": "test_rag_api",
        "end": END
    }
)
workflow.add_conditional_edges(
    "create_folder",
    route_flow,
    {
        "create_folder": "create_folder",
        "create_venv": "create_venv",
        "copy_requirements": "copy_requirements",
        "install_requirements": "install_requirements",
        "create_rag": "create_rag",
        "test_rag_api": "test_rag_api",
        "end": END
    }
)
workflow.add_conditional_edges(
    "create_venv",
    route_flow,
    {
        "create_venv": "create_venv",
        "copy_requirements": "copy_requirements",
        "install_requirements": "install_requirements",
        "create_rag": "create_rag",
        "test_rag_api": "test_rag_api",
        "end": END
    }
)
workflow.add_conditional_edges(
    "copy_requirements",
    route_flow,
    {
        "copy_requirements": "copy_requirements",
        "install_requirements": "install_requirements",
        "create_rag": "create_rag",
        "test_rag_api": "test_rag_api",
        "end": END
    }
)
workflow.add_conditional_edges(
    "install_requirements",
    route_flow,
    {
        "install_requirements": "install_requirements",
        "resolve_conflicts": "resolve_conflicts",
        "create_rag": "create_rag",
        "test_rag_api": "test_rag_api",
        "end": END
    }
)
workflow.add_conditional_edges(
    "resolve_conflicts",
    route_flow,
    {
        "install_requirements": "install_requirements",
        "copy_requirements": "copy_requirements",
        "create_rag": "create_rag",
        "test_rag_api": "test_rag_api",
        "end": END
    }
)
workflow.add_conditional_edges(
    "create_rag",
    route_flow,
    {
        "create_rag": "create_rag",
        "test_rag_api": "test_rag_api",
        "end": END
    }
)
workflow.add_conditional_edges(
    "test_rag_api",
    route_flow,
    {
        "test_rag_api": "test_rag_api",
        "end": END
    }
)

graph = workflow.compile()

# Run workflow
initial_state = {
    "messages": [HumanMessage(content="Start setup with SSH login, create folder, create venv, copy requirements, install requirements, create and test RAG API")],
    "setup_success": False,
    "folder_created": False,
    "venv_created": False,
    "file_copied": False,
    "pip_installed": False,
    "rag_created": False,
    "rag_tested": False,
    "error_message": "",
    "user_input_needed": False,
    "pip_retry_count": 0,
    "rag_retry_count": 0,
    "sftp_retry_count": 0,
    "command_outputs": [],
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "socks5_host": os.getenv("SOCKS5_HOST"),
    "socks5_port": os.getenv("SOCKS5_PORT"),
    "socks5_username": os.getenv("SOCKS5_USERNAME"),
    "socks5_password": os.getenv("SOCKS5_PASSWORD"),
    "bypass_proxy": os.getenv("BYPASS_PROXY", "false")
}

try:
    result = {}  # Initialize result to avoid NameError
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}
    logger.info(f"Invoking graph with config: {config}")
    result = graph.invoke(initial_state, config)
    
    if result.get("rag_tested", False):
        logger.info("Workflow completed successfully with RAG API tested")
        print("RAG API created and tested successfully")
    elif result.get("rag_created", False):
        logger.info("Workflow completed with RAG API created, testing pending")
        print("RAG API created, testing can be run separately")
    elif result.get("pip_installed", False):
        logger.info("Workflow completed with packages installed, RAG creation pending")
        print("Packages installed, RAG creation can be run separately")
    else:
        logger.error(f"Workflow failed: {result.get('error_message', 'Unknown error')}")
        print(f"Failed: {result.get('error_message', 'Unknown error')}")
    
    print("\nFull command outputs:")
    for output in result.get("command_outputs", []):
        print(output)
        
except Exception as e:
    logger.error(f"Workflow failed: {str(e)}")
    print(f"Failed: {str(e)}")
    print("\nFull command outputs:")
    for output in result.get("command_outputs", []):
        print(output)
finally:
    if 'http_client' in locals():
        http_client.close()