import os
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from .common import AgentState, logger, get_ssh_client

@tool
def create_venv() -> dict:
    """Create a virtual environment on the remote server."""
    remote_path = os.getenv("REMOTE_PATH")
    venv_path = f"{remote_path}/venv"
    logger.info(f"Creating venv at {venv_path} on remote server")
    try:
        ssh = get_ssh_client()
        python_cmd = os.getenv("PYTHON_VERSION")
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

def create_venv_node(state: AgentState) -> AgentState:
    logger.info("Entering create_venv_node")
    result = create_venv.invoke({})
    state["messages"].append(HumanMessage(content=f"Venv creation result: {result}"))
    state.update(result)
    state["command_outputs"].extend(result["command_outputs"])
    return state