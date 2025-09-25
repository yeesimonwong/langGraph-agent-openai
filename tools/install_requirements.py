import os
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from .common import AgentState, logger, get_ssh_client

@tool
def install_requirements() -> dict:
    """Run pip install for requirements.txt in the remote virtual environment."""
    remote_path = os.getenv("REMOTE_PATH")
    venv_path = f"{remote_path}/venv"
    requirements_path = f"{remote_path}/requirements.txt"
    logger.info(f"Running pip install -r {requirements_path} in venv at {venv_path}")
    
    try:
        ssh = get_ssh_client()
        command = f"source {venv_path}/bin/activate && pip install -r {requirements_path} --verbose"
        stdin, stdout, stderr = ssh.exec_command(command, timeout=300)  # 5-minute timeout
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
    except paramiko.ssh_exception.SSHException as e:
        logger.error(f"SSH timeout or error during pip install: {str(e)}")
        return {
            "pip_installed": False,
            "error_message": f"SSH timeout or error during pip install: {str(e)}",
            "command_outputs": [f"pip install -r {requirements_path}\nError: {str(e)}"]
        }
    except Exception as e:
        logger.error(f"Failed to run pip install: {str(e)}")
        return {
            "pip_installed": False,
            "error_message": f"Failed to run pip install: {str(e)}",
            "command_outputs": [f"pip install -r {requirements_path}\nError: {str(e)}"]
        }

def install_requirements_node(state: AgentState) -> AgentState:
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