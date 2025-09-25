import os
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from .common import AgentState, logger, get_ssh_client

@tool
def copy_requirements() -> dict:
    """Copy requirements.txt to the remote server."""
    remote_path = os.getenv("REMOTE_PATH")
    req_path = os.getenv("REQ_PATH")
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

def copy_requirements_node(state: AgentState) -> AgentState:
    logger.info("Entering copy_requirements_node")
    result = copy_requirements.invoke({})
    state["messages"].append(HumanMessage(content=f"Copy requirements result: {result}"))
    state.update(result)
    state["command_outputs"].extend(result["command_outputs"])
    return state