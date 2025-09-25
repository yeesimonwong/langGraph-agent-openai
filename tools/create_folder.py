import os
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from .common import AgentState, logger, get_ssh_client

@tool
def create_folder() -> dict:
    """Create a folder on the remote server."""
    remote_path = os.getenv("REMOTE_PATH")
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

def create_folder_node(state: AgentState) -> AgentState:
    logger.info("Entering create_folder_node")
    result = create_folder.invoke({})
    state["messages"].append(HumanMessage(content=f"Create folder result: {result}"))
    state.update(result)
    state["command_outputs"].extend(result["command_outputs"])
    return state