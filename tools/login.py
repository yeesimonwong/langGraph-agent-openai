import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from .common import AgentState, logger, get_ssh_client

class Credentials(BaseModel):
    hostname: str = Field(description="Hostname of the SSH server")
    username: str = Field(description="Username for SSH login")
    password: str = Field(description="Password for SSH login")
    port: int = Field(description="Port for SSH connection")

@tool
def perform_login(credentials: Credentials) -> dict:
    """Establish an SSH connection to a remote server."""
    logger.info(f"Attempting SSH login to {credentials.username}@{credentials.hostname}:{credentials.port}")
    try:
        ssh = get_ssh_client()
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

def ssh_login_node(state: AgentState) -> AgentState:
    logger.info("Entering ssh_login_node")
    credentials = {
        "hostname": os.getenv("SSH_HOST"),
        "username": os.getenv("SSH_USERNAME"),
        "password": os.getenv("SSH_PASSWORD"),
        "port": int(os.getenv("SSH_PORT"))
    }
    result = perform_login.invoke({"credentials": credentials})
    state["messages"].append(HumanMessage(content=f"SSH login result: {result}"))
    state.update(result)
    state["command_outputs"].extend(result["command_outputs"])
    return state