import os
import tempfile
import re
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from .common import AgentState, logger, get_ssh_client, get_llm

@tool
def resolve_conflicts(state: AgentState) -> dict:
    """Use LLM to resolve dependency conflicts by generating a new requirements.txt."""
    llm = get_llm(state)
    if not llm:
        error_msg = "LLM not initialized. OPENAI_API_KEY may be missing."
        logger.error(error_msg)
        return {
            "file_copied": False,
            "error_message": error_msg,
            "command_outputs": [error_msg]
        }
    error_message = state["error_message"]
    remote_path = os.getenv("REMOTE_PATH")
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

        # Prepare prompt for LLM
        prompt = f"""
        You are an expert in resolving Python package dependency conflicts. The following `pip install -r requirements.txt` command failed with this error:

        {error_message}

        The current `requirements.txt` contains:
        ```
        {current_requirements}
        ```

        Analyze the error and generate a new `requirements.txt` with compatible package versions to resolve the conflict. Include only the following packages required for a retrieval-only RAG API:
        - langchain-community==0.3.0
        - langchain-chroma==0.1.4
        - langchain-huggingface==0.0.3
        - fastapi==0.115.0
        - uvicorn==0.30.0

        Return only the new `requirements.txt` content as plain text, with one package per line in the format `package==version`. For example:
        langchain-community==0.3.0
        langchain-chroma==0.1.4
        langchain-huggingface==0.0.3
        fastapi==0.115.0
        uvicorn==0.30.0

        Do not include any explanatory text, comments, Markdown formatting, or other packages. If no resolution is possible, return an empty string.
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        new_requirements = response.content.strip()
        new_requirements = re.sub(r'^```.*?\n|```$', '', new_requirements, flags=re.MULTILINE).strip()
        logger.info(f"LLM generated requirements.txt (after cleaning):\n{new_requirements}")

        # Validate the LLM output
        valid_format = True
        if not new_requirements:
            valid_format = False
            logger.error("LLM returned empty requirements.txt")
        else:
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
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary requirements file {temp_file_path}: {str(e)}")
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

def resolve_conflicts_node(state: AgentState) -> AgentState:
    logger.info("Entering resolve_conflicts_node")
    result = resolve_conflicts.invoke({"state": state})
    state["messages"].append(HumanMessage(content=f"Resolve conflicts result: {result}"))
    state.update(result)
    state["command_outputs"].extend(result["command_outputs"])
    return state
