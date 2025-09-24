#!/usr/bin/env python3
"""
SSH RAG Agent: Deploys a retrieval-only RAG API to a remote server and tests it locally with OpenAI.
Requirements:
- Environment variables: SSH_HOST, SSH_PORT, SSH_USERNAME, SSH_PASSWORD, REMOTE_PATH, REQ_PATH, PDF_PATH, RAG_API_URL, PYTHON_VERSION, OPENAI_API_KEY, SOCKS5_HOST, SOCKS5_PORT, SOCKS5_USERNAME, SOCKS5_PASSWORD
- Files: requirements.txt (at REQ_PATH for both local and remote), PDF (at PDF_PATH)
- Remote server: Python 3 (e.g., python3), SSH server, curl, lsof, write/execute permissions for REMOTE_PATH, direct internet access for pip install
- Local machine: Internet access for OpenAI API via SOCKS5 proxy
- Remote server: Accessible via local network for SSH
"""

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
import shutil
import traceback
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables explicitly
env_path = os.path.join(os.path.dirname(__file__), ".env")
if not os.path.exists(env_path):
    logger.error(f".env file not found at {env_path}")
    raise FileNotFoundError(f".env file not found at {env_path}")
try:
    load_dotenv(env_path)
    logger.info(f"Loaded .env file from {env_path}")
except Exception as e:
    logger.error(f"Failed to load .env file from {env_path}: {str(e)}")
    raise

# Validate environment variables
required_env_vars = [
    "SSH_HOST", "SSH_PORT", "SSH_USERNAME", "SSH_PASSWORD",
    "REMOTE_PATH", "REQ_PATH", "PDF_PATH", "RAG_API_URL",
    "PYTHON_VERSION", "OPENAI_API_KEY",
    "SOCKS5_HOST", "SOCKS5_PORT", "SOCKS5_USERNAME", "SOCKS5_PASSWORD"
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

# Log loaded variables for debugging
logger.info(f"Loaded SSH_HOST={os.getenv('SSH_HOST')}, SSH_PORT={os.getenv('SSH_PORT')}, REMOTE_PATH={os.getenv('REMOTE_PATH')}, PDF_PATH={os.getenv('PDF_PATH')}, RAG_API_URL={os.getenv('RAG_API_URL')}, OPENAI_API_KEY=***, PYTHON_VERSION={os.getenv('PYTHON_VERSION')}")

# Set up SOCKS5 proxy for local OpenAI API access
def setup_proxy():
    try:
        proxy_host = os.getenv("SOCKS5_HOST")
        proxy_port = os.getenv("SOCKS5_PORT")
        proxy_username = os.getenv("SOCKS5_USERNAME")
        proxy_password = os.getenv("SOCKS5_PASSWORD")
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
    rag_created: bool
    rag_tested: bool
    error_message: str
    user_input_needed: bool
    pip_retry_count: int
    rag_retry_count: int
    sftp_retry_count: int
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
        hostname=os.getenv("SSH_HOST"),
        port=int(os.getenv("SSH_PORT")),
        username=os.getenv("SSH_USERNAME"),
        password=os.getenv("SSH_PASSWORD"),
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

@tool
def resolve_conflicts(error_message: str) -> dict:
    """Use LLM to resolve dependency conflicts by generating a new requirements.txt."""
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

@tool
def create_rag(pdf_path: str) -> dict:
    """Create a RAG system locally and deploy a FastAPI service to the remote server."""
    remote_path = os.getenv("REMOTE_PATH")
    venv_path = f"{remote_path}/venv"
    rag_path = f"{remote_path}/rag_system"
    remote_pdf_path = f"{remote_path}/{os.path.basename(pdf_path)}"
    
    logger.info(f"Creating RAG system locally and deploying API to {rag_path} from PDF: {pdf_path}")
    
    local_chroma_dir = None
    script_file_path = None
    rag_created = False
    try:
        # Check if PDF exists locally
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file {pdf_path} not found locally")
            return {
                "rag_created": False,
                "error_message": f"PDF file {pdf_path} not found locally",
                "command_outputs": [f"Create RAG from {pdf_path}\nError: PDF file not found"]
            }

        # Create RAG system locally
        logger.info("Loading PDF and creating vector store locally")
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
        except ImportError as e:
            logger.error(f"Failed to load PyPDFLoader: {str(e)}\n{traceback.format_exc()}")
            return {
                "rag_created": False,
                "error_message": f"Failed to load PyPDFLoader: {str(e)}",
                "command_outputs": [f"Create RAG from {pdf_path}\nError: {str(e)}\n{traceback.format_exc()}"]
            }
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        local_chroma_dir = tempfile.mkdtemp()
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            persist_directory=local_chroma_dir
        )
        
        # Generate FastAPI RAG script for retrieval
        rag_script_content = f'''#!/usr/bin/env python3
"""RAG API Service for PDF Document (Offline Retrieval)
Generated on {remote_path}
"""

import os
from fastapi import FastAPI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

class PDFRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        self.retriever = None
        self.setup_rag()
    
    def setup_rag(self):
        """Load precomputed vector store and setup retriever"""
        print("Loading vector store from ./chroma_db")
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={{"k": 3}})
        print("RAG retriever setup complete!")
    
    def retrieve(self, question: str):
        """Retrieve relevant document chunks"""
        if self.retriever:
            docs = self.retriever.get_relevant_documents(question)
            return [doc.page_content for doc in docs]
        return ["RAG retriever not initialized"]

rag = PDFRAG()

@app.get("/retrieve")
async def retrieve_chunks(question: str):
    chunks = rag.retrieve(question)
    return {{"chunks": chunks}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        # Write RAG script to temporary file
        rag_script_path = f"{rag_path}/rag_system.py"
        script_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py')
        script_file.write(rag_script_content)
        script_file_path = script_file.name
        script_file.close()

        # Connect to remote server
        ssh = get_ssh_client()
        sftp = ssh.open_sftp()

        # Upload PDF with timeout
        logger.info(f"Uploading PDF {pdf_path} to {remote_pdf_path}")
        try:
            sftp.put(pdf_path, remote_pdf_path, callback=lambda x, y: logger.info(f"Transferred {x} of {y} bytes for {remote_pdf_path}"))
        except Exception as e:
            logger.error(f"Failed to upload PDF {pdf_path}: {str(e)}\n{traceback.format_exc()}")
            sftp.close()
            ssh.close()
            shutil.rmtree(local_chroma_dir, ignore_errors=True)
            if os.path.exists(script_file_path):
                try:
                    os.unlink(script_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary script file {script_file_path}: {str(e)}")
            return {
                "rag_created": False,
                "error_message": f"Failed to upload PDF: {str(e)}",
                "command_outputs": [f"Upload PDF {pdf_path} to {remote_pdf_path}\nError: {str(e)}\n{traceback.format_exc()}"]
            }

        # Create RAG directory
        logger.info(f"Creating RAG directory {rag_path}")
        stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {rag_path}", timeout=10)
        stderr_output = stderr.read().decode()
        if stderr_output:
            logger.error(f"Failed to create RAG directory {rag_path}: {stderr_output}")
            sftp.close()
            ssh.close()
            shutil.rmtree(local_chroma_dir, ignore_errors=True)
            if os.path.exists(script_file_path):
                try:
                    os.unlink(script_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary script file {script_file_path}: {str(e)}")
            return {
                "rag_created": False,
                "error_message": f"Failed to create RAG directory: {stderr_output}",
                "command_outputs": [f"mkdir -p {rag_path}\nError: {stderr_output}"]
            }

        # Upload RAG script with timeout
        logger.info(f"Uploading RAG script {script_file_path} to {rag_script_path}")
        try:
            sftp.put(script_file_path, rag_script_path, callback=lambda x, y: logger.info(f"Transferred {x} of {y} bytes for {rag_script_path}"))
        except Exception as e:
            logger.error(f"Failed to upload RAG script {script_file_path}: {str(e)}\n{traceback.format_exc()}")
            sftp.close()
            ssh.close()
            shutil.rmtree(local_chroma_dir, ignore_errors=True)
            if os.path.exists(script_file_path):
                try:
                    os.unlink(script_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary script file {script_file_path}: {str(e)}")
            return {
                "rag_created": False,
                "error_message": f"Failed to upload RAG script: {str(e)}",
                "command_outputs": [f"Upload RAG script {script_file_path} to {rag_script_path}\nError: {str(e)}\n{traceback.format_exc()}"]
            }

        # Clean up temporary script file
        if os.path.exists(script_file_path):
            try:
                os.unlink(script_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary script file {script_file_path}: {str(e)}")

        # Upload Chroma vector store with timeout and detailed logging
        def upload_directory(local_dir, remote_dir, sftp, ssh, retry_count=0):
            max_retries = 3
            if retry_count >= max_retries:
                logger.error(f"Max retries ({max_retries}) reached for uploading directory {local_dir}")
                return False, f"Max retries ({max_retries}) reached for uploading {local_dir}"
            logger.info(f"Uploading directory {local_dir} to {remote_dir} (attempt {retry_count + 1}/{max_retries})")
            try:
                for root, dirs, files in os.walk(local_dir):
                    remote_subdir = os.path.join(remote_dir, os.path.relpath(root, local_dir)).replace("\\", "/")
                    logger.info(f"Creating remote subdirectory {remote_subdir}")
                    stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_subdir}", timeout=10)
                    stderr_output = stderr.read().decode()
                    if stderr_output:
                        logger.error(f"Failed to create subdirectory {remote_subdir}: {stderr_output}")
                        return False, f"Failed to create subdirectory {remote_subdir}: {stderr_output}"
                    for file in files:
                        local_file = os.path.join(root, file)
                        remote_file = f"{remote_subdir}/{file}"
                        logger.info(f"Uploading file {local_file} to {remote_file}")
                        try:
                            sftp.put(local_file, remote_file, callback=lambda x, y: logger.info(f"Transferred {x} of {y} bytes for {remote_file}"))
                        except Exception as e:
                            logger.error(f"Failed to upload file {local_file}: {str(e)}\n{traceback.format_exc()}")
                            return False, f"Failed to upload file {local_file}: {str(e)}"
                return True, ""
            except Exception as e:
                logger.error(f"Failed to upload directory {local_dir}: {str(e)}\n{traceback.format_exc()}")
                return False, f"Failed to upload directory {local_dir}: {str(e)}"

        success, error_message = upload_directory(local_chroma_dir, f"{rag_path}/chroma_db", sftp, ssh)
        sftp.close()
        if not success:
            ssh.close()
            shutil.rmtree(local_chroma_dir, ignore_errors=True)
            return {
                "rag_created": False,
                "error_message": error_message,
                "command_outputs": [f"Upload directory {local_chroma_dir} to {rag_path}/chroma_db\nError: {error_message}"]
            }

        # Set execute permissions for the script
        logger.info(f"Setting execute permissions for {rag_script_path}")
        stdin, stdout, stderr = ssh.exec_command(f"chmod +x {rag_script_path}", timeout=10)
        stderr_output = stderr.read().decode()
        if stderr_output:
            logger.error(f"Failed to set permissions for {rag_script_path}: {stderr_output}")
            ssh.close()
            shutil.rmtree(local_chroma_dir, ignore_errors=True)
            return {
                "rag_created": False,
                "error_message": f"Failed to set permissions: {stderr_output}",
                "command_outputs": [f"chmod +x {rag_script_path}\nError: {stderr_output}"]
            }

        # All setup steps completed successfully, mark rag_created as True
        logger.info("RAG setup completed successfully, marking rag_created=True")
        rag_created = True

        # Free port 8000 if in use
        logger.info("Checking and freeing port 8000 on remote server")
        free_port_command = f"lsof -i :8000 | grep LISTEN | awk '{{print $2}}' | xargs -r kill -9"
        stdin, stdout, stderr = ssh.exec_command(free_port_command, timeout=10)
        stderr_output = stderr.read().decode()
        stdout_output = stdout.read().decode()
        ssh.close()
        if stderr_output:
            logger.warning(f"Warning while freeing port 8000: {stderr_output}")
        logger.info(f"Port 8000 free command output: {stdout_output}")

        # Start RAG API with retries
        logger.info(f"Starting RAG API at {rag_path}")
        max_test_retries = 3
        test_output = ""
        test_error = ""
        for attempt in range(1, max_test_retries + 1):
            logger.info(f"Attempt {attempt}/{max_test_retries} to start and test RAG API")
            # Start the FastAPI server
            start_command = f"cd {rag_path} && source {venv_path}/bin/activate && nohup python rag_system.py > /dev/null 2>&1 &"
            try:
                ssh_start = get_ssh_client()
                channel = ssh_start.get_transport().open_session()
                channel.settimeout(10)
                channel.exec_command(start_command)
                time.sleep(1)  # Brief wait to ensure command is sent
                channel.close()
                ssh_start.close()
            except Exception as e:
                logger.warning(f"RAG API start attempt {attempt} failed: {str(e)}\n{traceback.format_exc()}")
                if attempt < max_test_retries:
                    # Free port again before retrying
                    try:
                        ssh_retry = get_ssh_client()
                        stdin, stdout, stderr = ssh_retry.exec_command(free_port_command, timeout=10)
                        stderr_output = stderr.read().decode()
                        ssh_retry.close()
                        if stderr_output:
                            logger.warning(f"Warning while freeing port 8000 before retry: {stderr_output}")
                    except Exception as e:
                        logger.warning(f"Failed to free port 8000 before retry: {str(e)}")
                continue

            # Wait to ensure server starts
            time.sleep(10)  # Wait 10 seconds locally

            # Test the API with curl
            test_command = f"curl http://localhost:8000/retrieve?question=Test"
            try:
                ssh_test = get_ssh_client()
                stdin, stdout, stderr = ssh_test.exec_command(test_command, timeout=30)
                test_output = stdout.read().decode()
                test_error = stderr.read().decode()
                ssh_test.close()
                if test_error and "error" in test_error.lower():
                    logger.warning(f"RAG API test attempt {attempt} failed: {test_error}")
                    if attempt < max_test_retries:
                        # Free port again before retrying
                        try:
                            ssh_retry = get_ssh_client()
                            stdin, stdout, stderr = ssh_retry.exec_command(free_port_command, timeout=10)
                            stderr_output = stderr.read().decode()
                            ssh_retry.close()
                            if stderr_output:
                                logger.warning(f"Warning while freeing port 8000 before retry: {stderr_output}")
                        except Exception as e:
                            logger.warning(f"Failed to free port 8000 before retry: {str(e)}")
                    continue
                else:
                    break
            except Exception as e:
                logger.warning(f"RAG API test attempt {attempt} failed: {str(e)}\n{traceback.format_exc()}")
                if attempt < max_test_retries:
                    # Free port again before retrying
                    try:
                        ssh_retry = get_ssh_client()
                        stdin, stdout, stderr = ssh_retry.exec_command(free_port_command, timeout=10)
                        stderr_output = stderr.read().decode()
                        ssh_retry.close()
                        if stderr_output:
                            logger.warning(f"Warning while freeing port 8000 before retry: {stderr_output}")
                    except Exception as e:
                        logger.warning(f"Failed to free port 8000 before retry: {str(e)}")
                continue
        else:
            logger.warning(f"RAG API test failed after {max_test_retries} attempts: Test output: {test_error}")
            shutil.rmtree(local_chroma_dir, ignore_errors=True)
            return {
                "rag_created": rag_created,  # Preserve setup success
                "error_message": f"RAG API test failed after {max_test_retries} attempts: Test output: {test_error}",
                "command_outputs": [f"Create RAG API from {pdf_path}\nTest output: {test_error}\nNote: RAG setup completed, server may be running but test failed"]
            }

        # Clean up local Chroma directory
        shutil.rmtree(local_chroma_dir, ignore_errors=True)

        logger.info(f"RAG API created successfully at {rag_path}")
        return {
            "rag_created": rag_created,  # Preserve setup success
            "error_message": "",
            "command_outputs": [f"RAG API created at {rag_path} from {pdf_path}\nTest output: {test_output}"]
        }

    except Exception as e:
        logger.error(f"Failed to create RAG API: {str(e)}\n{traceback.format_exc()}")
        if local_chroma_dir and os.path.exists(local_chroma_dir):
            shutil.rmtree(local_chroma_dir, ignore_errors=True)
        if script_file_path and os.path.exists(script_file_path):
            try:
                os.unlink(script_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary script file {script_file_path}: {str(e)}")
        return {
            "rag_created": rag_created,  # Preserve setup success if set
            "error_message": f"Failed to create RAG API: {str(e)}",
            "command_outputs": [f"Create RAG from {pdf_path}\nError: {str(e)}\n{traceback.format_exc()}"]
        }

@tool
def test_rag_api() -> dict:
    """Test the RAG API on the remote server by querying it from local and processing with OpenAI."""
    remote_api_url = os.getenv("RAG_API_URL")
    test_question = "What is the main topic of the PDF?"
    logger.info(f"Testing RAG API with question: {test_question}")
    
    try:
        # Call remote RAG API
        response = httpx.get(remote_api_url, params={"question": test_question}, timeout=10)
        response.raise_for_status()
        rag_response = response.json().get("chunks", ["No chunks received"])
        rag_response_text = "\n".join(rag_response) if isinstance(rag_response, list) else rag_response
        
        # Process with OpenAI LLM locally
        prompt = f"""
        You are an expert assistant. Based on the following retrieved context from a RAG system, answer the original question concisely and accurately.

        Original question: {test_question}
        Retrieved context: {rag_response_text}

        Answer:
        """
        final_response = llm.invoke([HumanMessage(content=prompt)])
        answer = final_response.content.strip()
        
        logger.info(f"RAG API test successful. Answer: {answer}")
        return {
            "rag_tested": True,
            "error_message": "",
            "command_outputs": [f"RAG API test with question '{test_question}'\nRetrieved chunks: {rag_response_text}\nFinal answer: {answer}"]
        }
    
    except Exception as e:
        logger.error(f"RAG API test failed: {str(e)}\n{traceback.format_exc()}")
        return {
            "rag_tested": False,
            "error_message": f"RAG API test failed: {str(e)}",
            "command_outputs": [f"RAG API test with question '{test_question}'\nError: {str(e)}\n{traceback.format_exc()}"]
        }

# Define nodes
def ssh_login_node(state: AgentState):
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

def create_rag_node(state: AgentState):
    logger.info("Entering create_rag_node")
    state["rag_retry_count"] = state.get("rag_retry_count", 0) + 1
    state["sftp_retry_count"] = state.get("sftp_retry_count", 0)
    if state["rag_retry_count"] > 3:
        logger.error("Max retries (3) reached for RAG creation")
        state["error_message"] = "Max retries (3) reached for RAG creation"
        state["command_outputs"].append("Max retries (3) reached for RAG creation")
        return state
    result = create_rag.invoke({"pdf_path": os.getenv("PDF_PATH")})
    state["messages"].append(HumanMessage(content=f"RAG creation result: {result}"))
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

def test_rag_api_node(state: AgentState):
    logger.info("Entering test_rag_api_node")
    result = test_rag_api.invoke({})
    state["messages"].append(HumanMessage(content=f"RAG API test result: {result}"))
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
            result = "test_rag_api" if state.get("rag_created", False) else "end"
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
    "command_outputs": []
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
    logger.error(f"Workflow failed: {str(e)}\n{traceback.format_exc()}")
    print(f"Failed: {str(e)}")
    print("\nFull command outputs:")
    for output in result.get("command_outputs", []):
        print(output)
finally:
    if 'http_client' in locals():
        http_client.close()
