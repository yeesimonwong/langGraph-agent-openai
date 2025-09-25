import os
import logging
import paramiko
import httpx
from httpx_socks import SyncProxyTransport
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from typing import Annotated, Sequence
from langchain_core.messages import HumanMessage
import operator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s] %(message)s"
)
logger = logging.getLogger(__name__)

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
    openai_api_key: str
    socks5_host: str
    socks5_port: str
    socks5_username: str
    socks5_password: str
    bypass_proxy: str

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

# Set up SOCKS5 proxy for local OpenAI API access
def setup_proxy(state: AgentState):
    bypass_proxy = state.get("bypass_proxy", "false").lower() == "true"
    if bypass_proxy:
        logger.info("Bypassing SOCKS5 proxy, using direct HTTP client")
        return httpx.Client(timeout=30.0)
    
    proxy_host = state.get("socks5_host")
    proxy_port = state.get("socks5_port")
    proxy_username = state.get("socks5_username")
    proxy_password = state.get("socks5_password")
    
    if not all([proxy_host, proxy_port, proxy_username, proxy_password]):
        logger.warning("SOCKS5 proxy variables missing in state, falling back to direct HTTP client. If OpenAI API calls fail, ensure internet access or set valid SOCKS5 credentials or BYPASS_PROXY=true.")
        return httpx.Client(timeout=30.0)
    
    try:
        proxy_url = f"socks5://{proxy_username}:{proxy_password}@{proxy_host}:{proxy_port}"
        proxy_transport = SyncProxyTransport.from_url(proxy_url)
        client = httpx.Client(transport=proxy_transport, timeout=10.0)
        response = client.get("https://api.ipify.org")
        response.raise_for_status()
        logger.info(f"SOCKS5 proxy credentials validated successfully: {proxy_url}")
        client.close()
        return httpx.Client(transport=proxy_transport, timeout=30.0)
    except Exception as e:
        logger.error(f"Failed to validate SOCKS5 proxy credentials: {str(e)}. Falling back to direct HTTP client. If OpenAI API calls fail, set BYPASS_PROXY=true or update SOCKS5 credentials.")
        return httpx.Client(timeout=30.0)

# Initialize LLM
def get_llm(state: AgentState):
    openai_api_key = state.get("openai_api_key")
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY is not set in the state. Skipping LLM initialization.")
        return None
    try:
        http_client = setup_proxy(state)
        logger.info("Initializing LLM with model gpt-4-turbo")
        llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
            openai_api_key=openai_api_key,
            http_client=http_client
        )
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}. Continuing without LLM for debugging.")
        return None