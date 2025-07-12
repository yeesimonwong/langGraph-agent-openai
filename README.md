# Autonomation Target
2025 July 12 -> use LLM to automate conflict resolution retry during "PIP install -r requirements.txt"

# setup
python3.11 -m venv langGraph-agent-openai  
source ./langGraph-agent-openai/bin/activate  
pip install --upgrade pip  
pip install -r requirements.txt  

# .env
OPENAI_API_KEY=  
TAVILY_API_KEY=  
SOCKS5_USERNAME=  
SOCKS5_PASSWORD=  
SOCKS5_HOST=  
SOCKS5_PORT=  
SSH_HOST=  
SSH_PORT=  
SSH_USERNAME=  
SSH_PASSWORD=  
REMOTE_PATH=  
REQ_PATH=  

# run
python agent.py
