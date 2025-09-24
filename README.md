# Autonomation Target

2025 Jul 12 -> use LLM to automate conflict resolution retry during "PIP install -r requirements.txt"  
2025 Sep 24 -> use LLM to automate a local RAG creation and deploy the RAG as API service on a remote server - verified

# setup
python3.11 -m venv lgAuto  
source ./lgAuto/bin/activate  
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
PDF_PATH=
RAG_API_URL=
PYTHON_VERSION=

# run
python agent.py
