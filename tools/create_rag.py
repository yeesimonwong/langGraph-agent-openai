import os
import paramiko
from typing import Dict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from .common import AgentState, logger, get_ssh_client

@tool
def create_rag(state: AgentState) -> Dict:
    """
    Create RAG system, upload to remote server, and start FastAPI service.
    Updates state with rag_created=True on success.
    """
    logger.info("Starting create_rag tool")
    try:
        # Retrieve environment variables
        remote_path = os.getenv("REMOTE_PATH")
        pdf_path = os.getenv("PDF_PATH")
        python_version = os.getenv("PYTHON_VERSION")
        rag_api_url = os.getenv("RAG_API_URL")
        
        if not all([remote_path, pdf_path, python_version, rag_api_url]):
            error_msg = "Missing required environment variables for create_rag"
            logger.error(error_msg)
            return {
                "rag_created": False,
                "error_message": error_msg,
                "command_outputs": state.get("command_outputs", []) + [error_msg]
            }

        # Define paths
        venv_path = os.path.join(remote_path, "venv")
        rag_system_path = os.path.join(remote_path, "rag_system")
        remote_pdf_path = os.path.join(rag_system_path, "input.pdf")
        rag_script_path = os.path.join(rag_system_path, "rag_api.py")

        # Create RAG script content
        rag_script = """
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
import uvicorn

app = FastAPI()

class Query(BaseModel):
    question: str

# Load and process PDF
loader = PyPDFLoader("input.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define prompt
prompt_template = \"""
Use the following pieces of context to answer the question. If you don't know the answer, say so.
{context}

Question: {question}
Answer: \"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create chain
def retrieve_and_answer(question: str) -> str:
    docs = vectorstore.similarity_search(question, k=3)
    context = "\\n".join([doc.page_content for doc in docs])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

@app.post("/retrieve")
async def retrieve(query: Query):
    answer = retrieve_and_answer(query.question)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        # Initialize SSH client
        ssh = get_ssh_client()
        sftp = ssh.open_sftp()

        try:
            # Create rag_system directory
            ssh.exec_command(f"mkdir -p {rag_system_path}")
            logger.info(f"Created directory {rag_system_path} on remote server")

            # Upload PDF
            try:
                sftp.put(pdf_path, remote_pdf_path)
                logger.info(f"Uploaded PDF from {pdf_path} to {remote_pdf_path}")
            except Exception as e:
                error_msg = f"Failed to upload PDF: {str(e)}"
                logger.error(error_msg)
                sftp.close()
                ssh.close()
                return {
                    "rag_created": False,
                    "error_message": error_msg,
                    "command_outputs": state.get("command_outputs", []) + [error_msg]
                }

            # Write RAG script
            try:
                with sftp.file(rag_script_path, 'w') as f:
                    f.write(rag_script)
                logger.info(f"Created RAG script at {rag_script_path}")
            except Exception as e:
                error_msg = f"Failed to create RAG script: {str(e)}"
                logger.error(error_msg)
                sftp.close()
                ssh.close()
                return {
                    "rag_created": False,
                    "error_message": error_msg,
                    "command_outputs": state.get("command_outputs", []) + [error_msg]
                }

            # Start FastAPI service
            venv_python = os.path.join(venv_path, "bin", python_version)
            start_cmd = (
                f"cd {rag_system_path} && "
                f"export OPENAI_API_KEY={os.getenv('OPENAI_API_KEY')} && "
                f"{venv_python} {rag_script_path} > rag_api.log 2>&1 &"
            )
            stdin, stdout, stderr = ssh.exec_command(start_cmd)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                error_output = stderr.read().decode()
                error_msg = f"Failed to start FastAPI service: {error_output}"
                logger.error(error_msg)
                sftp.close()
                ssh.close()
                return {
                    "rag_created": False,
                    "error_message": error_msg,
                    "command_outputs": state.get("command_outputs", []) + [error_msg]
                }

            logger.info("FastAPI service started successfully")
            sftp.close()
            ssh.close()
            return {
                "rag_created": True,
                "error_message": "",
                "command_outputs": state.get("command_outputs", []) + ["RAG API created and started"]
            }

        except Exception as e:
            error_msg = f"Error in create_rag: {str(e)}"
            logger.error(error_msg)
            sftp.close()
            ssh.close()
            return {
                "rag_created": False,
                "error_message": error_msg,
                "command_outputs": state.get("command_outputs", []) + [error_msg]
            }

    except Exception as e:
        error_msg = f"Failed to initialize SSH client: {str(e)}"
        logger.error(error_msg)
        return {
            "rag_created": False,
            "error_message": error_msg,
            "command_outputs": state.get("command_outputs", []) + [error_msg]
        }


def create_rag_node(state: AgentState) -> AgentState:
    logger.info("Entering create_rag_node")
    state["rag_retry_count"] = state.get("rag_retry_count", 0) + 1
    state["sftp_retry_count"] = state.get("sftp_retry_count", 0)
    if state["rag_retry_count"] > 3:
        logger.error("Max retries (3) reached for RAG creation")
        state["error_message"] = "Max retries (3) reached for RAG creation"
        state["command_outputs"].append("Max retries (3) reached for RAG creation")
        return state
    result = create_rag.invoke({"state": state})
    state["messages"].append(HumanMessage(content=f"RAG creation result: {result}"))
    state.update(result)
    state["command_outputs"].extend(result["command_outputs"])
    return state