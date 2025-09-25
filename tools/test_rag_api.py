import os
import httpx
import traceback
import time
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from .common import AgentState, logger, get_llm

@tool
def test_rag_api(state: AgentState) -> dict:
    """Test the RAG API on the remote server by querying it from local and processing with OpenAI."""
    llm = get_llm(state)
    if not llm:
        error_msg = "LLM not initialized. OPENAI_API_KEY may be missing."
        logger.error(error_msg)
        return {
            "rag_tested": False,
            "error_message": error_msg,
            "command_outputs": [error_msg]
        }
    remote_api_url = os.getenv("RAG_API_URL")
    test_question = "What is the main topic of the PDF?"
    logger.info(f"Testing RAG API with question: {test_question}")
    
    max_api_retries = 3
    try:
        # Call remote RAG API
        response = httpx.get(remote_api_url, params={"question": test_question}, timeout=10)
        response.raise_for_status()
        rag_response = response.json().get("chunks", ["No chunks received"])
        rag_response_text = "\n".join(rag_response) if isinstance(rag_response, list) else rag_response
        
        # Attempt to process with OpenAI LLM locally
        for attempt in range(1, max_api_retries + 1):
            logger.info(f"Attempt {attempt}/{max_api_retries} to call OpenAI API")
            try:
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
                error_msg = f"OpenAI API call attempt {attempt} failed: {str(e)}\n{traceback.format_exc()}"
                logger.warning(error_msg)
                if attempt < max_api_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                logger.error(f"OpenAI API call failed after {max_api_retries} attempts: {str(e)}. Set BYPASS_PROXY=true or update SOCKS5 credentials in .env")
                return {
                    "rag_tested": False,
                    "error_message": f"RAG API test failed after {max_api_retries} attempts: {str(e)}. RAG API responded successfully, but OpenAI API call failed.",
                    "command_outputs": [f"RAG API test with question '{test_question}'\nRetrieved chunks: {rag_response_text}\nError: {str(e)}\n{traceback.format_exc()}\nNote: RAG API responded, but OpenAI API call failed. Try setting BYPASS_PROXY=true or updating SOCKS5 credentials."]
                }
    
    except Exception as e:
        logger.error(f"RAG API request failed: {str(e)}\n{traceback.format_exc()}")
        return {
            "rag_tested": False,
            "error_message": f"RAG API request failed: {str(e)}",
            "command_outputs": [f"RAG API test with question '{test_question}'\nError: {str(e)}\n{traceback.format_exc()}"]
        }

def test_rag_api_node(state: AgentState) -> AgentState:
    logger.info("Entering test_rag_api_node")
    result = test_rag_api.invoke({"state": state})
    state["messages"].append(HumanMessage(content=f"RAG API test result: {result}"))
    state.update(result)
    state["command_outputs"].extend(result["command_outputs"])
    return state