import os
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import gradio as gr
from agents.query_agent import QueryAgent
from agents.summarization_agent import SummarizationAgent
from utils.setup import download_nltk_data
from utils.document_loader import initialize_documents, download_documents, extract_text_from_pdfs
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Legal Information Chatbot")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Create necessary directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Global variables for agents
query_agent = None
summarization_agent = None

def initialize_app():
    """Initialize the application and its dependencies."""
    global query_agent, summarization_agent

    try:
        # Step 1: Download required NLTK data
        logger.info("Downloading required NLTK data...")
        download_nltk_data()

        # Step 2: Download source PDF documents if necessary
        logger.info("Downloading source PDF documents if necessary...")
        download_documents(force_redownload=False)  # Ensure PDFs are present

        # Step 3: Extract text from PDFs and prepare for indexing (with new Markdown header logic)
        logger.info("Extracting text from PDFs and preparing for indexing...")
        # Use clean_existing=True to ensure PDFs are re-processed with the new inject_markdown_headers logic
        extract_text_from_pdfs(clean_existing=True)

        # Step 4: Initialize agents (was Step 3)
        logger.info("Initializing agents...")

        query_agent = QueryAgent()
        summarization_agent = SummarizationAgent()

        # Verify initialization
        if not query_agent or not query_agent.index:
            raise Exception("Query agent not properly initialized")
        if not summarization_agent or not summarization_agent.model:
            raise Exception("Summarization agent not properly initialized")

        logger.info("Application initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        return False

# Initialize the application
if not initialize_app():
    logger.error("Failed to initialize application. Please check the logs for details.")
    raise Exception("Application initialization failed")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def process_query(query: str = Form(...)):
    # Step 1: Query Agent retrieves relevant information
    relevant_info = query_agent.process(query)

    # Step 2: Summarization Agent simplifies the information
    response = summarization_agent.process(relevant_info, query)

    return {"response": response}

@app.get("/rebuild-knowledge")
async def rebuild_knowledge():
    """Endpoint to force redownload and reprocessing of documents."""
    from utils.document_loader import download_documents, extract_text_from_pdfs
    import shutil
    import os

    # Delete existing FAISS index
    faiss_dir = os.path.join("data", "faiss_index")
    if os.path.exists(faiss_dir):
        shutil.rmtree(faiss_dir)
        print(f"Deleted existing FAISS index at {faiss_dir}")

    # Redownload and reprocess documents
    download_documents(force_redownload=True)
    extracted_text = extract_text_from_pdfs(clean_existing=True)

    # Reinitialize agents
    global query_agent, summarization_agent
    query_agent = QueryAgent()
    summarization_agent = SummarizationAgent()

    return {"status": "success", "message": "Knowledge base rebuilt successfully"}

def process_query(query: str) -> str:
    """Process the user's query using the multi-agent system."""
    global query_agent, summarization_agent

    try:
        # Verify agents are available
        if not query_agent or not query_agent.index:
            logger.error("Query agent not available")
            return "I'm having trouble accessing my knowledge base. Please try again later."

        if not summarization_agent or not summarization_agent.model:
            logger.error("Summarization agent not available")
            return "I'm having trouble processing your request. Please try again later."

        # Step 1: Query Agent retrieves relevant information
        logger.info("Query Agent: Retrieving relevant information...")
        relevant_info = query_agent.process(query)

        if not relevant_info:
            return "I couldn't find specific information about that in my knowledge base. Please try asking about:\n" + \
                   "- Steps to file a lawsuit in India\n" + \
                   "- Documents required for filing a case\n" + \
                   "- ICAI guidelines for chartered accountants\n" + \
                   "- General litigation process in India"

        # Step 2: Summarization Agent simplifies the information
        logger.info("Summarization Agent: Processing and simplifying information...")
        response = summarization_agent.process(relevant_info, query)

        if not response or response.startswith("I apologize"):
            return "I'm having trouble processing your query. Please try rephrasing your question or ask about:\n" + \
                   "- Steps to file a lawsuit in India\n" + \
                   "- Documents required for filing a case\n" + \
                   "- ICAI guidelines for chartered accountants\n" + \
                   "- General litigation process in India"

        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return "I encountered an error while processing your query. Please try asking about:\n" + \
               "- Steps to file a lawsuit in India\n" + \
               "- Documents required for filing a case\n" + \
               "- ICAI guidelines for chartered accountants\n" + \
               "- General litigation process in India"

# Create the Gradio interface
demo = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(
        label="Your Legal Question",
        placeholder="Ask a question about Indian litigation or ICAI guidelines...",
        lines=3
    ),
    outputs=gr.Textbox(
        label="Response",
        lines=10
    ),
    title="Indian Legal & Accounting Assistant",
    description="""
    This AI assistant provides clear, concise answers about Indian legal procedures and ICAI guidelines.

    It utilizes a knowledge base built from specific PDF documents:
    - A Guide to Litigation in India
    - Key ICAI Guidelines for Chartered Accountants

    Developed by: Maqbool Saiyed
    """,
    examples=[]
    ,
    theme=gr.themes.Soft()
)

# For Hugging Face Spaces deployment
if __name__ == "__main__":
    demo.launch()
