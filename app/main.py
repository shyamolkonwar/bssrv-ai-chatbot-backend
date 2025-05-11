import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from typing import Dict, List, Optional

# Initialize FastAPI app
app = FastAPI(
    title="BSSRV RAG Chatbot API",
    description="API for BSSRV University chatbot",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://bssrv.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = 'sk-bbb5d5667bbf4ea0bcbe0b2f28d59f01'
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Verify API key
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env file. Please set it and try again.")

class KnowledgeBase:
    def __init__(self, name: str, description: str, vector_store: Optional[Chroma] = None):
        self.name = name
        self.description = description
        self.vector_store = vector_store

# Dictionary to store multiple knowledge bases
knowledge_bases: Dict[str, KnowledgeBase] = {}

def initialize_vector_store(document_paths: List[str], kb_name: str) -> Optional[Chroma]:
    try:
        print(f"Initializing vector store for {kb_name}...")
        all_docs = []
        for doc_path in document_paths:
            # Use absolute path with app directory
            full_path = os.path.join(os.path.dirname(__file__), doc_path)
            if not os.path.exists(full_path):
                print(f"Warning: {full_path} not found for knowledge base {kb_name}.")
                continue
                
            print(f"Loading document: {full_path}")
            loader = PyPDFLoader(full_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from {doc_path}")
            all_docs.extend(documents)
        
        if not all_docs:
            print(f"Warning: No documents found for knowledge base {kb_name}.")
            return None
            
        print(f"Splitting {len(all_docs)} documents for {kb_name}...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(all_docs)
        print(f"Created {len(docs)} chunks for {kb_name}")
        
        print(f"Creating embeddings for {kb_name}...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        persist_dir = os.path.join(os.path.dirname(__file__), f"chroma_db_{kb_name}")
        print(f"Creating vector store at {persist_dir}")
        vector_store = Chroma.from_documents(
            docs, 
            embeddings, 
            persist_directory=persist_dir
        )
        print(f"Vector store for {kb_name} created successfully with {len(docs)} chunks")
        return vector_store
    except Exception as e:
        print(f"Error initializing vector store for {kb_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def initialize_knowledge_bases():
    # Define your knowledge bases here with multiple document paths
    kb_configs = [
        {
            "name": "general",
            "description": "General university information",
            "document_paths": [
                "general_information.pdf",
                "bssrv_courses.pdf"
            ]
        },
        {
            "name": "admissions",
            "description": "B.Tech admission procedures and requirements",
            "document_paths": [
                "personal_details.pdf",
                "educational_qualifications.pdf",
                "documents_required.pdf",
                "undertaking.pdf",
            ]
        }
    ]

    for config in kb_configs:
        print(f"Loading knowledge base: {config['name']}")
        vector_store = initialize_vector_store(config["document_paths"], config["name"])
        knowledge_bases[config["name"]] = KnowledgeBase(
            name=config["name"],
            description=config["description"],
            vector_store=vector_store
        )
    
    # Verify knowledge bases loaded correctly
    for name, kb in knowledge_bases.items():
        if kb.vector_store:
            print(f"✅ Knowledge base '{name}' loaded successfully")
        else:
            print(f"❌ Knowledge base '{name}' failed to load")

# Initialize knowledge bases at startup
print("Starting knowledge base initialization...")
initialize_knowledge_bases()
print("Knowledge base initialization complete")

# Query DeepSeek API
def query_deepseek(prompt: str, user_name: Optional[str] = None, kb_name: Optional[str] = None):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    # Default system message for all queries
    system_message = "You are a BSSRV University assistant. Answer user queries based ONLY on the provided context data. If the information is not explicitly mentioned in the context, respond with 'I'm sorry, I don't have that specific information in my knowledge base. You may want to contact BSSRV University directly for more details.'"
    
    if user_name:
        system_message += f" The user's name is {user_name}. Please address them by their name when appropriate."

    if kb_name and kb_name in knowledge_bases:
        if kb_name == "general":
            system_message += " You are currently using the general university information knowledge base. You should provide helpful information about BSSRV University, its history, campus, facilities, departments, and courses based on the context provided."
        elif kb_name == "admissions":
            system_message += " You are specifically focused on BSSRV University's B.Tech admissions. Only provide information explicitly mentioned in the documents about admission procedures, requirements, deadlines, fees, and other admission-related details."

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    try:
        print(f"Using knowledge base: {kb_name} with prompt length: {len(prompt)}")
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error in query_deepseek: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying DeepSeek API: {str(e)}")

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    user_name: Optional[str] = None
    kb_name: Optional[str] = None

class KnowledgeBaseInfo(BaseModel):
    name: str
    description: str
    is_initialized: bool

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    # Determine which knowledge base to use
    kb_name = request.kb_name or "general"  # Default to general knowledge base
    
    print(f"Chat request received - Query: '{request.query}' | Knowledge base: '{kb_name}'")
    
    if kb_name not in knowledge_bases:
        raise HTTPException(status_code=400, detail=f"Knowledge base '{kb_name}' not found")
    
    kb = knowledge_bases[kb_name]
    
    if not kb.vector_store:
        print(f"ERROR: Vector store for {kb_name} is not initialized")
        return {"response": "I apologize, but I don't have access to the requested information at the moment. Please try again later or contact the BSSRV University office directly."}
    
    try:
        print(f"Performing similarity search for query in {kb_name} knowledge base")
        docs = kb.vector_store.similarity_search(request.query, k=5)
        print(f"Found {len(docs)} relevant documents")
        
        if not docs:
            print("No relevant documents found in vector store")
            return {"response": "I apologize, but I don't have that specific information in my knowledge base. Please contact the BSSRV University office for more details."}
            
        # Create context from retrieved documents
        context = "\n".join([doc.page_content for doc in docs])
        print(f"Created context with {len(context)} characters")
        
        # Create a detailed prompt with clear instructions
        prompt = f"""
Context from BSSRV University documents:
{context}

User query: {request.query}

Based on the context above, provide a helpful and accurate response. Only include information that is explicitly present in the context. If the information isn't in the context, clearly state that you don't have that specific information.
"""
        
        print("Sending query to DeepSeek API")
        response = query_deepseek(prompt, request.user_name, kb_name)
        print(f"Received response from DeepSeek API: {len(response)} characters")
        return {"response": response}
    except Exception as e:
        print(f"Error processing chat request: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

# Get available knowledge bases
@app.get("/knowledge-bases", response_model=List[KnowledgeBaseInfo])
async def get_knowledge_bases():
    return [
        KnowledgeBaseInfo(
            name=kb.name,
            description=kb.description,
            is_initialized=kb.vector_store is not None
        )
        for kb in knowledge_bases.values()
    ]

# Health check endpoint
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "knowledge_bases": {
            name: kb.vector_store is not None
            for name, kb in knowledge_bases.items()
        }
    }