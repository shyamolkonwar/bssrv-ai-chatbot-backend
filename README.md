# BSSRV Chatbot Backend

This is the FastAPI backend for the BSSRV University chatbot application. It uses LangChain for RAG (Retrieval-Augmented Generation) and DeepSeek API for generating responses.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the backend directory with your DeepSeek API key:
```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

4. Place your knowledge base PDF file (`knowledge_base.pdf`) in the backend directory.

## Running the Server

Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

The server will run on `http://localhost:8000` by default.

## API Endpoints

- `POST /chat`: Send a chat message and get a response
  - Request body: `{ "query": "your question here" }`
  - Response: `{ "response": "bot's response" }`

- `GET /health`: Health check endpoint
  - Response: `{ "status": "healthy" }`

## Development

- The server uses CORS middleware to allow requests from `http://localhost:3000` (Next.js frontend)
- Vector store is initialized at startup using the knowledge base PDF
- Responses are generated using the DeepSeek API with context from the vector store 