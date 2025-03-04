from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Generator
from dsk.api import DeepSeekAPI, AuthenticationError, RateLimitError, NetworkError, APIError
import os

app = FastAPI()

# Initialize the API with your auth token (store in environment variable for security)
API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key")
api = DeepSeekAPI(API_KEY)

class ChatRequest(BaseModel):
    prompt: str
    thinking_enabled: bool = True
    search_enabled: bool = False

def process_response(chunks: Generator[Dict[str, Any], None, None]) -> Dict[str, Any]:
    """Helper function to process response chunks"""
    thinking_lines = []
    text_content = []
    
    for chunk in chunks:
        if chunk['type'] == 'thinking':
            if chunk['content'] and chunk['content'] not in thinking_lines:
                thinking_lines.append(chunk['content'])
        elif chunk['type'] == 'text':
            text_content.append(chunk['content'])
    
    return {
        "thinking": thinking_lines,
        "response": ''.join(text_content)
    }

@app.post("/chat_completion")
def chat_completion(request: ChatRequest):
    try:
        session = api.create_chat_session()
        chunks = api.chat_completion(
            session, 
            request.prompt, 
            thinking_enabled=request.thinking_enabled, 
            search_enabled=request.search_enabled
        )
        return process_response(chunks)
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    except NetworkError:
        raise HTTPException(status_code=503, detail="Network error. Try again later")
    except APIError as e:
        raise HTTPException(status_code=e.status_code or 500, detail="API error occurred")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def status():
    return {"status": "ok", "message": "FastAPI DeepSeekAPI is running"}
