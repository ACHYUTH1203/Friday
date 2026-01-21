import os
import shutil
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from graph import app as graph_app

app = FastAPI(title="Friday: AI Career Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    user_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handles standard user queries (e.g., "Find me a job").
    """
    try:
        inputs = {
            "user_query": request.query,
            "user_id": request.user_id 
        }
        
        result = graph_app.invoke(inputs)

        final_msg = result.get("final_response", "I encountered an error processing your request.")
        final_user_id = result.get("user_id") 

        return ChatResponse(response=final_msg, user_id=final_user_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...), 
    user_id: Optional[str] = Form(None)
):
    """
    1. Streams file to a temporary location (RAM/Temp).
    2. Runs graph to ingest & save to MongoDB.
    3. Deletes file immediately.
    """
    temp_file_path = None
    
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name
            
        print(f" Processing temp file: {temp_file_path}")

        inputs = {
            "user_query": f"Analyzed resume: {file.filename}",
            "file_path": temp_file_path,
            "user_id": user_id
        }
        
        result = graph_app.invoke(inputs)
        
        final_user_id = result.get("user_id")
        
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print("🧹 Temp file cleaned up.")

        return {
            "message": "Resume uploaded and analyzed successfully.",
            "user_id": final_user_id,
            "analysis_summary": result.get("final_response")
        }

    except Exception as e:

        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)