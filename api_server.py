# api_server.py

import os
import time
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Import our custom modules
import persona_factory
import negotiation_engine

#nilesh

# --- Configuration & Initialization ---
load_dotenv()
app = FastAPI(
    title="Dynamic Negotiation API",
    description="An API to simulate a negotiation between two dynamic AI personas.",
)

# --- ADD THIS MIDDLEWARE BLOCK ---
# This specifically allows your new live website to make requests to your API.
origins = [
    "https://ai-debate-website.onrender.com",
    "http://localhost",
    "http://127.0.0.1",
    "null"  # Important for allowing requests from local file:///
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------------------


try:
    # Load keys from environment
    API_KEY_1 = os.getenv("GOOGLE_API_KEY")
    API_KEY_2 = os.getenv("GOOGLE_API_KEY_2", API_KEY_1) # Fallback
    
    if not API_KEY_1:
        print("Warning: GOOGLE_API_KEY environment variable not found.")
    else:
        # Configure with a default key. This will be switched as needed in the engine.
        genai.configure(api_key=API_KEY_1)

except Exception as e:
    print(f"Error during initial loading: {e}")
    API_KEY_1 = None
    API_KEY_2 = None


# --- Pydantic Models for API Request Body ---
class CharacterProfile(BaseModel):
    name: str
    profession: str
    background: str
    mood: str
    behavior: str
    objective: str
    strengths: str
    model_name: str = "gemini-2.5-Flash-Lite"

class NegotiationRequest(BaseModel):
    topic: str
    duration_seconds: int = 60
    character1: CharacterProfile
    character2: CharacterProfile

# --- Helper Functions ---
async def get_negotiation_summary(transcript: list, topic: str) -> str:
    if not transcript:
        return "The negotiation did not start or an error occurred."
    if not API_KEY_1:
        return "Summarization failed: API key not configured."
    try:
        # Ensure the summarizer uses the primary API key
        genai.configure(api_key=API_KEY_1)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        conversation_log = "\n".join([f"{item['speaker']}: {item['message']}" for item in transcript if 'error' not in item])
        
        prompt = f"""
        Based on the following negotiation transcript about '{topic}', please provide a brief, neutral summary of the outcome.
        Answer these questions:
        1. What was the final position of each party?
        2. Was a clear agreement reached? If so, what were the terms?
        3. If no agreement was reached, what were the main points of contention?
        Transcript:
        ---
        {conversation_log}
        ---
        """
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate summary: {e}"

# --- API Endpoint ---
@app.post("/negotiate")
async def start_negotiation_endpoint(request: NegotiationRequest):
    if not API_KEY_1 or not API_KEY_2:
        raise HTTPException(status_code=500, detail="Google API keys are not configured on the server.")
    
    start_time = time.time()
    
    instruction1 = persona_factory.create_system_instruction(**request.character1.model_dump(exclude={'model_name'}))
    instruction2 = persona_factory.create_system_instruction(**request.character2.model_dump(exclude={'model_name'}))
    
    try:
        model1 = genai.GenerativeModel(
            model_name=request.character1.model_name,
            system_instruction=instruction1
        )
        model2 = genai.GenerativeModel(
            model_name=request.character2.model_name,
            system_instruction=instruction2
        )
        
        chat1 = model1.start_chat(history=[])
        chat2 = model2.start_chat(history=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize AI models: {e}")
    
    initial_prompt = f"""
    As {request.character1.name}, make your opening statement to {request.character2.name}
    regarding the negotiation on '{request.topic}'. Clearly state your initial position
    based on your objective: '{request.character1.objective}'.
    """
    
    transcript = negotiation_engine.run_negotiation(
        model1_session=chat1,
        model2_session=chat2,
        model1_name=f"{request.character1.name} ({request.character1.background})",
        model2_name=f"{request.character2.name} ({request.character2.background})",
        initial_prompt=initial_prompt,
        duration_seconds=request.duration_seconds,
        api_key_1=API_KEY_1,
        api_key_2=API_KEY_2
    )
    
    summary = await get_negotiation_summary(transcript, request.topic)
    end_time = time.time()
    
    final_response = {
        "negotiation_summary": {
            "topic": request.topic,
            "duration_seconds": round(end_time - start_time),
            "outcome_analysis": summary,
        },
        "participants": [
            request.character1.model_dump(),
            request.character2.model_dump()
        ],
        "transcript": transcript
    }
    return final_response
