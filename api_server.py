# api_server.py

import os
import time
from typing import List
import google.generativeai as genai
from google.ai import generativelanguage as glm # Import for the service client
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import our custom modules
import persona_factory
import negotiation_engine

# --- Configuration & Initialization ---
load_dotenv()
app = FastAPI(
    title="Dynamic Negotiation API",
    description="An API to simulate a negotiation between two dynamic AI personas.",
)

try:
    # Load keys from environment
    API_KEY_1 = os.getenv("GOOGLE_API_KEY")
    API_KEY_2 = os.getenv("GOOGLE_API_KEY_2", API_KEY_1) # Fallback to key 1 if key 2 isn't set
    
    if not API_KEY_1:
        print("Warning: GOOGLE_API_KEY environment variable not found.")

except Exception as e:
    print(f"Error during initial loading: {e}")
    API_KEY_1 = None
    API_KEY_2 = None


# --- Pydantic Models for API Request Body ---
class CharacterProfile(BaseModel):
    name: str
    profession: str
    background: str # e.g., "from India" or "representing CyberCorp"
    mood: str
    behavior: str
    objective: str
    strengths: str
    model_name: str = "gemini-1.5-flash" # Default model

class NegotiationRequest(BaseModel):
    topic: str
    duration_seconds: int = 60
    character1: CharacterProfile
    character2: CharacterProfile

# --- Helper Functions ---
async def get_negotiation_summary(transcript: list, topic: str) -> str:
    """Uses an AI call to summarize the outcome of the negotiation."""
    if not transcript:
        return "The negotiation did not start or an error occurred."

    if not API_KEY_1:
        return "Summarization failed: API key not configured."

    try:
        # Use the robust client initialization for the summarizer as well
        summarizer_client = glm.GenerativeServiceClient(client_options={"api_key": API_KEY_1})
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            client=summarizer_client
        )
        
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
    """
    Starts a negotiation based on the provided character profiles and topic.
    """
    if not API_KEY_1 or not API_KEY_2:
        raise HTTPException(status_code=500, detail="Google API keys are not configured on the server. Please check environment variables.")

    start_time = time.time()
    
    instruction1 = persona_factory.create_system_instruction(**request.character1.model_dump(exclude={'model_name'}))
    instruction2 = persona_factory.create_system_instruction(**request.character2.model_dump(exclude={'model_name'}))

    try:
        # FINAL CORRECTED INITIALIZATION
        # 1. Create a separate service client for each API key.
        client1 = glm.GenerativeServiceClient(client_options={"api_key": API_KEY_1})
        client2 = glm.GenerativeServiceClient(client_options={"api_key": API_KEY_2})

        # 2. Pass the pre-configured client to the GenerativeModel using the 'client' argument.
        model1 = genai.GenerativeModel(
            model_name=request.character1.model_name,
            system_instruction=instruction1,
            client=client1
        )

        model2 = genai.GenerativeModel(
            model_name=request.character2.model_name,
            system_instruction=instruction2,
            client=client2
        )
        
        chat1 = model1.start_chat(history=[])
        chat2 = model2.start_chat(history=[])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize AI models: {e}")

    # Define the initial prompt for the first character to kick things off
    initial_prompt = f"""
    As {request.character1.name}, make your opening statement to {request.character2.name}
    regarding the negotiation on '{request.topic}'. Clearly state your initial position
    based on your objective: '{request.character1.objective}'.
    """
    
    # Run the negotiation engine
    transcript = negotiation_engine.run_negotiation(
        model1_session=chat1,
        model2_session=chat2,
        model1_name=f"{request.character1.name} ({request.character1.background})",
        model2_name=f"{request.character2.name} ({request.character2.background})",
        initial_prompt=initial_prompt,
        duration_seconds=request.duration_seconds
    )
    
    # Generate the final summary
    print("Generating final summary...")
    summary = await get_negotiation_summary(transcript, request.topic)
    
    end_time = time.time()
    
    # Assemble the final JSON response
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
