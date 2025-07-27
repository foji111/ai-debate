# negotiation_engine.py

import time
import random
import google.generativeai as genai

def simulate_thinking_pause(diplomat_name: str):
    """Creates a variable, realistic pause to simulate thinking."""
    pause_duration = random.uniform(2, 5) # Shortened for faster API responses
    print(f"[{diplomat_name} is considering a response...]")
    time.sleep(pause_duration)


def run_negotiation(
    model1_session,
    model2_session,
    model1_name: str,
    model2_name: str,
    initial_prompt: str,
    duration_seconds: int,
    api_key_1: str,
    api_key_2: str
) -> list:
    """
    Orchestrates the negotiation and returns a structured transcript.
    This now switches the global API key before each call.
    """
    print("--- Starting Negotiation Engine ---")
    negotiation_start_time = time.time()
    transcript = []
    turn_counter = 0

    try:
        # --- Turn 0 (Opening Statement from Model 1) ---
        simulate_thinking_pause(model1_name)
        # Set the global API key for Model 1
        genai.configure(api_key=api_key_1)
        response_text = model1_session.send_message(initial_prompt).text
        current_message = response_text
        
        turn_data = {
            "turn": turn_counter,
            "speaker": model1_name,
            "message": response_text
        }
        transcript.append(turn_data)
        print(f"{model1_name}: {response_text}\n")
        turn_counter += 1

        # --- Main Negotiation Loop ---
        while time.time() - negotiation_start_time < duration_seconds:
            # --- Model 2's Turn ---
            simulate_thinking_pause(model2_name)
            # Set the global API key for Model 2
            genai.configure(api_key=api_key_2)
            response_text = model2_session.send_message(current_message).text
            current_message = response_text
            
            turn_data = {
                "turn": turn_counter,
                "speaker": model2_name,
                "message": response_text
            }
            transcript.append(turn_data)
            print(f"{model2_name}: {response_text}\n")
            
            if time.time() - negotiation_start_time >= duration_seconds:
                break

            # --- Model 1's Turn ---
            simulate_thinking_pause(model1_name)
            # Set the global API key for Model 1 again
            genai.configure(api_key=api_key_1)
            response_text = model1_session.send_message(current_message).text
            current_message = response_text

            turn_data = {
                "turn": turn_counter + 1,
                "speaker": model1_name,
                "message": response_text
            }
            transcript.append(turn_data)
            print(f"{model1_name}: {response_text}\n")
            
            turn_counter += 2

    except Exception as e:
        print(f"An error occurred during negotiation: {e}")
        # Append error to transcript if needed
        transcript.append({"error": str(e)})

    finally:
        print("--- Negotiation Engine Finished ---")
        return transcript
