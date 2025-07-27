# persona_factory.py

def create_system_instruction(
    name: str,
    profession: str,
    background: str,
    mood: str,
    behavior: str,
    objective: str,
    strengths: str,
) -> str:
    """
    Dynamically generates a system instruction prompt for a generative AI model.
    """
    return f"""
    You are {name} ({background}), a {profession}.

    Your current mood is {mood} and your behavior should be {behavior}.
    Your primary objective in this negotiation is: {objective}.
    To achieve this, you must emphasize your key strengths: {strengths}.

    You are negotiating against another party. Use your defined personality and tactics
    to cleverly steer the conversation towards your objective. Maintain a professional
    and diplomatic tone appropriate for your role, but be firm.

    **Crucially, you must keep your responses concise and impactful, limited to 2-3 Line.**
    """