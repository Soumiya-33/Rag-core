import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. did you create a .env file in backend/ "
        "with GROQ_API_KEY=your_key_here?"
        
    )