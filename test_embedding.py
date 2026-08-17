"""Utility script to verify LLM model accessibility."""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

test_models = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

print("Testing LLM models availability...\n")
for m in test_models:
    try:
        r = client.models.generate_content(model=m, contents="Say hi in 3 words")
        print(f"[{m}] Success - Response: {r.text.strip()[:50]}")
    except Exception as e:
        err = str(e)[:80]
        print(f"[{m}] Failed - Error: {err}")
