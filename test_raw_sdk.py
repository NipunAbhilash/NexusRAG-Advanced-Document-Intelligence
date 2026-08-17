import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print(f"API Key starting with: {api_key[:5] if api_key else 'None'}")

print("Testing direct google-genai SDK...")
try:
    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='What is 2+2?'
    )
    print("Direct SDK generation SUCCESS!")
    
    response = client.models.embed_content(
        model='text-embedding-004',
        contents='test'
    )
    print("Direct SDK embedding SUCCESS!")
except Exception as e:
    print(f"Direct SDK failed: {e}")

print("Testing langchain-google-genai...")
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    res = llm.invoke("What is 2+2?")
    print("Langchain generation SUCCESS!")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    res2 = embeddings.embed_query("test")
    print("Langchain embedding SUCCESS!")
except Exception as e:
    print(f"Langchain failed: {e}")
