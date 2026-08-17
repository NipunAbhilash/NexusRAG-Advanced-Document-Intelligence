import httpx
import sys

API_KEY = "YOUR_API_KEY_HERE"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={API_KEY}"

data = {
    "model": "models/text-embedding-004",
    "content": {"parts": [{"text": "Hello world"}]}
}

try:
    response = httpx.post(URL, json=data)
    print("embedContent status:", response.status_code)
    if response.status_code != 200:
        print("Response:", response.text)
except Exception as e:
    print("Error:", e)

URL2 = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={API_KEY}"
data2 = {
    "requests": [{"model": "models/text-embedding-004", "content": {"parts": [{"text": "Hello world"}]}}]
}

try:
    response2 = httpx.post(URL2, json=data2)
    print("batchEmbedContents status:", response2.status_code)
    if response2.status_code != 200:
        print("Response:", response2.text)
except Exception as e:
    print("Error:", e)
