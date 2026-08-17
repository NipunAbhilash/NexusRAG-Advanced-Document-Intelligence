import urllib.request
import json

api_key = "YOUR_API_KEY_HERE"
model = "gemini-3.6-flash"

url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
payload = {
    "contents": [{"parts": [{"text": "Hello, how are you?"}]}]
}
data = json.dumps(payload).encode('utf-8')

print(f"Testing {model}...")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req) as response:
        print("Status Code:", response.getcode())
        print("Response:", response.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code)
    print("Response:", e.read().decode('utf-8'))
