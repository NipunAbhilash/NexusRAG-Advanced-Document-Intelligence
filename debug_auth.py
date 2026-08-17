import sys

def debug_auth():
    try:
        from google import genai
        client = genai.Client(api_key="AQ.testkey")
        print("vertexai:", client._api_client.vertexai)
        print("api_key:", client._api_client.api_key)
        
        has_use_auth = hasattr(client._api_client, "_use_google_auth_sync")
        print("Has _use_google_auth_sync:", has_use_auth)
        if has_use_auth:
            print("_use_google_auth_sync():", client._api_client._use_google_auth_sync())
            
        print("http_options:", client._api_client._http_options)
        
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    debug_auth()
