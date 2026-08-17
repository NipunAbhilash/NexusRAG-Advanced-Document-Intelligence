import urllib.request
import urllib.error
import json
import os

api_key = "AQ.Ab8RN6LV3j0USmxh7uV36DIl2ZqvnmCkNPjYSwbkohwM7As1g"

def test_api_key(headers):
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers=headers
    )
    try:
        response = urllib.request.urlopen(req)
        print("Success with headers:", list(headers.keys()))
        return True
    except urllib.error.HTTPError as e:
        print("Failed with headers:", list(headers.keys()), e.code, e.read().decode('utf-8'))
        return False

# Test 1: x-goog-api-key
test_api_key({"x-goog-api-key": api_key})

# Test 2: Authorization: Bearer
test_api_key({"Authorization": f"Bearer {api_key}"})
