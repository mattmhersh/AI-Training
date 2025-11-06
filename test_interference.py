import requests

endpoint_url = "https://api.runpod.ai/v2/k4exuts0nl1r0q/run"
api_token = ""  # Replace with your actual Runpod API key

headers = {
    "Authorization": f"Bearer {api_token}"
}

data = {
    "input": {
        "prompt": "Hello World"
    }
}

response = requests.post(endpoint_url, json=data, headers=headers)
print("Status code:", response.status_code)
print("Response:", response.text)
