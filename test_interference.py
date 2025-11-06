import os
import requests
import time

ENDPOINT_URL = "https://api.runpod.ai/v2/k4exuts0nl1r0q/run"
STATUS_URL = "https://api.runpod.ai/v2/k4exuts0nl1r0q/status"

API_KEY = os.getenv("RUNPOD_API_TOKEN")
if not API_KEY:
    print("ERROR: The environment variable RUNPOD_API_TOKEN is not set. Please set it before running this script.")
    exit(1)

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Send the prompt
data = {"input": {"prompt": "Hello World"}}
resp = requests.post(ENDPOINT_URL, json=data, headers=headers)
try:
    job_id = resp.json().get("id")
except Exception:
    print("Failed to decode response:", resp.text)
    exit(1)

# Poll for result
for _ in range(30):
    time.sleep(2)
    status_resp = requests.get(f"{STATUS_URL}/{job_id}", headers=headers)
    try:
        result = status_resp.json()
    except Exception:
        print("Failed to decode status response:", status_resp.text)
        break
    print(result)
    if result.get("status") == "COMPLETED":
        print("Output:", result.get("output"))
        break
    elif result.get("status") == "FAILED":
        print("Job Failed.")
        break
