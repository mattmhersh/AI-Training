import requests
import time

ENDPOINT_URL = "https://api.runpod.ai/v2/k4exuts0nl1r0q/run"
STATUS_URL = "https://api.runpod.ai/v2/k4exuts0nl1r0q/status"
API_KEY = ""

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Send the prompt
data = {"input": {"prompt": "Hello World"}}
resp = requests.post(ENDPOINT_URL, json=data, headers=headers)
job_id = resp.json().get("id")

# Poll for result
for _ in range(30):
    time.sleep(2)
    status_resp = requests.get(f"{STATUS_URL}/{job_id}", headers=headers)
    result = status_resp.json()
    print(result)
    if result.get("status") == "COMPLETED":
        print("Output:", result.get("output"))
        break
    elif result.get("status") == "FAILED":
        print("Job Failed.")
        break
