import requests
import json

# Configuration
ENDPOINT_URL = "https://api.runpod.ai/v2/k4exuts0nl1r0q/run"
API_KEY = ""

# Headers for authentication
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Test prompts
test_prompts = [
    "What is artificial intelligence?",
    "Explain the concept of machine learning in simple terms.",
    "Write a short poem about technology.",
    "What are the main benefits of cloud computing?",
    "Describe the difference between supervised and unsupervised learning."
]

def send_request(prompt):
    """Send a request to the Runpod endpoint"""
    data = {
        "input": {
            "prompt": prompt
        }
    }
    
    try:
        response = requests.post(ENDPOINT_URL, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def main():
    print("Starting Week #5 LLM Inference Tests\n")
    print("=" * 80)
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}/{len(test_prompts)}")
        print(f"Prompt: {prompt}")
        print("-" * 80)
        
        result = send_request(prompt)
        
        if result:
            # Extract relevant information
            status = result.get("status", "UNKNOWN")
            output = result.get("output", "No output received")
            
            print(f"Status: {status}")
            print(f"Response: {output}\n")
            
            # Store result for later analysis
            results.append({
                "prompt": prompt,
                "status": status,
                "output": output
            })
        else:
            print("Failed to get response\n")
    
    print("=" * 80)
    print(f"\nCompleted {len(results)} tests successfully")
    
    # Save results to file
    with open('inference_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to inference_results.json")

if __name__ == "__main__":
    main()
