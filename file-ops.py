import json

data = {
    "models": ["llama-70b", "mistral-7b"],
    "costs": {"llama": 6.90, "mistral": 0.50},
}

# Write JSON
with open("model_costs.json", "w") as f:
    json.dump(data, f, indent=2)

# Read JSON
try:
    with open("model_costs.json", "r") as f:
        loaded_data = json.load(f)
    print("Loaded ", loaded_data)
except FileNotFoundError:
    print("File not found")

# Process data
for model, cost in loaded_data["costs"].items():
    print(f"{model}: ${cost}/hour")
