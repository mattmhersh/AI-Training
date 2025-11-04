import csv
import json

# Write CSV
data = [
	{"date": "2025-11-04", "model": "llama-70b", "hours": 2.5, "cost": 1.73},
	{"date": "2025-11-05", "model": "mistral-7b", "hours": 1.0, "cost": 0.50},
]

with open("usage.csv", "w", newline="") as f:
	fields = ["date", "model", "hours", "cost"]
	writer = csv.DictWriter(f, fieldnames=fields)
	writer.writeheader()
	writer.writerows(data)

# Read and process
total_cost = 0
total_hours = 0
with open("usage.csv", "r") as f:
	reader = csv.DictReader(f)
	for row in reader:
		total_cost += float(row["cost"])
		total_hours += float(row["hours"])

# Output summary
summary = {
	"total_hours": total_hours,
	"total_cost": round(total_cost, 2),
	"avg_rate": round(total_cost / total_hours, 2)
}

with open("summary.json", "w") as f:
	json.dump(summary, f, indent=2)

print("Summary:", summary)