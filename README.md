# AI-Training

## AI Infrastructure Learning Roadmap

**6 Week Commitment**

**Start Date:** Monday, November 4, 2025  
**End Date:** Sunday, December 15, 2025

---

### Weekly Schedule

- **Weekdays:** 5:30-6:30 AM (1 hour)
- **Saturday:** 2:00-4:00 PM (2 hours)
- **Total:** 7 hours/week

---

## Week 1: Python Basics & Fundamentals (Nov 4-10)

**Goal:** Write basic Python scripts and understand core concepts

### Day 1 (Monday): Setup & Variables

**Tasks:**
- [ ] Install Python 3.11+ from [python.org/downloads](https://python.org/downloads)
- [ ] Verify installation: `python3 --version`
- [ ] Install VS Code from [code.visualstudio.com](https://code.visualstudio.com)
- [ ] Install Python extension in VS Code

**Create file:** `hello.py`

```python
# Variables and data types
name = "Your Name"
age = 35
is_learning = True
skills = ["Python", "Docker", "AI"]

print(f"Name: {name}, Age: {age}")
print(f"Skills: {', '.join(skills)}")

# Basic operations
for skill in skills:
print(f"Learning: {skill}")

```

Run: `python3 hello.py`


DAY 2 (Tuesday): Functions & Control Flow
Create file: functions.py
▼ ▼ ▼ CODE BLOCK ▼ ▼ ▼
------------------------------------------------------------
def calculate_gpu_cost(hours, rate_per_hour):
total = hours * rate_per_hour
return round(total, 2)

def check_budget(cost, budget=100):
if cost > budget:
return f"Over budget by ${cost - budget}"
else:
return f"Within budget. ${budget - cost} remaining"

# Test the functions
gpu_hours = 10
hourly_rate = 0.69
total_cost = calculate_gpu_cost(gpu_hours, hourly_rate)
print(f"GPU Cost: ${total_cost}")
print(check_budget(total_cost))
------------------------------------------------------------
▲ ▲ ▲ END CODE ▲ ▲ ▲

Run: python3 functions.py


DAY 3 (Wednesday): File I/O & Error Handling
Create file: file_ops.py
▼ ▼ ▼ CODE BLOCK ▼ ▼ ▼
------------------------------------------------------------
import json

data = {
"models": ["llama-70b", "mistral-7b"],
"costs": {"llama": 6.90, "mistral": 0.50}
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
------------------------------------------------------------
▲ ▲ ▲ END CODE ▲ ▲ ▲

Run: python3 file_ops.py


DAY 4 (Thursday): Basic OOP
Create file: classes.py
▼ ▼ ▼ CODE BLOCK ▼ ▼ ▼
------------------------------------------------------------
class LLMDeployment:
def __init__(self, model_name, gpu_type, cost_per_hour):
self.model_name = model_name
self.gpu_type = gpu_type
self.cost_per_hour = cost_per_hour
self.total_hours = 0

def add_usage(self, hours):
self.total_hours += hours

def calculate_cost(self):
return self.total_hours * self.cost_per_hour

def get_report(self):
return f"""
Model: {self.model_name}
GPU: {self.gpu_type}
Hours: {self.total_hours}
Cost: ${self.calculate_cost():.2f}
"""

deployment = LLMDeployment("Llama-3-70B", "RTX-4090", 0.69)
deployment.add_usage(5)
deployment.add_usage(3)
print(deployment.get_report())
------------------------------------------------------------
▲ ▲ ▲ END CODE ▲ ▲ ▲

Run: python3 classes.py


DAY 5-7 (Fri-Sun): Practice Project
Create file: csv_processor.py
▼ ▼ ▼ CODE BLOCK ▼ ▼ ▼
------------------------------------------------------------
import csv
import json

data = [
{"date": "2025-11-04", "model": "llama-70b",
"hours": 2.5, "cost": 1.73},
{"date": "2025-11-05", "model": "mistral-7b",
"hours": 1.0, "cost": 0.50},
]

# Write CSV
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
------------------------------------------------------------
▲ ▲ ▲ END CODE ▲ ▲ ▲

Run: python3 csv_processor.py

MILESTONE: All files created with correct output
DEADLINE: Sunday, November 10, 7 PM


==============================================
WEEK 2: PYTHON APIs WITH FASTAPI (Nov 11-17)
==============================================
GOAL: Build working REST API with 3 endpoints

DAY 1 (Monday): Setup FastAPI
Install packages:
pip install fastapi uvicorn python-multipart

Verify:
pip list | grep fastapi

Create file: main.py
▼ ▼ ▼ CODE BLOCK ▼ ▼ ▼
------------------------------------------------------------
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
return {"message": "AI Infrastructure API",
"status": "running"}

@app.get("/health")
def health_check():
return {"status": "healthy"}
------------------------------------------------------------
▲ ▲ ▲ END CODE ▲ ▲ ▲

Run: uvicorn main:app --reload

Test in browser:
- http://localhost:8000
- http://localhost:8000/docs


DAY 2 (Tuesday): GET Endpoint
Update main.py (add to existing code):
▼ ▼ ▼ CODE BLOCK ▼ ▼ ▼
------------------------------------------------------------
# In-memory data store
deployments = {
"1": {"id": "1", "model": "llama-70b",
"gpu": "RTX-4090", "status": "running"},
"2": {"id": "2", "model": "mistral-7b",
"gpu": "RTX-3090", "status": "stopped"}
}

@app.get("/deployments")
def get_all_deployments():
return {"deployments": list(deployments.values())}

@app.get("/deployments/{deployment_id}")
def get_deployment(deployment_id: str):
if deployment_id in deployments:
return deployments[deployment_id]
return {"error": "Not found"}, 404
------------------------------------------------------------
▲ ▲ ▲ END CODE ▲ ▲ ▲

Test with curl:
curl http://localhost:8000/deployments
curl http://localhost:8000/deployments/1


DAY 3 (Wednesday): POST Endpoint
Update main.py (add to existing code):
▼ ▼ ▼ CODE BLOCK ▼ ▼ ▼
------------------------------------------------------------
from pydantic import BaseModel
from datetime import datetime

class DeploymentCreate(BaseModel):
model: str
gpu: str
status: str = "starting"

@app.post("/deployments")
def create_deployment(deployment: DeploymentCreate):
new_id = str(len(deployments) + 1)
new_deployment = {
"id": new_id,
"model": deployment.model,
"gpu": deployment.gpu,
"status": deployment.status,
"created_at": datetime.now().isoformat()
}
deployments[new_id] = new_deployment
return new_deployment
------------------------------------------------------------
▲ ▲ ▲ END CODE ▲ ▲ ▲

Test with curl:
▼ ▼ ▼ COMMAND ▼ ▼ ▼
------------------------------------------------------------
curl -X POST http://localhost:8000/deployments \
-H "Content-Type: application/json" \
-d '{"model": "gpt-j-6b", "gpu": "A100"}'
------------------------------------------------------------
▲ ▲ ▲ END COMMAND ▲ ▲ ▲


DAY 4 (Thursday): Cost Calculation
Update main.py (add to existing code):
▼ ▼ ▼ CODE BLOCK ▼ ▼ ▼
------------------------------------------------------------
class CostCalculation(BaseModel):
hours: float
gpu_type: str

GPU_RATES = {
"RTX-4090": 0.69,
"RTX-3090": 0.49,
"A100": 1.89
}

@app.post("/calculate-cost")
def calculate_cost(calc: CostCalculation):
rate = GPU_RATES.get(calc.gpu_type, 0)
if rate == 0:
return {"error": "Unknown GPU type"}

total_cost = calc.hours * rate
return {
"gpu_type": calc.gpu_type,
"hours": calc.hours,
"rate_per_hour": rate,
"total_cost": round(total_cost, 2)
}
------------------------------------------------------------
▲ ▲ ▲ END CODE ▲ ▲ ▲

Test:
▼ ▼ ▼ COMMAND ▼ ▼ ▼
------------------------------------------------------------
curl -X POST http://localhost:8000/calculate-cost \
-H "Content-Type: application/json" \
-d '{"hours": 10, "gpu_type": "RTX-4090"}'
------------------------------------------------------------
▲ ▲ ▲ END COMMAND ▲ ▲ ▲


DAY 5-7 (Fri-Sun): Complete API with Error Handling
See full main.py at: github.com/YOUR_USERNAME/ai-api
(Reference complete code online to avoid email length)

Test all endpoints:
▼ ▼ ▼ COMMANDS ▼ ▼ ▼
------------------------------------------------------------
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/deployments

curl -X POST http://localhost:8000/deployments \
-H "Content-Type: application/json" \
-d '{"model": "llama-70b", "gpu": "RTX-4090"}'

curl -X POST http://localhost:8000/calculate-cost \
-H "Content-Type: application/json" \
-d '{"hours": 15.5, "gpu_type": "A100"}'
------------------------------------------------------------
▲ ▲ ▲ END COMMANDS ▲ ▲ ▲

MILESTONE: API with 3 endpoints, tested with curl
DEADLINE: Sunday, November 17, 7 PM


==============================================
WEEK 3: DOCKER FUNDAMENTALS (Nov 18-24)
==============================================
GOAL: Containerize FastAPI app

DAY 1 (Monday): Install Docker
Tasks:
□ Download Docker Desktop: docker.com/products/docker-desktop
□ Install and start Docker Desktop
□ Verify: docker --version
□ Test: docker run hello-world

Basic commands:
docker images # List images
docker ps # List running containers
docker ps -a # List all containers
docker logs <id> # View logs
docker stop <id> # Stop container
docker rm <id> # Remove container


DAY 2 (Tuesday): Run Pre-built Containers
▼ ▼ ▼ COMMANDS ▼ ▼ ▼
------------------------------------------------------------
# Run nginx
docker run -d -p 8080:80 --name my-nginx nginx

# Test in browser: http://localhost:8080

# View logs
docker logs my-nginx

# Stop and remove
docker stop my-nginx
docker rm my-nginx
------------------------------------------------------------
▲ ▲ ▲ END COMMANDS ▲ ▲ ▲


DAY 3 (Wednesday): Create Dockerfile
Create project structure:
mkdir ai-api
cd ai-api
(Move your main.py from Week 2 here)

Create requirements.txt:
------------------------------------------------------------
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
------------------------------------------------------------

Create Dockerfile:
▼ ▼ ▼ DOCKERFILE ▼ ▼ ▼
------------------------------------------------------------
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0",
"--port", "8000"]
------------------------------------------------------------
▲ ▲ ▲ END DOCKERFILE ▲ ▲ ▲

Create .dockerignore:
------------------------------------------------------------
__pycache__
*.pyc
.venv/
.git/
------------------------------------------------------------


DAY 4 (Thursday): Build and Run
▼ ▼ ▼ COMMANDS ▼ ▼ ▼
------------------------------------------------------------
# Build image
docker build -t ai-api:v1 .

# Verify
docker images | grep ai-api

# Run container
docker run -d -p 8000:8000 --name ai-api-container ai-api:v1

# Check status
docker ps

# View logs
docker logs ai-api-container

# Test API
curl http://localhost:8000/
curl http://localhost:8000/health

# Stop
docker stop ai-api-container
docker rm ai-api-container
------------------------------------------------------------
▲ ▲ ▲ END COMMANDS ▲ ▲ ▲


DAY 5 (Friday): Docker Compose
Create docker-compose.yml:
▼ ▼ ▼ YAML FILE ▼ ▼ ▼
------------------------------------------------------------
version: '3.8'
services:
api:
build: .
ports:
- "8000:8000"
environment:
- ENV=production
restart: unless-stopped
------------------------------------------------------------
▲ ▲ ▲ END YAML ▲ ▲ ▲

Commands:
docker-compose up -d # Start
docker-compose logs -f # View logs
docker-compose down # Stop


DAY 6-7 (Sat-Sun): Push to Docker Hub
▼ ▼ ▼ COMMANDS ▼ ▼ ▼
------------------------------------------------------------
# Create account at hub.docker.com
docker login

# Tag image
docker tag ai-api:v1 YOUR_USERNAME/ai-api:v1

# Push
docker push YOUR_USERNAME/ai-api:v1

# Test pull and run
docker rmi YOUR_USERNAME/ai-api:v1
docker run -d -p 8000:8000 YOUR_USERNAME/ai-api:v1
------------------------------------------------------------
▲ ▲ ▲ END COMMANDS ▲ ▲ ▲

MILESTONE: Containerized app on Docker Hub
DEADLINE: Sunday, November 24, 7 PM


==============================================
WEEK 4: LINUX COMMAND LINE (Nov 25-Dec 1)
==============================================
GOAL: Deploy Docker container on remote server

DAY 1 (Monday): Essential Commands
▼ ▼ ▼ COMMANDS ▼ ▼ ▼
------------------------------------------------------------
# Navigation
pwd # Print working directory
ls # List files
ls -la # List all with details
cd /path/to/dir # Change directory
cd ~ # Go home

# Files
touch file.txt # Create file
mkdir mydir # Create directory
cp file.txt backup.txt # Copy
mv file.txt new.txt # Move/rename
rm file.txt # Delete

# View files
cat file.txt # Display file
tail -f log.txt # Follow log updates
less file.txt # Page through (q to quit)

# Search
find . -name "*.txt" # Find files
grep "error" log.txt # Search in file
------------------------------------------------------------
▲ ▲ ▲ END COMMANDS ▲ ▲ ▲


DAY 2 (Tuesday): Process Management
▼ ▼ ▼ COMMANDS ▼ ▼ ▼
------------------------------------------------------------
# Processes
ps aux # List all processes
ps aux | grep python # Find Python processes
top # Process viewer (q to quit)
kill <PID> # Stop process

# System info
df -h # Disk space
du -sh * # Directory sizes
free -h # Memory usage

# Network
curl http://example.com # HTTP request
netstat -tuln # Show listening ports
------------------------------------------------------------
▲ ▲ ▲ END COMMANDS ▲ ▲ ▲


DAY 3 (Wednesday): Setup DigitalOcean
Tasks:
□ Sign up at digitalocean.com
□ Create Droplet:
- Ubuntu 24.04 LTS
- Basic $6/month plan
- Add SSH key

Generate SSH key (if needed):
▼ ▼ ▼ COMMAND ▼ ▼ ▼
------------------------------------------------------------
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter for defaults
# Copy public key:
cat ~/.ssh/id_ed25519.pub
------------------------------------------------------------
▲ ▲ ▲ END COMMAND ▲ ▲ ▲


DAY 4 (Thursday): SSH and Setup
▼ ▼ ▼ COMMANDS ▼ ▼ ▼
------------------------------------------------------------
# Connect
ssh root@YOUR_DROPLET_IP

# Update system
apt update
apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
docker --version

# Create user
adduser appuser
usermod -aG docker appuser
usermod -aG sudo appuser
su - appuser
------------------------------------------------------------
▲ ▲ ▲ END COMMANDS ▲ ▲ ▲


DAY 5 (Friday): Deploy Container
▼ ▼ ▼ COMMANDS ▼ ▼ ▼
------------------------------------------------------------
# Pull image
docker pull YOUR_USERNAME/ai-api:v1

# Run container
docker run -d \
--name ai-api \
--restart unless-stopped \
-p 80:8000 \
YOUR_USERNAME/ai-api:v1

# Check status
docker ps
docker logs ai-api
------------------------------------------------------------
▲ ▲ ▲ END COMMANDS ▲ ▲ ▲

Test from local machine:
curl http://YOUR_DROPLET_IP/
curl http://YOUR_DROPLET_IP/health


DAY 6-7 (Sat-Sun): Automation Scripts
Create helper script on server:
▼ ▼ ▼ FILE: ~/deploy.sh ▼ ▼ ▼
------------------------------------------------------------
#!/bin/bash
echo "Pulling latest..."
docker pull YOUR_USERNAME/ai-api:v1
echo "Redeploying..."
docker stop ai-api
docker rm ai-api
docker run -d --name ai-api --restart unless-stopped \
-p 80:8000 YOUR_USERNAME/ai-api:v1
echo "Done!"
docker ps
------------------------------------------------------------
▲ ▲ ▲ END FILE ▲ ▲ ▲

Make executable:
chmod +x ~/deploy.sh

Run updates:
./deploy.sh

MILESTONE: Container running on remote server
DEADLINE: Sunday, December 1, 7 PM


==============================================
WEEK 5: RUNPOD DEPLOYMENT (Dec 2-8)
==============================================
GOAL: Deploy and test pre-built LLM

DAY 1-2: RunPod Setup
Tasks:
□ Create account: runpod.io
□ Add $20 credit
□ Go to: Serverless > Templates
□ Find: "vLLM - Llama 3.1 70B"
□ Configure:
- GPU: RTX 4090
- Min Workers: 0
- Max Workers: 1
- Idle Timeout: 5 seconds
□ Note your endpoint URL


DAY 3-4: Test with Python
Full testing script available online to save space.
Key components:
- API authentication with Bearer token
- Inference requests with prompts
- Response time measurement
- Token generation tracking

Reference: github.com/YOUR_USERNAME/ai-api/test_runpod.py


DAY 5-7: Cost Tracking & Analysis
Create comprehensive cost tracking script.
Track:
- Duration per request
- Tokens generated
- Cost per request
- Tokens per second
- Total usage summary

Reference: github.com/YOUR_USERNAME/ai-api/cost_tracker.py

Run comprehensive tests and generate final report.

MILESTONE: Working RunPod deployment with cost docs
DEADLINE: Sunday, December 8, 7 PM


==============================================
WEEK 6: CUSTOM RUNPOD DEPLOYMENT (Dec 9-15)
==============================================
GOAL: Deploy YOUR container on RunPod

DAY 1-2: Modify for RunPod
Create RunPod-compatible handler.
Reference: github.com/YOUR_USERNAME/ai-api/runpod_main.py

Key changes:
- Add runpod package to requirements
- Implement handler function
- Update Dockerfile for RunPod


DAY 3: Build and Deploy
▼ ▼ ▼ COMMANDS ▼ ▼ ▼
------------------------------------------------------------
# Build
docker build -t YOUR_USERNAME/ai-api:runpod .

# Push
docker push YOUR_USERNAME/ai-api:runpod
------------------------------------------------------------
▲ ▲ ▲ END COMMANDS ▲ ▲ ▲

Deploy on RunPod:
- New Endpoint
- Container: YOUR_USERNAME/ai-api:runpod
- GPU: RTX 4090
- Workers: 0-1


DAY 4-5: Testing
Test custom endpoint with Python scripts.
Reference complete testing suite online.


DAY 6-7: Final Documentation
Create comprehensive project README with:
- Project structure
- Milestones achieved
- Cost summary
- Performance metrics
- Deployment URLs
- Skills demonstrated

Push everything to GitHub:
github.com/YOUR_USERNAME/ai-infrastructure-learning

MILESTONE: Complete project on GitHub with working deployment
DEADLINE: Sunday, December 15, 7 PM


==============================================
DAILY LOG TEMPLATE
==============================================
Date: ___________
Week: ___ Day: ___
Actual time: _______

Tasks completed:
□
□

What worked:


Blockers:


Tomorrow's task:


==============================================
WEEKLY CHECK-IN (Every Sunday 7 PM)
==============================================
Week #: ___
Date: ___________

Milestone achieved: YES / NO

If NO, why?


Hours this week: ___
Cumulative hours: ___

Next week's focus:


==============================================
ACCOUNTABILITY CHECKPOINTS
==============================================

Nov 10: Week 1 complete? YES / NO
Nov 17: Week 2 complete? YES / NO
Nov 24: Week 3 complete? YES / NO
Dec 1: Week 4 complete? YES / NO
Dec 8: Week 5 complete? YES / NO
Dec 15: FINAL COMPLETE? YES / NO

FAILURE CONDITION:
- Miss 2 consecutive weeks
- No logs for 5+ days
- No commits for 7+ days
- Total hours < 30 by Dec 1

==============================================
RESOURCES
==============================================
Python: docs.python.org/3/tutorial
FastAPI: fastapi.tiangolo.com
Docker: docs.docker.com/get-started
Linux: ubuntu.com/tutorials/command-line-for-beginners
RunPod: runpod.io/articles/guides

Complete code for all weeks:
github.com/YOUR_USERNAME/ai-infrastructure-learning

==============================================
BUDGET
==============================================
DigitalOcean: $5
RunPod: $20-30
TOTAL: $25-35

==============================================

FIRST ACTION:
Monday, November 4, 5:30 AM
Install Python, complete Day 1 tasks.