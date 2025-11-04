# AI Infrastructure Learning Roadmap

A structured 6-week program to learn Python, APIs, Docker, Linux, and AI deployment.

**Duration:** November 4 - December 15, 2025  
**Time Commitment:** 7 hours/week (5:30-6:30 AM weekdays + 2 hours Saturday)

---

## 📋 Table of Contents

- [Week 1: Python Basics](#week-1-python-basics-nov-4-10)
- [Week 2: FastAPI](#week-2-fastapi-nov-11-17)
- [Week 3: Docker](#week-3-docker-nov-18-24)
- [Week 4: Linux & Deployment](#week-4-linux--deployment-nov-25-dec-1)
- [Week 5: RunPod](#week-5-runpod-dec-2-8)
- [Week 6: Custom Deployment](#week-6-custom-deployment-dec-9-15)
- [Resources](#resources)
- [Tracking Progress](#tracking-progress)

---

## Week 1: Python Basics (Nov 4-10)

**Goal:** Write basic Python scripts and understand core concepts

### Setup Tasks
- [ ] Install Python 3.11+ from [python.org](https://python.org/downloads)
- [ ] Install VS Code from [code.visualstudio.com](https://code.visualstudio.com)
- [ ] Install Python extension in VS Code

### Day 1: Variables & Data Types

**File:** `hello.py`

```python
# Variables and data types
name = "Your Name"
age = 35
is_learning = True
skills = ["Python", "Docker", "AI"]

print(f"Name: {name}, Age: {age}")
print(f"Skills: {', '.join(skills)}")

for skill in skills:
    print(f"Learning: {skill}")
```

**Run:** `python3 hello.py`

### Day 2: Functions & Control Flow

**File:** `functions.py`

```python
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
```

### Day 3: File I/O & Error Handling

**File:** `file_ops.py`

```python
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
        print("Loaded:", loaded_data)
except FileNotFoundError:
    print("File not found")

# Process data
for model, cost in loaded_data["costs"].items():
    print(f"{model}: ${cost}/hour")
```

### Day 4: Object-Oriented Programming

**File:** `classes.py`

```python
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
```

### Days 5-7: Practice Project

**File:** `csv_processor.py`

```python
import csv
import json

data = [
    {"date": "2025-11-04", "model": "llama-70b", "hours": 2.5, "cost": 1.73},
    {"date": "2025-11-05", "model": "mistral-7b", "hours": 1.0, "cost": 0.50},
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
```

**Milestone:** All files created with correct output  
**Deadline:** Sunday, November 10, 7 PM

---

## Week 2: FastAPI (Nov 11-17)

**Goal:** Build a working REST API with multiple endpoints

### Day 1: Setup FastAPI

**Install packages:**
```bash
pip install fastapi uvicorn python-multipart
pip list | grep fastapi  # Verify installation
```

**File:** `main.py`

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Infrastructure API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

**Run:** `uvicorn main:app --reload`

**Test:** 
- http://localhost:8000
- http://localhost:8000/docs

### Day 2: GET Endpoints

Add to `main.py`:

```python
# In-memory data store
deployments = {
    "1": {"id": "1", "model": "llama-70b", "gpu": "RTX-4090", "status": "running"},
    "2": {"id": "2", "model": "mistral-7b", "gpu": "RTX-3090", "status": "stopped"}
}

@app.get("/deployments")
def get_all_deployments():
    return {"deployments": list(deployments.values())}

@app.get("/deployments/{deployment_id}")
def get_deployment(deployment_id: str):
    if deployment_id in deployments:
        return deployments[deployment_id]
    return {"error": "Not found"}
```

**Test:**
```bash
curl http://localhost:8000/deployments
curl http://localhost:8000/deployments/1
```

### Day 3: POST Endpoints

Add to `main.py`:

```python
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
```

**Test:**
```bash
curl -X POST http://localhost:8000/deployments \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-j-6b", "gpu": "A100"}'
```

### Day 4: Cost Calculation

Add to `main.py`:

```python
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
```

**Test:**
```bash
curl -X POST http://localhost:8000/calculate-cost \
  -H "Content-Type: application/json" \
  -d '{"hours": 10, "gpu_type": "RTX-4090"}'
```

**Milestone:** API with 3 endpoints, tested with curl  
**Deadline:** Sunday, November 17, 7 PM

---

## Week 3: Docker (Nov 18-24)

**Goal:** Containerize the FastAPI application

### Day 1: Install Docker

**Tasks:**
- [ ] Download [Docker Desktop](https://docker.com/products/docker-desktop)
- [ ] Install and start Docker Desktop
- [ ] Verify: `docker --version`
- [ ] Test: `docker run hello-world`

**Basic commands:**
```bash
docker images          # List images
docker ps              # List running containers
docker ps -a           # List all containers
docker logs <id>       # View logs
docker stop <id>       # Stop container
docker rm <id>         # Remove container
```

### Day 2: Run Pre-built Containers

```bash
# Run nginx
docker run -d -p 8080:80 --name my-nginx nginx

# Test in browser: http://localhost:8080

# View logs
docker logs my-nginx

# Stop and remove
docker stop my-nginx
docker rm my-nginx
```

### Day 3: Create Dockerfile

**Project structure:**
```bash
mkdir ai-api
cd ai-api
# Move your main.py from Week 2 here
```

**File:** `requirements.txt`
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
```

**File:** `Dockerfile`
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**File:** `.dockerignore`
```
__pycache__
*.pyc
.venv/
.git/
```

### Day 4: Build and Run

```bash
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
```

### Day 5: Docker Compose

**File:** `docker-compose.yml`
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=production
    restart: unless-stopped
```

**Commands:**
```bash
docker-compose up -d      # Start
docker-compose logs -f    # View logs
docker-compose down       # Stop
```

### Days 6-7: Push to Docker Hub

```bash
# Create account at hub.docker.com
docker login

# Tag image
docker tag ai-api:v1 YOUR_USERNAME/ai-api:v1

# Push
docker push YOUR_USERNAME/ai-api:v1

# Test pull and run
docker rmi YOUR_USERNAME/ai-api:v1
docker run -d -p 8000:8000 YOUR_USERNAME/ai-api:v1
```

**Milestone:** Containerized app on Docker Hub  
**Deadline:** Sunday, November 24, 7 PM

---

## Week 4: Linux & Deployment (Nov 25-Dec 1)

**Goal:** Deploy Docker container on a remote server

### Day 1: Essential Linux Commands

```bash
# Navigation
pwd                    # Print working directory
ls                     # List files
ls -la                 # List all with details
cd /path/to/dir        # Change directory
cd ~                   # Go home

# Files
touch file.txt         # Create file
mkdir mydir            # Create directory
cp file.txt backup.txt # Copy
mv file.txt new.txt    # Move/rename
rm file.txt            # Delete

# View files
cat file.txt           # Display file
tail -f log.txt        # Follow log updates
less file.txt          # Page through (q to quit)

# Search
find . -name "*.txt"   # Find files
grep "error" log.txt   # Search in file
```

### Day 2: Process Management

```bash
# Processes
ps aux                     # List all processes
ps aux | grep python       # Find Python processes
top                        # Process viewer (q to quit)
kill <PID>                 # Stop process

# System info
df -h                      # Disk space
du -sh *                   # Directory sizes
free -h                    # Memory usage

# Network
curl http://example.com    # HTTP request
netstat -tuln              # Show listening ports
```

### Day 3: Setup DigitalOcean

**Tasks:**
- [ ] Sign up at [digitalocean.com](https://digitalocean.com)
- [ ] Create Droplet: Ubuntu 24.04 LTS, Basic $6/month plan
- [ ] Add SSH key

**Generate SSH key:**
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter for defaults
cat ~/.ssh/id_ed25519.pub  # Copy this public key
```

### Day 4: SSH and Server Setup

```bash
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
```

### Day 5: Deploy Container

```bash
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
```

**Test from local machine:**
```bash
curl http://YOUR_DROPLET_IP/
curl http://YOUR_DROPLET_IP/health
```

### Days 6-7: Automation Scripts

**File:** `~/deploy.sh`
```bash
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
```

**Make executable and run:**
```bash
chmod +x ~/deploy.sh
./deploy.sh
```

**Milestone:** Container running on remote server  
**Deadline:** Sunday, December 1, 7 PM

---

## Week 5: RunPod (Dec 2-8)

**Goal:** Deploy and test a pre-built LLM

### Days 1-2: RunPod Setup

**Tasks:**
- [ ] Create account at [runpod.io](https://runpod.io)
- [ ] Add $20 credit
- [ ] Go to: Serverless > Templates
- [ ] Find: "vLLM - Llama 3.1 70B"
- [ ] Configure:
  - GPU: RTX 4090
  - Min Workers: 0
  - Max Workers: 1
  - Idle Timeout: 5 seconds
- [ ] Note your endpoint URL

### Days 3-4: Test with Python

Create testing script with:
- API authentication with Bearer token
- Inference requests with prompts
- Response time measurement
- Token generation tracking

*See `/examples/test_runpod.py` for full implementation*

### Days 5-7: Cost Tracking & Analysis

Create comprehensive cost tracking script to track:
- Duration per request
- Tokens generated
- Cost per request
- Tokens per second
- Total usage summary

*See `/examples/cost_tracker.py` for full implementation*

**Milestone:** Working RunPod deployment with cost documentation  
**Deadline:** Sunday, December 8, 7 PM

---

## Week 6: Custom Deployment (Dec 9-15)

**Goal:** Deploy YOUR container on RunPod

### Days 1-2: Modify for RunPod

Create RunPod-compatible handler with:
- Add `runpod` package to requirements
- Implement handler function
- Update Dockerfile for RunPod

*See `/examples/runpod_main.py` for reference implementation*

### Day 3: Build and Deploy

```bash
# Build
docker build -t YOUR_USERNAME/ai-api:runpod .

# Push
docker push YOUR_USERNAME/ai-api:runpod
```

**Deploy on RunPod:**
- Create New Endpoint
- Container: `YOUR_USERNAME/ai-api:runpod`
- GPU: RTX 4090
- Workers: 0-1

### Days 4-5: Testing

Test custom endpoint with Python scripts.

*See `/examples/test_custom_endpoint.py`*

### Days 6-7: Final Documentation

Create comprehensive project README with:
- Project structure
- Milestones achieved
- Cost summary
- Performance metrics
- Deployment URLs
- Skills demonstrated

**Milestone:** Complete project on GitHub with working deployment  
**Deadline:** Sunday, December 15, 7 PM

---

## Resources

### Official Documentation
- **Python:** [docs.python.org/3/tutorial](https://docs.python.org/3/tutorial)
- **FastAPI:** [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **Docker:** [docs.docker.com/get-started](https://docs.docker.com/get-started)
- **Linux:** [ubuntu.com/tutorials/command-line-for-beginners](https://ubuntu.com/tutorials/command-line-for-beginners)
- **RunPod:** [runpod.io/articles/guides](https://runpod.io/articles/guides)

### Budget
- **DigitalOcean:** $5-6/month
- **RunPod:** $20-30 for testing
- **Total:** $25-35

---

## Tracking Progress

### Daily Log Template

```
Date: ___________
Week: ___ Day: ___
Actual time: _______

Tasks completed:
□ 
□ 

What worked:


Blockers:


Tomorrow's task:
```

### Weekly Check-in (Every Sunday 7 PM)

```
Week #: ___
Date: ___________

Milestone achieved: YES / NO

If NO, why?


Hours this week: ___
Cumulative hours: ___

Next week's focus:
```

### Accountability Checkpoints

- [ ] **Nov 10:** Week 1 complete?
- [ ] **Nov 17:** Week 2 complete?
- [ ] **Nov 24:** Week 3 complete?
- [ ] **Dec 1:** Week 4 complete?
- [ ] **Dec 8:** Week 5 complete?
- [ ] **Dec 15:** FINAL COMPLETE?

### Failure Conditions
- Miss 2 consecutive weeks
- No logs for 5+ days
- No commits for 7+ days
- Total hours < 30 by Dec 1

---

## Getting Started

**First Action: Monday, November 4, 5:30 AM**

1. Install Python 3.11+
2. Install VS Code
3. Complete Day 1 tasks
4. Commit your first file to GitHub

Good luck! 🚀