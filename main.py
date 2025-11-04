from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Infrastructure API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

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
    return {"error": "Not found"}, 404

class DeploymentCreate(BaseModel):
    model: str
    gpu: str
    status: str = "starting"

@app.post("/deployments")
def create_deployment(deployment: DeploymentCreate):
    # (Removed duplicate code block outside of function)
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