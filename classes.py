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
