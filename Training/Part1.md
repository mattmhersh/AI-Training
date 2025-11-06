# AI Infrastructure & Engineering Curriculum
## Part 1: Foundations (Weeks 1-4)

**Duration:** 4 weeks  
**Time Commitment:** 7-10 hours/week  
**Budget:** $40-50  
**Prerequisites:** None

---

## Table of Contents - Part 1

- [Week 1: Python Fundamentals](#week-1-python-fundamentals)
- [Week 2: Building Production APIs](#week-2-building-production-apis)
- [Week 3: Containerization & Docker](#week-3-containerization--docker)
- [Week 4: Linux & Cloud Deployment](#week-4-linux--cloud-deployment)

---

## Week 1: Python Fundamentals

**Real-World Goal:** Build a CLI tool for AI cost analysis that you'll actually use

### Setup (Day 1)

```bash
# Install Python 3.11+
# Download from python.org

# Verify installation
python3 --version

# Install VS Code
# Download from code.visualstudio.com

# Install Python extension in VS Code
# Search "Python" in extensions, install Microsoft's official extension
```

### Day 1-2: Building a GPU Cost Calculator

**Real Use Case:** Calculate actual costs before running expensive AI workloads

**File:** `gpu_calculator.py`

```python
"""
GPU Cost Calculator - Real-world tool for estimating AI deployment costs
Use case: Before deploying an LLM, estimate actual costs for different scenarios
"""

from datetime import datetime
from typing import Dict, List
import json

# Real RunPod GPU pricing (as of 2024)
GPU_PRICING = {
    "RTX-4090": {"hourly": 0.69, "vram": 24, "tflops": 82.6},
    "RTX-3090": {"hourly": 0.49, "vram": 24, "tflops": 35.6},
    "A100-80GB": {"hourly": 1.89, "vram": 80, "tflops": 312},
    "A100-40GB": {"hourly": 1.69, "vram": 40, "tflops": 312},
    "H100": {"hourly": 4.25, "vram": 80, "tflops": 989},
}

def calculate_deployment_cost(
    gpu_type: str,
    hours_per_day: float,
    days: int,
    requests_per_hour: int = 0
) -> Dict:
    """
    Calculate total cost for a deployment scenario
    
    Example: Running a customer support chatbot
    - 8 hours/day (business hours)
    - 30 days/month
    - ~100 requests/hour during peak
    """
    if gpu_type not in GPU_PRICING:
        return {"error": f"Unknown GPU: {gpu_type}"}
    
    pricing = GPU_PRICING[gpu_type]
    total_hours = hours_per_day * days
    total_cost = total_hours * pricing["hourly"]
    
    # Calculate per-request cost if provided
    cost_per_request = None
    if requests_per_hour > 0:
        total_requests = requests_per_hour * total_hours
        cost_per_request = total_cost / total_requests
    
    return {
        "gpu_type": gpu_type,
        "scenario": {
            "hours_per_day": hours_per_day,
            "total_days": days,
            "total_hours": total_hours,
            "requests_per_hour": requests_per_hour or "N/A"
        },
        "gpu_specs": {
            "vram_gb": pricing["vram"],
            "tflops": pricing["tflops"],
            "hourly_rate": pricing["hourly"]
        },
        "costs": {
            "total": round(total_cost, 2),
            "daily_avg": round(total_cost / days, 2),
            "per_request": round(cost_per_request, 4) if cost_per_request else "N/A"
        },
        "timestamp": datetime.now().isoformat()
    }

def compare_gpus(scenario: Dict) -> List[Dict]:
    """Compare costs across all available GPUs"""
    results = []
    for gpu in GPU_PRICING.keys():
        result = calculate_deployment_cost(
            gpu, 
            scenario["hours_per_day"],
            scenario["days"],
            scenario.get("requests_per_hour", 0)
        )
        results.append(result)
    
    # Sort by total cost
    results.sort(key=lambda x: x["costs"]["total"])
    return results

def save_analysis(analysis: Dict, filename: str):
    """Save analysis for future reference"""
    with open(filename, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Analysis saved to {filename}")

# Real-world scenarios
if __name__ == "__main__":
    print("🤖 GPU Cost Calculator - Real Deployment Scenarios\n")
    
    # Scenario 1: Customer Support Chatbot (24/7)
    print("📊 Scenario 1: 24/7 Customer Support Chatbot")
    chatbot_scenario = {
        "hours_per_day": 24,
        "days": 30,
        "requests_per_hour": 150
    }
    results = compare_gpus(chatbot_scenario)
    print(f"Cheapest: {results[0]['gpu_type']} at ${results[0]['costs']['total']}/month")
    print(f"Cost per request: ${results[0]['costs']['per_request']}")
    print()
    
    # Scenario 2: Business Hours Document Analysis
    print("📊 Scenario 2: Business Hours Document Analyzer (8hrs/day)")
    doc_scenario = {
        "hours_per_day": 8,
        "days": 22,  # Business days
        "requests_per_hour": 50
    }
    results = compare_gpus(doc_scenario)
    print(f"Cheapest: {results[0]['gpu_type']} at ${results[0]['costs']['total']}/month")
    print(f"Daily cost: ${results[0]['costs']['daily_avg']}")
    print()
    
    # Scenario 3: High-performance batch processing
    print("📊 Scenario 3: Batch Processing (4hrs/day, high performance)")
    batch_scenario = {
        "hours_per_day": 4,
        "days": 30,
        "requests_per_hour": 0
    }
    # For batch jobs, you might want the fastest GPU
    result = calculate_deployment_cost("H100", 4, 30)
    print(f"H100 (Fastest): ${result['costs']['total']}/month")
    result = calculate_deployment_cost("RTX-4090", 4, 30)
    print(f"RTX-4090 (Budget): ${result['costs']['total']}/month")
    print(f"Savings: ${510 - result['costs']['total']:.2f}")
    print()
    
    # Save full analysis
    full_analysis = {
        "scenarios": {
            "chatbot_24_7": compare_gpus(chatbot_scenario),
            "business_hours": compare_gpus(doc_scenario),
            "batch_processing": [
                calculate_deployment_cost("H100", 4, 30),
                calculate_deployment_cost("RTX-4090", 4, 30)
            ]
        },
        "generated": datetime.now().isoformat()
    }
    save_analysis(full_analysis, "gpu_cost_analysis.json")
```

**Run it:**
```bash
python3 gpu_calculator.py
```

**Why this matters:** Before deploying any AI application, you need to estimate costs. This tool helps you make informed decisions about GPU selection based on your actual usage patterns.

### Day 3-4: Building a Usage Tracker

**Real Use Case:** Track actual API usage and costs over time

**File:** `usage_tracker.py`

```python
"""
API Usage Tracker - Monitor real API calls and costs
Use case: Track your actual OpenAI/RunPod API usage to stay within budget
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict

class UsageTracker:
    def __init__(self, data_file: str = "usage_data.json"):
        self.data_file = data_file
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Load existing usage data or create new"""
        if Path(self.data_file).exists():
            with open(self.data_file, "r") as f:
                return json.load(f)
        return {
            "api_calls": [],
            "monthly_limits": {
                "budget": 100.0,
                "max_calls": 10000
            }
        }
    
    def _save_data(self):
        """Persist data to disk"""
        with open(self.data_file, "w") as f:
            json.dump(self.data, f, indent=2)
    
    def log_api_call(
        self,
        service: str,
        endpoint: str,
        tokens_used: int,
        cost: float,
        latency_ms: float,
        status: str = "success"
    ):
        """Log a single API call with all relevant metrics"""
        call = {
            "timestamp": datetime.now().isoformat(),
            "service": service,
            "endpoint": endpoint,
            "tokens_used": tokens_used,
            "cost": cost,
            "latency_ms": latency_ms,
            "status": status
        }
        self.data["api_calls"].append(call)
        self._save_data()
        
        # Check if approaching limits
        self._check_limits()
    
    def _check_limits(self):
        """Warn if approaching budget or rate limits"""
        monthly_cost = self.get_monthly_cost()
        monthly_calls = self.get_monthly_call_count()
        
        budget = self.data["monthly_limits"]["budget"]
        max_calls = self.data["monthly_limits"]["max_calls"]
        
        if monthly_cost >= budget * 0.9:
            print(f"⚠️  WARNING: 90% of monthly budget used (${monthly_cost:.2f}/${budget})")
        
        if monthly_calls >= max_calls * 0.9:
            print(f"⚠️  WARNING: 90% of call limit reached ({monthly_calls}/{max_calls})")
    
    def get_monthly_cost(self) -> float:
        """Calculate current month's total cost"""
        current_month = datetime.now().strftime("%Y-%m")
        monthly_calls = [
            call for call in self.data["api_calls"]
            if call["timestamp"].startswith(current_month)
        ]
        return sum(call["cost"] for call in monthly_calls)
    
    def get_monthly_call_count(self) -> int:
        """Count current month's API calls"""
        current_month = datetime.now().strftime("%Y-%m")
        return len([
            call for call in self.data["api_calls"]
            if call["timestamp"].startswith(current_month)
        ])
    
    def get_stats(self) -> Dict:
        """Calculate usage statistics"""
        if not self.data["api_calls"]:
            return {"error": "No data available"}
        
        calls = self.data["api_calls"]
        successful_calls = [c for c in calls if c["status"] == "success"]
        
        total_cost = sum(c["cost"] for c in calls)
        total_tokens = sum(c["tokens_used"] for c in calls)
        avg_latency = sum(c["latency_ms"] for c in calls) / len(calls)
        success_rate = len(successful_calls) / len(calls) * 100
        
        # Group by service
        by_service = {}
        for call in calls:
            service = call["service"]
            if service not in by_service:
                by_service[service] = {"calls": 0, "cost": 0, "tokens": 0}
            by_service[service]["calls"] += 1
            by_service[service]["cost"] += call["cost"]
            by_service[service]["tokens"] += call["tokens_used"]
        
        return {
            "total_calls": len(calls),
            "successful_calls": len(successful_calls),
            "success_rate": round(success_rate, 2),
            "total_cost": round(total_cost, 2),
            "total_tokens": total_tokens,
            "avg_latency_ms": round(avg_latency, 2),
            "monthly_cost": round(self.get_monthly_cost(), 2),
            "monthly_calls": self.get_monthly_call_count(),
            "by_service": by_service,
            "budget_remaining": round(
                self.data["monthly_limits"]["budget"] - self.get_monthly_cost(), 
                2
            )
        }
    
    def export_to_csv(self, filename: str = "usage_report.csv"):
        """Export usage data to CSV for analysis"""
        if not self.data["api_calls"]:
            print("No data to export")
            return
        
        with open(filename, "w", newline="") as f:
            fieldnames = ["timestamp", "service", "endpoint", "tokens_used", 
                         "cost", "latency_ms", "status"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.data["api_calls"])
        
        print(f"✓ Exported {len(self.data['api_calls'])} records to {filename}")
    
    def generate_report(self):
        """Print a formatted usage report"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("📊 API USAGE REPORT")
        print("="*60)
        print(f"Total API Calls: {stats['total_calls']}")
        print(f"Success Rate: {stats['success_rate']}%")
        print(f"Total Cost: ${stats['total_cost']}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"Avg Latency: {stats['avg_latency_ms']}ms")
        print()
        print("CURRENT MONTH:")
        print(f"  Calls: {stats['monthly_calls']}")
        print(f"  Cost: ${stats['monthly_cost']}")
        print(f"  Budget Remaining: ${stats['budget_remaining']}")
        print()
        print("BY SERVICE:")
        for service, data in stats['by_service'].items():
            print(f"  {service}:")
            print(f"    Calls: {data['calls']}")
            print(f"    Cost: ${data['cost']:.2f}")
            print(f"    Tokens: {data['tokens']:,}")
        print("="*60 + "\n")

# Example usage
if __name__ == "__main__":
    tracker = UsageTracker()
    
    # Simulate some API calls (in real usage, you'd call this from your API client)
    print("Logging sample API calls...\n")
    
    tracker.log_api_call(
        service="OpenAI",
        endpoint="gpt-4",
        tokens_used=1250,
        cost=0.0375,  # $0.03/1K tokens
        latency_ms=1250
    )
    
    tracker.log_api_call(
        service="OpenAI",
        endpoint="gpt-3.5-turbo",
        tokens_used=800,
        cost=0.0016,  # $0.002/1K tokens
        latency_ms=450
    )
    
    tracker.log_api_call(
        service="RunPod",
        endpoint="llama-70b",
        tokens_used=2000,
        cost=0.138,  # Based on GPU time
        latency_ms=2100
    )
    
    # Generate report
    tracker.generate_report()
    
    # Export to CSV
    tracker.export_to_csv()
```

**Why this matters:** Without tracking, you can easily overspend on API calls. This tool helps you monitor usage in real-time and stay within budget.

### Day 5-7: Project - Personal AI Budget Manager

**Real Use Case:** Complete tool to manage all your AI/cloud spending

**File:** `ai_budget_manager.py`

```python
"""
AI Budget Manager - Complete budget tracking and forecasting tool
Use case: Manage spending across multiple AI services (OpenAI, RunPod, AWS, etc.)
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List
from pathlib import Path

class AIBudgetManager:
    def __init__(self, config_file: str = "budget_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.expenses_file = "expenses.json"
        self.expenses = self._load_expenses()
    
    def _load_config(self) -> Dict:
        """Load budget configuration"""
        if Path(self.config_file).exists():
            with open(self.config_file, "r") as f:
                return json.load(f)
        
        # Default config
        return {
            "monthly_budget": 200.0,
            "services": {
                "OpenAI": {"budget": 100.0, "priority": "high"},
                "RunPod": {"budget": 50.0, "priority": "medium"},
                "AWS": {"budget": 30.0, "priority": "low"},
                "Other": {"budget": 20.0, "priority": "low"}
            },
            "alerts": {
                "warning_threshold": 0.80,  # Warn at 80%
                "critical_threshold": 0.95   # Critical at 95%
            }
        }
    
    def _load_expenses(self) -> List[Dict]:
        """Load expense history"""
        if Path(self.expenses_file).exists():
            with open(self.expenses_file, "r") as f:
                return json.load(f)
        return []
    
    def _save_expenses(self):
        """Save expenses to disk"""
        with open(self.expenses_file, "w") as f:
            json.dump(self.expenses, f, indent=2)
    
    def add_expense(
        self,
        service: str,
        amount: float,
        description: str,
        date: str = None
    ):
        """Add a new expense"""
        expense = {
            "id": len(self.expenses) + 1,
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "service": service,
            "amount": amount,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        self.expenses.append(expense)
        self._save_expenses()
        
        # Check budget after adding
        self._check_service_budget(service)
    
    def get_monthly_expenses(self, year_month: str = None) -> List[Dict]:
        """Get expenses for a specific month"""
        if not year_month:
            year_month = datetime.now().strftime("%Y-%m")
        
        return [
            exp for exp in self.expenses
            if exp["date"].startswith(year_month)
        ]
    
    def get_service_total(self, service: str, year_month: str = None) -> float:
        """Get total spending for a service"""
        monthly_expenses = self.get_monthly_expenses(year_month)
        service_expenses = [
            exp for exp in monthly_expenses
            if exp["service"] == service
        ]
        return sum(exp["amount"] for exp in service_expenses)
    
    def get_monthly_total(self, year_month: str = None) -> float:
        """Get total spending for the month"""
        monthly_expenses = self.get_monthly_expenses(year_month)
        return sum(exp["amount"] for exp in monthly_expenses)
    
    def _check_service_budget(self, service: str):
        """Check if service is approaching budget limit"""
        if service not in self.config["services"]:
            return
        
        service_config = self.config["services"][service]
        service_budget = service_config["budget"]
        current_spending = self.get_service_total(service)
        percentage = current_spending / service_budget
        
        alerts = self.config["alerts"]
        if percentage >= alerts["critical_threshold"]:
            print(f"🚨 CRITICAL: {service} at {percentage*100:.1f}% of budget!")
        elif percentage >= alerts["warning_threshold"]:
            print(f"⚠️  WARNING: {service} at {percentage*100:.1f}% of budget")
    
    def forecast_monthly_cost(self) -> Dict:
        """Forecast end-of-month costs based on current spending rate"""
        today = datetime.now()
        current_month = today.strftime("%Y-%m")
        day_of_month = today.day
        
        # Get days in current month
        if today.month == 12:
            next_month = datetime(today.year + 1, 1, 1)
        else:
            next_month = datetime(today.year, today.month + 1, 1)
        days_in_month = (next_month - datetime(today.year, today.month, 1)).days
        
        current_spending = self.get_monthly_total(current_month)
        daily_rate = current_spending / day_of_month if day_of_month > 0 else 0
        projected_total = daily_rate * days_in_month
        
        return {
            "current_spending": round(current_spending, 2),
            "daily_rate": round(daily_rate, 2),
            "projected_total": round(projected_total, 2),
            "budget": self.config["monthly_budget"],
            "projected_overage": round(
                max(0, projected_total - self.config["monthly_budget"]), 
                2
            ),
            "days_remaining": days_in_month - day_of_month
        }
    
    def generate_dashboard(self):
        """Generate a comprehensive budget dashboard"""
        print("\n" + "="*70)
        print("💰 AI BUDGET DASHBOARD")
        print("="*70)
        
        # Monthly overview
        monthly_total = self.get_monthly_total()
        monthly_budget = self.config["monthly_budget"]
        percentage = (monthly_total / monthly_budget) * 100
        
        print(f"\n📊 MONTHLY OVERVIEW ({datetime.now().strftime('%B %Y')})")
        print(f"  Current Spending: ${monthly_total:.2f}")
        print(f"  Monthly Budget: ${monthly_budget:.2f}")
        print(f"  Remaining: ${monthly_budget - monthly_total:.2f}")
        print(f"  Used: {percentage:.1f}%")
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * min(percentage / 100, 1))
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"  [{bar}] {percentage:.1f}%")
        
        # Forecast
        forecast = self.forecast_monthly_cost()
        print(f"\n🔮 FORECAST")
        print(f"  Daily Rate: ${forecast['daily_rate']:.2f}/day")
        print(f"  Projected Total: ${forecast['projected_total']:.2f}")
        if forecast['projected_overage'] > 0:
            print(f"  ⚠️  Projected Overage: ${forecast['projected_overage']:.2f}")
        else:
            print(f"  ✓ On track to stay within budget")
        
        # By service
        print(f"\n📋 BY SERVICE")
        for service, config in self.config["services"].items():
            service_total = self.get_service_total(service)
            service_budget = config["budget"]
            service_pct = (service_total / service_budget) * 100 if service_budget > 0 else 0
            
            status = "✓"
            if service_pct >= 95:
                status = "🚨"
            elif service_pct >= 80:
                status = "⚠️"
            
            print(f"  {status} {service}:")
            print(f"     ${service_total:.2f} / ${service_budget:.2f} ({service_pct:.1f}%)")
        
        # Recent expenses
        print(f"\n📝 RECENT EXPENSES (Last 5)")
        recent = sorted(self.expenses, key=lambda x: x["timestamp"], reverse=True)[:5]
        for exp in recent:
            print(f"  {exp['date']} - {exp['service']}: ${exp['amount']:.2f}")
            print(f"    → {exp['description']}")
        
        print("="*70 + "\n")

# Example usage
if __name__ == "__main__":
    manager = AIBudgetManager()
    
    # Add some sample expenses
    print("Adding sample expenses...\n")
    
    manager.add_expense(
        service="OpenAI",
        amount=45.50,
        description="GPT-4 API calls for chatbot development"
    )
    
    manager.add_expense(
        service="RunPod",
        amount=18.75,
        description="Llama-70B testing and development"
    )
    
    manager.add_expense(
        service="AWS",
        amount=12.30,
        description="S3 storage and Lambda functions"
    )
    
    manager.add_expense(
        service="OpenAI",
        amount=32.10,
        description="GPT-4 fine-tuning experiments"
    )
    
    # Generate dashboard
    manager.generate_dashboard()
```

**Usage in real scenarios:**
```bash
python3 ai_budget_manager.py
```

**Week 1 Milestone:** ✅ You now have production-ready tools that you can actually use to track and manage AI costs. These aren't toy examples - they're tools you'll use throughout this entire curriculum.

---

*Continue to [Part 2: Weeks 2-4](Part2) for Building Production APIs, Docker, and Cloud Deployment*