# AI Infrastructure & Engineering Curriculum
## Part 3: AI Integration (Weeks 5-8)

**Build real AI-powered applications**

---

## Table of Contents - Part 3

- [Week 5: LLM Deployment & Cost Optimization](#week-5-llm-deployment--cost-optimization)
- [Week 6: Building AI Chat Systems](#week-6-building-ai-chat-systems)
- [Week 7: Vector Databases & RAG](#week-7-vector-databases--rag)
- [Week 8: Production ML Ops](#week-8-production-ml-ops)

---

## Week 5: LLM Deployment & Cost Optimization

**Real-World Goal:** Deploy a production LLM and build cost-effective inference systems

### Day 1-2: RunPod Setup

**Why RunPod?**
- Serverless = Pay only for active inference time
- RTX 4090: Best price/performance for 70B models
- Auto-scaling from 0 workers = $0 when idle

**Setup Steps:**

1. Create account at [runpod.io](https://runpod.io)
2. Add $20-30 credits (minimum)
3. Navigate to: **Serverless → Templates**
4. Find: **"vLLM - Llama 3.1 70B Instruct"**
5. Configure:
   - **GPU:** RTX 4090 (24GB VRAM)
   - **Min Workers:** 0 (critical for cost savings!)
   - **Max Workers:** 1
   - **Idle Timeout:** 5 seconds
   - **Container Disk:** 50GB
6. Deploy and note your **Endpoint ID** and **API Key**

**Cost Breakdown:**
- RTX 4090: $0.69/hour
- Only charged during active inference
- 5-second idle timeout = container stops quickly
- Example: 100 requests @ 3sec each = $0.06 total

### Day 3-4: Production LLM Client

**File:** `llm_client.py`

```python
"""
Production LLM Client - Cost-optimized inference with RunPod
Real use case: Customer-facing chatbot that needs to be cost-effective
"""

import os
import requests
import time
from typing import Dict, List, Optional
from datetime import datetime
import json

class RunPodLLMClient:
    def __init__(self, endpoint_id: str, api_key: str):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.request_log = []
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        timeout: int = 300
    ) -> Dict:
        """
        Generate text with cost and performance tracking
        
        Real use case: Customer support chatbot response
        """
        start_time = time.time()
        
        # Format for Llama-3.1 chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "input": {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        }
        
        try:
            # Submit job (RunPod is async)
            response = requests.post(
                f"{self.base_url}/run",
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            job_data = response.json()
            job_id = job_data.get("id")
            
            # Poll for results
            result = self._poll_job(job_id, timeout)
            
            elapsed_time = time.time() - start_time
            
            # Extract response
            output = result.get("output", {})
            choices = output.get("choices", [])
            text = ""
            if choices:
                message = choices[0].get("message", {})
                text = message.get("content", "")
            
            # Calculate cost
            exec_time_ms = result.get("executionTime", elapsed_time * 1000)
            cost = self._calculate_cost(exec_time_ms)
            
            # Get token usage
            usage = output.get("usage", {})
            
            # Log request
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "job_id": job_id,
                "prompt_length": len(prompt),
                "response_length": len(text),
                "max_tokens": max_tokens,
                "elapsed_time": round(elapsed_time, 2),
                "execution_time": round(exec_time_ms / 1000, 2),
                "cost": cost,
                "tokens": usage,
                "status": "success"
            }
            self.request_log.append(log_entry)
            
            return {
                "success": True,
                "text": text,
                "job_id": job_id,
                "elapsed_time": elapsed_time,
                "execution_time": exec_time_ms / 1000,
                "cost": cost,
                "tokens": usage,
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "elapsed_time": elapsed_time,
                "status": "failed",
                "error": str(e)
            }
            self.request_log.append(log_entry)
            
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": elapsed_time
            }
    
    def _poll_job(self, job_id: str, timeout: int = 300) -> Dict:
        """Poll job status until complete"""
        start = time.time()
        poll_interval = 1  # Start with 1 second
        
        while time.time() - start < timeout:
            response = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=self.headers
            )
            response.raise_for_status()
            
            data = response.json()
            status = data.get("status")
            
            if status == "COMPLETED":
                return data
            elif status == "FAILED":
                error_msg = data.get("error", "Unknown error")
                raise Exception(f"Job failed: {error_msg}")
            elif status == "CANCELLED":
                raise Exception("Job was cancelled")
            
            # Adaptive polling - increase interval as we wait longer
            if time.time() - start > 10:
                poll_interval = 2
            if time.time() - start > 30:
                poll_interval = 3
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Job {job_id} timed out after {timeout}s")
    
    def _calculate_cost(self, exec_time_ms: float) -> float:
        """
        Calculate actual cost based on RunPod pricing
        RTX 4090: $0.69/hour
        """
        GPU_RATE_PER_HOUR = 0.69
        
        # Convert milliseconds to hours
        hours = (exec_time_ms / 1000) / 3600
        cost = hours * GPU_RATE_PER_HOUR
        
        return round(cost, 6)
    
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> List[Dict]:
        """
        Process multiple prompts efficiently
        
        Real use case: Batch processing of customer inquiries
        """
        results = []
        total_cost = 0
        total_time = 0
        
        print(f"\n🤖 Processing {len(prompts)} prompts...")
        print("="*60)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"[{i}/{len(prompts)}] ", end="", flush=True)
            
            result = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt
            )
            
            results.append(result)
            
            if result["success"]:
                total_cost += result["cost"]
                total_time += result["elapsed_time"]
                print(f"✓ {result['elapsed_time']:.1f}s | ${result['cost']:.4f}")
            else:
                print(f"✗ Error: {result['error']}")
        
        print("="*60)
        print(f"✅ Batch complete!")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Total cost: ${total_cost:.4f}")
        print(f"   Avg cost/request: ${total_cost/len(prompts):.4f}")
        
        return results
    
    def get_cost_report(self) -> Dict:
        """Generate comprehensive cost analysis"""
        if not self.request_log:
            return {"error": "No requests logged"}
        
        successful = [r for r in self.request_log if r.get("status") == "success"]
        failed = [r for r in self.request_log if r.get("status") == "failed"]
        
        if not successful:
            return {"error": "No successful requests"}
        
        total_cost = sum(r["cost"] for r in successful)
        total_time = sum(r["elapsed_time"] for r in successful)
        total_exec_time = sum(r["execution_time"] for r in successful)
        
        avg_cost = total_cost / len(successful)
        avg_time = total_time / len(successful)
        avg_exec_time = total_exec_time / len(successful)
        
        # Calculate tokens if available
        total_input_tokens = sum(r.get("tokens", {}).get("prompt_tokens", 0) for r in successful)
        total_output_tokens = sum(r.get("tokens", {}).get("completion_tokens", 0) for r in successful)
        
        return {
            "summary": {
                "total_requests": len(self.request_log),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": round(len(successful) / len(self.request_log) * 100, 2)
            },
            "costs": {
                "total": round(total_cost, 4),
                "average_per_request": round(avg_cost, 6),
                "projected_100_requests": round(avg_cost * 100, 2),
                "projected_1000_requests": round(avg_cost * 1000, 2),
                "projected_monthly_10k": round(avg_cost * 10000, 2)
            },
            "performance": {
                "total_time": round(total_time, 2),
                "average_time": round(avg_time, 2),
                "average_exec_time": round(avg_exec_time, 2),
                "requests_per_minute": round(60 / avg_time if avg_time > 0 else 0, 2)
            },
            "tokens": {
                "total_input": total_input_tokens,
                "total_output": total_output_tokens,
                "avg_input": round(total_input_tokens / len(successful), 0) if successful else 0,
                "avg_output": round(total_output_tokens / len(successful), 0) if successful else 0
            }
        }
    
    def save_report(self, filename: str = "llm_cost_report.json"):
        """Save detailed cost report"""
        report = {
            "generated": datetime.now().isoformat(),
            "endpoint_id": self.endpoint_id,
            "summary": self.get_cost_report(),
            "detailed_log": self.request_log
        }
        
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved to {filename}")

# Example: Customer Support Use Case
if __name__ == "__main__":
    # Initialize client
    client = RunPodLLMClient(
        endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"),
        api_key=os.getenv("RUNPOD_API_KEY")
    )
    
    # Real-world scenario: Customer support responses
    system_prompt = """You are a helpful customer support agent for TechStart Inc, 
a SaaS company providing document processing APIs. Be professional, concise, 
and solution-oriented. Keep responses under 100 words."""
    
    customer_inquiries = [
        "How do I reset my password?",
        "What are your pricing plans and what's included?",
        "Can I upgrade my account mid-month and get prorated billing?",
        "How do I export all my data from your platform?",
        "Do you offer refunds if I'm not satisfied with the service?",
        "What's the maximum file size I can upload?",
        "How long do you retain my documents on your servers?",
        "Can I integrate your API with Zapier or Make.com?"
    ]
    
    print("🎯 Real-World Use Case: Customer Support Automation")
    print("="*60)
    
    # Process inquiries
    results = client.batch_generate(
        prompts=customer_inquiries,
        max_tokens=150,
        temperature=0.7,
        system_prompt=system_prompt
    )
    
    # Display results
    print("\n📝 RESPONSES:")
    print("="*60)
    for i, (inquiry, result) in enumerate(zip(customer_inquiries, results), 1):
        print(f"\n{i}. Customer: \"{inquiry}\"")
        if result["success"]:
            print(f"   Agent: {result['text']}")
            print(f"   ⏱ {result['elapsed_time']:.2f}s | 💰 ${result['cost']:.4f}")
        else:
            print(f"   ✗ Error: {result['error']}")
    
    # Cost analysis
    print("\n" + "="*60)
    print("💰 COST ANALYSIS")
    print("="*60)
    report = client.get_cost_report()
    
    print(f"\n📊 Summary:")
    print(f"   Total Requests: {report['summary']['total_requests']}")
    print(f"   Success Rate: {report['summary']['success_rate']}%")
    
    print(f"\n💵 Costs:")
    print(f"   Total: ${report['costs']['total']}")
    print(f"   Avg/Request: ${report['costs']['average_per_request']}")
    print(f"   Projected 1,000 requests: ${report['costs']['projected_1000_requests']}")
    print(f"   Projected 10,000/month: ${report['costs']['projected_monthly_10k']}")
    
    print(f"\n⚡ Performance:")
    print(f"   Avg Response Time: {report['performance']['average_time']:.2f}s")
    print(f"   Avg Execution Time: {report['performance']['average_exec_time']:.2f}s")
    print(f"   Throughput: {report['performance']['requests_per_minute']:.1f} req/min")
    
    print(f"\n🔤 Tokens:")
    print(f"   Avg Input: {report['tokens']['avg_input']}")
    print(f"   Avg Output: {report['tokens']['avg_output']}")
    
    # Save report
    client.save_report()
    
    # ROI Calculation
    print("\n" + "="*60)
    print("💡 ROI ANALYSIS")
    print("="*60)
    avg_cost = report['costs']['average_per_request']
    print(f"Cost per support ticket: ${avg_cost:.4f}")
    print(f"vs. Human agent ($15/hour, 6 tickets/hour): ${15/6:.2f}")
    print(f"Savings per ticket: ${15/6 - avg_cost:.2f}")
    print(f"Savings on 1,000 tickets/month: ${(15/6 - avg_cost) * 1000:.2f}")
```

**Run it:**
```bash
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="your-api-key"
python3 llm_client.py
```

### Day 5-7: Smart Caching System

**File:** `cached_llm_client.py`

```python
"""
Cached LLM Client - Reduce costs with intelligent caching
Real use case: FAQ bots where same questions are asked repeatedly
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Dict
from llm_client import RunPodLLMClient

class CachedLLMClient(RunPodLLMClient):
    def __init__(self, endpoint_id: str, api_key: str, cache_dir: str = "llm_cache"):
        super().__init__(endpoint_id, api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_hits = 0
        self.cache_misses = 0
        self.cost_saved = 0
    
    def _get_cache_key(self, prompt: str, max_tokens: int, temperature: float, system_prompt: str) -> str:
        """Generate unique cache key based on inputs"""
        content = f"{prompt}_{max_tokens}_{temperature}_{system_prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str, max_age_hours: int = 24) -> Optional[Dict]:
        """Retrieve cached response if exists and not too old"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        # Check cache age
        cache_age = time.time() - cache_file.stat().st_mtime
        if cache_age > max_age_hours * 3600:
            # Cache too old, delete it
            cache_file.unlink()
            return None
        
        with open(cache_file, "r") as f:
            return json.load(f)
    
    def _save_to_cache(self, cache_key: str, response: Dict):
        """Save response to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Add cache metadata
        response["_cache_metadata"] = {
            "cached_at": datetime.now().isoformat(),
            "cache_key": cache_key
        }
        
        with open(cache_file, "w") as f:
            json.dump(response, f, indent=2)
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
        cache_max_age_hours: int = 24,
        timeout: int = 300
    ) -> Dict:
        """Generate with caching support"""
        
        # Generate cache key
        cache_key = self._get_cache_key(
            prompt, 
            max_tokens, 
            temperature, 
            system_prompt or ""
        )
        
        # Try cache first
        if use_cache:
            cached = self._get_cached_response(cache_key, cache_max_age_hours)
            if cached:
                self.cache_hits += 1
                
                # Estimate cost saved (average cost)
                estimated_cost = 0.01  # Average cost per request
                self.cost_saved += estimated_cost
                
                # Calculate cache age
                cache_time = cached.get("_cache_metadata", {}).get("cached_at", "")
                cache_age_hours = 0
                if cache_time:
                    cache_dt = datetime.fromisoformat(cache_time)
                    cache_age_hours = (datetime.now() - cache_dt).total_seconds() / 3600
                
                # Add cache info to response
                cached["cached"] = True
                cached["cache_age_hours"] = round(cache_age_hours, 2)
                cached["cost_saved"] = estimated_cost
                
                return cached
        
        # Cache miss - make actual request
        self.cache_misses += 1
        response = super().generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            timeout=timeout
        )
        
        # Save to cache if successful
        if response["success"] and use_cache:
            self._save_to_cache(cache_key, response)
        
        response["cached"] = False
        return response
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate cache size
        cache_files = list(self.cache_dir.glob("*.json"))
        cache_size_bytes = sum(f.stat().st_size for f in cache_files)
        cache_size_mb = cache_size_bytes / (1024 * 1024)
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total_requests,
            "hit_rate": round(hit_rate, 2),
            "cost_saved": round(self.cost_saved, 4),
            "cache_size_mb": round(cache_size_mb, 2),
            "cached_entries": len(cache_files)
        }
    
    def clear_cache(self, older_than_hours: Optional[int] = None):
        """Clear cache entries"""
        deleted = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            should_delete = False
            
            if older_than_hours:
                age = time.time() - cache_file.stat().st_mtime
                if age > older_than_hours * 3600:
                    should_delete = True
            else:
                should_delete = True
            
            if should_delete:
                cache_file.unlink()
                deleted += 1
        
        return {"deleted": deleted, "remaining": len(list(self.cache_dir.glob("*.json")))}
    
    def preload_faqs(self, faq_list: List[Dict]):
        """Preload common FAQ responses into cache"""
        print(f"🔄 Preloading {len(faq_list)} FAQs into cache...")
        
        for i, faq in enumerate(faq_list, 1):
            question = faq["question"]
            system_prompt = faq.get("system_prompt")
            
            print(f"[{i}/{len(faq_list)}] Caching: {question[:50]}...")
            
            self.generate(
                prompt=question,
                system_prompt=system_prompt,
                use_cache=True
            )
        
        print("✅ FAQ preloading complete!")

# Example: FAQ Bot with Caching
if __name__ == "__main__":
    import os
    from datetime import datetime
    
    client = CachedLLMClient(
        endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"),
        api_key=os.getenv("RUNPOD_API_KEY")
    )
    
    system_prompt = "You are a helpful FAQ bot for an e-commerce platform. Give concise, accurate answers."
    
    # Common FAQs that get asked repeatedly
    print("🤖 FAQ Bot Demo - Testing Cache Efficiency")
    print("="*60)
    print("Simulating customer inquiries over time...\n")
    
    inquiries = [
        "What are your shipping options?",
        "How do I track my order?",
        "What are your shipping options?",  # Duplicate
        "What is your return policy?",
        "Do you ship internationally?",
        "How do I track my order?",  # Duplicate
        "What are your shipping options?",  # Duplicate again
        "How long does shipping take?",
        "What is your return policy?",  # Duplicate
        "Can I cancel my order?",
        "What are your shipping options?",  # Duplicate
        "Do you offer gift wrapping?"
    ]
    
    for i, question in enumerate(inquiries, 1):
        print(f"\n[Request {i}] Customer: {question}")
        
        result = client.generate(
            prompt=question,
            max_tokens=150,
            system_prompt=system_prompt,
            cache_max_age_hours=24
        )
        
        if result["success"]:
            cached_icon = "💾" if result["cached"] else "🌐"
            cost_display = "FREE (cached)" if result["cached"] else f"${result['cost']:.4f}"
            
            print(f"  {cached_icon} Response: {result['text'][:80]}...")
            print(f"  Cost: {cost_display} | Time: {result['elapsed_time']:.2f}s")
            
            if result["cached"]:
                print(f"  Cache age: {result['cache_age_hours']:.1f} hours")
    
    # Show cache statistics
    print("\n" + "="*60)
    print("📊 CACHE PERFORMANCE REPORT")
    print("="*60)
    
    stats = client.get_cache_stats()
    print(f"\n📈 Statistics:")
    print(f"   Total Requests: {stats['total_requests']}")
    print(f"   Cache Hits: {stats['cache_hits']}")
    print(f"   Cache Misses: {stats['cache_misses']}")
    print(f"   Hit Rate: {stats['hit_rate']}%")
    print(f"   Cached Entries: {stats['cached_entries']}")
    print(f"   Cache Size: {stats['cache_size_mb']:.2f} MB")
    
    print(f"\n💰 Cost Savings:")
    print(f"   Cost Saved: ${stats['cost_saved']:.4f}")
    
    # Calculate what it would have cost without cache
    total_without_cache = stats['total_requests'] * 0.01
    total_with_cache = stats['cache_misses'] * 0.01
    savings_percent = ((total_without_cache - total_with_cache) / total_without_cache * 100)
    
    print(f"   Cost without cache: ${total_without_cache:.4f}")
    print(f"   Cost with cache: ${total_with_cache:.4f}")
    print(f"   Savings: {savings_percent:.1f}%")
    
    print(f"\n💡 Business Impact:")
    monthly_requests = 10000
    cost_without = monthly_requests * 0.01
    cost_with = monthly_requests * (1 - stats['hit_rate']/100) * 0.01
    monthly_savings = cost_without - cost_with
    
    print(f"   Projected 10,000 requests/month:")
    print(f"     Without cache: ${cost_without:.2f}")
    print(f"     With cache: ${cost_with:.2f}")
    print(f"     Monthly savings: ${monthly_savings:.2f}")
    print(f"     Annual savings: ${monthly_savings * 12:.2f}")
```

**Why caching matters:**
- FAQ bots: 70-80% cache hit rate typical
- Cost reduction: 70-80% lower inference costs
- Faster responses: Instant vs. 2-3 seconds
- Better UX: No waiting for common questions

**Week 5 Milestone:** ✅ You've deployed a production LLM, built cost-tracking systems, and implemented intelligent caching. You understand the real economics of AI inference and can optimize costs effectively.

---

## Week 6: Building AI Chat Systems

**Real-World Goal:** Build a complete conversational AI system with memory, streaming, and analytics

### Day 1-3: Multi-Turn Conversation System

**File:** `chat_api.py`

```python
"""
Production Chat API - Complete conversational AI system
Real use case: Customer support chatbot with memory and context
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import json
from pathlib import Path
import uuid
from cached_llm_client import CachedLLMClient
import os

app = FastAPI(
    title="AI Chat API",
    version="2.0.0",
    description="Production chat system with conversation management"
)

# Initialize LLM client
llm_client = CachedLLMClient(
    endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"),
    api_key=os.getenv("RUNPOD_API_KEY")
)

# Data storage
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    cost: Optional[float] = None
    cached: Optional[bool] = False

class Conversation(BaseModel):
    conversation_id: str
    user_id: str
    title: Optional[str] = None
    messages: List[Message] = []
    created_at: str
    updated_at: str
    total_cost: float = 0.0
    system_prompt: Optional[str] = None
    metadata: Dict = {}

class ChatRequest(BaseModel):
    user_id: str
    message: str
    conversation_id: Optional[str] = None
    system_prompt: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    cost: float
    elapsed_time: float
    cached: bool
    tokens: Dict

def load_conversation(conversation_id: str) -> Optional[Conversation]:
    """Load conversation from disk"""
    conv_file = CONVERSATIONS_DIR / f"{conversation_id}.json"
    
    if not conv_file.exists():
        return None
    
    with open(conv_file, "r") as f:
        data = json.load(f)
        return Conversation(**data)

def save_conversation(conversation: Conversation):
    """Save conversation to disk"""
    conv_file = CONVERSATIONS_DIR / f"{conversation.conversation_id}.json"
    
    with open(conv_file, "w") as f:
        json.dump(conversation.dict(), f, indent=2)

def format_conversation_for_llm(conversation: Conversation, max_messages: int = 10) -> str:
    """
    Convert conversation history to prompt format
    Keep only recent messages to avoid token limits
    """
    # Get last N messages
    recent_messages = conversation.messages[-max_messages:]
    
    # Format as chat history
    history_parts = []
    for msg in recent_messages:
        role = "User" if msg.role == "user" else "Assistant"
        history_parts.append(f"{role}: {msg.content}")
    
    return "\n".join(history_parts)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a chat message and get AI response
    
    Real use case: Customer sends message, AI responds with full conversation context
    """
    
    # Load or create conversation
    if request.conversation_id:
        conversation = load_conversation(request.conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Verify user owns conversation
        if conversation.user_id != request.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
    else:
        # New conversation
        conversation = Conversation(
            conversation_id=str(uuid.uuid4()),
            user_id=request.user_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            system_prompt=request.system_prompt
        )
    
    # Add user message to history
    user_message = Message(
        role="user",
        content=request.message,
        timestamp=datetime.now().isoformat()
    )
    conversation.messages.append(user_message)
    
    # Build prompt with conversation history
    context = format_conversation_for_llm(conversation)
    full_prompt = f"{context}\nUser: {request.message}\nAssistant:"
    
    # Generate response
    result = llm_client.generate(
        prompt=full_prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        system_prompt=conversation.system_prompt or request.system_prompt,
        use_cache=True
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=f"LLM error: {result.get('error')}")
    
    # Add assistant message to history
    assistant_message = Message(
        role="assistant",
        content=result["text"],
        timestamp=datetime.now().isoformat(),
        cost=result.get("cost", 0),
        cached=result.get("cached", False)
    )
    conversation.messages.append(assistant_message)
    
    # Update conversation metadata
    conversation.updated_at = datetime.now().isoformat()
    if not result.get("cached", False):
        conversation.total_cost += result.get("cost", 0)
    
    # Auto-generate title from first exchange
    if not conversation.title and len(conversation.messages) >= 2:
        conversation.title = request.message[:50] + ("..." if len(request.message) > 50 else "")
    
    # Save conversation
    save_conversation(conversation)
    
    return ChatResponse(
        conversation_id=conversation.conversation_id,
        message=result["text"],
        cost=result.get("cost", 0),
        elapsed_time=result["elapsed_time"],
        cached=result.get("cached", False),
        tokens=result.get("tokens", {})
    )

@app.get("/conversations/{user_id}")
async def list_conversations(user_id: str, limit: int = 50):
    """List all conversations for a user"""
    user_conversations = []
    
    for conv_file in CONVERSATIONS_DIR.glob("*.json"):
        with open(conv_file, "r") as f:
            conv_data = json.load(f)
            
            if conv_data["user_id"] == user_id:
                # Return summary only
                user_conversations.append({
                    "conversation_id": conv_data["conversation_id"],
                    "title": conv_data.get("title", "Untitled"),
                    "message_count": len(conv_data["messages"]),
                    "total_cost": conv_data["total_cost"],
                    "created_at": conv_data["created_at"],
                    "updated_at": conv_data["updated_at"]
                })
    
    # Sort by most recent
    user_conversations.sort(key=lambda x: x["updated_at"], reverse=True)
    
    return {
        "user_id": user_id,
        "conversation_count": len(user_conversations),
        "conversations": user_conversations[:limit]
    }

@app.get("/conversations/{user_id}/{conversation_id}")
async def get_conversation(user_id: str, conversation_id: str):
    """Get full conversation history"""
    conversation = load_conversation(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conversation.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return conversation

@app.delete("/conversations/{user_id}/{conversation_id}")
async def delete_conversation(user_id: str, conversation_id: str):
    """Delete a conversation"""
    conversation = load_conversation(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conversation.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    conv_file = CONVERSATIONS_DIR / f"{conversation_id}.json"
    conv_file.unlink()
    
    return {"status": "deleted", "conversation_id": conversation_id}

@app.get("/analytics/{user_id}")
async def get_user_analytics(user_id: str):
    """Get comprehensive usage analytics for a user"""
    user_conversations = []
    
    for conv_file in CONVERSATIONS_DIR.glob("*.json"):
        with open(conv_file, "r") as f:
            conv_data = json.load(f)
            if conv_data["user_id"] == user_id:
                user_conversations.append(conv_data)
    
    if not user_conversations:
        return {
            "user_id": user_id,
            "total_conversations": 0,
            "total_messages": 0,
            "total_cost": 0.0
        }
    
    # Calculate statistics
    total_messages = sum(len(c["messages"]) for c in user_conversations)
    total_cost = sum(c["total_cost"] for c in user_conversations)
    
    # Count cached responses
    cached_count = 0
    for conv in user_conversations:
        for msg in conv["messages"]:
            if msg.get("cached", False):
                cached_count += 1
    
    avg_messages_per_conv = total_messages / len(user_conversations)
    
    # Calculate costs
    user_messages = sum(1 for c in user_conversations for m in c["messages"] if m["role"] == "user")
    
    return {
        "user_id": user_id,
        "summary": {
            "total_conversations": len(user_conversations),
            "total_messages": total_messages,
            "user_messages": user_messages,
            "ai_messages": total_messages - user_messages,
            "avg_messages_per_conversation": round(avg_messages_per_conv, 2)
        },
        "costs": {
            "total_cost": round(total_cost, 4),
            "avg_cost_per_conversation": round(total_cost / len(user_conversations), 4),
            "avg_cost_per_message": round(total_cost / user_messages, 4) if user_messages > 0 else 0
        },
        "caching": {
            "cached_responses": cached_count,
            "cache_hit_rate": round(cached_count / (total_messages - user_messages) * 100, 2) if total_messages > user_messages else 0
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Day 4-5: Interactive Chat Client

**File:** `chat_client.py`

```python
"""
Interactive Chat Client - Test and use the chat API
"""

import requests
from typing import Optional
import json

class ChatClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.user_id = None
        self.current_conversation_id = None
    
    def send_message(
        self,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> dict:
        """Send a message and get response"""
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "user_id": user_id,
                "message": message,
                "conversation_id": conversation_id,
                "system_prompt": system_prompt
            }
        )
        return response.json()
    
    def list_conversations(self, user_id: str) -> dict:
        """List all conversations for a user"""
        response = requests.get(f"{self.base_url}/conversations/{user_id}")
        return response.json()
    
    def get_conversation(self, user_id: str, conversation_id: str) -> dict:
        """Get full conversation history"""
        response = requests.get(
            f"{self.base_url}/conversations/{user_id}/{conversation_id}"
        )
        return response.json()
    
    def delete_conversation(self, user_id: str, conversation_id: str) -> dict:
        """Delete a conversation"""
        response = requests.delete(
            f"{self.base_url}/conversations/{user_id}/{conversation_id}"
        )
        return response.json()
    
    def get_analytics(self, user_id: str) -> dict:
        """Get usage analytics"""
        response = requests.get(f"{self.base_url}/analytics/{user_id}")
        return response.json()
    
    def interactive_chat(self, user_id: str, system_prompt: Optional[str] = None):
        """Start an interactive chat session"""
        self.user_id = user_id
        
        print("\n" + "="*70)
        print("🤖 AI CHAT INTERFACE")
        print("="*70)
        print("\nCommands:")
        print("  'quit' - Exit chat")
        print("  'new' - Start new conversation")
        print("  'history' - View conversation list")
        print("  'analytics' - View usage statistics")
        print("\nType your message to chat!")
        print("="*70 + "\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'new':
                self.current_conversation_id = None
                print("✨ Started new conversation\n")
                continue
            
            if user_input.lower() == 'history':
                self._show_history()
                continue
            
            if user_input.lower() == 'analytics':
                self._show_analytics()
                continue
            
            # Send message
            try:
                result = self.send_message(
                    message=user_input,
                    user_id=self.user_id,
                    conversation_id=self.current_conversation_id,
                    system_prompt=system_prompt
                )
                
                # Update conversation ID
                self.current_conversation_id = result["conversation_id"]
                
                # Display response
                cached_icon = "💾" if result["cached"] else "🌐"
                print(f"\nAI {cached_icon}: {result['message']}")
                
                # Show metadata
                cost_str = "FREE" if result["cached"] else f"${result['cost']:.4f}"
                print(f"[{cost_str} | {result['elapsed_time']:.2f}s]\n")
                
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")
    
    def _show_history(self):
        """Display conversation history"""
        try:
            data = self.list_conversations(self.user_id)
            
            print("\n" + "="*70)
            print(f"📋 CONVERSATION HISTORY ({data['conversation_count']} total)")
            print("="*70)
            
            for i, conv in enumerate(data['conversations'][:10], 1):
                print(f"\n{i}. {conv['title']}")
                print(f"   ID: {conv['conversation_id'][:16]}...")
                print(f"   Messages: {conv['message_count']}")
                print(f"   Cost: ${conv['total_cost']:.4f}")
                print(f"   Last updated: {conv['updated_at'][:19]}")
            
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error loading history: {str(e)}\n")
    
    def _show_analytics(self):
        """Display usage analytics"""
        try:
            analytics = self.get_analytics(self.user_id)
            
            print("\n" + "="*70)
            print("📊 USAGE ANALYTICS")
            print("="*70)
            
            summary = analytics['summary']
            costs = analytics['costs']
            caching = analytics['caching']
            
            print(f"\n📈 Summary:")
            print(f"   Total Conversations: {summary['total_conversations']}")
            print(f"   Total Messages: {summary['total_messages']}")
            print(f"   Your Messages: {summary['user_messages']}")
            print(f"   AI Responses: {summary['ai_messages']}")
            print(f"   Avg Messages/Conv: {summary['avg_messages_per_conversation']}")
            
            print(f"\n💰 Costs:")
            print(f"   Total Spent: ${costs['total_cost']}")
            print(f"   Avg/Conversation: ${costs['avg_cost_per_conversation']}")
            print(f"   Avg/Message: ${costs['avg_cost_per_message']}")
            
            print(f"\n💾 Caching:")
            print(f"   Cached Responses: {caching['cached_responses']}")
            print(f"   Cache Hit Rate: {caching['cache_hit_rate']}%")
            
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error loading analytics: {str(e)}\n")

# Example usage scenarios
if __name__ == "__main__":
    client = ChatClient()
    
    # Scenario 1: Customer Support Conversation
    print("="*70)
    print("📝 Demo 1: Multi-Turn Customer Support Conversation")
    print("="*70)
    
    user_id = "customer_456"
    system_prompt = """You are a helpful customer support agent for an e-commerce platform.
Be professional, empathetic, and solution-oriented. Keep responses concise."""
    
    conversation = [
        "Hi, I haven't received my order yet. Order number is #12345",
        "I ordered it 10 days ago",
        "The tracking says it's in transit but hasn't moved in 3 days",
        "Yes, I'd appreciate a replacement. How long will that take?",
        "Perfect, thank you for your help!"
    ]
    
    conversation_id = None
    
    for msg in conversation:
        print(f"\nCustomer: {msg}")
        
        result = client.send_message(
            message=msg,
            user_id=user_id,
            conversation_id=conversation_id,
            system_prompt=system_prompt
        )
        
        conversation_id = result["conversation_id"]
        
        cached = "💾" if result["cached"] else "🌐"
        print(f"Agent {cached}: {result['message']}")
        print(f"[Cost: ${result['cost']:.4f} | Time: {result['elapsed_time']:.2f}s]")
    
    # Scenario 2: Analytics
    print("\n" + "="*70)
    print("📊 Demo 2: User Analytics")
    print("="*70)
    
    analytics = client.get_analytics(user_id)
    
    print(f"\nUser: {user_id}")
    print(f"Conversations: {analytics['summary']['total_conversations']}")
    print(f"Total Messages: {analytics['summary']['total_messages']}")
    print(f"Total Cost: ${analytics['costs']['total_cost']}")
    print(f"Cache Hit Rate: {analytics['caching']['cache_hit_rate']}%")
    
    # Scenario 3: Interactive Mode
    print("\n" + "="*70)
    print("💬 Demo 3: Interactive Chat Mode")
    print("="*70)
    print("\nStarting interactive session...")
    print("(In real use, you'd interact here. Skipping for demo.)")
    
    # Uncomment to try interactive mode:
    # client.interactive_chat(
    #     user_id="interactive_user",
    #     system_prompt="You are a friendly AI assistant."
    # )
```

### Day 6-7: Conversation Export & Analysis

**File:** `conversation_analyzer.py`

```python
"""
Conversation Analyzer - Export and analyze chat data
Real use case: Generate insights from customer conversations
"""

import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import csv

class ConversationAnalyzer:
    def __init__(self, conversations_dir: str = "conversations"):
        self.conversations_dir = Path(conversations_dir)
    
    def load_all_conversations(self, user_id: Optional[str] = None) -> List[Dict]:
        """Load all conversations, optionally filtered by user"""
        conversations = []
        
        for conv_file in self.conversations_dir.glob("*.json"):
            with open(conv_file, "r") as f:
                conv_data = json.load(f)
                
                if user_id is None or conv_data["user_id"] == user_id:
                    conversations.append(conv_data)
        
        return conversations
    
    def analyze_message_patterns(self, conversations: List[Dict]) -> Dict:
        """Analyze message patterns and timing"""
        if not conversations:
            return {"error": "No conversations to analyze"}
        
        message_lengths = []
        response_times = []
        costs = []
        
        for conv in conversations:
            messages = conv["messages"]
            
            for i in range(len(messages)):
                msg = messages[i]
                
                # Collect message lengths
                message_lengths.append(len(msg["content"]))
                
                # Collect costs (for AI messages)
                if msg["role"] == "assistant" and msg.get("cost"):
                    costs.append(msg["cost"])
                
                # Calculate response times
                if i > 0:
                    prev_msg = messages[i-1]
                    try:
                        prev_time = datetime.fromisoformat(prev_msg["timestamp"])
                        curr_time = datetime.fromisoformat(msg["timestamp"])
                        response_time = (curr_time - prev_time).total_seconds()
                        if response_time < 300:  # Less than 5 minutes
                            response_times.append(response_time)
                    except:
                        pass
        
        return {
            "message_statistics": {
                "total_messages": len(message_lengths),
                "avg_message_length": round(sum(message_lengths) / len(message_lengths), 0) if message_lengths else 0,
                "min_length": min(message_lengths) if message_lengths else 0,
                "max_length": max(message_lengths) if message_lengths else 0
            },
            "timing": {
                "avg_response_time": round(sum(response_times) / len(response_times), 2) if response_times else 0,
                "min_response_time": round(min(response_times), 2) if response_times else 0,
                "max_response_time": round(max(response_times), 2) if response_times else 0
            },
            "costs": {
                "total_ai_cost": round(sum(costs), 4),
                "avg_cost_per_response": round(sum(costs) / len(costs), 6) if costs else 0,
                "total_responses": len(costs)
            }
        }
    
    def export_to_csv(self, conversations: List[Dict], filename: str = "conversations_export.csv"):
        """Export conversations to CSV for further analysis"""
        rows = []
        
        for conv in conversations:
            for msg in conv["messages"]:
                rows.append({
                    "conversation_id": conv["conversation_id"],
                    "user_id": conv["user_id"],
                    "timestamp": msg["timestamp"],
                    "role": msg["role"],
                    "message_length": len(msg["content"]),
                    "cost": msg.get("cost", 0),
                    "cached": msg.get("cached", False)
                })
        
        with open(filename, "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        
        print(f"✓ Exported {len(rows)} messages to {filename}")
        return len(rows)
    
    def generate_report(self, user_id: Optional[str] = None):
        """Generate comprehensive analysis report"""
        conversations = self.load_all_conversations(user_id)
        
        if not conversations:
            print("No conversations found")
            return
        
        print("\n" + "="*70)
        print("📊 CONVERSATION ANALYSIS REPORT")
        print("="*70)
        
        if user_id:
            print(f"\nUser: {user_id}")
        else:
            print(f"\nAll Users")
        
        print(f"Total Conversations: {len(conversations)}")
        
        # Analyze patterns
        patterns = self.analyze_message_patterns(conversations)
        
        print(f"\n📝 Message Statistics:")
        stats = patterns["message_statistics"]
        print(f"   Total Messages: {stats['total_messages']}")
        print(f"   Avg Length: {stats['avg_message_length']} characters")
        print(f"   Range: {stats['min_length']} - {stats['max_length']} characters")
        
        print(f"\n⏱️  Timing:")
        timing = patterns["timing"]
        print(f"   Avg Response Time: {timing['avg_response_time']}s")
        print(f"   Range: {timing['min_response_time']}s - {timing['max_response_time']}s")
        
        print(f"\n💰 Costs:")
        costs = patterns["costs"]
        print(f"   Total AI Cost: ${costs['total_ai_cost']}")
        print(f"   Total Responses: {costs['total_responses']}")
        print(f"   Avg Cost/Response: ${costs['avg_cost_per_response']}")
        
        print("\n" + "="*70)

if __name__ == "__main__":
    analyzer = ConversationAnalyzer()
    
    # Generate report
    analyzer.generate_report()
    
    # Export data
    conversations = analyzer.load_all_conversations()
    analyzer.export_to_csv(conversations)
```

**Week 6 Milestone:** ✅ You've built a production-grade chat API with:
- Multi-turn conversation management
- Conversation history and context
- User analytics and insights
- Cost tracking per conversation
- Interactive chat interface
- Data export capabilities

This is the foundation for any conversational AI product (customer support, virtual assistants, chatbots).

---

*Continue to Week 7 for Vector Databases & RAG Systems*