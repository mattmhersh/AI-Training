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

*Continue to Week 6 for building complete AI chat systems with conversation management, streaming, and analytics.*