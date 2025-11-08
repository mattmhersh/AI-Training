# AI Infrastructure & Engineering Curriculum
## Part 5: Capstone Projects (Weeks 17-20)

**Build complete production systems that showcase your skills**

---

## Table of Contents - Part 5

- [Overview](#overview)
- [Project 1: AI Customer Support Platform](#project-1-ai-customer-support-platform-weeks-17-18)
- [Project 2: AI Content Analysis Pipeline](#project-2-ai-content-analysis-pipeline-weeks-19-20)
- [Deployment & Portfolio](#deployment--portfolio)
- [Next Steps](#next-steps)

---

## Overview

These capstone projects integrate everything you've learned:
- Python programming
- API development
- Docker containerization
- Cloud deployment
- LLM integration
- RAG systems
- Monitoring & logging
- Cost optimization

Each project is portfolio-worthy and demonstrates real-world skills that employers value.

---

## Project 1: AI Customer Support Platform (Weeks 17-18)

**Goal:** Build a complete AI-powered customer support system

### Features

1. **Multi-Channel Support**
   - Web chat interface
   - Email integration
   - Slack/Discord bot
   - API for third-party integration

2. **Intelligent Routing**
   - AI determines ticket severity
   - Auto-routes to appropriate department
   - Escalates complex issues to humans
   - Tracks resolution time

3. **Knowledge Base Integration**
   - RAG-powered answers from documentation
   - Learns from past tickets
   - Suggests relevant KB articles
   - Updates KB based on common questions

4. **Analytics Dashboard**
   - Real-time metrics
   - Cost per ticket
   - Resolution rates
   - Customer satisfaction scores
   - Agent performance

### Architecture

```
┌─────────────┐
│  Web Chat   │
│   Widget    │
└──────┬──────┘
       │
┌──────▼──────┐     ┌───────────────┐
│   FastAPI   │────▶│  PostgreSQL   │
│   Backend   │     │   Database    │
└──────┬──────┘     └───────────────┘
       │
       ├────▶ LLM (RunPod)
       ├────▶ Vector DB (ChromaDB)
       ├────▶ Redis (Caching)
       └────▶ Monitoring (Logs)
```

### Week 17: Core Functionality

**Day 1-2: Database Design**

```python
# File: models.py
"""
Database models for support system
"""

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import enum

Base = declarative_base()

class TicketStatus(enum.Enum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"
    RESOLVED = "resolved"
    CLOSED = "closed"

class TicketPriority(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class Ticket(Base):
    __tablename__ = "tickets"
    
    id = Column(Integer, primary_key=True)
    ticket_number = Column(String(20), unique=True, nullable=False)
    customer_email = Column(String(255), nullable=False)
    customer_name = Column(String(255))
    subject = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(Enum(TicketStatus), default=TicketStatus.NEW)
    priority = Column(Enum(TicketPriority), default=TicketPriority.MEDIUM)
    category = Column(String(100))
    assigned_to = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime)
    ai_cost = Column(Float, default=0.0)
    
class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True)
    ticket_id = Column(Integer, nullable=False)
    sender_type = Column(String(20), nullable=False)  # customer, agent, ai
    sender_name = Column(String(255))
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    ai_generated = Column(Integer, default=0)  # Boolean
    ai_confidence = Column(Float)
    
class KnowledgeArticle(Base):
    __tablename__ = "knowledge_articles"
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(100))
    views = Column(Integer, default=0)
    helpful_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Initialize database
def init_db():
    engine = create_engine('sqlite:///support.db')  # Use PostgreSQL in production
    Base.metadata.create_all(engine)
    return engine

if __name__ == "__main__":
    init_db()
    print("✓ Database initialized")
```

**Day 3-4: Ticket Management API**

```python
# File: support_api.py
"""
Customer Support API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime
import uuid
from sqlalchemy.orm import sessionmaker
from models import Ticket, Message, TicketStatus, TicketPriority, init_db

app = FastAPI(title="AI Support Platform", version="1.0.0")

# Database setup
engine = init_db()
SessionLocal = sessionmaker(bind=engine)

class TicketCreate(BaseModel):
    customer_email: EmailStr
    customer_name: str
    subject: str
    description: str

class TicketResponse(BaseModel):
    ticket_number: str
    status: str
    priority: str
    subject: str
    created_at: datetime
    estimated_response_time: str

class MessageCreate(BaseModel):
    ticket_number: str
    content: str

@app.post("/tickets", response_model=TicketResponse)
async def create_ticket(
    ticket: TicketCreate,
    background_tasks: BackgroundTasks
):
    """
    Create a new support ticket
    
    Real use case: Customer submits issue via web form
    """
    db = SessionLocal()
    
    try:
        # Generate ticket number
        ticket_number = f"SUP-{uuid.uuid4().hex[:8].upper()}"
        
        # Create ticket
        new_ticket = Ticket(
            ticket_number=ticket_number,
            customer_email=ticket.customer_email,
            customer_name=ticket.customer_name,
            subject=ticket.subject,
            description=ticket.description,
            status=TicketStatus.NEW
        )
        
        db.add(new_ticket)
        db.commit()
        db.refresh(new_ticket)
        
        # Analyze ticket in background
        background_tasks.add_task(
            analyze_and_route_ticket,
            new_ticket.id,
            ticket.description
        )
        
        return TicketResponse(
            ticket_number=ticket_number,
            status="new",
            priority="medium",
            subject=ticket.subject,
            created_at=new_ticket.created_at,
            estimated_response_time="< 2 hours"
        )
    
    finally:
        db.close()

async def analyze_and_route_ticket(ticket_id: int, description: str):
    """
    Analyze ticket and determine priority/category using AI
    """
    from cached_llm_client import CachedLLMClient
    import os
    
    llm = CachedLLMClient(
        endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"),
        api_key=os.getenv("RUNPOD_API_KEY")
    )
    
    prompt = f"""Analyze this customer support ticket and provide:
1. Priority (LOW, MEDIUM, HIGH, URGENT)
2. Category (billing, technical, account, feature_request, bug)
3. Sentiment (positive, neutral, negative)

Ticket: {description}

Respond in format: PRIORITY|CATEGORY|SENTIMENT"""
    
    result = llm.generate(
        prompt=prompt,
        max_tokens=50,
        temperature=0.3
    )
    
    if result["success"]:
        try:
            priority, category, sentiment = result["text"].strip().split("|")
            
            # Update ticket
            db = SessionLocal()
            ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
            if ticket:
                ticket.priority = TicketPriority[priority]
                ticket.category = category.lower()
                ticket.ai_cost += result["cost"]
                db.commit()
            db.close()
            
        except Exception as e:
            print(f"Error parsing AI response: {e}")

@app.post("/tickets/{ticket_number}/messages")
async def add_message(ticket_number: str, message: MessageCreate):
    """
    Add message to ticket and get AI response
    """
    db = SessionLocal()
    
    try:
        # Get ticket
        ticket = db.query(Ticket).filter(
            Ticket.ticket_number == ticket_number
        ).first()
        
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Add customer message
        customer_msg = Message(
            ticket_id=ticket.id,
            sender_type="customer",
            sender_name=ticket.customer_name,
            content=message.content,
            ai_generated=False
        )
        db.add(customer_msg)
        
        # Generate AI response using RAG
        from document_qa import DocumentQA
        from cached_llm_client import CachedLLMClient
        import os
        
        rag = DocumentQA()
        llm = CachedLLMClient(
            endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"),
            api_key=os.getenv("RUNPOD_API_KEY")
        )
        
        ai_response = rag.answer_question(
            question=message.content,
            llm_client=llm,
            n_results=3
        )
        
        if ai_response["success"]:
            # Add AI response
            ai_msg = Message(
                ticket_id=ticket.id,
                sender_type="ai",
                sender_name="AI Assistant",
                content=ai_response["answer"],
                ai_generated=True,
                ai_confidence=0.85
            )
            db.add(ai_msg)
            ticket.ai_cost += ai_response["cost"]
            ticket.updated_at = datetime.utcnow()
            
        db.commit()
        
        return {
            "status": "success",
            "ai_response": ai_response["answer"] if ai_response["success"] else None
        }
    
    finally:
        db.close()

@app.get("/tickets/{ticket_number}")
async def get_ticket(ticket_number: str):
    """Get ticket details with message history"""
    db = SessionLocal()
    
    try:
        ticket = db.query(Ticket).filter(
            Ticket.ticket_number == ticket_number
        ).first()
        
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        messages = db.query(Message).filter(
            Message.ticket_id == ticket.id
        ).order_by(Message.created_at).all()
        
        return {
            "ticket": {
                "ticket_number": ticket.ticket_number,
                "subject": ticket.subject,
                "status": ticket.status.value,
                "priority": ticket.priority.value,
                "created_at": ticket.created_at,
                "ai_cost": ticket.ai_cost
            },
            "messages": [
                {
                    "sender_type": msg.sender_type,
                    "sender_name": msg.sender_name,
                    "content": msg.content,
                    "created_at": msg.created_at,
                    "ai_generated": bool(msg.ai_generated)
                }
                for msg in messages
            ]
        }
    
    finally:
        db.close()

@app.get("/analytics/dashboard")
async def get_dashboard():
    """Get support analytics dashboard"""
    db = SessionLocal()
    
    try:
        # Calculate metrics
        total_tickets = db.query(Ticket).count()
        open_tickets = db.query(Ticket).filter(
            Ticket.status.in_([TicketStatus.NEW, TicketStatus.IN_PROGRESS])
        ).count()
        resolved_tickets = db.query(Ticket).filter(
            Ticket.status == TicketStatus.RESOLVED
        ).count()
        
        # Calculate AI cost savings
        total_ai_cost = db.query(Ticket).with_entities(
            db.func.sum(Ticket.ai_cost)
        ).scalar() or 0
        
        # Estimate human cost saved (assume $15/hour, 10 min per ticket)
        human_cost_avoided = total_tickets * (15 / 6)
        savings = human_cost_avoided - total_ai_cost
        
        return {
            "tickets": {
                "total": total_tickets,
                "open": open_tickets,
                "resolved": resolved_tickets,
                "resolution_rate": round(resolved_tickets / total_tickets * 100, 1) if total_tickets > 0 else 0
            },
            "costs": {
                "ai_cost": round(total_ai_cost, 2),
                "human_cost_avoided": round(human_cost_avoided, 2),
                "savings": round(savings, 2),
                "roi": round(savings / total_ai_cost * 100, 0) if total_ai_cost > 0 else 0
            }
        }
    
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Day 5-7: Web Interface**

```html
<!-- File: index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Support Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50">
    <div class="container mx-auto px-4 py-8 max-w-4xl">
        <!-- Header -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h1 class="text-3xl font-bold text-gray-800">🤖 AI Support Platform</h1>
            <p class="text-gray-600 mt-2">Get instant help from our AI assistant</p>
        </div>
        
        <!-- New Ticket Form -->
        <div id="newTicketForm" class="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 class="text-2xl font-semibold mb-4">Submit a Support Request</h2>
            
            <form onsubmit="createTicket(event)">
                <div class="mb-4">
                    <label class="block text-gray-700 mb-2">Your Name</label>
                    <input type="text" id="customerName" required
                           class="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-blue-500">
                </div>
                
                <div class="mb-4">
                    <label class="block text-gray-700 mb-2">Email</label>
                    <input type="email" id="customerEmail" required
                           class="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-blue-500">
                </div>
                
                <div class="mb-4">
                    <label class="block text-gray-700 mb-2">Subject</label>
                    <input type="text" id="subject" required
                           class="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-blue-500">
                </div>
                
                <div class="mb-4">
                    <label class="block text-gray-700 mb-2">Description</label>
                    <textarea id="description" rows="5" required
                              class="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-blue-500"></textarea>
                </div>
                
                <button type="submit" 
                        class="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition">
                    Submit Request
                </button>
            </form>
        </div>
        
        <!-- Ticket View (hidden initially) -->
        <div id="ticketView" class="hidden bg-white rounded-lg shadow-lg p-6">
            <div class="flex justify-between items-center mb-4">
                <div>
                    <h2 class="text-2xl font-semibold">Ticket <span id="ticketNumber"></span></h2>
                    <p class="text-gray-600" id="ticketSubject"></p>
                </div>
                <span id="ticketStatus" class="px-4 py-2 rounded-full text-sm font-semibold"></span>
            </div>
            
            <!-- Messages -->
            <div id="messages" class="space-y-4 mb-6 max-h-96 overflow-y-auto"></div>
            
            <!-- Reply Form -->
            <form onsubmit="sendMessage(event)" class="flex gap-2">
                <input type="text" id="messageContent" placeholder="Type your message..." required
                       class="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:border-blue-500">
                <button type="submit" 
                        class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition">
                    Send
                </button>
            </form>
        </div>
    </div>
    
    <script>
        let currentTicketNumber = null;
        
        async function createTicket(event) {
            event.preventDefault();
            
            const data = {
                customer_name: document.getElementById('customerName').value,
                customer_email: document.getElementById('customerEmail').value,
                subject: document.getElementById('subject').value,
                description: document.getElementById('description').value
            };
            
            try {
                const response = await fetch('http://localhost:8000/tickets', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                currentTicketNumber = result.ticket_number;
                
                // Show ticket view
                document.getElementById('newTicketForm').classList.add('hidden');
                document.getElementById('ticketView').classList.remove('hidden');
                document.getElementById('ticketNumber').textContent = result.ticket_number;
                document.getElementById('ticketSubject').textContent = result.subject;
                document.getElementById('ticketStatus').textContent = result.status.toUpperCase();
                document.getElementById('ticketStatus').className = 'px-4 py-2 rounded-full text-sm font-semibold bg-yellow-100 text-yellow-800';
                
                // Load ticket details
                loadTicket(result.ticket_number);
                
            } catch (error) {
                alert('Error creating ticket: ' + error.message);
            }
        }
        
        async function loadTicket(ticketNumber) {
            try {
                const response = await fetch(`http://localhost:8000/tickets/${ticketNumber}`);
                const data = await response.json();
                
                // Display messages
                const messagesDiv = document.getElementById('messages');
                messagesDiv.innerHTML = '';
                
                data.messages.forEach(msg => {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = msg.sender_type === 'customer' 
                        ? 'bg-blue-50 p-4 rounded-lg' 
                        : 'bg-gray-50 p-4 rounded-lg';
                    
                    const senderIcon = msg.sender_type === 'ai' ? '🤖' : '👤';
                    
                    messageDiv.innerHTML = `
                        <div class="flex items-start gap-3">
                            <span class="text-2xl">${senderIcon}</span>
                            <div class="flex-1">
                                <div class="font-semibold text-gray-800">${msg.sender_name}</div>
                                <div class="text-gray-700 mt-1">${msg.content}</div>
                                <div class="text-xs text-gray-500 mt-2">${new Date(msg.created_at).toLocaleString()}</div>
                            </div>
                        </div>
                    `;
                    
                    messagesDiv.appendChild(messageDiv);
                });
                
                // Scroll to bottom
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                
            } catch (error) {
                console.error('Error loading ticket:', error);
            }
        }
        
        async function sendMessage(event) {
            event.preventDefault();
            
            const content = document.getElementById('messageContent').value;
            
            try {
                const response = await fetch(`http://localhost:8000/tickets/${currentTicketNumber}/messages`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        ticket_number: currentTicketNumber,
                        content: content
                    })
                });
                
                document.getElementById('messageContent').value = '';
                
                // Reload messages
                setTimeout(() => loadTicket(currentTicketNumber), 1000);
                
            } catch (error) {
                alert('Error sending message: ' + error.message);
            }
        }
    </script>
</body>
</html>
```

### Week 18: Polish & Deploy

**Day 1-2:** Add email notifications, Slack integration  
**Day 3-4:** Deploy to DigitalOcean with Docker Compose  
**Day 5-7:** Testing, documentation, demo video

---

## Project 2: AI Content Analysis Pipeline (Weeks 19-20)

**Goal:** Build a system that analyzes content at scale

### Features

1. **Multi-Format Support**
   - Documents (PDF, Word, Text)
   - Images (with OCR)
   - Videos (transcript generation)
   - Web pages (scraping)

2. **Analysis Capabilities**
   - Sentiment analysis
   - Topic extraction
   - Summary generation
   - Key insights identification
   - Compliance checking

3. **Batch Processing**
   - Queue-based processing
   - Progress tracking
   - Result export (CSV, JSON, PDF reports)

4. **API & Dashboard**
   - Upload interface
   - Real-time progress
   - Analytics visualization
   - Cost tracking

### Architecture

```
Upload → Queue → Worker Pool → LLM → Results DB → Dashboard
   ↓        ↓          ↓          ↓         ↓          ↓
  S3   →  Redis  →  Docker  → RunPod →  Postgres → Charts
```

[Continue with detailed implementation...]

---

## Deployment & Portfolio

### Docker Compose Setup

```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "80:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/support
      - REDIS_URL=redis://redis:6379
      - RUNPOD_API_KEY=${RUNPOD_API_KEY}
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=support
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Portfolio Presentation

**GitHub Repository Structure:**
```
project-name/
├── README.md (with demo GIF)
├── docs/
│   ├── architecture.md
│   ├── api-documentation.md
│   └── deployment-guide.md
├── src/
├── tests/
├── docker-compose.yml
└── .env.example
```

**README Template:**
```markdown
# AI Customer Support Platform

Production-ready AI-powered customer support system with RAG, multi-channel support, and analytics.

## 🚀 Features
- AI-powered ticket routing and responses
- RAG-based knowledge base integration
- Real-time analytics dashboard
- 80% cost savings vs human agents

## 📊 Tech Stack
- Python 3.11, FastAPI, PostgreSQL
- RunPod (LLM), ChromaDB (Vector DB)
- Docker, DigitalOcean
- React (frontend)

## 🎯 Metrics
- <2s avg response time
- 85% customer satisfaction
- $0.02 avg cost per ticket
- 500+ ROI

## 🛠️ Setup
[Installation instructions]

## 📸 Screenshots
[Demo GIFs and images]

## 📈 Performance
[Benchmarks and cost analysis]
```

---

## Congratulations! 🎉

**You've completed the entire AI Infrastructure & Engineering Curriculum!**

### What You've Accomplished:

✅ Built 2 production-ready AI applications  
✅ Deployed to cloud infrastructure  
✅ Implemented monitoring and cost tracking  
✅ Created portfolio-worthy projects  
✅ Mastered AI engineering fundamentals  

### Your Skills:
- Python development
- API design & implementation
- Docker & containerization
- Cloud deployment (DigitalOcean)
- LLM integration & optimization
- RAG systems
- Database design
- Cost optimization
- Production monitoring

### Total Investment:
- **Time:** 20 weeks (140-200 hours)
- **Money:** $200-300 total
- **Value:** Enterprise AI engineering skills

### What's Next?

1. **Job Applications:** You're ready for AI Engineer, ML Ops, or Backend roles
2. **Freelancing:** Build custom AI solutions for clients
3. **Startups:** Launch your own AI SaaS product
4. **Continue Learning:** Fine-tuning, multi-agent systems, advanced optimization

### Resources for Continued Growth:
- Join AI/ML communities (Discord, Reddit)
- Contribute to open-source AI projects
- Build your personal brand (blog, YouTube, Twitter)
- Stay updated (follow AI researchers, read papers)

---

**You're now an AI Infrastructure Engineer!** 🚀

Keep building, keep learning, and most importantly - use these skills to create value in the world. Good luck!