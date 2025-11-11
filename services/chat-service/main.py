# services/chat-service/main.py
"""
Chat Service - AI Financial Advisor powered by Google Gemini
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uvicorn
import os
import httpx
import google.generativeai as genai
import json
import logging

from shared.database import get_db
from shared.security import SecurityManager
from shared.models import User

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    logger.info("✅ Gemini API configured successfully")
else:
    logger.warning("⚠️ GEMINI_API_KEY not found, using fallback responses")
    model = None

app = FastAPI(
    title="Chat Service - AI Financial Advisor",
    description="Get personalized financial advice powered by Google Gemini",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Store conversation history in memory (in production, use Redis or database)
conversation_store: Dict[str, List[Dict]] = {}

# Models
class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: datetime
    sources: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None

# Auth dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Verify JWT token and return current user"""
    token = credentials.credentials
    payload = SecurityManager.decode_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user = db.query(User).filter(User.id == int(payload.get("sub"))).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user

async def get_user_financial_context(user_id: int, token: str) -> Dict[str, Any]:
    """Fetch comprehensive financial context from all services"""
    context = {
        "user_id": user_id,
        "transactions": [],
        "budgets": [],
        "investments": [],
        "debts": [],
        "analytics": {},
        "profile": {}
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    timeout = httpx.Timeout(5.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            # Get user profile
            profile_response = await client.get(
                "http://user-service:8102/profile",
                headers=headers
            )
            if profile_response.status_code == 200:
                context['profile'] = profile_response.json()
                logger.info("✅ Fetched user profile")
        except Exception as e:
            logger.error(f"❌ Failed to fetch profile: {e}")
        
        try:
            # Get recent transactions
            trans_response = await client.get(
                "http://transaction-service:8103/transactions?limit=20",
                headers=headers
            )
            if trans_response.status_code == 200:
                context['transactions'] = trans_response.json()
                logger.info("✅ Fetched transactions")
        except Exception as e:
            logger.error(f"❌ Failed to fetch transactions: {e}")
        
        try:
            # Get budgets
            budget_response = await client.get(
                "http://budget-service:8104/budgets?active_only=true",
                headers=headers
            )
            if budget_response.status_code == 200:
                context['budgets'] = budget_response.json()
                logger.info("✅ Fetched budgets")
        except Exception as e:
            logger.error(f"❌ Failed to fetch budgets: {e}")
        
        try:
            # Get investments
            investment_response = await client.get(
                "http://investment-service:8105/investments/portfolio/summary",
                headers=headers
            )
            if investment_response.status_code == 200:
                context['investments'] = investment_response.json()
                logger.info("✅ Fetched investments")
        except Exception as e:
            logger.error(f"❌ Failed to fetch investments: {e}")
        
        try:
            # Get debts
            debt_response = await client.get(
                "http://debt-service:8106/debts/summary/overview",
                headers=headers
            )
            if debt_response.status_code == 200:
                context['debts'] = debt_response.json()
                logger.info("✅ Fetched debts")
        except Exception as e:
            logger.error(f"❌ Failed to fetch debts: {e}")
        
        try:
            # Get analytics dashboard
            analytics_response = await client.get(
                "http://analytics-service:8107/dashboard",
                headers=headers
            )
            if analytics_response.status_code == 200:
                context['analytics'] = analytics_response.json()
                logger.info("✅ Fetched analytics")
        except Exception as e:
            logger.error(f"❌ Failed to fetch analytics: {e}")
    
    return context

def build_gemini_prompt(user_message: str, context: Dict[str, Any], conversation_history: List[Dict]) -> str:
    """Build a comprehensive prompt for Gemini with financial context"""
    
    # Extract key financial data
    analytics = context.get('analytics', {})
    current_month = analytics.get('current_month', {})
    income = current_month.get('income', 0)
    expenses = current_month.get('expenses', 0)
    savings = current_month.get('savings', 0)
    
    budgets = context.get('budgets', [])
    transactions = context.get('transactions', [])
    investments = context.get('investments', {})
    debts = context.get('debts', {})
    profile = context.get('profile', {})
    
    # Build conversation history
    history_text = ""
    if conversation_history:
        for msg in conversation_history[-5:]:  # Last 5 messages
            history_text += f"User: {msg.get('user_message', '')}\n"
            history_text += f"Assistant: {msg.get('ai_response', '')}\n\n"
    
    # Build comprehensive prompt
    prompt = f"""You are an expert AI Financial Advisor helping a user with their personal finances. 

**Your Role:**
- Provide personalized financial advice based on the user's actual data
- Be friendly, encouraging, and practical
- Use specific numbers from their data when relevant
- Format currency in Indian Rupees (₹)
- Give actionable recommendations

**User's Financial Profile:**
Name: {profile.get('full_name', 'User')}
Email: {profile.get('email', 'N/A')}

**Current Month Financial Summary:**
- Total Income: ₹{income:,.2f}
- Total Expenses: ₹{expenses:,.2f}
- Savings: ₹{savings:,.2f}
- Savings Rate: {(savings/income*100) if income > 0 else 0:.1f}%

**Active Budgets:**
{json.dumps(budgets, indent=2) if budgets else "No active budgets set"}

**Recent Transactions (Last 20):**
{json.dumps(transactions[:5], indent=2) if transactions else "No recent transactions"}

**Investment Portfolio:**
{json.dumps(investments, indent=2) if investments else "No investments tracked"}

**Debt Overview:**
{json.dumps(debts, indent=2) if debts else "No debts tracked"}

**Previous Conversation:**
{history_text if history_text else "This is the start of the conversation"}

**User's Current Question:**
{user_message}

**Instructions:**
1. Answer the user's question using their actual financial data
2. If they ask about spending, reference their transactions and budgets
3. If they ask about investments, use their portfolio data
4. If they ask for advice, give specific, actionable recommendations
5. If data is missing, explain what information you need
6. Keep responses concise but informative (max 300 words)
7. Use bullet points for lists
8. Always be encouraging and positive

**Response:**"""
    
    return prompt

async def generate_gemini_response(prompt: str) -> str:
    """Generate response using Google Gemini API"""
    try:
        if model is None:
            raise Exception("Gemini API not configured")
        
        # Generate content with safety settings
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40,
                max_output_tokens=1024,
            ),
            safety_settings={
                'HARASSMENT': 'BLOCK_NONE',
                'HATE_SPEECH': 'BLOCK_NONE',
                'SEXUAL': 'BLOCK_NONE',
                'DANGEROUS': 'BLOCK_NONE'
            }
        )
        
        if response and response.text:
            logger.info("✅ Gemini response generated successfully")
            return response.text.strip()
        else:
            logger.warning("⚠️ Empty response from Gemini")
            return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
    except Exception as e:
        logger.error(f"❌ Gemini API error: {e}")
        return generate_fallback_response(prompt)

def generate_fallback_response(prompt: str) -> str:
    """Fallback responses when Gemini is unavailable"""
    user_message = prompt.split("**User's Current Question:**")[-1].strip()
    message_lower = user_message.lower()
    
    if 'spent' in message_lower or 'spending' in message_lower:
        return """I can help you analyze your spending! Based on your recent transactions, here's what I can see:

📊 **Quick Analysis:**
- Check your recent transactions in the Transactions section
- Review your budget usage to see where you're spending most
- Look for patterns in your daily, weekly, and monthly spending

💡 **Tip:** I need the Gemini API to be configured for detailed AI-powered analysis. Please check your GEMINI_API_KEY environment variable."""
    
    elif 'budget' in message_lower:
        return """Let me help you with budgeting! 

📝 **Budget Basics:**
- Set budgets for different categories (food, transport, entertainment)
- Try the 50-30-20 rule: 50% needs, 30% wants, 20% savings
- Track your spending against budgets regularly

💡 **Note:** For personalized AI recommendations, please configure the Gemini API key."""
    
    elif 'invest' in message_lower or 'sip' in message_lower:
        return """Great question about investing! 

💰 **Investment Basics:**
- Start with an emergency fund (6 months expenses)
- Consider SIP in mutual funds for long-term wealth creation
- Diversify across equity, debt, and gold
- Start small and increase gradually

💡 **For detailed AI-powered investment advice, please configure the GEMINI_API_KEY.**"""
    
    else:
        return """Hello! I'm your AI Financial Advisor powered by Google Gemini.

I can help you with:
Understanding your spending patterns
Budget recommendations
Investment advice
Savings strategies
Debt management
Financial goal planning

**Note:** The Gemini API needs to be configured for full AI-powered responses. Please set your GEMINI_API_KEY in the environment variables.

What would you like to know about your finances?"""

def generate_suggestions(context: Dict[str, Any]) -> List[str]:
    """Generate contextual question suggestions based on user's financial data"""
    suggestions = []
    
    analytics = context.get('analytics', {})
    current_month = analytics.get('current_month', {})
    expenses = current_month.get('expenses', 0)
    savings = current_month.get('savings', 0)
    income = current_month.get('income', 0)
    
    budgets = context.get('budgets', [])
    investments = context.get('investments', {})
    debts = context.get('debts', {})
    
    # Context-aware suggestions
    if expenses > 0:
        suggestions.append("How much have I spent this month?")
        suggestions.append("Show me my biggest expense categories")
    
    if budgets:
        suggestions.append("Am I within my budget limits?")
        suggestions.append("Which budgets am I overspending on?")
    
    if savings > 0:
        suggestions.append("How much am I saving compared to my income?")
        suggestions.append("How can I increase my savings rate?")
    
    if not investments or not investments.get('total_value'):
        suggestions.append("How should I start investing?")
        suggestions.append("Explain SIP and mutual funds to me")
    else:
        suggestions.append("How is my investment portfolio performing?")
        suggestions.append("Should I rebalance my portfolio?")
    
    if debts:
        suggestions.append("What's my total debt and how can I pay it off faster?")
    
    # General suggestions
    suggestions.extend([
        "Give me a financial summary for this month",
        "What are the best ways to save money?",
        "How much should I invest each month?",
        "Create a budget plan for me"
    ])
    
    return suggestions[:8]  # Return top 8 suggestions

@app.post("/chat/message", response_model=ChatResponse)
async def chat_message(
    chat_data: ChatMessage,
    current_user: User = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Send a message to the AI Financial Advisor"""
    
    try:
        # Get user's financial context
        token = credentials.credentials
        logger.info(f"Processing chat for user {current_user.id}")
        context = await get_user_financial_context(current_user.id, token)
        
        # Get conversation history
        conv_id = f"{current_user.id}_{chat_data.conversation_id}"
        conversation_history = conversation_store.get(conv_id, [])
        
        # Build prompt for Gemini
        prompt = build_gemini_prompt(chat_data.message, context, conversation_history)
        
        # Generate response using Gemini
        response_text = await generate_gemini_response(prompt)
        
        # Store conversation
        if conv_id not in conversation_store:
            conversation_store[conv_id] = []
        
        conversation_store[conv_id].append({
            "user_message": chat_data.message,
            "ai_response": response_text,
            "timestamp": datetime.utcnow().isoformat(),
            "context_summary": {
                "income": context.get('analytics', {}).get('current_month', {}).get('income', 0),
                "expenses": context.get('analytics', {}).get('current_month', {}).get('expenses', 0),
                "savings": context.get('analytics', {}).get('current_month', {}).get('savings', 0)
            }
        })
        
        # Generate contextual suggestions
        suggestions = generate_suggestions(context)
        
        return ChatResponse(
            response=response_text,
            conversation_id=chat_data.conversation_id,
            timestamp=datetime.utcnow(),
            sources=["Transactions", "Budgets", "Investments", "Analytics", "Google Gemini AI"],
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error(f"Error in chat_message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat message: {str(e)}"
        )

@app.get("/chat/history")
async def get_chat_history(
    conversation_id: str = "default",
    current_user: User = Depends(get_current_user)
):
    """Get chat conversation history"""
    conv_id = f"{current_user.id}_{conversation_id}"
    history = conversation_store.get(conv_id, [])
    
    return {
        "conversation_id": conversation_id,
        "messages": history,
        "total_messages": len(history),
        "user_id": current_user.id
    }

@app.delete("/chat/history/{conversation_id}")
async def clear_chat_history(
    conversation_id: str,
    current_user: User = Depends(get_current_user)
):
    """Clear specific conversation history"""
    conv_id = f"{current_user.id}_{conversation_id}"
    if conv_id in conversation_store:
        del conversation_store[conv_id]
        return {"message": f"Chat history for conversation {conversation_id} cleared successfully"}
    
    return {"message": "Conversation not found"}

@app.delete("/chat/clear-all")
async def clear_all_history(current_user: User = Depends(get_current_user)):
    """Clear all conversation history for the current user"""
    user_conversations = [key for key in conversation_store.keys() if key.startswith(f"{current_user.id}_")]
    
    for conv_id in user_conversations:
        del conversation_store[conv_id]
    
    return {
        "message": f"Cleared {len(user_conversations)} conversations",
        "conversations_cleared": len(user_conversations)
    }

@app.get("/chat/suggestions")
async def get_suggestions(
    current_user: User = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get contextual question suggestions for the user"""
    try:
        token = credentials.credentials
        context = await get_user_financial_context(current_user.id, token)
        suggestions = generate_suggestions(context)
        
        return {
            "suggestions": suggestions,
            "total": len(suggestions)
        }
    except Exception as e:
        logger.error(f"Error generating suggestions: {e}")
        return {
            "suggestions": [
                "What's my financial summary?",
                "How much have I spent this month?",
                "Should I invest in mutual funds?",
                "How can I save more money?"
            ]
        }

@app.get("/chat/conversations")
async def list_conversations(current_user: User = Depends(get_current_user)):
    """List all conversations for the current user"""
    user_conversations = [
        {
            "conversation_id": key.split('_', 1)[1],
            "message_count": len(messages),
            "last_message": messages[-1] if messages else None
        }
        for key, messages in conversation_store.items()
        if key.startswith(f"{current_user.id}_")
    ]
    
    return {
        "conversations": user_conversations,
        "total": len(user_conversations)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gemini_status = "configured" if model is not None else "not_configured"
    
    return {
        "status": "healthy",
        "service": "chat-service",
        "gemini_api": gemini_status,
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "AI Financial Advisor Chat Service",
        "version": "1.0.0",
        "description": "Get personalized financial advice powered by Google Gemini",
        "powered_by": "Google Gemini Pro",
        "features": [
            "Context-aware financial advice",
            "Real-time spending analysis",
            "Investment recommendations",
            "Budget optimization",
            "Debt management strategies",
            "Personalized insights"
        ],
        "endpoints": {
            "chat": "/chat/message",
            "history": "/chat/history",
            "suggestions": "/chat/suggestions",
            "conversations": "/chat/conversations",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("CHAT_SERVICE_PORT", 8110))
    logger.info(f"Starting Chat Service on port {port}")
    logger.info(f"Gemini API: {'Configured' if model else 'Not Configured'}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")