"""
Transaction Service - Complete Implementation
Handles income and expense transactions with full CRUD operations
"""
from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, text
from pydantic import BaseModel, field_validator
from datetime import datetime, date
from typing import Optional, List
from enum import Enum
from contextlib import asynccontextmanager
import uvicorn
import os
import logging

from shared.database import get_db, Base, engine
from shared.models import User, Transaction
from shared.security import SecurityManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# Lifespan Context Manager
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown"""
    # Startup
    logger.info("🚀 Starting Transaction Service")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            conn.commit()
        logger.info("✅ Database connected")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables created/verified")
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Transaction Service")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Transaction Service",
    description="Manage income and expense transactions",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Enums
# ==========================================

class TransactionType(str, Enum):
    income = "income"
    expense = "expense"

class PaymentMethod(str, Enum):
    cash = "cash"
    credit_card = "credit_card"
    debit_card = "debit_card"
    upi = "upi"
    bank_transfer = "bank_transfer"
    online = "online"

# ==========================================
# Pydantic Models
# ==========================================

class TransactionCreate(BaseModel):
    type: TransactionType
    category: str
    amount: float
    description: Optional[str] = None
    date: Optional[datetime] = None
    merchant_name: Optional[str] = None
    payment_method: Optional[PaymentMethod] = None
    location: Optional[str] = None
    tags: Optional[List[str]] = None
    
    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return round(v, 2)

class TransactionResponse(BaseModel):
    id: int
    user_id: int
    type: str
    category: str
    amount: float
    description: Optional[str] = None
    date: datetime
    merchant_name: Optional[str] = None
    payment_method: Optional[str] = None
    location: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class TransactionUpdate(BaseModel):
    category: Optional[str] = None
    amount: Optional[float] = None
    description: Optional[str] = None
    date: Optional[datetime] = None
    merchant_name: Optional[str] = None
    payment_method: Optional[PaymentMethod] = None

# ==========================================
# Auth Dependency
# ==========================================

async def get_current_user(token: str = Depends(SecurityManager.get_token), db: Session = Depends(get_db)):
    """Get current authenticated user"""
    try:
        payload = SecurityManager.decode_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_id = int(payload.get("sub"))
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

# ==========================================
# Endpoints
# ==========================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            conn.commit()
        db_status = "connected"
    except Exception as e:
        logger.error(f"Health check DB error: {e}")
        db_status = "disconnected"
    
    return {
        "status": "healthy",
        "service": "transaction-service",
        "database": db_status
    }

@app.get("/")
async def root():
    return {
        "service": "transaction-service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "create": "POST /transactions",
            "list": "GET /transactions",
            "get_one": "GET /transactions/{id}",
            "update": "PUT /transactions/{id}",
            "delete": "DELETE /transactions/{id}",
            "summary": "GET /transactions/summary/stats",
            "docs": "/docs"
        }
    }

@app.post("/transactions", response_model=TransactionResponse, status_code=status.HTTP_201_CREATED)
async def create_transaction(
    transaction: TransactionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new transaction"""
    try:
        new_transaction = Transaction(
            user_id=current_user.id,
            type=transaction.type.value,
            category=transaction.category,
            amount=transaction.amount,
            description=transaction.description,
            date=transaction.date or datetime.utcnow(),
            merchant_name=transaction.merchant_name,
            payment_method=transaction.payment_method.value if transaction.payment_method else None,
            location=transaction.location,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(new_transaction)
        db.commit()
        db.refresh(new_transaction)
        
        logger.info(f"✅ Transaction created: {new_transaction.id} for user {current_user.id}")
        
        return new_transaction
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions", response_model=List[TransactionResponse])
async def get_transactions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    type: Optional[TransactionType] = None,
    category: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all transactions for current user with filters
    
    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum number of records to return
    - **type**: Filter by transaction type (income/expense)
    - **category**: Filter by category
    - **start_date**: Filter from this date
    - **end_date**: Filter until this date
    """
    try:
        query = db.query(Transaction).filter(Transaction.user_id == current_user.id)
        
        # Apply filters
        if type:
            query = query.filter(Transaction.type == type.value)
        
        if category:
            query = query.filter(Transaction.category == category)
        
        if start_date:
            query = query.filter(Transaction.date >= start_date)
        
        if end_date:
            query = query.filter(Transaction.date <= end_date)
        
        # Order by date descending (newest first)
        query = query.order_by(desc(Transaction.date))
        
        # Apply pagination
        transactions = query.offset(skip).limit(limit).all()
        
        logger.info(f"✅ Retrieved {len(transactions)} transactions for user {current_user.id}")
        
        return transactions
        
    except Exception as e:
        logger.error(f"Error retrieving transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions/{transaction_id}", response_model=TransactionResponse)
async def get_transaction(
    transaction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific transaction by ID"""
    transaction = db.query(Transaction).filter(
        and_(
            Transaction.id == transaction_id,
            Transaction.user_id == current_user.id
        )
    ).first()
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return transaction

@app.put("/transactions/{transaction_id}", response_model=TransactionResponse)
async def update_transaction(
    transaction_id: int,
    transaction_update: TransactionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a transaction"""
    transaction = db.query(Transaction).filter(
        and_(
            Transaction.id == transaction_id,
            Transaction.user_id == current_user.id
        )
    ).first()
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    # Update fields
    update_data = transaction_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if field == 'payment_method' and value:
            setattr(transaction, field, value.value)
        else:
            setattr(transaction, field, value)
    
    transaction.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(transaction)
    
    logger.info(f"✅ Transaction {transaction_id} updated")
    
    return transaction

@app.delete("/transactions/{transaction_id}")
async def delete_transaction(
    transaction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a transaction"""
    transaction = db.query(Transaction).filter(
        and_(
            Transaction.id == transaction_id,
            Transaction.user_id == current_user.id
        )
    ).first()
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    db.delete(transaction)
    db.commit()
    
    logger.info(f"✅ Transaction {transaction_id} deleted")
    
    return {"message": "Transaction deleted successfully", "id": transaction_id}

@app.get("/transactions/summary/stats")
async def get_transaction_summary(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get transaction summary statistics"""
    query = db.query(Transaction).filter(Transaction.user_id == current_user.id)
    
    if start_date:
        query = query.filter(Transaction.date >= start_date)
    if end_date:
        query = query.filter(Transaction.date <= end_date)
    
    transactions = query.all()
    
    total_income = sum(t.amount for t in transactions if t.type == "income")
    total_expenses = sum(t.amount for t in transactions if t.type == "expense")
    
    # Category breakdown
    category_breakdown = {}
    for t in transactions:
        if t.category not in category_breakdown:
            category_breakdown[t.category] = {"income": 0, "expense": 0, "count": 0}
        
        if t.type == "income":
            category_breakdown[t.category]["income"] += t.amount
        else:
            category_breakdown[t.category]["expense"] += t.amount
        
        category_breakdown[t.category]["count"] += 1
    
    return {
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "net_savings": round(total_income - total_expenses, 2),
        "transaction_count": len(transactions),
        "category_breakdown": category_breakdown,
        "start_date": start_date,
        "end_date": end_date
    }

if __name__ == "__main__":
    port = int(os.getenv("TRANSACTION_SERVICE_PORT", 8103))
    logger.info(f"Starting Transaction Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")