"""
Budget Service - Complete Implementation
Handles budget creation, tracking, and alerts
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
from shared.models import User, Budget, Transaction
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
    logger.info("🚀 Starting Budget Service")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connected")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables created/verified")
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Budget Service")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Budget Service",
    description="Manage budgets and track spending",
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

class BudgetPeriod(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    yearly = "yearly"

# ==========================================
# Pydantic Models
# ==========================================

class BudgetCreate(BaseModel):
    category: str
    amount: float
    period: BudgetPeriod
    start_date: datetime
    end_date: datetime
    alert_threshold: Optional[float] = 80  # Alert at 80% by default
    
    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return round(v, 2)
    
    @field_validator('alert_threshold')
    @classmethod
    def validate_threshold(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Alert threshold must be between 0 and 100')
        return v

class BudgetResponse(BaseModel):
    id: int
    user_id: int
    category: str
    amount: float
    spent: float
    remaining: float
    percentage_used: float
    period: str
    start_date: datetime
    end_date: datetime
    alert_threshold: float
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class BudgetUpdate(BaseModel):
    amount: Optional[float] = None
    alert_threshold: Optional[float] = None
    end_date: Optional[datetime] = None

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
# Helper Functions
# ==========================================

def calculate_budget_spent(budget: Budget, db: Session) -> float:
    """Calculate total spent for a budget category within the period"""
    transactions = db.query(Transaction).filter(
        and_(
            Transaction.user_id == budget.user_id,
            Transaction.category == budget.category,
            Transaction.type == "expense",
            Transaction.date >= budget.start_date,
            Transaction.date <= budget.end_date
        )
    ).all()
    
    return sum(t.amount for t in transactions)

def enrich_budget_data(budget: Budget, db: Session) -> dict:
    """Add calculated fields to budget"""
    spent = calculate_budget_spent(budget, db)
    remaining = budget.amount - spent
    percentage_used = (spent / budget.amount * 100) if budget.amount > 0 else 0
    
    return {
        "id": budget.id,
        "user_id": budget.user_id,
        "category": budget.category,
        "amount": budget.amount,
        "spent": round(spent, 2),
        "remaining": round(remaining, 2),
        "percentage_used": round(percentage_used, 2),
        "period": budget.period,
        "start_date": budget.start_date,
        "end_date": budget.end_date,
        "alert_threshold": budget.alert_threshold,
        "is_active": budget.is_active,
        "created_at": budget.created_at
    }

# ==========================================
# Endpoints
# ==========================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        logger.error(f"Health check DB error: {e}")
        db_status = "disconnected"
    
    return {
        "status": "healthy",
        "service": "budget-service",
        "database": db_status
    }

@app.get("/")
async def root():
    return {
        "service": "budget-service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "create": "POST /budgets",
            "list": "GET /budgets",
            "get_one": "GET /budgets/{id}",
            "update": "PUT /budgets/{id}",
            "delete": "DELETE /budgets/{id}",
            "summary": "GET /budgets/summary/overview",
            "docs": "/docs"
        }
    }

@app.post("/budgets", response_model=BudgetResponse, status_code=status.HTTP_201_CREATED)
async def create_budget(
    budget: BudgetCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new budget"""
    try:
        # Check if budget already exists for this category and period
        existing = db.query(Budget).filter(
            and_(
                Budget.user_id == current_user.id,
                Budget.category == budget.category,
                Budget.is_active == True,
                Budget.end_date >= datetime.utcnow()
            )
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Active budget already exists for category '{budget.category}'"
            )
        
        new_budget = Budget(
            user_id=current_user.id,
            category=budget.category,
            amount=budget.amount,
            period=budget.period.value,
            start_date=budget.start_date,
            end_date=budget.end_date,
            alert_threshold=budget.alert_threshold,
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(new_budget)
        db.commit()
        db.refresh(new_budget)
        
        logger.info(f"✅ Budget created: {new_budget.id} for user {current_user.id}")
        
        # Return enriched budget data
        return enrich_budget_data(new_budget, db)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating budget: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/budgets", response_model=List[BudgetResponse])
async def get_budgets(
    active_only: bool = Query(True),
    category: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all budgets for current user
    
    - **active_only**: Only return active budgets (default: true)
    - **category**: Filter by category
    """
    try:
        query = db.query(Budget).filter(Budget.user_id == current_user.id)
        
        if active_only:
            query = query.filter(
                and_(
                    Budget.is_active == True,
                    Budget.end_date >= datetime.utcnow()
                )
            )
        
        if category:
            query = query.filter(Budget.category == category)
        
        query = query.order_by(desc(Budget.created_at))
        
        budgets = query.all()
        
        logger.info(f"✅ Retrieved {len(budgets)} budgets for user {current_user.id}")
        
        # Enrich each budget with calculated fields
        enriched_budgets = [enrich_budget_data(b, db) for b in budgets]
        
        return enriched_budgets
        
    except Exception as e:
        logger.error(f"Error retrieving budgets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/budgets/{budget_id}", response_model=BudgetResponse)
async def get_budget(
    budget_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific budget by ID"""
    budget = db.query(Budget).filter(
        and_(
            Budget.id == budget_id,
            Budget.user_id == current_user.id
        )
    ).first()
    
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    
    return enrich_budget_data(budget, db)

@app.put("/budgets/{budget_id}", response_model=BudgetResponse)
async def update_budget(
    budget_id: int,
    budget_update: BudgetUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a budget"""
    budget = db.query(Budget).filter(
        and_(
            Budget.id == budget_id,
            Budget.user_id == current_user.id
        )
    ).first()
    
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    
    # Update fields
    update_data = budget_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(budget, field, value)
    
    budget.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(budget)
    
    logger.info(f"✅ Budget {budget_id} updated")
    
    return enrich_budget_data(budget, db)

@app.delete("/budgets/{budget_id}")
async def delete_budget(
    budget_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a budget"""
    budget = db.query(Budget).filter(
        and_(
            Budget.id == budget_id,
            Budget.user_id == current_user.id
        )
    ).first()
    
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    
    db.delete(budget)
    db.commit()
    
    logger.info(f"✅ Budget {budget_id} deleted")
    
    return {"message": "Budget deleted successfully", "id": budget_id}

@app.get("/budgets/summary/overview")
async def get_budget_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get overall budget summary"""
    active_budgets = db.query(Budget).filter(
        and_(
            Budget.user_id == current_user.id,
            Budget.is_active == True,
            Budget.end_date >= datetime.utcnow()
        )
    ).all()
    
    total_budget = sum(b.amount for b in active_budgets)
    total_spent = sum(calculate_budget_spent(b, db) for b in active_budgets)
    total_remaining = total_budget - total_spent
    
    # Budget status counts
    on_track = sum(1 for b in active_budgets if calculate_budget_spent(b, db) / b.amount * 100 < b.alert_threshold)
    warning = sum(1 for b in active_budgets if b.alert_threshold <= calculate_budget_spent(b, db) / b.amount * 100 < 100)
    exceeded = sum(1 for b in active_budgets if calculate_budget_spent(b, db) >= b.amount)
    
    return {
        "total_budgets": len(active_budgets),
        "total_budget_amount": round(total_budget, 2),
        "total_spent": round(total_spent, 2),
        "total_remaining": round(total_remaining, 2),
        "overall_percentage_used": round((total_spent / total_budget * 100) if total_budget > 0 else 0, 2),
        "status_counts": {
            "on_track": on_track,
            "warning": warning,
            "exceeded": exceeded
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("BUDGET_SERVICE_PORT", 8104))
    logger.info(f"Starting Budget Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")