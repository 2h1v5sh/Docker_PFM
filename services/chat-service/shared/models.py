"""
Shared Database Models
All SQLAlchemy models for the Personal Finance Manager
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Date, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from shared.database import Base

# ==========================================
# Enums
# ==========================================

class TransactionType(str, enum.Enum):
    income = "income"
    expense = "expense"

class BudgetPeriod(str, enum.Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    yearly = "yearly"

class InvestmentType(str, enum.Enum):
    stocks = "stocks"
    mutual_funds = "mutual_funds"
    bonds = "bonds"
    crypto = "crypto"
    real_estate = "real_estate"
    gold = "gold"
    other = "other"

class DebtType(str, enum.Enum):
    credit_card = "credit_card"
    personal_loan = "personal_loan"
    home_loan = "home_loan"
    car_loan = "car_loan"
    student_loan = "student_loan"
    other = "other"

# ==========================================
# User Model
# ==========================================

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    # FIXED: Made username optional with nullable=True
    username = Column(String(100), unique=True, index=True, nullable=True)
    full_name = Column(String(255))
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    phone = Column(String(20))
    currency = Column(String(10), default="INR")  # Changed from USD to INR
    timezone = Column(String(50), default="Asia/Kolkata")  # Changed from UTC
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # ADDED: role field
    role = Column(String(50), default="user")
    
    # Relationships
    transactions = relationship("Transaction", back_populates="user", cascade="all, delete-orphan")
    budgets = relationship("Budget", back_populates="user", cascade="all, delete-orphan")
    investments = relationship("Investment", back_populates="user", cascade="all, delete-orphan")
    debts = relationship("Debt", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")

# ==========================================
# Transaction Model
# ==========================================

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    type = Column(String(20), nullable=False)  # income or expense
    category = Column(String(100), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    description = Column(Text)
    date = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    merchant_name = Column(String(255))
    payment_method = Column(String(50))  # cash, credit_card, debit_card, upi, etc.
    location = Column(String(255))
    receipt_url = Column(String(500))
    tags = Column(Text)  # JSON string of tags
    is_recurring = Column(Boolean, default=False)
    recurring_frequency = Column(String(20))  # daily, weekly, monthly, yearly
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="transactions")

# ==========================================
# Budget Model
# ==========================================

class Budget(Base):
    __tablename__ = "budgets"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    category = Column(String(100), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    period = Column(String(20), nullable=False)  # daily, weekly, monthly, yearly
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    alert_threshold = Column(Float, default=80)  # Alert at 80% by default
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="budgets")

# ==========================================
# Investment Model
# ==========================================

class Investment(Base):
    __tablename__ = "investments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # stocks, mutual_funds, bonds, crypto, etc.
    symbol = Column(String(20))  # Stock ticker or fund code
    quantity = Column(Float, nullable=False)
    purchase_price = Column(Float, nullable=False)
    current_price = Column(Float)
    purchase_date = Column(Date, nullable=False)
    platform = Column(String(100))  # Where it's held (Robinhood, Vanguard, etc.)
    notes = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="investments")

# ==========================================
# Debt Model
# ==========================================

class Debt(Base):
    __tablename__ = "debts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # credit_card, personal_loan, home_loan, etc.
    principal_amount = Column(Float, nullable=False)
    current_balance = Column(Float, nullable=False)
    interest_rate = Column(Float, nullable=False)
    minimum_payment = Column(Float)
    due_date = Column(Date)
    payment_frequency = Column(String(20))  # monthly, quarterly, etc.
    creditor = Column(String(255))
    account_number = Column(String(100))
    start_date = Column(Date, nullable=False)
    target_payoff_date = Column(Date)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="debts")

# ==========================================
# Notification Model
# ==========================================

class Notification(Base):
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    type = Column(String(50), nullable=False)  # budget_alert, bill_reminder, etc.
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False)
    priority = Column(String(20), default="normal")  # low, normal, high, urgent
    related_entity_type = Column(String(50))  # budget, transaction, debt, etc.
    related_entity_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    read_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="notifications")

# ==========================================
# Financial Goal Model
# ==========================================

class FinancialGoal(Base):
    __tablename__ = "financial_goals"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    target_amount = Column(Float, nullable=False)
    current_amount = Column(Float, default=0)
    target_date = Column(Date, nullable=False)
    category = Column(String(100))  # emergency_fund, vacation, house, etc.
    priority = Column(String(20), default="medium")  # low, medium, high
    is_active = Column(Boolean, default=True)
    is_completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)

# ==========================================
# Chat History Model
# ==========================================

class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String(100), index=True)
    role = Column(String(20), nullable=False)  # user or assistant
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    extra_data = Column(Text)  # renamed from 'metadata' to avoid conflicts

# ==========================================
# ML Prediction Model
# ==========================================

class MLPrediction(Base):
    __tablename__ = "ml_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    prediction_type = Column(String(50), nullable=False)  # spending_forecast, anomaly_detection, etc.
    input_data = Column(Text)  # JSON string
    prediction_result = Column(Text)  # JSON string
    confidence_score = Column(Float)
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

# ==========================================
# Audit Log Model
# ==========================================

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    action = Column(String(100), nullable=False)  # login, create_transaction, etc.
    entity_type = Column(String(50))  # transaction, budget, etc.
    entity_id = Column(Integer)
    ip_address = Column(String(50))
    user_agent = Column(String(500))
    details = Column(Text)  # JSON string
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
