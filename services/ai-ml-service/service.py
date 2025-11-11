from sqlalchemy.orm import Session
from sqlalchemy import func, extract, and_
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from typing import List, Dict
import sys
sys.path.append('../..')

from shared.models import Transaction, TransactionType, TransactionCategory
from schemas import CategorizeRequest, BudgetRecommendation

class AiMlService:
    
    def __init__(self):
        self.models_dir = "/app/models"
        os.makedirs(self.models_dir, exist_ok=True)
        self.encoder = LabelEncoder()

    def _get_model_path(self, user_id: int) -> str:
        return os.path.join(self.models_dir, f"user_{user_id}_categorizer.joblib")

    def _train_categorizer(self, db: Session, user_id: int):
        """Trains and saves a categorization model for a user."""
        transactions = db.query(Transaction).filter(
            Transaction.user_id == user_id,
            Transaction.type == TransactionType.EXPENSE
        ).limit(1000).all()
        
        if len(transactions) < 20:
            return None # Not enough data
            
        data = [{
            "description": t.description or "",
            "merchant": t.merchant_name or "",
            "amount": t.amount,
            "category": t.category.value
        } for t in transactions]
        
        df = pd.DataFrame(data)
        
        # Simple feature engineering (in production, use TF-IDF on description/merchant)
        df['desc_len'] = df['description'].str.len()
        df['merch_len'] = df['merchant'].str.len()
        
        features = ['amount', 'desc_len', 'merch_len']
        target = 'category'
        
        X = df[features]
        y = self.encoder.fit_transform(df[target])
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Save model and encoder classes
        model_path = self._get_model_path(user_id)
        joblib.dump((model, self.encoder.classes_), model_path)
        
        return model, self.encoder.classes_

    def categorize_transaction(self, db: Session, user_id: int, data: CategorizeRequest) -> dict:
        """Predicts transaction category using a user-specific model."""
        model_path = self._get_model_path(user_id)
        model = None
        classes = None
        
        if os.path.exists(model_path):
            model, classes = joblib.load(model_path)
        else:
            result = self._train_categorizer(db, user_id)
            if result:
                model, classes = result
        
        if not model:
            return {"suggested_category": "OTHER", "confidence": 0.1} # Default
            
        # Create feature vector for prediction
        desc_len = len(data.description or "")
        merch_len = len(data.merchant_name or "")
        amount = data.amount
        
        features = pd.DataFrame([[amount, desc_len, merch_len]], columns=['amount', 'desc_len', 'merch_len'])
        
        prediction_proba = model.predict_proba(features)
        confidence = np.max(prediction_proba)
        predicted_index = np.argmax(prediction_proba)
        category = classes[predicted_index]
        
        return {"suggested_category": category, "confidence": float(confidence)}

    def get_budget_recommendations(self, db: Session, user_id: int) -> List[BudgetRecommendation]:
        """Recommends budget amounts based on 3-month avg spending."""
        start_date = datetime.utcnow() - timedelta(days=90)
        
        # Get average monthly spending per category for the last 3 months
        avg_spending = db.query(
            Transaction.category,
            func.sum(Transaction.amount).label('total_spent')
        ).filter(
            Transaction.user_id == user_id,
            Transaction.type == TransactionType.EXPENSE,
            Transaction.date >= start_date
        ).group_by(Transaction.category).all()
        
        recommendations = []
        for category, total_spent in avg_spending:
            avg_monthly = total_spent / 3.0
            recommended_budget = round(avg_monthly * 1.1, -2) # Add 10% buffer, round to nearest 100
            
            recommendations.append(BudgetRecommendation(
                category=category.value,
                recommended_budget=recommended_budget,
                based_on_average=round(avg_monthly, 2)
            ))
            
        return recommendations