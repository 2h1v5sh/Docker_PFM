from pydantic import BaseModel
from typing import Dict, Any

# --- Internal Event Schemas (for RabbitMQ) ---
# These are not API schemas, but Pydantic models
# for validating incoming queue messages.

class NotificationEvent(BaseModel):
    type: str  # e.g., "password_reset", "budget_alert"
    user_id: int
    email: str
    data: Dict[str, Any] # Contains 'otp', 'category', 'percentage', etc.

# --- Manual Trigger (for testing) ---
class ManualNotification(BaseModel):
    type: str
    user_id: int
    data: Dict[str, Any]