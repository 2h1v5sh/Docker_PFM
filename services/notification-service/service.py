import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from typing import Dict, Any
import sys
sys.path.append('../..')

from schemas import NotificationEvent

class NotificationService:
    
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASSWORD")
        self.smtp_from = os.getenv("SMTP_FROM")
        
        # self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        # self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        # self.twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")
        
    def _send_email(self, to_email: str, subject: str, html_body: str):
        """Internal helper to send email via SMTP."""
        if not all([self.smtp_host, self.smtp_port, self.smtp_user, self.smtp_pass, self.smtp_from]):
            print(f"Email config missing. Skipping email to {to_email}")
            return
            
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.smtp_from
            msg['To'] = to_email
            
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            print(f"Email sent successfully to {to_email}")
        except Exception as e:
            print(f"Error sending email to {to_email}: {e}")

    def _send_sms(self, to_phone: str, body: str):
        """Internal helper to send SMS via Twilio."""
        print(f"SMS Service: Sending to {to_phone}: {body}")
        # Twilio client logic would go here
        # try:
        #    client = Client(self.twilio_sid, self.twilio_token)
        #    message = client.messages.create(body=body, from_=self.twilio_phone, to=to_phone)
        #    print(f"SMS sent: {message.sid}")
        # except Exception as e:
        #    print(f"Error sending SMS: {e}")

    def process_notification(self, event: NotificationEvent):
        """Routes the notification event to the correct handler."""
        print(f"Processing notification: {event.type} for {event.email}")
        
        if event.type == "password_reset":
            self.send_password_reset_email(event.email, event.data.get("otp"))
        elif event.type == "budget_alert":
            self.send_budget_alert_email(
                event.email,
                event.data.get("category"),
                event.data.get("percentage_used")
            )
        # Add more event types here
        # elif event.type == "emi_reminder":
        #    self.send_sms(event.phone, ...)
            
        else:
            print(f"Unknown notification type: {event.type}")

    def send_password_reset_email(self, email: str, otp: str):
        if not otp:
            print("OTP missing for password reset")
            return
            
        subject = "Your Password Reset Code"
        body = f"""
        <html><body>
        <p>Hi,</p>
        <p>Your password reset code is: <strong>{otp}</strong></p>
        <p>This code is valid for 10 minutes.</p>
        <p>If you did not request this, please ignore this email.</p>
        </body></html>
        """
        self._send_email(email, subject, body)

    def send_budget_alert_email(self, email: str, category: str, percentage: float):
        if category is None or percentage is None:
            print("Budget alert data missing")
            return
            
        subject = f"Budget Alert for {category.title()}"
        body = f"""
        <html><body>
        <p>Hi,</p>
        <p>This is a notification that you have used <strong>{percentage:.1f}%</strong> 
        of your budget for the <strong>{category.title()}</strong> category.</p>
        <p>Please review your spending to stay on track.</p>
        </body></html>
        """
        self._send_email(email, subject, body)