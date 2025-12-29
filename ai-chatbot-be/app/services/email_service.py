import smtplib
import random
import string
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from jinja2 import Template
from app.core.config import settings
from app.services.supabase_client import supabase_client


class EmailService:
    def __init__(self):
        self.conf = ConnectionConfig(
            MAIL_USERNAME=settings.mail_username,
            MAIL_PASSWORD=settings.mail_password,
            MAIL_FROM=settings.mail_from,
            MAIL_PORT=settings.mail_port,
            MAIL_SERVER=settings.mail_server,
            MAIL_FROM_NAME=settings.mail_from_name,
            MAIL_STARTTLS=settings.mail_starttls,
            MAIL_SSL_TLS=settings.mail_ssl_tls,
            USE_CREDENTIALS=True,
            VALIDATE_CERTS=True,
        )
        self.fastmail = FastMail(self.conf)

    def generate_otp(self, length: int = 6) -> str:
        """Generate a random OTP"""
        return "".join(random.choices(string.digits, k=length))

    def store_otp(self, email: str, otp: str, purpose: str = "verification"):
        """Store OTP in database with expiration"""
        try:
            # Delete any existing OTPs for this email and purpose
            supabase_client.table("otps").delete().eq("email", email).eq(
                "purpose", purpose
            ).execute()

            # Store new OTP with 10 minutes expiration
            expiry_time = datetime.utcnow() + timedelta(minutes=10)
            otp_data = {
                "email": email,
                "otp": otp,
                "purpose": purpose,
                "expires_at": expiry_time.isoformat(),
                "used": False,
            }

            supabase_client.table("otps").insert(otp_data).execute()
            return True
        except Exception as e:
            print(f"Error storing OTP: {str(e)}")
            return False

    async def verify_otp(self, email: str, otp: str, purpose: str = "verification") -> bool:
        """Verify OTP and mark it as used"""
        try:
            # Get the OTP from database
            result = (
                supabase_client.table("otps")
                .select("*")
                .eq("email", email)
                .eq("otp", otp)
                .eq("purpose", purpose)
                .eq("used", False)
                .execute()
            )

            if not result.data:
                return False

            otp_record = result.data[0]

            # Check if OTP is expired
            expiry_time = datetime.fromisoformat(otp_record["expires_at"].replace("Z", "+00:00"))
            if datetime.utcnow().replace(tzinfo=expiry_time.tzinfo) > expiry_time:
                return False

            # Mark OTP as used
            supabase_client.table("otps").update({"used": True}).eq(
                "id", otp_record["id"]
            ).execute()

            return True
        except Exception as e:
            print(f"Error verifying OTP: {str(e)}")
            return False

    async def send_verification_email(self, email: str, full_name: str = None):
        """Send email verification OTP"""
        try:
            otp = self.generate_otp()

            # Store OTP in database
            self.store_otp(email, otp, "verification")

            # Email template
            template = Template(
                """
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background-color: #f8f9fa; padding: 30px; border-radius: 10px; text-align: center;">
                    <h1 style="color: #333; margin-bottom: 30px;">Email Verification</h1>
                    <p style="font-size: 16px; color: #666; margin-bottom: 30px;">
                        {% if full_name %}Hi {{ full_name }},{% else %}Hi,{% endif %}
                    </p>
                    <p style="font-size: 16px; color: #666; margin-bottom: 30px;">
                        Thank you for registering with AI Chatbot! Please use the following OTP to verify your email address:
                    </p>
                    <div style="background-color: #007bff; color: white; padding: 20px; border-radius: 5px; font-size: 24px; font-weight: bold; letter-spacing: 5px; margin: 30px 0;">
                        {{ otp }}
                    </div>
                    <p style="font-size: 14px; color: #999;">
                        This OTP will expire in 10 minutes. If you didn't request this verification, please ignore this email.
                    </p>
                </div>
            </body>
            </html>
            """
            )

            html_content = template.render(otp=otp, full_name=full_name)

            message = MessageSchema(
                subject="Email Verification - AI Chatbot",
                recipients=[email],
                body=html_content,
                subtype="html",
            )

            await self.fastmail.send_message(message)
            return True

        except Exception as e:
            print(f"Error sending verification email: {str(e)}")
            return False

    async def send_password_reset_email(self, email: str, full_name: str = None):
        """Send password reset OTP"""
        try:
            otp = self.generate_otp()

            # Store OTP in database
            self.store_otp(email, otp, "password_reset")

            # Email template
            template = Template(
                """
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background-color: #f8f9fa; padding: 30px; border-radius: 10px; text-align: center;">
                    <h1 style="color: #333; margin-bottom: 30px;">Password Reset</h1>
                    <p style="font-size: 16px; color: #666; margin-bottom: 30px;">
                        {% if full_name %}Hi {{ full_name }},{% else %}Hi,{% endif %}
                    </p>
                    <p style="font-size: 16px; color: #666; margin-bottom: 30px;">
                        You requested a password reset for your AI Chatbot account. Please use the following OTP to reset your password:
                    </p>
                    <div style="background-color: #dc3545; color: white; padding: 20px; border-radius: 5px; font-size: 24px; font-weight: bold; letter-spacing: 5px; margin: 30px 0;">
                        {{ otp }}
                    </div>
                    <p style="font-size: 14px; color: #999;">
                        This OTP will expire in 10 minutes. If you didn't request a password reset, please ignore this email.
                    </p>
                </div>
            </body>
            </html>
            """
            )

            html_content = template.render(otp=otp, full_name=full_name)

            message = MessageSchema(
                subject="Password Reset - AI Chatbot",
                recipients=[email],
                body=html_content,
                subtype="html",
            )

            await self.fastmail.send_message(message)
            return True

        except Exception as e:
            print(f"Error sending password reset email: {str(e)}")
            return False

    async def send_password_changed_notification(self, email: str, full_name: str = None):
        """Send password changed notification"""
        try:
            # Email template
            template = Template(
                """
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background-color: #f8f9fa; padding: 30px; border-radius: 10px; text-align: center;">
                    <h1 style="color: #333; margin-bottom: 30px;">Password Changed Successfully</h1>
                    <p style="font-size: 16px; color: #666; margin-bottom: 30px;">
                        {% if full_name %}Hi {{ full_name }},{% else %}Hi,{% endif %}
                    </p>
                    <p style="font-size: 16px; color: #666; margin-bottom: 30px;">
                        Your password has been successfully changed for your AI Chatbot account.
                    </p>
                    <div style="background-color: #28a745; color: white; padding: 20px; border-radius: 5px; margin: 30px 0;">
                        <strong>✓ Password Updated</strong>
                    </div>
                    <p style="font-size: 14px; color: #999;">
                        If you didn't make this change, please contact our support team immediately.
                    </p>
                </div>
            </body>
            </html>
            """
            )

            html_content = template.render(full_name=full_name)

            message = MessageSchema(
                subject="Password Changed - AI Chatbot",
                recipients=[email],
                body=html_content,
                subtype="html",
            )

            await self.fastmail.send_message(message)
            return True

        except Exception as e:
            print(f"Error sending password changed notification: {str(e)}")
            return False


email_service = EmailService()
