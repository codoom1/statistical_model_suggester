"""Transactional email delivery for Vercel and local development."""

import logging
import os
from uuid import uuid4

import requests
from flask import current_app
from flask_mail import Mail, Message


mail = Mail()
logger = logging.getLogger(__name__)
RESEND_API_URL = "https://api.resend.com/emails"


def init_mail(app) -> None:
    """Initialize Flask-Mail for the optional local SMTP fallback."""
    mail.init_app(app)


def get_email_provider() -> str:
    """Resolve the configured delivery provider."""
    configured = os.environ.get("EMAIL_PROVIDER", "").strip().lower()
    if configured:
        return configured
    if os.environ.get("RESEND_API_KEY"):
        return "resend"
    if os.environ.get("MAIL_USERNAME") and os.environ.get("MAIL_PASSWORD"):
        return "smtp"
    return "disabled"


def _send_with_resend(
    subject: str,
    recipient: str,
    html_body: str,
    text_body: str | None,
) -> None:
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    sender = current_app.config.get("MAIL_DEFAULT_SENDER", "").strip()
    if not api_key:
        raise RuntimeError("RESEND_API_KEY is not configured.")
    if not sender:
        raise RuntimeError("MAIL_DEFAULT_SENDER is not configured.")

    payload = {
        "from": sender,
        "to": [recipient],
        "subject": subject,
        "html": html_body,
    }
    if text_body:
        payload["text"] = text_body

    response = requests.post(
        RESEND_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": str(uuid4()),
        },
        json=payload,
        timeout=(5, 20),
    )
    if not response.ok:
        logger.warning(
            "Resend rejected an email with status %s.",
            response.status_code,
        )
        response.raise_for_status()


def _send_with_smtp(
    subject: str,
    recipient: str,
    html_body: str,
    text_body: str | None,
) -> None:
    message = Message(
        subject=subject,
        recipients=[recipient],
        html=html_body,
        body=text_body
        or "Please view this message in an HTML-compatible email client.",
    )
    mail.send(message)


def send_email(
    subject: str,
    recipient: str,
    html_body: str,
    text_body: str | None = None,
) -> bool:
    """Send an email synchronously and report whether delivery was accepted."""
    if current_app.config.get("MAIL_SUPPRESS_SEND"):
        logger.info("Email delivery suppressed for %s.", recipient)
        return True

    provider = get_email_provider()
    try:
        if provider == "resend":
            _send_with_resend(subject, recipient, html_body, text_body)
        elif provider == "smtp":
            _send_with_smtp(subject, recipient, html_body, text_body)
        elif provider == "disabled":
            logger.warning(
                "Email delivery is disabled; notification for %s was not sent.",
                recipient,
            )
            return False
        else:
            logger.error("Unsupported EMAIL_PROVIDER: %s", provider)
            return False
    except Exception as exc:
        logger.error(
            "Email delivery through %s failed for %s: %s",
            provider,
            recipient,
            exc,
        )
        return False

    logger.info("Email accepted by %s for %s.", provider, recipient)
    return True


def send_expert_approved_email(user, email: str) -> bool:
    """Notify a user that their expert application was approved."""
    subject = "Your Expert Application Has Been Approved!"
    html_body = f"""
    <p>Congratulations {user.username}!</p>
    <p>Your application to become an expert on the Statistical Model Suggester
    platform has been <strong>approved</strong>.</p>
    <p>You can now provide guidance, offer consultations, and share your
    expertise with the community.</p>
    <p>Best regards,<br>Statistical Model Suggester Team</p>
    """
    return send_email(subject, email, html_body)


def send_expert_rejected_email(user, email: str) -> bool:
    """Notify a user that their expert application was not approved."""
    subject = "Update on Your Expert Application"
    html_body = f"""
    <p>Dear {user.username},</p>
    <p>Thank you for your interest in becoming an expert on the Statistical
    Model Suggester platform.</p>
    <p>After reviewing your application, we are unable to approve your expert
    status at this time. You are welcome to apply again with additional
    information about your qualifications and experience.</p>
    <p>Best regards,<br>Statistical Model Suggester Team</p>
    """
    return send_email(subject, email, html_body)
