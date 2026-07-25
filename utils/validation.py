"""Shared validation helpers for user-controlled account fields."""

import re


MIN_PASSWORD_LENGTH = 10
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def normalize_email(value):
    return (value or "").strip().lower()


def is_valid_email(value):
    return bool(
        isinstance(value, str)
        and len(value) <= 120
        and EMAIL_PATTERN.fullmatch(value)
    )


def is_valid_username(value):
    return bool(
        isinstance(value, str)
        and 3 <= len(value.strip()) <= 80
        and USERNAME_PATTERN.fullmatch(value.strip())
    )


def is_valid_password(value):
    return isinstance(value, str) and len(value) >= MIN_PASSWORD_LENGTH
