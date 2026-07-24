"""Durable per-user usage controls for paid AI features."""

import os
from datetime import datetime, timedelta

from models import AIUsageEvent, db


def hourly_ai_limit() -> int:
    """Return the configured per-user hourly unit budget."""
    try:
        return min(
            max(int(os.environ.get("AI_REQUESTS_PER_USER_PER_HOUR", "20")), 1),
            1_000,
        )
    except ValueError:
        return 20


def consume_user_ai_quota(user_id: int, units: int = 1) -> tuple[bool, int]:
    """Consume durable AI usage units and return allowed/remaining."""
    requested_units = max(units, 1)
    window_start = datetime.utcnow() - timedelta(hours=1)
    used = AIUsageEvent.query.filter(
        AIUsageEvent.user_id == user_id,
        AIUsageEvent.created_at >= window_start,
    ).count()
    limit = hourly_ai_limit()
    if used + requested_units > limit:
        return False, max(limit - used, 0)

    db.session.add_all(
        AIUsageEvent(user_id=user_id) for _ in range(requested_units)
    )
    db.session.commit()
    return True, limit - used - requested_units
