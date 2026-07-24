"""Regression tests for production AI and email integrations."""

from unittest.mock import Mock, patch

import pytest

from models import AIUsageEvent
from utils.ai_service import (
    DEFAULT_API_URL,
    HuggingFaceError,
    call_huggingface_api,
)
from utils.email_service import RESEND_API_URL, send_email


def test_hugging_face_uses_current_chat_completions_shape(monkeypatch):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_test_token")
    monkeypatch.setenv("HUGGINGFACE_MODEL", "test/model:fastest")
    response = Mock(
        ok=True,
        status_code=200,
    )
    response.json.return_value = {
        "choices": [{"message": {"content": "Use logistic regression."}}]
    }

    with patch("utils.ai_service.requests.post", return_value=response) as post:
        result = call_huggingface_api("Which model should I use?")

    assert result == "Use logistic regression."
    args, kwargs = post.call_args
    assert args[0] == DEFAULT_API_URL
    assert kwargs["headers"]["Authorization"] == "Bearer hf_test_token"
    assert kwargs["json"]["model"] == "test/model:fastest"
    assert kwargs["json"]["messages"][1]["role"] == "user"
    assert kwargs["timeout"][1] <= 55


def test_hugging_face_requires_token_when_enabled(monkeypatch):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)

    with pytest.raises(HuggingFaceError, match="HUGGINGFACE_API_KEY") as error:
        call_huggingface_api("Test")

    assert error.value.status_code == 503


def test_hugging_face_maps_provider_rate_limit(monkeypatch):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_test_token")
    response = Mock(ok=False, status_code=429, text="")
    response.json.return_value = {"error": {"message": "Too many requests"}}

    with patch("utils.ai_service.requests.post", return_value=response):
        with pytest.raises(HuggingFaceError, match="usage limits") as error:
            call_huggingface_api("Test")

    assert error.value.status_code == 429


def test_resend_email_delivery_is_synchronous(app, monkeypatch):
    monkeypatch.setenv("EMAIL_PROVIDER", "resend")
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    response = Mock(ok=True, status_code=200)
    app.config.update(
        MAIL_SUPPRESS_SEND=False,
        MAIL_DEFAULT_SENDER="Stats <noreply@example.com>",
    )

    with app.app_context(), patch(
        "utils.email_service.requests.post", return_value=response
    ) as post:
        delivered = send_email(
            "Test subject",
            "recipient@example.com",
            "<p>Test body</p>",
            "Test body",
        )

    assert delivered is True
    args, kwargs = post.call_args
    assert args[0] == RESEND_API_URL
    assert kwargs["headers"]["Authorization"] == "Bearer re_test_key"
    assert kwargs["json"] == {
        "from": "Stats <noreply@example.com>",
        "to": ["recipient@example.com"],
        "subject": "Test subject",
        "html": "<p>Test body</p>",
        "text": "Test body",
    }
    assert "Idempotency-Key" in kwargs["headers"]


def test_email_disabled_returns_false(app, monkeypatch):
    monkeypatch.setenv("EMAIL_PROVIDER", "disabled")
    app.config["MAIL_SUPPRESS_SEND"] = False

    with app.app_context():
        assert send_email("Test", "recipient@example.com", "<p>Test</p>") is False


def _login(client, user):
    return client.post(
        "/auth/login",
        data={"username": user["username"], "password": user["password"]},
    )


def test_chatbot_requires_authentication(client, monkeypatch):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")

    response = client.post(
        "/chatbot/ask",
        json={"question": "Which model?", "context": "Model selection"},
    )

    assert response.status_code == 401
    assert response.get_json()["success"] is False


def test_chatbot_enforces_durable_user_quota(client, test_user, monkeypatch):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_test_token")
    monkeypatch.setenv("AI_REQUESTS_PER_USER_PER_HOUR", "1")
    _login(client, test_user)

    with patch(
        "routes.chatbot_routes.call_huggingface_api",
        return_value="Use a generalized linear model.",
    ) as generate:
        first = client.post(
            "/chatbot/ask",
            json={"question": "Which model?", "context": "Model selection"},
        )
        second = client.post(
            "/chatbot/ask",
            json={"question": "And why?", "context": "Model selection"},
        )

    assert first.status_code == 200
    assert first.get_json()["requests_remaining"] == 0
    assert second.status_code == 429
    assert generate.call_count == 1
    assert AIUsageEvent.query.filter_by(user_id=test_user["id"]).count() == 1


def test_admin_ai_page_never_renders_api_key(client, admin_user, monkeypatch):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_do_not_render_this_secret")
    _login(client, admin_user)

    response = client.get("/admin/ai_settings")

    assert response.status_code == 200
    assert b"hf_do_not_render_this_secret" not in response.data
    assert b"Configured" in response.data


def test_questionnaire_ai_enhancement_requires_login(client, monkeypatch):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")

    with patch("routes.questionnaire_routes.generate_questionnaire") as generate:
        response = client.post(
            "/questionnaire/design",
            data={
                "research_topic": "Public health",
                "research_description": "Measure an intervention outcome",
                "target_audience": "Adults",
                "questionnaire_purpose": "Evaluation",
                "use_ai_enhancement": "on",
                "num_ai_questions": "3",
            },
        )

    assert response.status_code == 302
    assert "/auth/login" in response.headers["Location"]
    generate.assert_not_called()


def test_questionnaire_ai_enhancement_consumes_weighted_quota(
    client, test_user, monkeypatch
):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("AI_REQUESTS_PER_USER_PER_HOUR", "3")
    _login(client, test_user)

    with patch(
        "routes.questionnaire_routes.generate_questionnaire",
        return_value=[],
    ) as generate:
        first = client.post(
            "/questionnaire/design",
            data={
                "research_topic": "Public health",
                "research_description": "Measure an intervention outcome",
                "target_audience": "Adults",
                "questionnaire_purpose": "Evaluation",
                "use_ai_enhancement": "on",
                "num_ai_questions": "3",
            },
        )
        second = client.post(
            "/questionnaire/design",
            data={
                "research_topic": "Public health",
                "research_description": "Measure another outcome",
                "target_audience": "Adults",
                "questionnaire_purpose": "Evaluation",
                "use_ai_enhancement": "on",
                "num_ai_questions": "1",
            },
        )

    assert first.status_code == 302
    assert second.status_code == 302
    assert generate.call_count == 1
    assert AIUsageEvent.query.filter_by(user_id=test_user["id"]).count() == 3
