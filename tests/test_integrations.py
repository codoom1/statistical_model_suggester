"""Regression tests for production AI and email integrations."""

from unittest.mock import Mock, patch

import httpx
from openai import NotFoundError, RateLimitError
import pytest
from sqlalchemy import inspect

from models import AIUsageEvent, db
from utils.ai_service import (
    OpenAIServiceError,
    call_openai_api,
)
from utils.email_service import RESEND_API_URL, send_email
from utils.recommendation_ai import review_recommendation


def test_openai_uses_responses_api(monkeypatch):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    response = Mock(output_text="Use logistic regression.")
    client = Mock()
    client.responses.create.return_value = response

    with patch("utils.ai_service.OpenAI", return_value=client) as openai_client:
        result = call_openai_api(
            "Which model should I use?",
            safety_identifier="user-7",
        )

    assert result == "Use logistic regression."
    openai_client.assert_called_once_with(
        api_key="sk-test",
        timeout=45.0,
        max_retries=1,
    )
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-test"
    assert kwargs["input"] == "Which model should I use?"
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["max_output_tokens"] == 400
    assert kwargs["safety_identifier"] == "user-7"
    assert kwargs["store"] is False


def test_openai_supports_strict_structured_outputs(monkeypatch):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    response = Mock(output_text='{"answer":"ok"}')
    client = Mock()
    client.responses.create.return_value = response
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    with patch("utils.ai_service.OpenAI", return_value=client):
        result = call_openai_api(
            "Return a structured answer.",
            response_schema=schema,
            schema_name="test_answer",
        )

    assert result == '{"answer":"ok"}'
    assert client.responses.create.call_args.kwargs["text"] == {
        "format": {
            "type": "json_schema",
            "name": "test_answer",
            "schema": schema,
            "strict": True,
        }
    }


def test_openai_allows_a_larger_feature_specific_output_budget(monkeypatch):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKENS", "400")
    response = Mock(output_text='{"answer":"complete"}', status="completed")
    client = Mock()
    client.responses.create.return_value = response

    with patch("utils.ai_service.OpenAI", return_value=client):
        result = call_openai_api(
            "Return the complete review.",
            max_output_tokens=1_500,
        )

    assert result == '{"answer":"complete"}'
    assert (
        client.responses.create.call_args.kwargs["max_output_tokens"]
        == 1_500
    )


def test_openai_rejects_an_incomplete_provider_response(monkeypatch):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    response = Mock(
        output_text='{"answer":"cut off',
        status="incomplete",
    )
    client = Mock()
    client.responses.create.return_value = response

    with patch("utils.ai_service.OpenAI", return_value=client):
        with pytest.raises(
            OpenAIServiceError,
            match="exceeded its output limit",
        ) as error:
            call_openai_api("Return the complete review.")

    assert error.value.status_code == 502


def test_recommendation_ai_is_limited_to_verified_candidates():
    model_database = {
        "Linear Regression": {
            "description": "Models a continuous outcome with linear effects.",
            "assumptions": ["Linear relationship"],
        },
        "Random Forest": {
            "description": "Models nonlinear relationships using trees.",
            "assumptions": [],
        },
    }
    provider_response = {
        "recommended_model": "Linear Regression",
        "summary": "The stated design favors a linear model.",
        "why_it_fits": ["The outcome is continuous."],
        "assumptions_to_check": ["Inspect residual patterns."],
        "alternative_tradeoffs": [
            {
                "model": "Random Forest",
                "when_to_prefer": "Prefer it if nonlinear effects dominate.",
            },
            {
                "model": "Invented Model",
                "when_to_prefer": "Never.",
            },
        ],
    }

    with patch(
        "utils.recommendation_ai.call_openai_api",
        return_value=__import__("json").dumps(provider_response),
    ) as generate:
        review = review_recommendation(
            research_question="What predicts blood pressure?",
            analysis_inputs={"analysis_goal": "predict"},
            candidate_models=[
                "Linear Regression",
                "Random Forest",
                "Invented Model",
            ],
            model_database=model_database,
            safety_identifier="user-hash",
        )

    assert review["recommended_model"] == "Linear Regression"
    assert review["alternative_tradeoffs"] == [
        {
            "model": "Random Forest",
            "when_to_prefer": "Prefer it if nonlinear effects dominate.",
        }
    ]
    schema = generate.call_args.kwargs["response_schema"]
    assert generate.call_args.kwargs["max_output_tokens"] == 1_500
    assert schema["properties"]["recommended_model"]["enum"] == [
        "Linear Regression",
        "Random Forest",
    ]


def test_recommendation_ai_rejects_an_unverified_selection():
    with patch(
        "utils.recommendation_ai.call_openai_api",
        return_value=(
            '{"recommended_model":"Invented Model","summary":"No.",'
            '"why_it_fits":[],"assumptions_to_check":[],'
            '"alternative_tradeoffs":[]}'
        ),
    ):
        with pytest.raises(ValueError, match="unverified model"):
            review_recommendation(
                research_question="What predicts the outcome?",
                analysis_inputs={"analysis_goal": "predict"},
                candidate_models=["Linear Regression"],
                model_database={"Linear Regression": {"description": "Linear."}},
                safety_identifier="user-hash",
            )


def test_openai_requires_key_when_enabled(monkeypatch):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(OpenAIServiceError, match="OPENAI_API_KEY") as error:
        call_openai_api("Test")

    assert error.value.status_code == 503


def test_openai_maps_provider_rate_limit(monkeypatch):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(429, request=request)

    with patch("utils.ai_service.OpenAI") as openai_client:
        openai_client.return_value.responses.create.side_effect = (
            RateLimitError(
                "Too many requests",
                response=response,
                body=None,
            )
        )
        with pytest.raises(OpenAIServiceError, match="usage limits") as error:
            call_openai_api("Test")

    assert error.value.status_code == 429


def test_openai_falls_back_when_configured_model_is_unavailable(monkeypatch):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "unavailable-model")
    monkeypatch.setenv("OPENAI_FALLBACK_MODEL", "gpt-5-mini")
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    not_found_response = httpx.Response(404, request=request)
    unavailable = NotFoundError(
        "Model not found",
        response=not_found_response,
        body=None,
    )
    client = Mock()
    client.responses.create.side_effect = [
        unavailable,
        Mock(output_text="Use logistic regression."),
    ]

    with patch("utils.ai_service.OpenAI", return_value=client):
        result = call_openai_api("Which model should I use?")

    assert result == "Use logistic regression."
    assert [
        call.kwargs["model"] for call in client.responses.create.call_args_list
    ] == ["unavailable-model", "gpt-5-mini"]


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
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AI_REQUESTS_PER_USER_PER_HOUR", "1")
    _login(client, test_user)

    with patch(
        "routes.chatbot_routes.call_openai_api",
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


def test_chatbot_guides_model_replacement_questions(
    client,
    test_user,
    monkeypatch,
):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _login(client, test_user)
    page_context = (
        "Recommended model: Linear Regression. "
        "Analysis goal: predict. Outcome type: continuous. "
        "Verified compatible alternatives: Ridge Regression, Random Forest."
    )

    with patch(
        "routes.chatbot_routes.call_openai_api",
        return_value="- Ridge Regression: preferable with collinearity.",
    ) as generate:
        response = client.post(
            "/chatbot/ask",
            json={
                "question": "What other model can replace the recommended one?",
                "context": page_context,
            },
        )

    assert response.status_code == 200
    call = generate.call_args
    assert page_context in call.args[0]
    assert "What other model can replace" in call.args[0]
    system_prompt = call.kwargs["system_prompt"]
    assert "verified compatible alternatives" in system_prompt
    assert "3–6 concise bullets" in system_prompt
    assert "conditional option" in system_prompt


def test_model_recommendation_can_use_ai_review(
    client,
    test_user,
    sample_analysis_data,
    monkeypatch,
):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _login(client, test_user)
    data = {**sample_analysis_data, "use_ai_review": "on"}

    def build_review(**kwargs):
        selected_model = kwargs["candidate_models"][0]
        return {
            "recommended_model": selected_model,
            "summary": "The verified leader remains the best fit.",
            "why_it_fits": ["It matches the stated outcome and goal."],
            "assumptions_to_check": ["Use held-out validation."],
            "alternative_tradeoffs": [],
        }

    with patch(
        "routes.main_routes.review_recommendation",
        side_effect=build_review,
    ) as review:
        response = client.post("/results", data=data)

    assert response.status_code == 200
    assert b"AI-assisted review" in response.data
    assert b"The verified leader remains the best fit." in response.data
    assert review.call_count == 1
    assert AIUsageEvent.query.filter_by(user_id=test_user["id"]).count() == 1


def test_model_recommendation_falls_back_when_ai_review_fails(
    client,
    test_user,
    sample_analysis_data,
    monkeypatch,
):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _login(client, test_user)
    data = {**sample_analysis_data, "use_ai_review": "on"}

    with patch(
        "routes.main_routes.review_recommendation",
        side_effect=OpenAIServiceError("Provider unavailable.", 503),
    ):
        response = client.post("/results", data=data)

    assert response.status_code == 200
    assert b"rules-based recommendation is still complete" in response.data


def test_admin_ai_page_never_renders_api_key(client, admin_user, monkeypatch):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk_do_not_render_this_secret")
    _login(client, admin_user)

    response = client.get("/admin/ai_settings")

    assert response.status_code == 200
    assert b"sk_do_not_render_this_secret" not in response.data
    assert b"Configured" in response.data


def test_admin_can_initialize_missing_ai_usage_storage(
    app,
    admin_client,
    monkeypatch,
):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with app.app_context():
        AIUsageEvent.__table__.drop(bind=db.engine)

    status_response = admin_client.get("/admin/ai_settings")

    assert status_response.status_code == 200
    assert b"Needs initialization" in status_response.data
    assert b"Not initialized" in status_response.data

    initialize_response = admin_client.post(
        "/admin/initialize-ai-storage",
        json={"confirm": True},
    )

    assert initialize_response.status_code == 200
    assert initialize_response.get_json()["storage_ready"] is True
    with app.app_context():
        assert inspect(db.engine).has_table("ai_usage_events")

    ready_response = admin_client.get("/admin/ai_settings")
    assert b"Overall readiness" in ready_response.data
    assert b"Ready" in ready_response.data


def test_ai_storage_initialization_requires_explicit_confirmation(admin_client):
    response = admin_client.post(
        "/admin/initialize-ai-storage",
        json={"confirm": False},
    )

    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_chatbot_explains_missing_ai_usage_storage(
    app,
    client,
    test_user,
    monkeypatch,
):
    monkeypatch.setenv("AI_ENHANCEMENT_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _login(client, test_user)
    with app.app_context():
        AIUsageEvent.__table__.drop(bind=db.engine)

    response = client.post(
        "/chatbot/ask",
        json={"question": "Which model?", "context": "Model selection"},
    )

    assert response.status_code == 503
    assert "usage storage is not ready" in response.get_json()["response"]


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
