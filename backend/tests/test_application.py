import asyncio
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from subarag.api.app import create_app
from subarag.api.settings import Settings
from subarag.rag import Citation, RagQuery, RagResult
from test_media import make_webm


def settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, elevenlabs_api_key="test-only-key", **overrides)


def run_request(method: str, path: str, *, engine=None, config=None, **kwargs):
    async def run():
        app = create_app(settings=config or settings(), rag_engine=engine)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def test_application_status_distinguishes_running_services_and_missing_rag():
    response = run_request("GET", "/api/status")
    assert response.status_code == 200
    assert response.json() == {
        "transcription": {"provider": "elevenlabs"},
        "rag": {"status": "unavailable"},
        "data": {"status": "not_configured", "revision": None},
    }


def test_stub_accepts_question_without_inventing_answers_or_citations():
    response = run_request(
        "POST", "/api/queries", json={"question": "  गोवा की राजधानी क्या है?  ", "language": "hi"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "गोवा की राजधानी क्या है?"
    assert body["status"] == "unavailable"
    assert body["answer"] is None
    assert body["citations"] == []
    assert body["message"] == "Your question is ready. Answers are coming soon."
    assert body["request_id"]
    assert body["latency_ms"] >= 0


@pytest.mark.parametrize("question", ["", " \n ", "a" * 2001, None])
def test_query_validation_rejects_invalid_input(question):
    response = run_request("POST", "/api/queries", json={"question": question})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_retrieval_engine_can_be_replaced_without_changing_api():
    class TestEngine:
        status = "ready"
        received = []

        async def answer(self, query):
            self.received.append(query)
            return RagResult(
                status="answered",
                answer="A fixture answer.",
                citations=[Citation(document_id="fixture-1", text="A fixture source.")],
            )

    engine = TestEngine()
    response = run_request(
        "POST", "/api/queries", engine=engine,
        json={"question": "A fixture question?", "language": " HI-IN "},
    )
    assert engine.received == [RagQuery(question="A fixture question?", language="hi-in")]
    assert response.json()["status"] == "answered"
    assert response.json()["citations"][0]["document_id"] == "fixture-1"


def test_engine_failure_is_sanitized():
    class BrokenEngine:
        status = "ready"

        async def answer(self, query):
            raise RuntimeError("secret-token fixture crash")

    response = run_request(
        "POST", "/api/queries", engine=BrokenEngine(), json={"question": "Test?"}
    )
    assert response.status_code == 502
    assert response.json()["error"]["retryable"] is True
    assert "secret-token" not in response.text


def test_engine_timeout_is_retryable_without_retranscription():
    class SlowEngine:
        status = "ready"

        async def answer(self, query):
            await asyncio.sleep(1)

    response = run_request(
        "POST", "/api/queries", engine=SlowEngine(),
        config=settings(query_timeout_seconds=0.01), json={"question": "Test?"},
    )
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "query_timeout"


@pytest.mark.parametrize("payload", [
    {"status": "answered", "answer": "Unsupported answer"},
    {"status": "unavailable", "answer": "Fabricated answer"},
    {"status": "no_answer", "citations": [{"document_id": "1", "text": "Invented source"}]},
])
def test_result_contract_rejects_ungrounded_or_misleading_results(payload):
    with pytest.raises(ValidationError):
        RagResult(**payload)


def test_built_frontend_and_api_share_origin_without_api_html_fallback(tmp_path: Path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/app.js").write_text("console.info('fixture')")
    (tmp_path / "index.html").write_text("<!doctype html><h1>SubaRAG fixture</h1>")
    config = settings(frontend_dist=tmp_path)
    response = run_request("GET", "/", config=config)
    assert response.status_code == 200
    assert "SubaRAG fixture" in response.text
    assert response.headers["cache-control"] == "no-cache"
    assert run_request("GET", "/assets/app.js", config=config).status_code == 200
    missing = run_request("GET", "/api/missing", config=config)
    assert missing.status_code == 404
    assert missing.headers["content-type"] == "application/json"


def test_browser_audio_reaches_elevenlabs_once_then_query_boundary():
    async def run():
        calls = []

        async def provider(request: httpx.Request):
            calls.append(request.url)
            assert request.url.host == "api.elevenlabs.io"
            return httpx.Response(200, json={"text": "What is Goa?", "language_code": "eng"})

        provider_client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
        app = create_app(settings=settings(), client_factory=lambda: provider_client)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                transcript = await client.post(
                    "/api/transcriptions",
                    files={"audio": ("capture.webm", make_webm(), "audio/webm")},
                )
                assert transcript.status_code == 200
                assert transcript.json()["provider"] == "elevenlabs"
                result = await client.post("/api/queries", json={
                    "question": transcript.json()["text"],
                    "language": transcript.json()["language"],
                })
                assert result.json()["status"] == "unavailable"
                assert len(calls) == 1
                override = await client.post(
                    "/api/transcriptions", data={"provider": "sarvam"},
                    files={"audio": ("capture.webm", make_webm(), "audio/webm")},
                )
                assert override.status_code == 400
                assert len(calls) == 1

    asyncio.run(run())


def test_provider_concurrency_is_bounded_and_slots_are_released():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def provider(request):
            entered.set()
            await release.wait()
            return httpx.Response(200, json={"text": "Fixture", "language_code": "eng"})

        provider_client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
        app = create_app(
            settings=settings(max_concurrent_transcriptions=1),
            client_factory=lambda: provider_client,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                async def transcribe():
                    return await client.post(
                        "/api/transcriptions",
                        files={"audio": ("capture.webm", make_webm(), "audio/webm")},
                    )

                first = asyncio.create_task(transcribe())
                await entered.wait()
                busy = await transcribe()
                assert busy.status_code == 503
                assert busy.json()["error"]["code"] == "service_busy"
                release.set()
                assert (await first).status_code == 200
                assert (await transcribe()).status_code == 200

    asyncio.run(run())
