"""A shared password gate for the hosted preview, not a public account system."""

from base64 import b64decode
from binascii import Error
from secrets import compare_digest

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class PreviewAccessMiddleware:
    def __init__(self, app: ASGIApp, *, password: str) -> None:
        self.app = app
        self.credentials = f"subarag:{password}".encode("utf-8")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or (
            scope["path"] == "/healthz" and scope["method"] in {"GET", "HEAD"}
        ):
            await self.app(scope, receive, send)
            return

        authorization = dict(scope.get("headers", [])).get(b"authorization", b"")
        scheme, _, token = authorization.partition(b" ")
        credentials = b""
        if scheme.lower() == b"basic":
            try:
                credentials = b64decode(token, validate=True)
            except (ValueError, Error):
                pass
        if compare_digest(credentials, self.credentials):
            await self.app(scope, receive, send)
            return

        await JSONResponse(
            {"error": {"code": "authentication_required", "message": "Sign in to this preview.", "retryable": False}},
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="SubaRAG", charset="UTF-8"', "Cache-Control": "no-store"},
        )(scope, receive, send)
