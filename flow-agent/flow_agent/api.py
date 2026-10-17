"""FastAPI application: a native Flow API and an OpenAI-compatible one.

Two surfaces over one engine. The native routes speak the engine's own
vocabulary — aspect ratios, durations, quality keys — and the OpenAI routes
translate the shape an OpenAI client already sends, so an existing SDK or a chat
UI works without knowing anything about Flow.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .engine import EngineError, FlowEngine

app = FastAPI(
    title="flow-agent",
    version="1.0.0",
    description="Image and video generation on Google Flow.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = FlowEngine()


# OpenAI 규격 이미지 크기와 Google Flow 종횡비 매핑 테이블.
#
# OpenAI DALL-E 표준 해상도와 Flow 네이티브 비율(1:1, 16:9, 9:16 등)을 상호 변환합니다.
SIZE_TO_ASPECT = {
    "1024x1024": "1:1",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
    "1365x1024": "4:3",
    "1024x1365": "3:4",
    "512x512": "1:1",
    "256x256": "1:1",
    # 텍스트 형태의 별칭 지원
    "square": "1:1",
    "landscape": "16:9",
    "portrait": "9:16",
}

MEDIA_DIR = engine.output_dir


# --------------------------------------------------------------------------- #
# request models
# --------------------------------------------------------------------------- #


class ImageGenerationRequest(BaseModel):
    """OpenAI 호환 이미지 생성 요청 모델.

    OpenAI 공식 규격(prompt, model, n, size, response_format)을 충족하며,
    Swagger UI 미리보기 및 외부 클라이언트의 즉각적인 파일 수신을 위해
    'image' 포맷(바이너리 스트림 직접 반환)을 함께 지원합니다.
    """

    model_config = ConfigDict(extra="allow")

    prompt: str = Field(
        default="a glowing crystal lotus on calm water",
        description="이미지 생성을 위한 텍스트 프롬프트",
    )
    model: str = Field(
        default="beluga",
        description="Flow 이미지 생성 모델 (기본값: 'beluga')",
    )
    n: int = Field(default=1, ge=1, le=4, description="생성할 이미지 개수 (1-4)")
    size: str = Field(
        default="1024x1024",
        description="이미지 해상도 (1024x1024, 1792x1024, 1024x1792 등)",
    )
    response_format: str = Field(
        default="image",
        description=(
            "응답 형식: 'image' (Swagger UI 및 브라우저에서 즉시 미리보고 다운로드 가능한 이미지 파일 직접 반환), "
            "'url' (OpenAI 표준 URL JSON), 'b64_json' (Base64 JSON)"
        ),
    )
    # Flow 고유 설정 (OpenAI 클라이언트는 무시)
    all_accounts: bool = False
    cookies: Optional[str] = Field(default=None, description="인증 세션 쿠키 (선택 사항)")

    @field_validator("model", mode="before")
    @classmethod
    def sanitize_model(cls, v: Any) -> str:
        # 빈 문자열 또는 Swagger UI 기본값('string') 입력 시 beluga로 대체
        val = str(v).strip().lower() if v else "beluga"
        if val in ("narwhal", "string", "default", "none", "null", ""):
            return "beluga"
        return val

    @field_validator("cookies", mode="before")
    @classmethod
    def sanitize_cookies(cls, v: Any) -> Optional[str]:
        # Swagger UI의 기본 입력값('string') 및 더미 값 필터링
        if not v or str(v).strip().lower() in ("string", "null", "none", "undefined", '""', "''"):
            return None
        return str(v).strip()


class FlowImageRequest(BaseModel):
    prompt: str
    aspect: str = "1:1"
    count: int = Field(default=1, ge=1, le=4)
    model: str = "beluga"
    all_accounts: bool = False
    cookies: Optional[str] = None

    @field_validator("model", mode="before")
    @classmethod
    def sanitize_model(cls, v: Any) -> str:
        # Fallback to BELUGA if empty or set to legacy/unsupported models or placeholders
        val = str(v).strip().lower() if v else "beluga"
        if val in ("narwhal", "string", "default", "none", "null", ""):
            return "beluga"
        return val

    @field_validator("cookies", mode="before")
    @classmethod
    def sanitize_cookies(cls, v: Any) -> Optional[str]:
        # Filter out Swagger UI default placeholder ('string') and empty values
        if not v or str(v).strip().lower() in ("string", "null", "none", "undefined", '""', "''"):
            return None
        return str(v).strip()


class FlowVideoRequest(BaseModel):
    prompt: str
    aspect: str = "landscape"
    duration: str = "8s"
    quality: str = "720p"
    count: int = Field(default=1, ge=1, le=4)
    start_image: Optional[str] = None
    all_accounts: bool = False
    cookies: Optional[str] = None

    @field_validator("cookies", mode="before")
    @classmethod
    def sanitize_cookies(cls, v: Any) -> Optional[str]:
        # Filter out Swagger UI default placeholder ('string') and empty values
        if not v or str(v).strip().lower() in ("string", "null", "none", "undefined", '""', "''"):
            return None
        return str(v).strip()


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "flow"
    messages: list[ChatMessage]
    stream: bool = False
    # Flow-specific.
    size: str = "1024x1024"
    all_accounts: bool = False
    cookies: Optional[str] = None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _media_url(request: Request, path: str) -> str:
    """클라이언트가 생성된 미디어 파일을 직접 다운로드할 수 있는 완전한 절대 HTTP URL을 반환합니다.

    로컬호스트뿐만 아니라 외부 LAN IP 및 리버스 프록시 헤더(X-Forwarded-Proto, X-Forwarded-Host)를
    모두 고려하여 외부 클라이언트가 즉시 접속 가능한 주소를 조합합니다.
    """
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    base = f"{proto}://{host}".rstrip("/")
    return f"{base}/api/v1/media/{Path(path).name}"


def _no_image_detail(result: dict) -> str:
    """Why a run that exited zero produced nothing usable.

    The engine reports a refusal in the result rather than by failing, so this
    is where the two are told apart. The status and job id are named because
    they are what a caller needs to look the run up; `status` is `empty` for a
    refusal and `succeeded` for a job whose files could not be read.
    """
    return (
        "the engine produced no image "
        f"(status={result.get('status')!r}, job={result.get('job_id')!r}, "
        f"account={result.get('account_id')!r})"
    )


def _image_payload(request: Request, result: dict, response_format: str) -> dict:
    """Turn an engine result into the OpenAI images response."""
    files = result.get("files") or []
    data: list[dict] = []

    for item in files:
        path = item.get("path")
        if not path:
            continue
        if response_format == "b64_json":
            try:
                data.append(
                    {"b64_json": base64.b64encode(Path(path).read_bytes()).decode()}
                )
            except OSError:
                continue
        else:
            data.append({"url": _media_url(request, path)})

    return {"created": int(time.time()), "data": data, "flow": result}


def _prompt_from_messages(messages: list[ChatMessage]) -> str:
    """The text of the last user turn, which is what the engine generates from."""
    for message in reversed(messages):
        if message.role != "user":
            continue
        content = message.content
        if isinstance(content, str):
            return content
        # The newer OpenAI shape is a list of parts; take the text ones.
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            joined = " ".join(p for p in parts if p).strip()
            if joined:
                return joined
    return ""


def _completion_body(model: str, markdown: str) -> dict:
    return {
        "id": f"chatcmpl-flow-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": markdown},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse(body: dict) -> StreamingResponse:
    """Stream one completion as the two SSE frames an OpenAI client expects.

    The content is delivered in one delta rather than token by token: the engine
    produces an image, not prose, so there is nothing to stream progressively —
    but a client that asked for `stream: true` still needs the framing.
    """

    async def frames():
        head = {
            "id": body["id"],
            "object": "chat.completion.chunk",
            "created": body["created"],
            "model": body["model"],
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(head)}\n\n"

        chunk = {
            **head,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": body["choices"][0]["message"]["content"]},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        tail = {
            **head,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(tail)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(frames(), media_type="text/event-stream")


# --------------------------------------------------------------------------- #
# OpenAI-compatible
# --------------------------------------------------------------------------- #


@app.post(
    "/v1/images/generations",
    responses={
        200: {
            "content": {
                "application/json": {},
                "image/jpeg": {},
            },
            "description": "OpenAI 호환 JSON(URL/Base64) 또는 Swagger UI 미리보기용 이미지 파일",
        }
    },
)
@app.post(
    "/api/v1/images/generations",
    responses={
        200: {
            "content": {
                "application/json": {},
                "image/jpeg": {},
            },
            "description": "OpenAI 호환 JSON(URL/Base64) 또는 Swagger UI 미리보기용 이미지 파일",
        }
    },
)
async def openai_images(payload: ImageGenerationRequest, request: Request):
    """OpenAI 규격 이미지 생성 엔드포인트.

    - response_format='image' (또는 'file', 'binary', 'preview'):
      생성된 이미지 파일(image/jpeg)을 직접 스트리밍하여 Swagger UI 및 브라우저에서 즉시 그림으로 확인하고 다운로드할 수 있습니다.
    - response_format='url' (기본값):
      OpenAI 표준 형식에 맞추어 외부에서 접속 가능한 다운로드 URL을 포함한 JSON을 반환합니다.
    - response_format='b64_json':
      Base64로 인코딩된 이미지 문자열 JSON을 반환합니다.
    """
    aspect = SIZE_TO_ASPECT.get(payload.size.strip().lower())
    if aspect is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported size {payload.size!r}; use one of "
                f"{', '.join(sorted(SIZE_TO_ASPECT))}"
            ),
        )

    try:
        result = await engine.generate_image(
            prompt=payload.prompt,
            aspect=aspect,
            count=payload.n,
            model=payload.model,
            all_accounts=payload.all_accounts,
            cookies=payload.cookies,
        )
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    files = result.get("files") or []
    if not files:
        raise HTTPException(status_code=502, detail=_no_image_detail(result))

    fmt = (payload.response_format or "").strip().lower()

    # Swagger UI 미리보기 및 외부 브라우저 직접 다운로드를 위한 이미지 바이너리 즉시 반환
    if fmt in ("image", "file", "binary", "raw", "preview"):
        first_file = files[0].get("path")
        if first_file:
            img_path = Path(first_file)
            if img_path.is_file():
                return FileResponse(
                    img_path,
                    media_type="image/jpeg",
                    filename=img_path.name,
                )

    payload_out = _image_payload(request, result, payload.response_format)
    # A run that produced nothing is not a success. Returning 200 with an empty
    # `data` would tell an OpenAI client the request worked and there simply
    # were no images, which is the opposite of what happened.
    if not payload_out["data"]:
        raise HTTPException(status_code=502, detail=_no_image_detail(result))

    return payload_out


@app.post("/v1/chat/completions")
async def openai_chat(payload: ChatCompletionRequest, request: Request):
    """Generate from the last user turn and answer with a Markdown link."""
    prompt = _prompt_from_messages(payload.messages)
    if not prompt:
        raise HTTPException(status_code=400, detail="no user message with text to generate from")

    aspect = SIZE_TO_ASPECT.get(payload.size.strip().lower(), "1:1")

    try:
        result = await engine.generate_image(
            prompt=prompt,
            aspect=aspect,
            all_accounts=payload.all_accounts,
            cookies=payload.cookies,
        )
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    files = result.get("files") or []
    if not files:
        markdown = _no_image_detail(result)
    else:
        links = "\n".join(
            f"![generated image]({_media_url(request, f['path'])})" for f in files if f.get("path")
        )
        markdown = links or "The engine reported files but named no paths."

    body = _completion_body(payload.model, markdown)
    return _sse(body) if payload.stream else body


# --------------------------------------------------------------------------- #
# native
# --------------------------------------------------------------------------- #


@app.post("/v1/generate/image")
@app.post("/v1/image")
@app.post("/api/v1/image")
async def flow_image(payload: FlowImageRequest, request: Request):
    """Google Flow 고유 매개변수를 사용하는 네이티브 이미지 생성 엔드포인트.

    생성된 파일 목록(files)의 각 항목에 외부 클라이언트가 직접 다운로드할 수 있는
    완전한 HTTP URL(url)을 자동으로 추가하여 반환합니다.
    """
    try:
        result = await engine.generate_image(
            prompt=payload.prompt,
            aspect=payload.aspect,
            count=payload.count,
            model=payload.model,
            all_accounts=payload.all_accounts,
            cookies=payload.cookies,
        )
        # 외부 클라이언트가 직접 파일을 내려받을 수 있도록 다운로드 URL 주입
        if isinstance(result, dict) and "files" in result and isinstance(result["files"], list):
            for f in result["files"]:
                if isinstance(f, dict) and "path" in f:
                    f["url"] = _media_url(request, f["path"])
        return result
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/v1/generate/video")
@app.post("/v1/video")
@app.post("/api/v1/video")
async def flow_video(payload: FlowVideoRequest, request: Request):
    """비디오 생성 엔드포인트."""
    try:
        result = await engine.generate_video(
            prompt=payload.prompt,
            aspect=payload.aspect,
            duration=payload.duration,
            quality=payload.quality,
            count=payload.count,
            start_image=payload.start_image,
            all_accounts=payload.all_accounts,
            cookies=payload.cookies,
        )
        if isinstance(result, dict) and "files" in result and isinstance(result["files"], list):
            for f in result["files"]:
                if isinstance(f, dict) and "path" in f:
                    f["url"] = _media_url(request, f["path"])
        return result
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/v1/balance")
@app.get("/api/v1/balance")
async def flow_balance(refresh: bool = False):
    """Credits per account, cached by default and probed with `?refresh=true`."""
    try:
        return await engine.get_balance(refresh=refresh)
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/v1/stats")
@app.get("/api/v1/stats")
async def flow_stats():
    """Generation history and engine totals, read from the database."""
    try:
        return await engine.get_stats()
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/v1/projects")
@app.get("/api/v1/projects")
async def flow_projects(cookies: Optional[str] = None):
    try:
        return {"projects": await engine.get_projects(cookies=cookies)}
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/v1/media/{filename:path}")
@app.get("/api/v1/media/{filename:path}")
async def flow_media(filename: str):
    """생성된 미디어 파일을 안전하게 서빙하는 엔드포인트.

    엔진의 다양한 출력 디렉터리 경로 후보를 순회하여 파일을 탐색하고 전송합니다.
    디렉터리 탐색 공격(..)은 상위 디렉터리 경계 검사를 통해 차단합니다.
    """
    search_dirs = [
        MEDIA_DIR,
        engine.root / "output",
        Path(__file__).resolve().parent.parent / "output",
    ]
    for sdir in search_dirs:
        try:
            cand = (sdir / filename).resolve()
            if cand.is_file() and str(cand).startswith(str(sdir.resolve())):
                ext = cand.suffix.lower()
                media_type = (
                    "image/jpeg"
                    if ext in (".jpg", ".jpeg")
                    else "video/mp4"
                    if ext == ".mp4"
                    else "application/octet-stream"
                )
                return FileResponse(cand, media_type=media_type, filename=cand.name)
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="no such media file")


@app.get("/health")
async def health():
    return {"status": "ok", "engine": engine.available(), "binary": str(engine.binary)}


# --------------------------------------------------------------------------- #
# Web Preview Dashboard (웹 브라우저 실시간 미리보기 및 다운로드 페이지)
# --------------------------------------------------------------------------- #

PREVIEW_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flow Agent - OpenAI 이미지 생성 미리보기</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --accent: #e11d48;
            --accent-hover: #be123c;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border: #334155;
            --success: #22c55e;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem 1rem;
        }
        .container {
            width: 100%;
            max-width: 840px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
        }
        header {
            text-align: center;
            margin-bottom: 2rem;
        }
        h1 {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        .badge {
            display: inline-block;
            background: rgba(225, 29, 72, 0.2);
            color: var(--accent);
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        p.subtitle {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }
        .form-group {
            margin-bottom: 1.25rem;
        }
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--text-primary);
        }
        textarea, select {
            width: 100%;
            padding: 0.85rem;
            border-radius: 8px;
            background: #0b1120;
            border: 1px solid var(--border);
            color: #fff;
            font-size: 1rem;
            resize: vertical;
        }
        textarea:focus, select:focus {
            outline: none;
            border-color: var(--accent);
        }
        .row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }
        button.btn-generate {
            width: 100%;
            padding: 1rem;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            transition: background 0.2s, transform 0.1s;
            margin-top: 0.5rem;
        }
        button.btn-generate:hover {
            background: var(--accent-hover);
        }
        button.btn-generate:disabled {
            background: #475569;
            cursor: not-allowed;
        }
        .result-section {
            margin-top: 2rem;
            border-top: 1px solid var(--border);
            padding-top: 2rem;
            display: none;
        }
        .status-box {
            padding: 0.85rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            font-size: 0.95rem;
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            color: #60a5fa;
            text-align: center;
        }
        .image-container {
            text-align: center;
            background: #0b1120;
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid var(--border);
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
        }
        .action-bar {
            margin-top: 1rem;
            display: flex;
            gap: 1rem;
            justify-content: center;
        }
        .btn-action {
            padding: 0.6rem 1.2rem;
            border-radius: 6px;
            border: 1px solid var(--border);
            background: var(--bg-secondary);
            color: var(--text-primary);
            text-decoration: none;
            font-weight: 600;
            font-size: 0.9rem;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            cursor: pointer;
        }
        .btn-action.download {
            background: var(--success);
            border-color: var(--success);
            color: #fff;
        }
        .links {
            margin-top: 2rem;
            text-align: center;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        .links a {
            color: #38bdf8;
            text-decoration: none;
            margin: 0 0.5rem;
        }
        .links a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>⚡ Flow Agent <span class="badge">OpenAI DALL-E 호환</span></h1>
            <p class="subtitle">OpenAI 규격(<code>/v1/images/generations</code>)으로 이미지를 생성하고 즉시 브라우저에서 미리보기 및 다운로드합니다.</p>
        </header>

        <form id="genForm">
            <div class="form-group">
                <label for="prompt">프롬프트 (Prompt)</label>
                <textarea id="prompt" rows="3" required placeholder="예: a serene Japanese zen garden in autumn, cinematic lighting, 8k">a glowing crystal lotus on calm water</textarea>
            </div>

            <div class="row">
                <div class="form-group">
                    <label for="size">해상도 / 비율</label>
                    <select id="size">
                        <option value="1024x1024" selected>1024x1024 (정사각형 1:1)</option>
                        <option value="1792x1024">1792x1024 (와이드 16:9)</option>
                        <option value="1024x1792">1024x1792 (세로형 9:16)</option>
                        <option value="1365x1024">1365x1024 (가로형 4:3)</option>
                        <option value="1024x1365">1024x1365 (세로형 3:4)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="format">수신 방식</label>
                    <select id="format">
                        <option value="image" selected>이미지 직접 수신 (즉시 미리보기/다운로드)</option>
                        <option value="url">OpenAI 표준 JSON (URL 링크)</option>
                    </select>
                </div>
            </div>

            <button type="submit" id="btnSubmit" class="btn-generate">🎨 OpenAI 방식으로 이미지 생성하기</button>
        </form>

        <div id="resultSection" class="result-section">
            <div id="statusBox" class="status-box">생성 중...</div>
            <div id="imageBox" class="image-container" style="display: none;">
                <img id="resultImage" src="" alt="생성된 이미지" />
                <div class="action-bar">
                    <a id="btnDownload" href="#" download="flow_image.jpg" class="btn-action download">📥 이미지 다운로드</a>
                    <a id="btnNewTab" href="#" target="_blank" class="btn-action">🔍 새 탭에서 원본 보기</a>
                </div>
            </div>
        </div>

        <div class="links">
            <a href="/docs" target="_blank">📖 Swagger UI API 문서</a> |
            <a href="/redoc" target="_blank">📘 ReDoc</a> |
            <a href="/health" target="_blank">🩺 서버 상태(/health)</a>
        </div>
    </div>

    <script>
        const form = document.getElementById('genForm');
        const btnSubmit = document.getElementById('btnSubmit');
        const resultSection = document.getElementById('resultSection');
        const statusBox = document.getElementById('statusBox');
        const imageBox = document.getElementById('imageBox');
        const resultImage = document.getElementById('resultImage');
        const btnDownload = document.getElementById('btnDownload');
        const btnNewTab = document.getElementById('btnNewTab');

        let timerId = null;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const prompt = document.getElementById('prompt').value.trim();
            const size = document.getElementById('size').value;
            const format = document.getElementById('format').value;

            if (!prompt) return;

            btnSubmit.disabled = true;
            btnSubmit.innerText = '⏳ 이미지 생성 중... 잠시만 기다려 주세요';
            resultSection.style.display = 'block';
            imageBox.style.display = 'none';

            let startTime = Date.now();
            statusBox.style.background = 'rgba(59, 130, 246, 0.1)';
            statusBox.style.color = '#60a5fa';
            statusBox.innerText = `Google Flow 원격 생성 요청 중... (소요 시간: 0초)`;

            timerId = setInterval(() => {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                statusBox.innerText = `Google Flow 원격 생성 및 다운로드 중... (소요 시간: ${elapsed}초)`;
            }, 1000);

            try {
                const payload = {
                    prompt: prompt,
                    model: 'beluga',
                    n: 1,
                    size: size,
                    response_format: format
                };

                const res = await fetch('/v1/images/generations', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                clearInterval(timerId);
                const elapsedSec = ((Date.now() - startTime) / 1000).toFixed(1);

                if (!res.ok) {
                    const errText = await res.text();
                    statusBox.style.background = 'rgba(239, 68, 68, 0.1)';
                    statusBox.style.color = '#f87171';
                    statusBox.innerText = `[생성 오류 ${res.status}] ${errText}`;
                    btnSubmit.disabled = false;
                    btnSubmit.innerText = '🎨 OpenAI 방식으로 이미지 생성하기';
                    return;
                }

                statusBox.style.background = 'rgba(34, 197, 94, 0.1)';
                statusBox.style.color = '#4ade80';
                statusBox.innerText = `✓ 생성 성공! (소요 시간: ${elapsedSec}초)`;

                let imgUrl = '';
                if (format === 'image') {
                    const blob = await res.blob();
                    imgUrl = URL.createObjectURL(blob);
                } else {
                    const data = await res.json();
                    if (data.data && data.data[0] && data.data[0].url) {
                        imgUrl = data.data[0].url;
                    }
                }

                if (imgUrl) {
                    resultImage.src = imgUrl;
                    btnDownload.href = imgUrl;
                    btnNewTab.href = imgUrl;
                    imageBox.style.display = 'block';
                }

            } catch (err) {
                clearInterval(timerId);
                statusBox.style.background = 'rgba(239, 68, 68, 0.1)';
                statusBox.style.color = '#f87171';
                statusBox.innerText = `[통신 오류] ${err.message}`;
            } finally {
                btnSubmit.disabled = false;
                btnSubmit.innerText = '🎨 OpenAI 방식으로 이미지 생성하기';
            }
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
@app.get("/preview", response_class=HTMLResponse)
async def preview_dashboard():
    """브라우저에서 OpenAI API 방식으로 이미지를 즉시 생성하고 확인하는 실시간 미리보기 대시보드."""
    return HTMLResponse(content=PREVIEW_HTML)
