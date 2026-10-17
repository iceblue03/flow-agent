# -*- coding: utf-8 -*-
"""
Google Flow Agent API Key 인증 및 접근 제어 모듈.

이 모듈은 외부 네트워크 노출 시 무단 이미지 및 비디오 생성을 방지하기 위한
API Key의 생성, 안전한 파일 저장, 조회, 비활성화(폐기),
그리고 FastAPI 요청 수신 시 암호학적 상시 시간(constant-time) 검증 기능을 제공합니다.

OpenAI Python SDK 표준(Authorization: Bearer <key>) 및 범용 HTTP 클라이언트의
X-API-Key 헤더를 모두 지원하며, 등록된 키가 없는 경우에는 기존 로컬 개발 환경과의
하위 호환성을 위해 무인증(개발 모드)으로 투명하게 동작합니다.
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

# API 키 접두사 및 기본 저장 경로
KEY_PREFIX = "sk-flow-"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_KEY_FILE = DEFAULT_DATA_DIR / "api_keys.json"

# OpenAPI(Swagger UI) 자물쇠 버튼 활성화를 위한 보안 스키마 정의
# auto_error=False로 설정하여 미입력 시에도 내부 verify_api_key에서 하위 호환성(미설정 시 통과)을 제어합니다.
security_bearer = HTTPBearer(
    auto_error=False,
    description="OpenAI 표준 규격 Bearer API Key ('sk-flow-...' 형식)",
)
security_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="HTTP 헤더 기반 API Key ('sk-flow-...' 형식)",
)


def get_key_file_path(custom_path: Optional[Path] = None) -> Path:
    """
    API 키 저장소 파일의 절대 경로를 반환하며 부모 디렉터리가 없으면 생성합니다.

    Args:
        custom_path: 사용자 지정 파일 경로 (기본값: data/api_keys.json)

    Returns:
        Path: 보장된 키 파일 경로
    """
    path = custom_path or DEFAULT_KEY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def generate_secret_key(prefix: str = KEY_PREFIX) -> str:
    """
    암호학적으로 안전한 무작위 난수 기반의 API Key 문자열을 생성합니다.

    Python 표준 secrets 모듈을 사용하여 32바이트(256비트) 엔트로피의
    URL-safe Base64 문자열을 생성하고 접두사를 부여합니다.
    """
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}{random_part}"


def load_api_keys(key_file: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    디스크에 저장된 API 키 목록을 불러옵니다.

    파일이 존재하지 않거나 유효하지 않은 JSON 형식인 경우 빈 리스트를 반환합니다.
    """
    path = get_key_file_path(key_file)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "keys" in data and isinstance(data["keys"], list):
            return data["keys"]
        if isinstance(data, list):
            return data
        return []
    except Exception:
        # 파일 손상 또는 읽기 실패 시 안전하게 빈 목록 반환
        return []


def save_api_keys(keys: List[Dict[str, Any]], key_file: Optional[Path] = None) -> None:
    """
    API 키 목록을 디스크에 원자적(atomic) 방식으로 안전하게 저장합니다.

    임시 파일에 먼저 기록한 뒤 os.replace를 수행하여 쓰기 도중 프로세스가
    중단되더라도 원본 파일이 깨지지 않도록 보호합니다.
    """
    path = get_key_file_path(key_file)
    tmp_path = path.with_suffix(".tmp")

    payload = {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "keys": keys,
    }

    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    # 파일 권한을 소유자 읽기/쓰기로 제한 (보안 강화)
    try:
        os.chmod(tmp_path, 0o600)
    except OSError:
        pass
    tmp_path.replace(path)


def create_api_key(
    name: str = "default",
    key_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    새로운 API 키를 생성하고 저장소에 등록한 뒤 메타데이터를 반환합니다.

    Args:
        name: 키 식별자 별칭 (예: 'my-laptop', 'production-bot')
        key_file: 저장할 파일 경로 (기본값: data/api_keys.json)

    Returns:
        Dict[str, Any]: 생성된 키 정보 (key, name, created_at, active)
    """
    key_value = generate_secret_key()
    new_entry = {
        "key": key_value,
        "name": name.strip() or "default",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active": True,
    }

    keys = load_api_keys(key_file)
    keys.append(new_entry)
    save_api_keys(keys, key_file)
    return new_entry


def list_api_keys(key_file: Optional[Path] = None) -> List[Dict[str, Any]]:
    """등록된 모든 API 키 메타데이터 목록을 반환합니다."""
    return load_api_keys(key_file)


def mask_key(key: str) -> str:
    """
    화면 출력 또는 로깅 시 보안을 위해 API 키의 중간 부분을 마스킹합니다.
    예: 'sk-flow-abc123456789...wxyz' -> 'sk-flow-abc1...wxyz'
    """
    if len(key) <= 16:
        return "****"
    return f"{key[:12]}...{key[-4:]}"


def revoke_api_key(target: str, key_file: Optional[Path] = None) -> bool:
    """
    API Key 문자열 또는 키 이름(별칭)을 기준으로 키를 비활성화(revoked) 처리합니다.

    Args:
        target: 비활성화할 API 키 전체 문자열 또는 이름(name)
        key_file: 대상 파일 경로

    Returns:
        bool: 성공적으로 대상을 찾아 비활성화했으면 True, 미발견 시 False
    """
    keys = load_api_keys(key_file)
    found = False

    for item in keys:
        if item.get("key") == target or item.get("name") == target:
            item["active"] = False
            item["revoked_at"] = datetime.now(timezone.utc).isoformat()
            found = True

    if found:
        save_api_keys(keys, key_file)
    return found


def delete_api_key(target: str, key_file: Optional[Path] = None) -> bool:
    """
    API Key를 저장소에서 영구 삭제합니다.
    """
    keys = load_api_keys(key_file)
    initial_len = len(keys)
    keys = [item for item in keys if item.get("key") != target and item.get("name") != target]

    if len(keys) < initial_len:
        save_api_keys(keys, key_file)
        return True
    return False


def get_active_keys(key_file: Optional[Path] = None) -> List[str]:
    """
    현재 인증에 유효한 모든 활성 API 키 문자열 집합을 반환합니다.
    환경 변수 'FLOW_API_KEY'가 설정되어 있는 경우 해당 마스터 키도 포함됩니다.
    """
    active_keys: List[str] = []

    # 1. 환경 변수 기반 마스터 키 확인
    env_master = os.environ.get("FLOW_API_KEY", "").strip()
    if env_master:
        active_keys.append(env_master)

    # 2. 파일 저장소 기반 등록 키 확인
    for item in load_api_keys(key_file):
        if item.get("active", True) and "key" in item:
            active_keys.append(item["key"])

    return active_keys


def is_auth_enforced(key_file: Optional[Path] = None) -> bool:
    """
    현재 서버가 API Key 인증 필수 모드인지 확인합니다.

    환경 변수 'FLOW_API_KEY'가 지정되어 있거나, 등록된 활성 키가 1개 이상 존재하면
    인증 필수 모드(True)로 동작합니다.
    키가 전무한 초기 상태에서는 로컬 하위 호환성을 위해 무인증(False)으로 동작합니다.
    """
    return len(get_active_keys(key_file)) > 0


def verify_api_key(
    request: Request,
    bearer: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    x_api_key: Optional[str] = Security(security_header),
) -> Optional[str]:
    """
    FastAPI 의존성(Depends) 함수로서 수신된 요청의 API Key 유효성을 검증합니다.

    검사 우선순위:
    1. HTTP 'Authorization: Bearer <token>' 헤더 (OpenAI SDK 표준)
    2. HTTP 'X-API-Key: <token>' 헤더 (일반 HTTP 클라이언트/cURL)
    3. URL 쿼리 파라미터 '?api_key=<token>' (브라우저 직접 다운로드 및 미디어 접근용)

    보안 비교:
    - 타이밍 공격(timing attack)을 방지하기 위해 secrets.compare_digest를 사용합니다.

    오류 처리:
    - 인증 필수 모드에서 유효한 키가 누락되었거나 일치하지 않을 경우 HTTP 401 오류를 발생시킵니다.
    """
    valid_keys = get_active_keys()

    # 등록된 키가 없으면 로컬 개발 하위 호환성을 위해 통과
    if not valid_keys:
        return None

    # 클라이언트가 전달한 토큰 추출
    client_token: Optional[str] = None

    if bearer and bearer.credentials:
        client_token = bearer.credentials.strip()
    elif x_api_key:
        client_token = x_api_key.strip()
    else:
        # 쿼리 파라미터 대체 확인
        query_key = request.query_params.get("api_key")
        if query_key:
            client_token = query_key.strip()

    if not client_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key가 누락되었습니다. 'Authorization: Bearer <key>' 또는 'X-API-Key' 헤더를 제공해야 합니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 상시 시간(constant-time) 비교를 통한 토큰 유효성 검사
    for valid_key in valid_keys:
        if secrets.compare_digest(client_token, valid_key):
            return client_token

    # 일치하는 키가 없는 경우 접근 거부
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="제공된 API Key가 유효하지 않거나 비활성화되었습니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )
