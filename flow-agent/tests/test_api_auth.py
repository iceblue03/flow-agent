# -*- coding: utf-8 -*-
"""
Google Flow Agent API Key 인증 기능 단위 테스트 (Unit Tests).

이 테스트 모듈은 FastAPI surface 상에서 다음 핵심 동작을 검증합니다:
1. API Key가 등록되지 않은 상태(개발 모드)에서 하위 호환성 유지 (무인증 요청 허용).
2. API Key가 등록되었거나 FLOW_API_KEY 환경 변수가 설정된 상태에서 비인가 요청 차단 (401 반환).
3. 유효하지 않은 API Key 전달 시 401 오류 반환.
4. 유효한 'Authorization: Bearer <key>' 헤더 전달 시 인가 성공.
5. 유효한 'X-API-Key: <key>' 헤더 전달 시 인가 성공.
6. 공개 엔드포인트(/health, /)는 API Key 없이 상시 접근 가능.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from flow_agent import api, auth
from flow_agent.auth import (
    create_api_key,
    delete_api_key,
    get_active_keys,
    is_auth_enforced,
    revoke_api_key,
    verify_api_key,
)


class FakeEngine:
    """테스트를 위한 가짜 FlowEngine 스텁 클래스."""

    binary = Path("/tmp/fake/flow")

    def available(self) -> bool:
        return True

    async def generate_image(self, **kwargs):
        return {
            "job_id": "test-job-1",
            "account_id": "test-acct-1",
            "status": "succeeded",
            "files": [{"path": "/tmp/fake_output.jpg"}],
        }

    async def generate_video(self, **kwargs):
        return {
            "job_id": "test-job-2",
            "status": "submitted",
            "files": [],
        }

    async def get_balance(self, refresh=False):
        return {"accounts": [], "total_credits": 100, "unread": 0}

    async def get_stats(self):
        return {"total_generations": 1}

    async def get_projects(self, cookies=None):
        return []


class TestApiKeyAuthentication(unittest.TestCase):
    """API Key 인증 및 엔드포인트 보안 검증 테스트 케이스."""

    def setUp(self):
        # 임시 디렉터리 생성하여 격리된 키 파일 환경 구축
        self.temp_dir = tempfile.mkdtemp()
        self.key_file = Path(self.temp_dir) / "api_keys.json"

        # 테스트용 스텁 엔진 주입
        self.fake_engine = FakeEngine()
        api.engine = self.fake_engine

        # DEFAULT_KEY_FILE을 임시 파일로 패치
        self.auth_file_patcher = patch.object(auth, "DEFAULT_KEY_FILE", self.key_file)
        self.auth_file_patcher.start()

        # 환경 변수 백업 및 초기화
        self.orig_env_key = os.environ.get("FLOW_API_KEY")
        if "FLOW_API_KEY" in os.environ:
            del os.environ["FLOW_API_KEY"]

        # FastAPI TestClient 생성
        self.client = TestClient(api.app)

    def tearDown(self):
        self.auth_file_patcher.stop()
        if self.orig_env_key is not None:
            os.environ["FLOW_API_KEY"] = self.orig_env_key
        elif "FLOW_API_KEY" in os.environ:
            del os.environ["FLOW_API_KEY"]

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_unauthenticated_mode_when_no_keys_configured(self):
        """키가 전혀 등록되지 않은 경우 하위 호환성을 위해 요청이 통과하는지 검증합니다."""
        self.assertFalse(is_auth_enforced(self.key_file))

        # 인증 헤더 없이 요청
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)

        # 이미지 생성 엔드포인트도 개발 모드에서는 통과
        res = self.client.post(
            "/api/v1/image",
            json={"prompt": "test prompt", "aspect": "1:1"},
        )
        self.assertEqual(res.status_code, 200)

    def test_auth_enforced_blocks_missing_key(self):
        """키가 등록된 상태에서 인증 헤더가 누락되면 401 오류가 발생하는지 검증합니다."""
        # 키 생성
        entry = create_api_key(name="test-device", key_file=self.key_file)
        self.assertTrue(is_auth_enforced(self.key_file))

        # 보호 대상 엔드포인트에 인증 없이 요청 -> 401 반환해야 함
        res = self.client.post(
            "/api/v1/image",
            json={"prompt": "test prompt", "aspect": "1:1"},
        )
        self.assertEqual(res.status_code, 401)
        self.assertIn("API Key가 누락되었습니다", res.json()["detail"])

    def test_auth_blocks_invalid_key(self):
        """유효하지 않은 API Key로 요청 시 401 오류가 발생하는지 검증합니다."""
        create_api_key(name="test-device", key_file=self.key_file)

        headers = {"Authorization": "Bearer sk-flow-invalid-token-12345"}
        res = self.client.post(
            "/api/v1/image",
            json={"prompt": "test prompt", "aspect": "1:1"},
            headers=headers,
        )
        self.assertEqual(res.status_code, 401)
        self.assertIn("유효하지 않거나 비활성화되었습니다", res.json()["detail"])

    def test_auth_succeeds_with_valid_bearer_token(self):
        """OpenAI 표준 'Authorization: Bearer <key>' 전달 시 성공적으로 통과하는지 검증합니다."""
        entry = create_api_key(name="test-device", key_file=self.key_file)
        valid_key = entry["key"]

        headers = {"Authorization": f"Bearer {valid_key}"}
        res = self.client.post(
            "/api/v1/image",
            json={"prompt": "test prompt", "aspect": "1:1"},
            headers=headers,
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["job_id"], "test-job-1")

    def test_auth_succeeds_with_x_api_key_header(self):
        """'X-API-Key: <key>' 헤더 전달 시 성공적으로 통과하는지 검증합니다."""
        entry = create_api_key(name="test-device", key_file=self.key_file)
        valid_key = entry["key"]

        headers = {"X-API-Key": valid_key}
        res = self.client.get("/v1/balance", headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["total_credits"], 100)

    def test_env_var_flow_api_key_acts_as_master_key(self):
        """환경 변수 FLOW_API_KEY가 마스터 키로 정상 작동하는지 검증합니다."""
        os.environ["FLOW_API_KEY"] = "sk-flow-master-secret-key"
        self.assertTrue(is_auth_enforced(self.key_file))

        headers = {"Authorization": "Bearer sk-flow-master-secret-key"}
        res = self.client.get("/v1/balance", headers=headers)
        self.assertEqual(res.status_code, 200)

    def test_revoked_key_is_rejected(self):
        """비활성화(revoked)된 키는 거부되는지 검증합니다."""
        entry = create_api_key(name="temp-key", key_file=self.key_file)
        valid_key = entry["key"]

        # 비활성화 실행
        revoke_api_key("temp-key", key_file=self.key_file)

        headers = {"Authorization": f"Bearer {valid_key}"}
        res = self.client.get("/v1/balance", headers=headers)
        # 키가 비활성화되어 유효 키가 0개가 되면 하위 호환 모드가 되거나 거부됨
        # 만약 다른 활성 키가 있다면 명확히 401
        create_api_key(name="another-active-key", key_file=self.key_file)

        res = self.client.get("/v1/balance", headers=headers)
        self.assertEqual(res.status_code, 401)

    def test_public_endpoints_remain_accessible(self):
        """키가 강제되는 상황에서도 /health 등 공개 엔드포인트는 인증 없이 접근 가능한지 검증합니다."""
        create_api_key(name="test-device", key_file=self.key_file)

        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
