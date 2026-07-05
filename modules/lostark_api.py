from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests
import yaml


@dataclass
class ApiResult:
    ok: bool
    status_code: int
    data: Any = None
    error: str = ""


class LostArkApiClient:
    def __init__(self, jwt_token: str, config_path: str | Path = "configs/api_endpoints.yaml", timeout: int = 15):
        self.jwt_token = self._normalize_token(jwt_token)
        self.timeout = timeout
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.base_url = self.config.get("base_url", "https://developer-lostark.game.onstove.com").rstrip("/")
        # v70.5: keep-alive/커넥션 풀링용 공유 세션. 매 요청마다 TCP/TLS 핸드셰이크를
        # 다시 하지 않으므로 동일 호스트로의 다중 요청이 훨씬 빨라집니다.
        self._session = requests.Session()
        self._session.headers.update({
            "accept": "application/json",
            "authorization": self.jwt_token,
        })
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    @staticmethod
    def _normalize_token(token: str) -> str:
        token = (token or "").strip()
        if not token:
            return ""
        lower = token.lower()
        if lower.startswith("bearer "):
            return token
        return f"bearer {token}"

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "accept": "application/json",
            "authorization": self.jwt_token,
        }

    def get(self, path: str) -> ApiResult:
        url = f"{self.base_url}{path}"
        try:
            res = self._session.get(url, timeout=self.timeout)
            if res.status_code == 204:
                return ApiResult(True, res.status_code, None)
            try:
                payload = res.json()
            except json.JSONDecodeError:
                payload = res.text
            if 200 <= res.status_code < 300:
                return ApiResult(True, res.status_code, payload)
            return ApiResult(False, res.status_code, payload, error=f"HTTP {res.status_code}")
        except Exception as e:  # noqa: BLE001 - 화면에 오류 표시용
            return ApiResult(False, 0, None, error=str(e))

    def fetch_armory_bundle(self, character_name: str) -> Dict[str, ApiResult]:
        """v70.5: 9개 엔드포인트를 순차 호출 대신 병렬로 동시에 요청합니다.

        기존에는 for 루프로 summary→profiles→...→arkgrid를 하나씩 기다려서
        총 소요시간 = 9개 요청 시간의 '합'이었습니다.
        병렬 처리하면 총 소요시간 ≈ 가장 느린 단일 요청 시간으로 줄어듭니다.
        반환 구조(dict[key] -> ApiResult)는 기존과 100% 동일합니다.
        """
        encoded_name = quote(character_name.strip(), safe="")
        endpoints = self.config.get("endpoints", {})
        tasks = {
            key: meta["path"].format(characterName=encoded_name)
            for key, meta in endpoints.items()
            if meta.get("enabled", True)
        }
        results: Dict[str, ApiResult] = {}
        if not tasks:
            return results

        max_workers = min(16, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {
                executor.submit(self.get, path): key
                for key, path in tasks.items()
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:  # noqa: BLE001 - 개별 요청 실패 격리
                    results[key] = ApiResult(False, 0, None, error=str(e))

        # 원래 endpoints 정의 순서를 유지해서 다운스트림 표시/디버그 일관성을 보장합니다.
        ordered = {key: results[key] for key in tasks if key in results}
        return ordered


def serializable_bundle(bundle: Dict[str, ApiResult]) -> Dict[str, Any]:
    """Streamlit session/debug용으로 ApiResult를 dict로 변환합니다."""
    out: Dict[str, Any] = {}
    for key, result in bundle.items():
        out[key] = {
            "ok": result.ok,
            "status_code": result.status_code,
            "data": result.data,
            "error": result.error,
        }
    return out
