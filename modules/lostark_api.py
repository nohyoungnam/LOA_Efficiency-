from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
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
    def __init__(self, jwt_token: str, config_path: str | Path = "configs/api_endpoints.yaml", timeout: int = 8):
        self.jwt_token = self._normalize_token(jwt_token)
        self.timeout = timeout
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.base_url = self.config.get("base_url", "https://developer-lostark.game.onstove.com").rstrip("/")
        self.last_timings: Dict[str, float] = {}

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
        if token.lower().startswith("bearer "):
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
        t0 = time.perf_counter()
        try:
            res = self._session.get(url, timeout=self.timeout)
            self.last_timings[path] = round((time.perf_counter() - t0) * 1000.0, 3)
            if res.status_code == 204:
                return ApiResult(True, res.status_code, None)
            try:
                payload = res.json()
            except json.JSONDecodeError:
                payload = res.text
            if 200 <= res.status_code < 300:
                return ApiResult(True, res.status_code, payload)
            return ApiResult(False, res.status_code, payload, error=f"HTTP {res.status_code}")
        except Exception as e:  # noqa: BLE001
            self.last_timings[path] = round((time.perf_counter() - t0) * 1000.0, 3)
            return ApiResult(False, 0, None, error=str(e))

    def _bundle_from_summary(self, summary_result: ApiResult) -> Dict[str, ApiResult]:
        if not summary_result.ok or not isinstance(summary_result.data, dict):
            return {"summary": summary_result}

        data = summary_result.data
        mapping = {
            "profiles": "ArmoryProfile",
            "equipment": "ArmoryEquipment",
            "combat_skills": "ArmorySkills",
            "engravings": "ArmoryEngraving",
            "cards": "ArmoryCard",
            "gems": "ArmoryGem",
            "arkpassive": "ArkPassive",
            "arkgrid": "ArkGrid",
        }
        bundle: Dict[str, ApiResult] = {"summary": summary_result}
        for key, summary_key in mapping.items():
            bundle[key] = ApiResult(True, summary_result.status_code, data.get(summary_key))
        return bundle

    def fetch_armory_bundle(self, character_name: str) -> Dict[str, ApiResult]:
        encoded_name = quote(character_name.strip(), safe="")
        endpoints = self.config.get("endpoints", {})

        summary_meta = endpoints.get("summary") or {}
        if self.config.get("use_summary_first", False) and summary_meta.get("enabled", True):
            summary_path = summary_meta["path"].format(characterName=encoded_name)
            summary_result = self.get(summary_path)
            if (
                summary_result.ok
                and isinstance(summary_result.data, dict)
                and "ArmoryProfile" in summary_result.data
            ):
                return self._bundle_from_summary(summary_result)

        tasks = {
            key: meta["path"].format(characterName=encoded_name)
            for key, meta in endpoints.items()
            if key != "summary" and meta.get("enabled", True)
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
                except Exception as e:  # noqa: BLE001
                    results[key] = ApiResult(False, 0, None, error=str(e))

        return {key: results[key] for key in tasks if key in results}


def serializable_bundle(bundle: Dict[str, ApiResult]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, result in bundle.items():
        out[key] = {
            "ok": result.ok,
            "status_code": result.status_code,
            "data": result.data,
            "error": result.error,
        }
    return out
