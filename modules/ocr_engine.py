from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml
from PIL import Image


@dataclass
class OcrLine:
    text: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float


def _bbox_to_xyxy(bbox: Any) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# v70.5r: EasyOCR → RapidOCR 교체
# ---------------------------------------------------------------------------

def rapidocr_available() -> bool:
    """rapidocr-onnxruntime 설치 여부를 확인합니다."""
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


# 하위 호환: app.py 에서 easyocr_available 이름으로 import 합니다.
easyocr_available = rapidocr_available


def run_rapidocr(
    image: Image.Image,
    lang: List[str] | None = None,
    gpu: bool = False,
    use_korean_model: bool = True,
) -> List[OcrLine]:
    """이미지를 RapidOCR 로 인식합니다.

    lang 파라미터는 하위 호환성을 위해 받지만 무시됩니다.
    RapidOCR 은 Korean 모델(data/ko_rec.onnx 또는 자동 다운로드)로 한글을 인식합니다.
    """
    import numpy as np
    from .rapidocr_adapter import get_rapidocr_reader  # type: ignore

    reader = get_rapidocr_reader(gpu=False, use_korean_model=use_korean_model)
    arr = np.array(image.convert("RGB"))
    result = reader.readtext(arr, detail=1, paragraph=False)
    lines: List[OcrLine] = []
    for item in result:
        try:
            bbox, text, conf = item[0], str(item[1]), float(item[2]) if len(item) > 2 else 0.5
        except Exception:
            continue
        x1, y1, x2, y2 = _bbox_to_xyxy(bbox)
        text = text.strip()
        if text:
            lines.append(OcrLine(text=text, x1=x1, y1=y1, x2=x2, y2=y2, conf=float(conf)))
    return lines


# 하위 호환: 기존 run_easyocr 이름으로 호출 가능
run_easyocr = run_rapidocr


def load_ocr_config(config_path: str | Path = "configs/ocr_columns.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_header(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(text)).lower()


def header_key_from_text(text: str, config: Dict[str, Any]) -> str | None:
    t = _normalize_header(text)
    for key, meta in config.get("columns", {}).items():
        for alias in meta.get("aliases", []):
            a = _normalize_header(alias)
            if a and (a in t or t in a):
                return key
    return None


def group_lines_by_row(lines: List[OcrLine], y_tolerance: float = 18.0) -> List[List[OcrLine]]:
    sorted_lines = sorted(lines, key=lambda l: (l.y1 + l.y2) / 2)
    rows: List[List[OcrLine]] = []
    for line in sorted_lines:
        cy = (line.y1 + line.y2) / 2
        placed = False
        for row in rows:
            row_cy = sum((r.y1 + r.y2) / 2 for r in row) / len(row)
            if abs(cy - row_cy) <= y_tolerance:
                row.append(line)
                placed = True
                break
        if not placed:
            rows.append([line])
    for row in rows:
        row.sort(key=lambda l: l.x1)
    return rows


def detect_header_positions(rows: List[List[OcrLine]], config: Dict[str, Any]) -> Tuple[Dict[str, float], int]:
    """헤더 행과 컬럼 x좌표를 찾습니다. 실패하면 빈 dict."""
    best_score = 0
    best_map: Dict[str, float] = {}
    best_idx = -1
    for idx, row in enumerate(rows[:12]):
        col_map: Dict[str, float] = {}
        for line in row:
            key = header_key_from_text(line.text, config)
            if key:
                col_map[key] = (line.x1 + line.x2) / 2
        if len(col_map) > best_score:
            best_score = len(col_map)
            best_map = col_map
            best_idx = idx
    return best_map, best_idx


def nearest_col(x: float, col_positions: Dict[str, float]) -> str | None:
    if not col_positions:
        return None
    return min(col_positions.keys(), key=lambda key: abs(x - col_positions[key]))


def parse_battle_table_from_lines(lines: List[OcrLine], config_path: str | Path = "configs/ocr_columns.yaml") -> pd.DataFrame:
    config = load_ocr_config(config_path)
    rows = group_lines_by_row(lines)
    col_positions, header_idx = detect_header_positions(rows, config)
    labels = {k: v.get("label", k) for k, v in config.get("columns", {}).items()}
    ordered_keys = list(config.get("columns", {}).keys())

    # 헤더를 충분히 못 찾으면 좌측부터 기본 순서로 추정
    if len(col_positions) < 3:
        xs = sorted({round((l.x1 + l.x2) / 2, 0) for l in lines})
        # 너무 많은 x가 나오므로 일반 전투분석기 순서 기준으로 대략 분포 생성
        if lines:
            min_x = min(l.x1 for l in lines)
            max_x = max(l.x2 for l in lines)
            step = (max_x - min_x) / max(1, len(ordered_keys) - 1)
            col_positions = {key: min_x + i * step for i, key in enumerate(ordered_keys)}
            header_idx = 0

    data_rows: List[Dict[str, str]] = []
    start_idx = header_idx + 1 if header_idx >= 0 else 0
    for row in rows[start_idx:]:
        # 너무 위/아래 UI 잡음 제거용: 한 행에 2개 미만이면 제외
        if len(row) < 2:
            continue
        item: Dict[str, List[str]] = {key: [] for key in ordered_keys}
        for line in row:
            cx = (line.x1 + line.x2) / 2
            key = nearest_col(cx, col_positions)
            if key:
                item[key].append(line.text)
        merged = {labels[key]: " ".join(item[key]).strip() for key in ordered_keys}
        # 이름/피해량이 둘 다 없으면 스킵
        if not merged.get(labels.get("name", "name")) and not merged.get(labels.get("damage_text", "damage_text")):
            continue
        data_rows.append(merged)
    return pd.DataFrame(data_rows)


def image_from_upload(uploaded_file: Any) -> Image.Image:
    return Image.open(io.BytesIO(uploaded_file.getvalue()))


def empty_battle_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"이름": "허리케인 소드", "피해량": "", "초당 피해량": "", "백어택 적중률": "", "헤드어택 적중률": "", "치명타 적중률": "", "사용 횟수": "", "쿨타임 비율": "", "피해량 지분": ""},
        {"이름": "브루탈 임팩트", "피해량": "", "초당 피해량": "", "백어택 적중률": "", "헤드어택 적중률": "", "치명타 적중률": "", "사용 횟수": "", "쿨타임 비율": "", "피해량 지분": ""},
    ])
