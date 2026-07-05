from __future__ import annotations

import io
import re
import hashlib
from pathlib import Path
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageEnhance

# 전투분석기 전체 화면 스크린샷 기준 좌표입니다.
# 1920x1080 기준으로 잡고, 실제 이미지 크기가 달라도 비율로 환산합니다.
# 좌표가 조금 안 맞으면 여기 숫자만 수정하면 됩니다.

ATTACK_GRID = {
    # 공격정보 탭의 실제 데이터 첫 행 top, 행 높이, 마지막 행 범위
    "row_start": 291 / 1080,
    "row_height": 43 / 1080,
    "row_count": 14,
    "columns": [
        # key, label, x1, x2
        ("name", "이름", 268 / 1920, 472 / 1920),
        ("damage_text", "피해량", 474 / 1920, 642 / 1920),
        ("dps_text", "초당 피해량", 646 / 1920, 812 / 1920),
        ("back_attack_rate", "백어택 적중률", 814 / 1920, 982 / 1920),
        ("crit_rate", "치명타 적중률", 984 / 1920, 1150 / 1920),
        ("casts", "사용 횟수", 1152 / 1920, 1320 / 1920),
        ("cooldown_rate", "쿨타임 비율", 1322 / 1920, 1490 / 1920),
        ("share_rate", "피해량 지분", 1492 / 1920, 1644 / 1920),
    ],
}



ATTACK_ICON_GRID = {
    # 공격정보 표의 스킬 아이콘 영역. 이름 칸 왼쪽 아이콘을 잘라 API 아이콘과 비교합니다.
    "x1": 228 / 1920,
    "x2": 268 / 1920,
}


# v30: 아이콘/셀 crop 여유값. 전투분석기 행이 조금만 밀려도 글자나 아이콘이 잘리면
# OCR/아이콘 매칭이 급격히 떨어져서, 기본적으로 각 셀을 5% 정도 넓게 자릅니다.
CELL_CROP_EXPAND_RATIO = 0.05
ICON_CROP_EXPAND_RATIO = 0.12

SUMMARY_ROIS = {
    # 종합정보 탭
    "elapsed_text": (250 / 1920, 210 / 1080, 500 / 1920, 236 / 1080),
    "total_damage_text": (260 / 1920, 290 / 1080, 545 / 1920, 390 / 1080),
    "dps_text": (600 / 1920, 290 / 1080, 920 / 1920, 390 / 1080),
    "damage_increase_efficiency": (995 / 1920, 290 / 1080, 1245 / 1920, 360 / 1080),
    "stagger_text": (1320 / 1920, 290 / 1080, 1635 / 1920, 390 / 1080),
    "crit_rate": (230 / 1920, 452 / 1080, 930 / 1920, 522 / 1080),
    "head_attack_rate": (230 / 1920, 524 / 1080, 930 / 1920, 594 / 1080),
    "back_attack_rate": (930 / 1920, 596 / 1080, 1640 / 1920, 666 / 1080),
    "damage_increase_uptime": (230 / 1920, 740 / 1080, 930 / 1920, 810 / 1080),
    "burst_uptime": (230 / 1920, 812 / 1080, 930 / 1920, 882 / 1080),
}

STANDARD_COLUMNS = [
    "이름",
    "피해량",
    "초당 피해량",
    "백어택 적중률",
    "헤드어택 적중률",
    "치명타 적중률",
    "사용 횟수",
    "쿨타임 비율",
    "피해량 지분",
]


def get_easyocr_reader(gpu: bool = True) -> Any:
    """OCR reader를 생성합니다 (RapidOCR 사용, EasyOCR 호환 API).

    v70.5r: EasyOCR(torch) → RapidOCR(onnxruntime) 으로 교체.
    torch 불필요 → HF Spaces 무료 CPU 배포 메모리 절감.
    gpu 파라미터는 하위 호환성 유지를 위해 남겨두지만 무시됩니다(RapidOCR은 CPU ONNX).
    """
    try:
        from .rapidocr_adapter import get_rapidocr_reader  # type: ignore
    except ImportError:
        from rapidocr_adapter import get_rapidocr_reader  # type: ignore
    return get_rapidocr_reader(gpu=False, use_korean_model=True)


# 하위 호환: get_ocr_reader 로도 호출 가능
get_ocr_reader = get_easyocr_reader


def _norm_box_to_pixels(image: Image.Image, box: Tuple[float, float, float, float], pad: int = 2) -> Tuple[int, int, int, int]:
    w, h = image.size
    x1, y1, x2, y2 = box
    return (
        max(0, int(x1 * w) + pad),
        max(0, int(y1 * h) + pad),
        min(w, int(x2 * w) - pad),
        min(h, int(y2 * h) - pad),
    )


def crop_norm(image: Image.Image, box: Tuple[float, float, float, float], pad: int = 2) -> Image.Image:
    return image.crop(_norm_box_to_pixels(image, box, pad=pad))


def preprocess_crop(crop: Image.Image, scale: int = 4) -> Image.Image:
    # 전투분석기 글자가 작아서 셀 단위로 크게 확대합니다.
    # EasyOCR은 원본 1920x1080이라도 셀 crop 안에서는 글자가 매우 작아서
    # crop 자체를 3~5배 키운 뒤 대비/선명도를 올리는 편이 안정적입니다.
    crop = crop.convert("RGB")
    if scale > 1:
        crop = crop.resize((max(1, crop.width * scale), max(1, crop.height * scale)), Image.Resampling.LANCZOS)
    crop = ImageEnhance.Brightness(crop).enhance(1.05)
    crop = ImageEnhance.Contrast(crop).enhance(1.65)
    crop = ImageEnhance.Sharpness(crop).enhance(1.65)
    return crop


def _ocr_text_once(reader: Any, crop: Image.Image, *, numeric: bool = False) -> str:
    arr = np.array(crop)
    try:
        if numeric:
            result = reader.readtext(
                arr,
                detail=0,
                paragraph=False,
                allowlist="0123456789.,:%-억만초분 ",
            )
        else:
            result = reader.readtext(arr, detail=0, paragraph=False)
    except TypeError:
        result = reader.readtext(arr, detail=0, paragraph=False)
    return clean_ocr_text(" ".join(str(x).strip() for x in result if str(x).strip()))


def ocr_crop(reader: Any, image: Image.Image, box: Tuple[float, float, float, float], *, numeric: bool = False, scale: int = 4) -> str:
    """셀 OCR.

    v24: 숫자 칸은 확대 배율/대비를 두 번 시도해서 더 그럴듯한 값을 선택합니다.
    그래도 틀린 값은 app.py에서 종합정보 총피해+지분으로 다시 보정합니다.
    """
    base = crop_norm(image, box, pad=1)
    crop = preprocess_crop(base, scale=scale)
    text = _ocr_text_once(reader, crop, numeric=numeric)

    if numeric:
        # 숫자 칸은 한 번 더 크게/강하게 읽어보고 숫자 토큰이 더 많은 쪽을 선택합니다.
        alt = preprocess_crop(base, scale=min(10, max(scale + 1, 6)))
        alt = ImageEnhance.Contrast(alt).enhance(1.20)
        alt = ImageEnhance.Sharpness(alt).enhance(1.25)
        alt_text = _ocr_text_once(reader, alt, numeric=True)
        def score(t: str) -> int:
            return len(re.findall(r"\d", str(t))) + (3 if re.search(r"억|만|%", str(t)) else 0)
        if score(alt_text) > score(text):
            text = alt_text
    return clean_ocr_text(text)


def clean_ocr_text(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("％", "%")
    text = text.replace("，", ",")
    text = text.replace("ㆍ", ".")
    text = text.replace("。", ".")
    text = re.sub(r"\s+", " ", text)
    return text


def extract_elapsed_seconds(text: str) -> float | None:
    # 예: 전투 시간 05 : 04, 05:04
    text = clean_ocr_text(text)
    m = re.search(r"(\d{1,2})\s*[:：]\s*(\d{2})", text)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def extract_first_percent(text: str) -> float | None:
    text = clean_ocr_text(text).replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if not m:
        # % 기호가 OCR에서 빠진 경우도 있어서 소수점 숫자만 마지막 보조로 잡습니다.
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if not nums:
            return None
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return float(m.group(1))


def extract_korean_number_text(text: str) -> str:
    """OCR 덩어리에서 4,641.26억 / 8,200.25만 같은 첫 번째 한국식 숫자만 뽑습니다."""
    text = clean_ocr_text(text)
    m = re.search(r"\d[\d,]*(?:\.\d+)?\s*(?:억|만)", text)
    if m:
        return m.group(0).replace(" ", "")
    m = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    return m.group(0) if m else text


def parse_summary_fixed_grid(image: Image.Image, reader: Any) -> Dict[str, Any]:
    raw: Dict[str, str] = {}
    for key, box in SUMMARY_ROIS.items():
        raw[key] = ocr_crop(reader, image, box, numeric=True, scale=4)

    from modules.calculators import parse_korean_number

    meta: Dict[str, Any] = {
        "raw": raw,
        "elapsed_seconds": extract_elapsed_seconds(raw.get("elapsed_text", "")),
        "total_damage_text": extract_korean_number_text(raw.get("total_damage_text", "")),
        "dps_text": extract_korean_number_text(raw.get("dps_text", "")),
        "damage_increase_efficiency": extract_first_percent(raw.get("damage_increase_efficiency", "")),
        "crit_rate": extract_first_percent(raw.get("crit_rate", "")),
        "head_attack_rate": extract_first_percent(raw.get("head_attack_rate", "")),
        "back_attack_rate": extract_first_percent(raw.get("back_attack_rate", "")),
        "damage_increase_uptime": extract_first_percent(raw.get("damage_increase_uptime", "")),
        "burst_uptime": extract_first_percent(raw.get("burst_uptime", "")),
    }
    meta["total_damage"] = parse_korean_number(meta.get("total_damage_text"))
    meta["dps"] = parse_korean_number(meta.get("dps_text"))

    # v26: 전투시간은 반드시 이미지 OCR 결과만 사용합니다.
    # 총피해량 / DPS 역산은 하지 않습니다. OCR 실패 시 수동 보정 칸에서 입력합니다.
    if meta.get("elapsed_seconds") is None:
        meta["elapsed_source"] = "ocr_failed"
    else:
        meta["elapsed_source"] = "ocr"
    return meta


def parse_attack_fixed_grid(image: Image.Image, reader: Any, row_count: int | None = None, scale: int = 4) -> pd.DataFrame:
    grid = ATTACK_GRID
    if row_count is None:
        row_count = int(grid["row_count"])

    rows: List[Dict[str, str]] = []
    for i in range(row_count):
        y1 = grid["row_start"] + i * grid["row_height"]
        y2 = y1 + grid["row_height"]
        row: Dict[str, str] = {col: "" for col in STANDARD_COLUMNS}
        row["_ocr_row_index"] = str(i)

        for key, label, x1, x2 in grid["columns"]:
            text = ocr_crop(reader, image, (x1, y1, x2, y2), numeric=(key != "name"), scale=scale)
            if key in ["damage_text", "dps_text"]:
                text = extract_korean_number_text(text)
            row[label] = text

        # 공격정보 탭에는 헤드어택 열이 없으므로 빈 값 유지.
        # v24: 이름 OCR이 비어도 아이콘 보정으로 복구할 수 있으므로
        # 이름만 보고 버리지 않고, 숫자/지분/횟수 중 하나라도 있으면 행을 유지합니다.
        meaningful = False
        for col in STANDARD_COLUMNS:
            val = str(row.get(col) or "").strip()
            if val and val not in ["-", "_", "None", "nan"]:
                meaningful = True
                break
        if meaningful:
            rows.append(row)

    df = pd.DataFrame(rows, columns=STANDARD_COLUMNS + ["_ocr_row_index"])
    # 완전히 빈 행 제거
    if not df.empty:
        nonempty = df[STANDARD_COLUMNS].astype(str).apply(lambda r: any(x.strip() and x.strip() not in ["", "-", "_", "None", "nan"] for x in r), axis=1)
        df = df[nonempty].reset_index(drop=True)
    return df



# ==============================
# v26: OCR 디버그/시각화용 함수
# ==============================
def make_summary_ocr_debug(image: Image.Image, reader: Any, scale: int = 7) -> List[Dict[str, Any]]:
    """종합정보 ROI별 crop/전처리/인식 텍스트를 UI에서 확인하기 위한 디버그 데이터."""
    labels = {
        "elapsed_text": "전투 시간",
        "total_damage_text": "총 피해량",
        "dps_text": "DPS",
        "damage_increase_efficiency": "피해 증가 유효율",
        "crit_rate": "치명타 적중률",
        "head_attack_rate": "헤드어택 적중률",
        "back_attack_rate": "백어택 적중률",
    }
    rows: List[Dict[str, Any]] = []
    for key, box in SUMMARY_ROIS.items():
        raw_crop = crop_norm(image, box, pad=1)
        proc_crop = preprocess_crop(raw_crop, scale=scale)
        text = _ocr_text_once(reader, proc_crop, numeric=True)
        rows.append({
            "key": key,
            "label": labels.get(key, key),
            "text": text,
            "raw_crop": raw_crop,
            "processed_crop": proc_crop,
        })
    return rows


def make_attack_ocr_debug(image: Image.Image, reader: Any, row_count: int = 18, scale: int = 7) -> List[Dict[str, Any]]:
    """공격정보 표를 어떤 셀로 자르고 무엇을 OCR했는지 보여주는 디버그 데이터."""
    grid = ATTACK_GRID
    rows: List[Dict[str, Any]] = []
    for i in range(int(row_count)):
        y1 = grid["row_start"] + i * grid["row_height"]
        y2 = y1 + grid["row_height"]
        row_crop = crop_norm(image, (ATTACK_ICON_GRID["x1"], y1, grid["columns"][-1][3], y2), pad=0)
        icon_crop = crop_attack_icon(image, i)
        cells: List[Dict[str, Any]] = []
        for key, label, x1, x2 in grid["columns"]:
            raw_crop = crop_norm(image, (x1, y1, x2, y2), pad=1)
            proc_crop = preprocess_crop(raw_crop, scale=scale)
            numeric = key != "name"
            text = _ocr_text_once(reader, proc_crop, numeric=numeric)
            if key in ["damage_text", "dps_text"]:
                text = extract_korean_number_text(text)
            cells.append({
                "key": key,
                "label": label,
                "text": text,
                "raw_crop": raw_crop,
                "processed_crop": proc_crop,
            })
        rows.append({
            "row_index": i,
            "row_crop": row_crop,
            "icon_crop": icon_crop,
            "cells": cells,
        })
    return rows


# ==============================
# v18: 전투분석기 OCR 스킬명 자동 보정
# ==============================
# OCR이 "허리켜인 소드", "불레이노 이런선"처럼 읽어도
# API에서 불러온 현재 캐릭터의 스킬 목록과 비교해 가장 가까운 스킬명으로 보정합니다.
_COMMON_OCR_NAME_REPLACEMENTS = {
    "허리켜인": "허리케인",
    "허리캐인": "허리케인",
    "불레이드": "블레이드",
    "블레이드": "블레이드",
    "불로": "블로",
    "브루발": "브루탈",
    "브루뱃": "브루탈",
    "브루렬": "브루탈",
    "임트": "임팩트",
    "임팩": "임팩트",
    "킬로린": "길로틴",
    "길로린": "길로틴",
    "불레이노": "볼케이노",
    "볼레이노": "볼케이노",
    "블레이노": "볼케이노",
    "이런선": "이럽션",
    "이럽선": "이럽션",
    "폐이람": "페이탈",
    "폐이탈": "페이탈",
    "페이람": "페이탈",
}


def normalize_skill_name_for_match(value: Any) -> str:
    text = clean_ocr_text(str(value or ""))
    for bad, good in _COMMON_OCR_NAME_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = re.sub(r"[^0-9A-Za-z가-힣]", "", text).lower()
    return text


def best_skill_name_match(ocr_name: Any, skill_names: List[str], threshold: float = 0.62) -> Tuple[str, float, bool]:
    original = str(ocr_name or "").strip()
    norm_original = normalize_skill_name_for_match(original)
    if not norm_original:
        return original, 0.0, False

    best_name = original
    best_score = 0.0
    for skill in skill_names or []:
        skill = str(skill or "").strip()
        if not skill:
            continue
        norm_skill = normalize_skill_name_for_match(skill)
        if not norm_skill:
            continue
        score = SequenceMatcher(None, norm_original, norm_skill).ratio()
        # 한쪽이 다른 쪽을 거의 포함하면 OCR 공백/일부 누락으로 보고 가산합니다.
        if norm_original in norm_skill or norm_skill in norm_original:
            score = max(score, 0.88)
        if score > best_score:
            best_score = score
            best_name = skill

    return best_name, round(best_score * 100.0, 2), best_score >= threshold


def correct_battle_skill_names(df: pd.DataFrame, skill_names: List[str], threshold: float = 0.62) -> pd.DataFrame:
    if df is None or df.empty or not skill_names:
        return df
    out = df.copy()
    name_col = "이름" if "이름" in out.columns else "name" if "name" in out.columns else None
    if name_col is None:
        return out

    original_names = []
    corrected_names = []
    scores = []
    corrected_flags = []
    for value in out[name_col].tolist():
        matched, score, ok = best_skill_name_match(value, skill_names, threshold=threshold)
        original = str(value or "").strip()
        original_names.append(original)
        scores.append(score)
        corrected_flags.append(bool(ok and matched != original))
        corrected_names.append(matched if ok else original)

    # UI에서는 보정용 컬럼을 보여주지 않습니다.
    # OCR 원문/점수는 필요하면 여기에서 다시 추가할 수 있지만, 현재 프로그램은
    # 검수 편의를 위해 최종 스킬명만 표에 남깁니다.
    out[name_col] = corrected_names
    return out



# ==============================
# v20: 스킬 아이콘 기반 전투분석기 이름 보정
# ==============================
# OCR 텍스트가 심하게 깨져도 전투분석기 행의 스킬 아이콘을 잘라 API 스킬 아이콘과 비교합니다.
# 최종 이름은 텍스트 유사도 + 아이콘 유사도를 함께 보고 정합니다.

@lru_cache(maxsize=512)
def _download_icon_image(url: str) -> Image.Image | None:
    """API 스킬 아이콘을 로컬 cache/skill_icons에 저장해두고 재사용합니다.

    - 처음 한 번은 아이콘 URL에서 내려받습니다.
    - 이후에는 로컬 파일을 열어서 비교하므로 OCR 재실행이 훨씬 빠르고 안정적입니다.
    - 아이콘 CDN URL은 API 응답에 포함된 공개 이미지 주소라 별도 API KEY가 필요하지 않습니다.
    """
    if not url:
        return None
    try:
        root = Path(__file__).resolve().parents[1]
        cache_dir = root / "cache" / "skill_icons"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ext = ".png"
        m = re.search(r"\.(png|jpg|jpeg|webp)(?:$|[?])", url, flags=re.I)
        if m:
            ext = "." + m.group(1).lower().replace("jpeg", "jpg")
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{key}{ext}"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return Image.open(cache_path).convert("RGB")

        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def _square_resize_for_icon(img: Image.Image, size: int = 48, margin_ratio: float = 0.08) -> Image.Image:
    """전투분석기 아이콘과 API 아이콘 비교용 전처리.

    v23에서는 테두리/선택행 파란 오버레이 영향을 줄이기 위해 margin_ratio를 바꿔가며
    여러 후보 이미지를 만들 수 있게 했습니다.
    """
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = max(0, (w - side) // 2)
    top = max(0, (h - side) // 2)
    img = img.crop((left, top, left + side, top + side))
    if img.width > 8 and img.height > 8 and margin_ratio > 0:
        margin = max(1, int(min(img.width, img.height) * margin_ratio))
        if img.width - margin * 2 > 4 and img.height - margin * 2 > 4:
            img = img.crop((margin, margin, img.width - margin, img.height - margin))
    img = ImageEnhance.Contrast(img).enhance(1.30)
    img = ImageEnhance.Sharpness(img).enhance(1.20)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def _dhash(img: Image.Image, hash_size: int = 8) -> int:
    """간단한 difference hash. 아이콘 색이 조금 달라도 모양이 같으면 가까워집니다."""
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def _hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def _icon_similarity(a: Image.Image, b: Image.Image) -> float:
    """0~1 아이콘 유사도.

    v23 핵심:
    - 전투분석기 아이콘과 API 아이콘이 같은 경우에만 이름을 확정하기 위해
      기존보다 더 보수적인 점수를 사용합니다.
    - 색상 히스토그램, 밝기 패턴, dHash, OpenCV template matching을 함께 봅니다.
    """
    try:
        # 같은 아이콘인데 행 선택 파란색/테두리 때문에 점수가 낮아질 수 있어
        # margin을 여러 개로 바꿔 비교한 뒤 가장 높은 점수를 사용합니다.
        best_score = 0.0
        for margin_a in (0.00, 0.06, 0.10, 0.16):
            for margin_b in (0.00, 0.06, 0.10, 0.16):
                ia = _square_resize_for_icon(a, size=64, margin_ratio=margin_a)
                ib = _square_resize_for_icon(b, size=64, margin_ratio=margin_b)
                aa = np.asarray(ia, dtype=np.float32)
                bb = np.asarray(ib, dtype=np.float32)

                rms = float(np.sqrt(np.mean((aa - bb) ** 2)))
                color_score = max(0.0, 1.0 - rms / 255.0)

                ag = aa.mean(axis=2)
                bg = bb.mean(axis=2)
                agn = (ag - ag.mean()) / (ag.std() + 1e-6)
                bgn = (bg - bg.mean()) / (bg.std() + 1e-6)
                corr = float(np.mean(agn * bgn))
                pattern_score = max(0.0, min(1.0, (corr + 1.0) / 2.0))

                ha = _dhash(ia)
                hb = _dhash(ib)
                hash_score = 1.0 - _hamming_distance(ha, hb) / 64.0
                hash_score = max(0.0, min(1.0, hash_score))

                hist_score = 0.0
                tmpl_score = 0.0
                try:
                    import cv2

                    hsv_a = cv2.cvtColor(np.asarray(ia), cv2.COLOR_RGB2HSV)
                    hsv_b = cv2.cvtColor(np.asarray(ib), cv2.COLOR_RGB2HSV)
                    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [24, 24], [0, 180, 0, 256])
                    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [24, 24], [0, 180, 0, 256])
                    cv2.normalize(hist_a, hist_a)
                    cv2.normalize(hist_b, hist_b)
                    hist_score = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
                    hist_score = max(0.0, min(1.0, (hist_score + 1.0) / 2.0))

                    gray_a = cv2.cvtColor(np.asarray(ia), cv2.COLOR_RGB2GRAY)
                    gray_b = cv2.cvtColor(np.asarray(ib), cv2.COLOR_RGB2GRAY)
                    tmpl_score = float(cv2.matchTemplate(gray_a, gray_b, cv2.TM_CCOEFF_NORMED)[0][0])
                    tmpl_score = max(0.0, min(1.0, tmpl_score))
                except Exception:
                    pass

                score = (
                    0.14 * color_score
                    + 0.24 * pattern_score
                    + 0.20 * hash_score
                    + 0.18 * hist_score
                    + 0.24 * tmpl_score
                )
                best_score = max(best_score, score)
        return round(max(0.0, min(1.0, best_score)), 4)
    except Exception:
        return 0.0


def crop_attack_icon(image: Image.Image, row_index: int) -> Image.Image | None:
    try:
        grid = ATTACK_GRID
        y1 = grid["row_start"] + int(row_index) * grid["row_height"]
        y2 = y1 + grid["row_height"]
        box = (ATTACK_ICON_GRID["x1"], y1, ATTACK_ICON_GRID["x2"], y2)
        return crop_norm(image, box, pad=0)
    except Exception:
        return None


def _candidate_name_icon_pairs(skill_candidates: List[Any]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for item in skill_candidates or []:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("스킬명") or "").strip()
            icon = str(item.get("icon") or item.get("아이콘") or item.get("Icon") or "").strip()
        else:
            name = str(item or "").strip()
            icon = ""
        if name:
            pairs.append({"name": name, "icon": icon})
    return pairs


def best_skill_name_match_with_icon(
    ocr_name: Any,
    row_icon: Image.Image | None,
    skill_candidates: List[Any],
    *,
    text_threshold: float = 0.55,
    icon_threshold: float = 0.78,
) -> Tuple[str, float, float, bool, str]:
    """OCR 이름을 API 스킬 후보로 보정합니다.

    반환: (이름, 텍스트점수, 아이콘점수, 보정여부, 보정근거)

    v22 원칙:
    1) 아이콘이 충분히 같으면 그 API 스킬명을 최우선 사용
    2) 아이콘이 애매하면 텍스트가 확실할 때만 보정
    3) 둘 다 애매하면 원본 유지/필터링 대상으로 둠
    """
    pairs = _candidate_name_icon_pairs(skill_candidates)
    if not pairs:
        return str(ocr_name or "").strip(), 0.0, 0.0, False, "no_candidates"

    original = str(ocr_name or "").strip()
    norm_original = normalize_skill_name_for_match(original)

    scored: List[Dict[str, Any]] = []
    for pair in pairs:
        name = pair["name"]
        norm_skill = normalize_skill_name_for_match(name)
        text_score = 0.0
        if norm_original and norm_skill:
            text_score = SequenceMatcher(None, norm_original, norm_skill).ratio()
            if norm_original in norm_skill or norm_skill in norm_original:
                text_score = max(text_score, 0.88)

        icon_score = 0.0
        if row_icon is not None and pair.get("icon"):
            api_icon = _download_icon_image(pair["icon"])
            if api_icon is not None:
                icon_score = _icon_similarity(row_icon, api_icon)

        combined = 0.35 * text_score + 0.65 * icon_score
        scored.append({"name": name, "text": text_score, "icon": icon_score, "combined": combined})

    scored.sort(key=lambda x: (x["icon"], x["combined"], x["text"]), reverse=True)
    best_icon = scored[0]
    second_icon = scored[1]["icon"] if len(scored) > 1 else 0.0
    icon_margin = best_icon["icon"] - second_icon

    # 1) 아이콘 확정. 점수와 2등과의 차이를 같이 봐서 오보정을 줄입니다.
    icon_confident = best_icon["icon"] >= icon_threshold and (icon_margin >= 0.06 or best_icon["icon"] >= min(0.96, icon_threshold + 0.10))
    if icon_confident:
        return best_icon["name"], round(best_icon["text"] * 100.0, 2), round(best_icon["icon"] * 100.0, 2), True, "icon"

    # 2) 텍스트 확정. 아이콘이 애매할 때만 사용합니다.
    best_text = max(scored, key=lambda x: x["text"])
    if best_text["text"] >= text_threshold:
        return best_text["name"], round(best_text["text"] * 100.0, 2), round(best_text["icon"] * 100.0, 2), True, "text"

    return original, round(best_icon.get("text", 0.0) * 100.0, 2), round(best_icon.get("icon", 0.0) * 100.0, 2), False, "unmatched"

def correct_battle_skill_names_with_icons(
    df: pd.DataFrame,
    attack_image: Image.Image,
    skill_candidates: List[Any],
    *,
    threshold: float = 0.55,
    icon_threshold: float = 0.78,
    drop_unmatched: bool = True,
) -> pd.DataFrame:
    """전투분석기 스킬명 보정.

    - 아이콘이 API 스킬 아이콘과 확실히 같으면 그 이름을 사용합니다.
    - 아이콘이 애매하면 OCR 텍스트 유사도로 보정합니다.
    - 둘 다 실패한 행은 기본값으로 제거합니다. 이러면 스킬룬/기타/잘못 읽힌 하단 행이 계산에 섞이는 것을 줄일 수 있습니다.
    """
    if df is None or df.empty or not skill_candidates:
        return df
    out = df.copy()
    name_col = "이름" if "이름" in out.columns else "name" if "name" in out.columns else None
    if name_col is None:
        return out

    rows = []
    for idx, row in out.iterrows():
        try:
            row_index = int(float(row.get("_ocr_row_index", idx)))
        except Exception:
            row_index = int(idx)
        icon_crop = crop_attack_icon(attack_image, row_index)
        matched, text_score, icon_score, ok, reason = best_skill_name_match_with_icon(
            row.get(name_col),
            icon_crop,
            skill_candidates,
            text_threshold=threshold,
            icon_threshold=icon_threshold,
        )
        matched = _canonical_display_name_v86(matched)
        new_row = row.copy()
        if ok:
            new_row[name_col] = matched
        elif name_col in new_row:
            new_row[name_col] = _canonical_display_name_v86(new_row.get(name_col))
        elif drop_unmatched:
            continue
        # 숨김/디버그용. app.py에서 기본 표에는 표시하지 않습니다.
        new_row["_name_match_text_score"] = text_score
        new_row["_name_match_icon_score"] = icon_score
        new_row["_name_match_reason"] = reason
        rows.append(new_row)

    if not rows:
        return out.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)

def apply_summary_overrides(summary: Dict[str, Any], meta: Dict[str, Any] | None) -> Dict[str, Any]:
    """종합정보 이미지에서 읽은 정확한 총피해/DPS/전체 치명/백어택률을 결과 요약에 덮어씁니다."""
    if not meta:
        return summary
    out = dict(summary)
    if meta.get("total_damage"):
        out["total_damage"] = meta["total_damage"]
    if meta.get("dps"):
        out["dps"] = meta["dps"]
    if meta.get("back_attack_rate") is not None:
        out["weighted_back_attack_rate"] = meta["back_attack_rate"]
    if meta.get("head_attack_rate") is not None:
        out["weighted_head_attack_rate"] = meta["head_attack_rate"]
    if meta.get("crit_rate") is not None:
        out["weighted_crit_rate"] = meta["crit_rate"]
    return out


# ==============================
# update: movable battle-analyzer window support + debug export helpers
# ==============================
# 기존 고정 좌표는 1920x1080 전체 스크린샷에서 전투분석기 창이 기본 위치에 있을 때의 좌표입니다.
# 전투분석기 창은 이동 가능하므로, 먼저 창 제목 "전투 분석기"를 OCR로 찾아 창 위치를 추정하고,
# 기존 좌표를 창 내부 상대 좌표로 변환해서 crop합니다.

BASE_SCREEN_W = 1920.0
BASE_SCREEN_H = 1080.0
# 기본 위치에서의 전투분석기 창 바운딩 박스. 실제 크기는 거의 고정이고 위치만 이동합니다.
BASE_WINDOW_BOX = (214.0, 68.0, 1670.0, 958.0)
BASE_WINDOW_W = BASE_WINDOW_BOX[2] - BASE_WINDOW_BOX[0]
BASE_WINDOW_H = BASE_WINDOW_BOX[3] - BASE_WINDOW_BOX[1]
BASE_WINDOW_TITLE_CENTER_Y = 88.0
BASE_TITLE_OFFSET_Y = BASE_WINDOW_TITLE_CENTER_Y - BASE_WINDOW_BOX[1]


def _normalize_for_window_detection(text: Any) -> str:
    t = str(text or "")
    return re.sub(r"[^0-9A-Za-z가-힣]", "", t).lower()


def detect_analyzer_window(image: Image.Image, reader: Any | None = None) -> Tuple[int, int, int, int] | None:
    """전투분석기 창 위치 자동 감지.

    1순위: 전체 이미지 OCR에서 `전투 분석기` 제목을 찾고, 제목 중심점을 기준으로 창을 역산합니다.
    실패하면 None을 반환하고 기존 전체 화면 고정 좌표를 사용합니다.
    """
    if reader is None:
        return None
    try:
        arr = np.array(image.convert("RGB"))
        results = reader.readtext(arr, detail=1, paragraph=False)
    except Exception:
        return None

    best = None
    best_score = -1.0
    for item in results:
        try:
            bbox, text, conf = item[0], str(item[1]), float(item[2]) if len(item) > 2 else 0.5
        except Exception:
            continue
        norm = _normalize_for_window_detection(text)
        # 제목 OCR은 `전투 분석기`, `전투분석기` 등으로 잡힙니다.
        if "전투" in norm and "분석" in norm:
            xs = [float(pt[0]) for pt in bbox]
            ys = [float(pt[1]) for pt in bbox]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            # 너무 아래쪽의 다른 문구는 제외합니다. 제목은 보통 창 상단에 있습니다.
            if cy > image.height * 0.35:
                continue
            score = conf + (0.4 if "전투분석" in norm else 0.0) + (0.2 if 5 <= len(norm) <= 8 else 0.0)
            if score > best_score:
                best = (cx, cy, text, conf)
                best_score = score

    if best is None:
        return None

    cx, cy, _text, _conf = best
    win_w = BASE_WINDOW_W / BASE_SCREEN_W * image.width
    win_h = BASE_WINDOW_H / BASE_SCREEN_H * image.height
    top_offset = BASE_TITLE_OFFSET_Y / BASE_SCREEN_H * image.height

    x1 = int(round(cx - win_w / 2.0))
    y1 = int(round(cy - top_offset))
    x2 = int(round(x1 + win_w))
    y2 = int(round(y1 + win_h))

    # 화면 밖으로 나가지 않게 보정합니다.
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > image.width:
        shift = x2 - image.width
        x1 = max(0, x1 - shift)
        x2 = image.width
    if y2 > image.height:
        shift = y2 - image.height
        y1 = max(0, y1 - shift)
        y2 = image.height

    # 말도 안 되게 작게 잡히면 실패로 처리합니다.
    if (x2 - x1) < image.width * 0.55 or (y2 - y1) < image.height * 0.55:
        return None
    return (x1, y1, x2, y2)


def _full_norm_box_to_window_rel(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    fx1, fy1, fx2, fy2 = box[0] * BASE_SCREEN_W, box[1] * BASE_SCREEN_H, box[2] * BASE_SCREEN_W, box[3] * BASE_SCREEN_H
    bx1, by1, bx2, by2 = BASE_WINDOW_BOX
    return (
        (fx1 - bx1) / (bx2 - bx1),
        (fy1 - by1) / (by2 - by1),
        (fx2 - bx1) / (bx2 - bx1),
        (fy2 - by1) / (by2 - by1),
    )


def _window_rel_to_pixels(image: Image.Image, rel_box: Tuple[float, float, float, float], window_box: Tuple[int, int, int, int], pad: int = 2) -> Tuple[int, int, int, int]:
    wx1, wy1, wx2, wy2 = window_box
    ww, wh = wx2 - wx1, wy2 - wy1
    rx1, ry1, rx2, ry2 = rel_box
    x1 = wx1 + rx1 * ww
    y1 = wy1 + ry1 * wh
    x2 = wx1 + rx2 * ww
    y2 = wy1 + ry2 * wh
    return (
        max(0, int(round(x1)) + pad),
        max(0, int(round(y1)) + pad),
        min(image.width, int(round(x2)) - pad),
        min(image.height, int(round(y2)) - pad),
    )


# 기존 함수명 override. old code에서도 runtime global lookup으로 이 함수를 사용합니다.
def _norm_box_to_pixels(image: Image.Image, box: Tuple[float, float, float, float], pad: int = 2, window_box: Tuple[int, int, int, int] | None = None) -> Tuple[int, int, int, int]:
    if window_box is not None:
        return _window_rel_to_pixels(image, _full_norm_box_to_window_rel(box), window_box, pad=pad)
    w, h = image.size
    x1, y1, x2, y2 = box
    return (
        max(0, int(x1 * w) + pad),
        max(0, int(y1 * h) + pad),
        min(w, int(x2 * w) - pad),
        min(h, int(y2 * h) - pad),
    )


def crop_norm(image: Image.Image, box: Tuple[float, float, float, float], pad: int = 2, window_box: Tuple[int, int, int, int] | None = None) -> Image.Image:
    return image.crop(_norm_box_to_pixels(image, box, pad=pad, window_box=window_box))


def ocr_crop(reader: Any, image: Image.Image, box: Tuple[float, float, float, float], *, numeric: bool = False, scale: int = 4, window_box: Tuple[int, int, int, int] | None = None) -> str:
    base = crop_norm(image, box, pad=1, window_box=window_box)
    crop = preprocess_crop(base, scale=scale)
    text = _ocr_text_once(reader, crop, numeric=numeric)

    if numeric:
        alt = preprocess_crop(base, scale=min(10, max(scale + 1, 6)))
        alt = ImageEnhance.Contrast(alt).enhance(1.20)
        alt = ImageEnhance.Sharpness(alt).enhance(1.25)
        alt_text = _ocr_text_once(reader, alt, numeric=True)
        def score(t: str) -> int:
            return len(re.findall(r"\d", str(t))) + (3 if re.search(r"억|만|%", str(t)) else 0)
        if score(alt_text) > score(text):
            text = alt_text
    return clean_ocr_text(text)


def extract_elapsed_seconds(text: str) -> float | None:
    """전투 시간 추출. OCR이 콜론을 6처럼 읽는 케이스까지 보정합니다."""
    text = clean_ocr_text(text)
    t = text.replace("O", "0").replace("o", "0").replace("ㅣ", "1")
    # 정상: 05:04
    m = re.search(r"(\d{1,2})\s*[:：]\s*(\d{2})", t)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    # OCR: 05 6 04 / 05 b 04
    m = re.search(r"(\d{1,2})\s*[6bB]\s*(\d{2})", t)
    if m:
        mm, ss = int(m.group(1)), int(m.group(2))
        if 0 <= ss < 60:
            return mm * 60 + ss
    # OCR: 전투 시간 05 04처럼 콜론이 사라진 경우. 마지막쪽 두 자리 쌍을 우선 사용합니다.
    nums = re.findall(r"\d{1,2}", t)
    pairs = []
    for a, b in zip(nums, nums[1:]):
        try:
            mm, ss = int(a), int(b)
        except Exception:
            continue
        if 0 <= mm < 60 and 0 <= ss < 60:
            pairs.append((mm, ss))
    if pairs:
        mm, ss = pairs[-1]
        return mm * 60 + ss
    return None


def parse_summary_fixed_grid(image: Image.Image, reader: Any, scale: int = 7, auto_window: bool = True) -> Dict[str, Any]:
    window_box = detect_analyzer_window(image, reader) if auto_window else None
    raw: Dict[str, str] = {}
    for key, box in SUMMARY_ROIS.items():
        raw[key] = ocr_crop(reader, image, box, numeric=True, scale=scale, window_box=window_box)

    from modules.calculators import parse_korean_number

    meta: Dict[str, Any] = {
        "raw": raw,
        "window_box": window_box,
        "window_detected": window_box is not None,
        "elapsed_seconds": extract_elapsed_seconds(raw.get("elapsed_text", "")),
        "total_damage_text": extract_korean_number_text(raw.get("total_damage_text", "")),
        "dps_text": extract_korean_number_text(raw.get("dps_text", "")),
        "damage_increase_efficiency": extract_first_percent(raw.get("damage_increase_efficiency", "")),
        "crit_rate": extract_first_percent(raw.get("crit_rate", "")),
        "head_attack_rate": extract_first_percent(raw.get("head_attack_rate", "")),
        "back_attack_rate": extract_first_percent(raw.get("back_attack_rate", "")),
        "damage_increase_uptime": extract_first_percent(raw.get("damage_increase_uptime", "")),
        "burst_uptime": extract_first_percent(raw.get("burst_uptime", "")),
    }
    meta["total_damage"] = parse_korean_number(meta.get("total_damage_text"))
    meta["dps"] = parse_korean_number(meta.get("dps_text"))
    meta["elapsed_source"] = "ocr" if meta.get("elapsed_seconds") is not None else "ocr_failed"
    return meta


def parse_attack_fixed_grid(image: Image.Image, reader: Any, row_count: int | None = None, scale: int = 7, auto_window: bool = True) -> pd.DataFrame:
    grid = ATTACK_GRID
    if row_count is None:
        row_count = int(grid["row_count"])
    window_box = detect_analyzer_window(image, reader) if auto_window else None

    rows: List[Dict[str, str]] = []
    for i in range(row_count):
        y1 = grid["row_start"] + i * grid["row_height"]
        y2 = y1 + grid["row_height"]
        row: Dict[str, Any] = {col: "" for col in STANDARD_COLUMNS}
        row["_ocr_row_index"] = str(i)
        if window_box:
            row["_window_x1"], row["_window_y1"], row["_window_x2"], row["_window_y2"] = window_box

        for key, label, x1, x2 in grid["columns"]:
            text = ocr_crop(reader, image, (x1, y1, x2, y2), numeric=(key != "name"), scale=scale, window_box=window_box)
            if key in ["damage_text", "dps_text"]:
                text = extract_korean_number_text(text)
            row[label] = text

        meaningful = False
        for col in STANDARD_COLUMNS:
            val = str(row.get(col) or "").strip()
            if val and val not in ["-", "_", "None", "nan"]:
                meaningful = True
                break
        if meaningful:
            rows.append(row)

    df = pd.DataFrame(rows, columns=STANDARD_COLUMNS + ["_ocr_row_index", "_window_x1", "_window_y1", "_window_x2", "_window_y2"])
    if not df.empty:
        nonempty = df[STANDARD_COLUMNS].astype(str).apply(lambda r: any(x.strip() and x.strip() not in ["", "-", "_", "None", "nan"] for x in r), axis=1)
        df = df[nonempty].reset_index(drop=True)
    return df


def _row_window_box(row: pd.Series | Dict[str, Any]) -> Tuple[int, int, int, int] | None:
    try:
        vals = [row.get("_window_x1"), row.get("_window_y1"), row.get("_window_x2"), row.get("_window_y2")]
        if all(v is not None and str(v) not in ["", "nan", "None"] for v in vals):
            return tuple(int(float(v)) for v in vals)  # type: ignore[return-value]
    except Exception:
        pass
    return None


def crop_attack_icon(image: Image.Image, row_index: int, window_box: Tuple[int, int, int, int] | None = None) -> Image.Image | None:
    try:
        grid = ATTACK_GRID
        y1 = grid["row_start"] + int(row_index) * grid["row_height"]
        y2 = y1 + grid["row_height"]
        return crop_norm(image, (ATTACK_ICON_GRID["x1"], y1, ATTACK_ICON_GRID["x2"], y2), pad=1, window_box=window_box)
    except Exception:
        return None


def correct_battle_skill_names_with_icons(
    df: pd.DataFrame,
    attack_image: Image.Image,
    skill_candidates: List[Any],
    *,
    threshold: float = 0.55,
    icon_threshold: float = 0.78,
    drop_unmatched: bool = True,
) -> pd.DataFrame:
    if df is None or df.empty or not skill_candidates:
        return df
    out = df.copy()
    name_col = "이름" if "이름" in out.columns else "name" if "name" in out.columns else None
    if name_col is None:
        return out

    rows = []
    for idx, row in out.iterrows():
        try:
            row_index = int(float(row.get("_ocr_row_index", idx)))
        except Exception:
            row_index = int(idx)
        icon_crop = crop_attack_icon(attack_image, row_index, window_box=_row_window_box(row))
        matched, text_score, icon_score, ok, reason = best_skill_name_match_with_icon(
            row.get(name_col),
            icon_crop,
            skill_candidates,
            text_threshold=threshold,
            icon_threshold=icon_threshold,
        )
        new_row = row.copy()
        if ok:
            new_row[name_col] = matched
        elif drop_unmatched:
            continue
        new_row["_name_match_text_score"] = text_score
        new_row["_name_match_icon_score"] = icon_score
        new_row["_name_match_reason"] = reason
        rows.append(new_row)

    if not rows:
        return out.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


def make_summary_ocr_debug(image: Image.Image, reader: Any, scale: int = 7) -> List[Dict[str, Any]]:
    labels = {
        "elapsed_text": "전투 시간",
        "total_damage_text": "총 피해량",
        "dps_text": "DPS",
        "damage_increase_efficiency": "피해 증가 유효율",
        "crit_rate": "치명타 적중률",
        "head_attack_rate": "헤드어택 적중률",
        "back_attack_rate": "백어택 적중률",
    }
    window_box = detect_analyzer_window(image, reader)
    rows: List[Dict[str, Any]] = []
    for key, box in SUMMARY_ROIS.items():
        raw_crop = crop_norm(image, box, pad=1, window_box=window_box)
        proc_crop = preprocess_crop(raw_crop, scale=scale)
        text = _ocr_text_once(reader, proc_crop, numeric=True)
        rows.append({
            "key": key,
            "label": labels.get(key, key),
            "text": text,
            "raw_crop": raw_crop,
            "processed_crop": proc_crop,
            "window_box": window_box,
        })
    return rows


def make_attack_ocr_debug(image: Image.Image, reader: Any, row_count: int = 18, scale: int = 7) -> List[Dict[str, Any]]:
    grid = ATTACK_GRID
    window_box = detect_analyzer_window(image, reader)
    rows: List[Dict[str, Any]] = []
    for i in range(int(row_count)):
        y1 = grid["row_start"] + i * grid["row_height"]
        y2 = y1 + grid["row_height"]
        row_crop = crop_norm(image, (ATTACK_ICON_GRID["x1"], y1, grid["columns"][-1][3], y2), pad=0, window_box=window_box)
        icon_crop = crop_attack_icon(image, i, window_box=window_box)
        cells: List[Dict[str, Any]] = []
        for key, label, x1, x2 in grid["columns"]:
            raw_crop = crop_norm(image, (x1, y1, x2, y2), pad=1, window_box=window_box)
            proc_crop = preprocess_crop(raw_crop, scale=scale)
            numeric = key != "name"
            text = _ocr_text_once(reader, proc_crop, numeric=numeric)
            if key in ["damage_text", "dps_text"]:
                text = extract_korean_number_text(text)
            cells.append({
                "key": key,
                "label": label,
                "text": text,
                "raw_crop": raw_crop,
                "processed_crop": proc_crop,
            })
        rows.append({
            "row_index": i,
            "row_crop": row_crop,
            "icon_crop": icon_crop,
            "cells": cells,
            "window_box": window_box,
        })
    return rows


# ==============================
# v28: OCR 정밀 보정 override
# ==============================
# real_ocr_debug.zip 분석 결과:
# - crop 위치는 대체로 맞지만, 숫자 OCR에서 파란 피해량 막대/흰 글자 외곽광 때문에
#   426.77억 -> 77억, 75.00% -> 75 6 0096처럼 깨지는 패턴이 확인됐습니다.
# - 그래서 숫자 칸은 일반 OCR 전처리 1회가 아니라, 흰색/노란색 글자만 분리한
#   숫자 전용 전처리까지 병행하고, 퍼센트 문자는 후처리로 복구합니다.

from PIL import ImageOps, ImageFilter


def _preprocess_numeric_text_mask(crop: Image.Image, scale: int = 7) -> Image.Image:
    """숫자/단위 OCR 전용 전처리.

    전투분석기 숫자는 대부분 흰색/노란색이고, 배경은 검정/파랑 막대입니다.
    RGB/HSV 마스크로 밝은 글자만 남겨서 EasyOCR이 배경 막대를 숫자로 착각하지 않게 합니다.
    """
    img = crop.convert("RGB")
    arr = np.asarray(img)
    try:
        import cv2

        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # 흰 글자: 밝고 채도가 낮음 / 노란 글자: 밝고 Hue가 노란 영역
        white = (v > 125) & (s < 115)
        yellow = (v > 110) & (h >= 12) & (h <= 42) & (s > 55)
        mask = (white | yellow).astype("uint8") * 255
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        # 너무 가는 획은 닫기 처리
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        out = Image.fromarray(mask, mode="L")
        out = ImageOps.expand(out, border=4, fill=0)
        if scale > 1:
            out = out.resize((max(1, out.width * scale), max(1, out.height * scale)), Image.Resampling.LANCZOS)
        out = ImageOps.autocontrast(out)
        return out.convert("RGB")
    except Exception:
        gray = ImageOps.grayscale(img)
        gray = ImageEnhance.Contrast(gray).enhance(2.0)
        # 밝은 글자만 남김
        bw = gray.point(lambda p: 255 if p > 135 else 0)
        bw = bw.filter(ImageFilter.MaxFilter(3))
        bw = ImageOps.expand(bw, border=4, fill=0)
        if scale > 1:
            bw = bw.resize((max(1, bw.width * scale), max(1, bw.height * scale)), Image.Resampling.LANCZOS)
        return bw.convert("RGB")


def _preprocess_text_light_mask(crop: Image.Image, scale: int = 7) -> Image.Image:
    """스킬명 OCR 전용 전처리. 흰/하늘색 글자를 살리고 배경 막대를 줄입니다."""
    img = crop.convert("RGB")
    arr = np.asarray(img)
    try:
        import cv2
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # 이름은 흰색/연한 하늘색이 많습니다.
        white = (v > 110) & (s < 145)
        cyan = (v > 105) & (h >= 80) & (h <= 115) & (s > 35)
        mask = (white | cyan).astype("uint8") * 255
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        out = Image.fromarray(mask, mode="L")
        out = ImageOps.expand(out, border=4, fill=0)
        if scale > 1:
            out = out.resize((max(1, out.width * scale), max(1, out.height * scale)), Image.Resampling.LANCZOS)
        out = ImageOps.autocontrast(out)
        return out.convert("RGB")
    except Exception:
        return preprocess_crop(crop, scale=scale)


def _numeric_ocr_score(text: str, *, kind: str = "generic") -> float:
    t = clean_ocr_text(text)
    digit_count = len(re.findall(r"\d", t))
    if digit_count == 0:
        return -100.0
    score = float(digit_count)
    if re.search(r"억|만", t):
        score += 8.0
    if "%" in t or re.search(r"(?:96|9o|％)", t):
        score += 4.0
    if "." in t:
        score += 3.0
    if "," in t:
        score += 1.0
    # 너무 긴 잡음은 감점
    if len(t) > 24:
        score -= (len(t) - 24) * 0.15
    if kind == "percent":
        val = parse_percent_ocr_value(t)
        if val is not None:
            score += 10.0
            if 0 <= val <= 100:
                score += 5.0
    elif kind in {"damage", "dps", "korean_number"}:
        if re.search(r"\d[\d,]*(?:\.\d+)?\s*(억|만)", t):
            score += 10.0
    return score


def _ocr_text_variants(reader: Any, raw_crop: Image.Image, *, numeric: bool, scale: int, kind: str = "generic") -> List[Tuple[str, Image.Image, str, float]]:
    """여러 전처리 결과를 OCR하고 점수가 높은 후보를 고릅니다."""
    variants: List[Tuple[str, Image.Image]] = []
    variants.append(("normal", preprocess_crop(raw_crop, scale=scale)))
    variants.append(("normal_more", preprocess_crop(raw_crop, scale=min(10, max(scale + 1, 7)))))
    if numeric:
        variants.append(("numeric_mask", _preprocess_numeric_text_mask(raw_crop, scale=min(10, max(scale, 7)))))
        variants.append(("numeric_mask_big", _preprocess_numeric_text_mask(raw_crop, scale=min(12, max(scale + 1, 8)))))
    else:
        variants.append(("text_mask", _preprocess_text_light_mask(raw_crop, scale=min(10, max(scale, 7)))))

    out: List[Tuple[str, Image.Image, str, float]] = []
    for name, img in variants:
        try:
            txt = _ocr_text_once(reader, img, numeric=numeric)
        except Exception:
            txt = ""
        score = _numeric_ocr_score(txt, kind=kind) if numeric else (len(txt) + (3 if re.search(r"[가-힣]", txt) else 0))
        out.append((name, img, txt, score))
    return out


def parse_percent_ocr_value(text: Any) -> float | None:
    """전투분석기 퍼센트 OCR 보정.

    EasyOCR에서 자주 나오는 패턴:
    - 100.0096 -> 100.00
    - 75 6 0096 -> 75.00
    - 76 9 6696 -> 76.66
    - 96 4 0096 -> 96.00
    - 78.8696 -> 78.86
    """
    t = clean_ocr_text(str(text or ""))
    if not t or t in {"-", "_", "None", "nan"}:
        return None
    t = (
        t.replace("O", "0")
        .replace("o", "0")
        .replace("％", "%")
        .replace("ㅣ", "1")
        .replace("l", "1")
        .replace("I", "1")
    )
    # 정상 소수점이 있으면 먼저 사용. 뒤에 붙은 96은 % 오인식일 수 있어 제거합니다.
    dot_nums = re.findall(r"\d{1,3}\.\d+", t)
    if dot_nums:
        s = dot_nums[-1]
        # 78.8696 -> 78.86, 100.0096 -> 100.00
        m = re.match(r"(\d{1,3})\.(\d+)", s)
        if m:
            ip = int(m.group(1))
            dec = m.group(2)
            if dec.endswith("96") and len(dec) > 2:
                dec = dec[:-2]
            dec = (dec + "00")[:2]
            val = ip + int(dec) / 100.0
            return min(100.0, max(0.0, round(val, 2)))
        try:
            return min(100.0, max(0.0, round(float(s), 2)))
        except Exception:
            pass

    nums = re.findall(r"\d+", t)
    if not nums:
        return None
    # 첫 숫자는 보통 정수부입니다. 0/1 같은 작은 잡음 뒤에 큰 숫자가 오면 큰 숫자를 사용합니다.
    ip_token = nums[0]
    if len(nums) >= 2 and int(ip_token) <= 5 and int(nums[1]) >= 10:
        ip_token = nums[1]
        rest_tokens = nums[2:]
    else:
        rest_tokens = nums[1:]
    try:
        ip = int(ip_token[:3])
    except Exception:
        return None
    if ip > 100 and len(ip_token) > 3:
        # 1000096 같은 덩어리
        ip = int(ip_token[:3]) if ip_token.startswith("100") else int(ip_token[:2])

    dec = "00"
    if rest_tokens:
        # 75 6 0096 / 76 9 6696 같은 경우 가운데 한 자리 잡음을 버립니다.
        if len(rest_tokens) >= 2 and len(rest_tokens[0]) == 1 and rest_tokens[0] in {"4", "5", "6", "8", "9"}:
            dec_src = rest_tokens[1]
        else:
            dec_src = rest_tokens[0]
        if dec_src.endswith("96") and len(dec_src) > 2:
            dec_src = dec_src[:-2]
        elif dec_src.endswith("6") and len(dec_src) > 2:
            dec_src = dec_src[:-1]
        # 009 -> 00, 6696 -> 66
        dec = (dec_src + "00")[:2]
    try:
        val = ip + int(dec) / 100.0
    except Exception:
        val = float(ip)
    return min(100.0, max(0.0, round(val, 2)))


def format_percent_from_ocr(text: Any) -> str:
    val = parse_percent_ocr_value(text)
    if val is None:
        return clean_ocr_text(text)
    return f"{val:.2f}%"


def extract_first_percent(text: str) -> float | None:
    return parse_percent_ocr_value(text)


def extract_elapsed_seconds(text: str) -> float | None:
    """전투 시간 추출 강화.

    디버그에서 `정보 (전투 시간 05 : 04)`가 `,1 1 만 05 4 049`처럼 깨지는 케이스가 확인되어,
    05/04처럼 보이는 마지막 시간쌍을 더 적극적으로 찾습니다.
    """
    t = clean_ocr_text(text)
    if not t:
        return None
    t = t.replace("O", "0").replace("o", "0").replace("ㅣ", "1").replace("I", "1").replace("l", "1")
    m = re.search(r"(\d{1,2})\s*[:：]\s*(\d{2})", t)
    if m:
        mm, ss = int(m.group(1)), int(m.group(2))
        if 0 <= ss < 60:
            return mm * 60 + ss
    m = re.search(r"전투\s*시간[^0-9]*(\d{1,2})\D+(\d{1,2})", t)
    if m:
        mm, ss = int(m.group(1)), int(m.group(2))
        if 0 <= mm < 60 and 0 <= ss < 60:
            return mm * 60 + ss
    # OCR 덩어리 안에서 05 04 / 02 37 같은 후보를 찾음.
    nums = re.findall(r"\d{1,3}", t)
    candidates: List[Tuple[int, int]] = []
    for i in range(len(nums) - 1):
        a, b = nums[i], nums[i + 1]
        # 049 -> 04로 잘라야 하는 경우
        for bb in {b, b[:2], b[-2:]}:
            try:
                mm, ss = int(a), int(bb)
            except Exception:
                continue
            if 0 <= mm < 30 and 0 <= ss < 60:
                candidates.append((mm, ss))
    # 전투분석기 시간은 보통 원문 뒤쪽 괄호 안에 있으므로 뒤쪽 후보 우선.
    if candidates:
        mm, ss = candidates[-1]
        return mm * 60 + ss
    return None


def extract_korean_number_text(text: str) -> str:
    """OCR 덩어리에서 한국식 숫자 텍스트 추출 강화."""
    t = clean_ocr_text(text)
    if not t:
        return ""
    t = t.replace("O", "0").replace("o", "0").replace("ㅣ", "1").replace("I", "1").replace("l", "1")
    # 4,641.26억 / 4641.26억 / 8,200.25만 / 1.23조
    matches = re.findall(r"\d[\d,]*(?:\.\d+)?\s*(?:조|억|만)", t)
    if matches:
        # 가장 긴 매치를 사용하면 뒤에 붙은 실제 원시 숫자보다 표시 숫자를 더 잘 잡습니다.
        best = max(matches, key=lambda x: (len(re.findall(r"\d", x)), len(x)))
        return best.replace(" ", "")
    nums = re.findall(r"\d[\d,]*(?:\.\d+)?", t)
    return nums[0] if nums else t


def full_raw_number_from_summary(text: str) -> float | None:
    """종합정보 카드에서 '정확한 원시 숫자'를 뽑습니다(v45).

    카드에는 축약값(예: 1,015.59억)과 그 아래 전체 원시 숫자(101,559,999,844)가 함께
    표시됩니다. 중국어 OCR이 '억/만/조' 글자를 '9' 등으로 오인식해도, 소수점 없는 가장 긴
    정수(원시 숫자)는 그대로 읽히므로 이 값을 사용해 단위를 정확히 역산합니다.
    반환: 원시 숫자(float) 또는 None.
    """
    t = clean_ocr_text(str(text or ""))
    t = t.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1").replace("ㅣ", "1")
    best: float | None = None
    best_digits = -1
    for m in re.finditer(r"[\d,]+", t):
        tok = m.group(0).strip(",")
        digits = tok.replace(",", "")
        if not digits.isdigit():
            continue
        # 소수점 없는 긴 정수 = 전체 원시 숫자(보통 9자리 이상). 5자리 이상만 후보로 봅니다.
        if len(digits) >= 5 and len(digits) > best_digits:
            best_digits = len(digits)
            best = float(digits)
    return best


def ocr_crop(
    reader: Any,
    image: Image.Image,
    box: Tuple[float, float, float, float],
    *,
    numeric: bool = False,
    scale: int = 7,
    window_box: Tuple[int, int, int, int] | None = None,
    kind: str = "generic",
) -> str:
    """v28 셀 OCR. 여러 전처리 후보 중 가장 그럴듯한 결과를 선택합니다."""
    raw_crop = crop_norm(image, box, pad=1, window_box=window_box)
    variants = _ocr_text_variants(reader, raw_crop, numeric=numeric, scale=scale, kind=kind)
    if not variants:
        return ""
    variants.sort(key=lambda x: x[3], reverse=True)
    text = clean_ocr_text(variants[0][2])
    if numeric and kind == "percent":
        return format_percent_from_ocr(text)
    return text


def parse_summary_fixed_grid(image: Image.Image, reader: Any, scale: int = 7, auto_window: bool = True) -> Dict[str, Any]:
    window_box = detect_analyzer_window(image, reader) if auto_window else None
    raw: Dict[str, str] = {}
    percent_keys = {"damage_increase_efficiency", "crit_rate", "head_attack_rate", "back_attack_rate", "damage_increase_uptime", "burst_uptime"}
    for key, box in SUMMARY_ROIS.items():
        kind = "percent" if key in percent_keys else "korean_number" if key in {"total_damage_text", "dps_text", "stagger_text"} else "generic"
        raw[key] = ocr_crop(reader, image, box, numeric=True, scale=scale, window_box=window_box, kind=kind)

    from modules.calculators import parse_korean_number

    meta: Dict[str, Any] = {
        "raw": raw,
        "window_box": window_box,
        "window_detected": window_box is not None,
        "elapsed_seconds": extract_elapsed_seconds(raw.get("elapsed_text", "")),
        "total_damage_text": extract_korean_number_text(raw.get("total_damage_text", "")),
        "dps_text": extract_korean_number_text(raw.get("dps_text", "")),
        "damage_increase_efficiency": extract_first_percent(raw.get("damage_increase_efficiency", "")),
        "crit_rate": extract_first_percent(raw.get("crit_rate", "")),
        "head_attack_rate": extract_first_percent(raw.get("head_attack_rate", "")),
        "back_attack_rate": extract_first_percent(raw.get("back_attack_rate", "")),
        "damage_increase_uptime": extract_first_percent(raw.get("damage_increase_uptime", "")),
        "burst_uptime": extract_first_percent(raw.get("burst_uptime", "")),
    }
    meta["total_damage"] = parse_korean_number(meta.get("total_damage_text"))
    meta["dps"] = parse_korean_number(meta.get("dps_text"))
    meta["elapsed_source"] = "ocr" if meta.get("elapsed_seconds") is not None else "ocr_failed"
    return meta


def parse_attack_fixed_grid(image: Image.Image, reader: Any, row_count: int | None = None, scale: int = 7, auto_window: bool = True) -> pd.DataFrame:
    grid = ATTACK_GRID
    if row_count is None:
        row_count = int(grid["row_count"])
    window_box = detect_analyzer_window(image, reader) if auto_window else None

    percent_keys = {"back_attack_rate", "crit_rate", "cooldown_rate", "share_rate"}
    korean_number_keys = {"damage_text", "dps_text"}
    rows: List[Dict[str, Any]] = []
    for i in range(int(row_count)):
        y1 = grid["row_start"] + i * grid["row_height"]
        y2 = y1 + grid["row_height"]
        row: Dict[str, Any] = {col: "" for col in STANDARD_COLUMNS}
        row["_ocr_row_index"] = str(i)
        if window_box:
            row["_window_x1"], row["_window_y1"], row["_window_x2"], row["_window_y2"] = window_box

        for key, label, x1, x2 in grid["columns"]:
            kind = "percent" if key in percent_keys else "korean_number" if key in korean_number_keys else "text"
            text = ocr_crop(reader, image, (x1, y1, x2, y2), numeric=(key != "name"), scale=scale, window_box=window_box, kind=kind)
            if key in korean_number_keys:
                text = extract_korean_number_text(text)
            elif key in percent_keys:
                text = format_percent_from_ocr(text)
            row[label] = text

        meaningful = False
        for col in STANDARD_COLUMNS:
            val = str(row.get(col) or "").strip()
            if val and val not in ["-", "_", "None", "nan"]:
                meaningful = True
                break
        if meaningful:
            rows.append(row)

    cols = STANDARD_COLUMNS + ["_ocr_row_index", "_window_x1", "_window_y1", "_window_x2", "_window_y2"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        nonempty = df[STANDARD_COLUMNS].astype(str).apply(lambda r: any(x.strip() and x.strip() not in ["", "-", "_", "None", "nan"] for x in r), axis=1)
        df = df[nonempty].reset_index(drop=True)
    return df


def make_summary_ocr_debug(image: Image.Image, reader: Any, scale: int = 7) -> List[Dict[str, Any]]:
    labels = {
        "elapsed_text": "전투 시간",
        "total_damage_text": "총 피해량",
        "dps_text": "DPS",
        "damage_increase_efficiency": "피해 증가 유효율",
        "stagger_text": "무력화",
        "crit_rate": "치명타 적중률",
        "head_attack_rate": "헤드어택 적중률",
        "back_attack_rate": "백어택 적중률",
        "damage_increase_uptime": "피해 증가 가동률",
        "burst_uptime": "폭주 가동률",
    }
    window_box = detect_analyzer_window(image, reader)
    percent_keys = {"damage_increase_efficiency", "crit_rate", "head_attack_rate", "back_attack_rate", "damage_increase_uptime", "burst_uptime"}
    rows: List[Dict[str, Any]] = []
    for key, box in SUMMARY_ROIS.items():
        raw_crop = crop_norm(image, box, pad=1, window_box=window_box)
        kind = "percent" if key in percent_keys else "korean_number" if key in {"total_damage_text", "dps_text", "stagger_text"} else "generic"
        variants = _ocr_text_variants(reader, raw_crop, numeric=True, scale=scale, kind=kind)
        variants.sort(key=lambda x: x[3], reverse=True)
        best_name, best_crop, best_text, best_score = variants[0]
        if kind == "percent":
            best_text = format_percent_from_ocr(best_text)
        rows.append({
            "key": key,
            "label": labels.get(key, key),
            "text": best_text,
            "raw_crop": raw_crop,
            "processed_crop": best_crop,
            "window_box": window_box,
            "preprocess": best_name,
            "score": best_score,
        })
    return rows


def make_attack_ocr_debug(image: Image.Image, reader: Any, row_count: int = 18, scale: int = 7) -> List[Dict[str, Any]]:
    grid = ATTACK_GRID
    window_box = detect_analyzer_window(image, reader)
    percent_keys = {"back_attack_rate", "crit_rate", "cooldown_rate", "share_rate"}
    korean_number_keys = {"damage_text", "dps_text"}
    rows: List[Dict[str, Any]] = []
    for i in range(int(row_count)):
        y1 = grid["row_start"] + i * grid["row_height"]
        y2 = y1 + grid["row_height"]
        row_crop = crop_norm(image, (ATTACK_ICON_GRID["x1"], y1, grid["columns"][-1][3], y2), pad=0, window_box=window_box)
        icon_crop = crop_attack_icon(image, i, window_box=window_box)
        cells: List[Dict[str, Any]] = []
        for key, label, x1, x2 in grid["columns"]:
            raw_crop = crop_norm(image, (x1, y1, x2, y2), pad=1, window_box=window_box)
            kind = "percent" if key in percent_keys else "korean_number" if key in korean_number_keys else "text"
            variants = _ocr_text_variants(reader, raw_crop, numeric=(key != "name"), scale=scale, kind=kind)
            variants.sort(key=lambda x: x[3], reverse=True)
            best_name, best_crop, best_text, best_score = variants[0]
            if key in korean_number_keys:
                best_text = extract_korean_number_text(best_text)
            elif key in percent_keys:
                best_text = format_percent_from_ocr(best_text)
            cells.append({
                "key": key,
                "label": label,
                "text": best_text,
                "raw_crop": raw_crop,
                "processed_crop": best_crop,
                "preprocess": best_name,
                "score": best_score,
            })
        rows.append({
            "row_index": i,
            "row_crop": row_crop,
            "icon_crop": icon_crop,
            "cells": cells,
            "window_box": window_box,
        })
    return rows


# ==============================
# v30: dynamic icon-row detection override
# ==============================
# 고정 row_start만으로 표를 나누면 전투분석기 창 위치/행 높이/스크롤 상태에 따라 하단 행이 깨질 수 있습니다.
# v30부터는 공격정보 표의 왼쪽 스킬 아이콘을 먼저 찾아서 행 중심을 잡고, 그 행 기준으로 각 열을 crop합니다.

from typing import Optional


def _window_rel_x_from_full_norm(x_norm: float, window_box: Tuple[int, int, int, int]) -> int:
    wx1, wy1, wx2, wy2 = window_box
    rel_x = (x_norm * BASE_SCREEN_W - BASE_WINDOW_BOX[0]) / BASE_WINDOW_W
    return int(round(wx1 + rel_x * (wx2 - wx1)))


def _window_rel_y_from_full_norm(y_norm: float, window_box: Tuple[int, int, int, int]) -> int:
    wx1, wy1, wx2, wy2 = window_box
    rel_y = (y_norm * BASE_SCREEN_H - BASE_WINDOW_BOX[1]) / BASE_WINDOW_H
    return int(round(wy1 + rel_y * (wy2 - wy1)))


def _expand_pixel_box(image: Image.Image, box: Tuple[int, int, int, int], ratio: float = 0.0, min_px: int = 0) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    if ratio <= 0 and min_px <= 0:
        return box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    dx = max(min_px, int(round(w * ratio)))
    dy = max(min_px, int(round(h * ratio)))
    return (max(0, x1 - dx), max(0, y1 - dy), min(image.width, x2 + dx), min(image.height, y2 + dy))


def _clip_box(image: Image.Image, box: Tuple[int, int, int, int], pad: int = 0, expand_ratio: float = 0.0) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = _expand_pixel_box(image, box, ratio=expand_ratio, min_px=2 if expand_ratio > 0 else 0)
    return (max(0, x1 + pad), max(0, y1 + pad), min(image.width, x2 - pad), min(image.height, y2 - pad))


def _crop_pixel_box(image: Image.Image, box: Tuple[int, int, int, int], pad: int = 0, expand_ratio: float = 0.0) -> Image.Image:
    return image.crop(_clip_box(image, box, pad=pad, expand_ratio=expand_ratio))


def detect_attack_icon_rows(
    image: Image.Image,
    reader: Any | None = None,
    *,
    window_box: Tuple[int, int, int, int] | None = None,
    max_rows: int = 24,
) -> List[Dict[str, Any]]:
    """공격정보 표의 왼쪽 스킬 아이콘을 감지해 행 좌표를 반환합니다.

    반환 row에는 row_box/icon_box가 전체 이미지 픽셀 좌표로 들어갑니다.
    OCR 이름보다 아이콘 위치가 훨씬 안정적이므로, 이후 API 스킬 아이콘 매칭에도 이 crop을 그대로 사용합니다.
    """
    if window_box is None:
        window_box = detect_analyzer_window(image, reader)
    if window_box is None:
        return []

    wx1, wy1, wx2, wy2 = window_box
    ww, wh = wx2 - wx1, wy2 - wy1
    if ww <= 0 or wh <= 0:
        return []

    arr = np.asarray(image.convert("RGB"))
    win = arr[wy1:wy2, wx1:wx2]
    if win.size == 0:
        return []

    try:
        import cv2
        hsv = cv2.cvtColor(win, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        # 스킬 아이콘/이름 선택 배경은 채도와 밝기가 배경보다 높습니다.
        mask = ((sat > 45) & (val > 45)).astype(np.uint8)
    except Exception:
        gray = np.asarray(image.crop(window_box).convert("L"))
        mask = (gray > 55).astype(np.uint8)

    # 기존 1920x1080 기준 아이콘 x좌표를 창 내부 좌표로 변환하고, 여유를 둡니다.
    icon_x1 = _window_rel_x_from_full_norm(ATTACK_ICON_GRID["x1"], window_box) - wx1
    icon_x2 = _window_rel_x_from_full_norm(ATTACK_ICON_GRID["x2"], window_box) - wx1
    sx1 = max(0, int(icon_x1) - 16)
    sx2 = min(ww, int(icon_x2) + 22)

    # 공격정보 표 행 영역만 검사합니다. 너무 위쪽 탭/헤더나 너무 아래쪽 페이지 버튼은 제외합니다.
    base_row_start = _window_rel_y_from_full_norm(ATTACK_GRID["row_start"], window_box) - wy1
    base_row_h = max(24, int(round((ATTACK_GRID["row_height"] * BASE_SCREEN_H / BASE_WINDOW_H) * wh)))
    sy1 = max(0, int(base_row_start - base_row_h * 2.2))
    sy2 = min(wh, int(base_row_start + base_row_h * max(max_rows + 2, 8)))
    band = mask[sy1:sy2, sx1:sx2]
    if band.size == 0:
        return []

    scores = band.sum(axis=1).astype(float)
    if len(scores) < 5:
        return []
    kernel = max(5, min(11, int(base_row_h // 4) * 2 + 1))
    smooth = np.convolve(scores, np.ones(kernel) / kernel, mode="same")
    max_score = float(np.max(smooth)) if len(smooth) else 0.0
    if max_score <= 0:
        return []
    thr = max(8.0, max_score * 0.32)

    segments: List[Tuple[int, int, float]] = []
    in_seg = False
    start = 0
    for i, s in enumerate(smooth):
        if s > thr and not in_seg:
            start = i
            in_seg = True
        elif s <= thr and in_seg:
            if i - start >= max(8, base_row_h // 3):
                segments.append((start + sy1, i + sy1, float(np.max(smooth[start:i]))))
            in_seg = False
    if in_seg and len(smooth) - start >= max(8, base_row_h // 3):
        segments.append((start + sy1, len(smooth) + sy1, float(np.max(smooth[start:]))))

    if not segments:
        return []

    # 파란 선택 배경이나 이름 막대가 연결되면 두 행이 하나의 segment로 붙을 수 있습니다.
    # 긴 segment는 기준 행 높이로 다시 쪼개서 `행 3+행 4`가 한 행으로 합쳐지는 문제를 막습니다.
    split_centers: List[float] = []
    for a, b, _mx in segments:
        length = b - a
        if length > base_row_h * 1.35:
            n = max(2, int(round(length / float(base_row_h))))
            step = length / float(n)
            for k in range(n):
                split_centers.append(a + step * (k + 0.5))
        else:
            split_centers.append((a + b) / 2.0)

    # 중심점 중복 제거 및 행 높이 추정
    centers = sorted(split_centers)
    dedup: List[float] = []
    for c in centers:
        if not dedup or abs(c - dedup[-1]) > base_row_h * 0.55:
            dedup.append(c)
        else:
            dedup[-1] = (dedup[-1] + c) / 2.0

    if len(dedup) >= 2:
        diffs = [dedup[i + 1] - dedup[i] for i in range(len(dedup) - 1) if 20 <= dedup[i + 1] - dedup[i] <= 70]
        row_h = int(round(float(np.median(diffs)))) if diffs else base_row_h
    else:
        row_h = base_row_h
    row_h = int(max(30, min(54, row_h)))

    rows: List[Dict[str, Any]] = []
    for idx, c in enumerate(dedup[: int(max_rows)]):
        top_rel = int(round(c - row_h * 0.52))
        bot_rel = int(round(top_rel + row_h))
        if top_rel < 0 or bot_rel > wh:
            continue
        # 행 전체 box는 첫 열 아이콘부터 마지막 지분 열까지.
        row_x1 = max(0, sx1 - 4)
        row_x2 = min(ww, _window_rel_x_from_full_norm(ATTACK_GRID["columns"][-1][3], window_box) - wx1 + 6)
        # v30: 아이콘 crop이 너무 딱 맞으면 테두리/선택행/스킬명 일부 때문에 매칭이 흔들립니다.
        # 좌우/상하를 조금 더 포함하고, 실제 비교 단계에서 중앙 정사각형으로 다시 정규화합니다.
        icon_box = (
            wx1 + max(0, int(icon_x1) - 8),
            wy1 + max(0, top_rel + 1),
            wx1 + min(ww, int(icon_x2) + 8),
            wy1 + min(wh, bot_rel - 1),
        )
        row_box = (wx1 + row_x1, wy1 + top_rel, wx1 + row_x2, wy1 + bot_rel)
        rows.append({
            "row_index": idx,
            "row_center_y": wy1 + int(round(c)),
            "row_box": row_box,
            "icon_box": icon_box,
            "window_box": window_box,
            "row_height": row_h,
            "source": "icon_detect",
        })
    return rows


def _column_pixel_box_for_row(
    image: Image.Image,
    window_box: Tuple[int, int, int, int],
    row_box: Tuple[int, int, int, int],
    x1_norm: float,
    x2_norm: float,
    *,
    pad_x: int = 1,
    pad_y: int = 1,
) -> Tuple[int, int, int, int]:
    x1 = _window_rel_x_from_full_norm(x1_norm, window_box)
    x2 = _window_rel_x_from_full_norm(x2_norm, window_box)
    y1, y2 = row_box[1], row_box[3]
    return _clip_box(image, (x1, y1, x2, y2), pad=0 if (pad_x == 0 and pad_y == 0) else 0)


def _ocr_attack_cell_by_box(reader: Any, image: Image.Image, box: Tuple[int, int, int, int], *, key: str, scale: int) -> str:
    raw = _crop_pixel_box(image, box, pad=1, expand_ratio=CELL_CROP_EXPAND_RATIO)
    percent_keys = {"back_attack_rate", "crit_rate", "cooldown_rate", "share_rate", "head_attack_rate"}
    korean_number_keys = {"damage_text", "dps_text"}
    kind = "percent" if key in percent_keys else "korean_number" if key in korean_number_keys else "text"
    variants = _ocr_text_variants(reader, raw, numeric=(key != "name"), scale=scale, kind=kind)
    if not variants:
        return ""
    variants.sort(key=lambda x: x[3], reverse=True)
    text = clean_ocr_text(variants[0][2])
    if key in korean_number_keys:
        return extract_korean_number_text(text)
    if key in percent_keys:
        return format_percent_from_ocr(text)
    return text


def parse_attack_fixed_grid(image: Image.Image, reader: Any, row_count: int | None = None, scale: int = 7, auto_window: bool = True) -> pd.DataFrame:
    """공격정보 OCR. v30에서는 아이콘 행 감지를 우선 사용합니다."""
    grid = ATTACK_GRID
    if row_count is None:
        row_count = int(grid["row_count"])
    window_box = detect_analyzer_window(image, reader) if auto_window else None
    detected_rows = detect_attack_icon_rows(image, reader, window_box=window_box, max_rows=int(row_count)) if window_box else []

    # 아이콘 행 감지가 실패하면 기존 고정 행으로 fallback합니다.
    if not detected_rows:
        detected_rows = []
        for i in range(int(row_count)):
            y1 = grid["row_start"] + i * grid["row_height"]
            y2 = y1 + grid["row_height"]
            if window_box:
                row_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, grid["columns"][-1][3], y2), pad=0, window_box=window_box)
                icon_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, ATTACK_ICON_GRID["x2"], y2), pad=1, window_box=window_box)
            else:
                row_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, grid["columns"][-1][3], y2), pad=0)
                icon_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, ATTACK_ICON_GRID["x2"], y2), pad=1)
            detected_rows.append({"row_index": i, "row_box": row_box, "icon_box": icon_box, "window_box": window_box, "source": "fixed_fallback"})

    rows: List[Dict[str, Any]] = []
    for i, rinfo in enumerate(detected_rows[: int(row_count)]):
        row: Dict[str, Any] = {col: "" for col in STANDARD_COLUMNS}
        row["_ocr_row_index"] = str(i)
        row["_row_source"] = rinfo.get("source", "")
        if window_box:
            row["_window_x1"], row["_window_y1"], row["_window_x2"], row["_window_y2"] = window_box
        ix1, iy1, ix2, iy2 = rinfo["icon_box"]
        row["_icon_x1"], row["_icon_y1"], row["_icon_x2"], row["_icon_y2"] = ix1, iy1, ix2, iy2
        row["_row_x1"], row["_row_y1"], row["_row_x2"], row["_row_y2"] = rinfo["row_box"]

        for key, label, x1, x2 in grid["columns"]:
            if window_box:
                box = _column_pixel_box_for_row(image, window_box, rinfo["row_box"], x1, x2)
            else:
                # fallback full-image normalized x, dynamic row y
                px1 = int(round(x1 * image.width)); px2 = int(round(x2 * image.width))
                box = _clip_box(image, (px1, rinfo["row_box"][1], px2, rinfo["row_box"][3]))
            row[label] = _ocr_attack_cell_by_box(reader, image, box, key=key, scale=scale)

        meaningful = False
        for col in STANDARD_COLUMNS:
            val = str(row.get(col) or "").strip()
            if val and val not in ["-", "_", "None", "nan"]:
                meaningful = True
                break
        if meaningful:
            rows.append(row)

    cols = STANDARD_COLUMNS + [
        "_ocr_row_index", "_row_source",
        "_window_x1", "_window_y1", "_window_x2", "_window_y2",
        "_icon_x1", "_icon_y1", "_icon_x2", "_icon_y2",
        "_row_x1", "_row_y1", "_row_x2", "_row_y2",
    ]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        nonempty = df[STANDARD_COLUMNS].astype(str).apply(lambda r: any(x.strip() and x.strip() not in ["", "-", "_", "None", "nan"] for x in r), axis=1)
        df = df[nonempty].reset_index(drop=True)
    return df


def crop_attack_icon(image: Image.Image, row_index: int, window_box: Tuple[int, int, int, int] | None = None) -> Image.Image | None:
    """기존 호환용 fallback 아이콘 crop."""
    try:
        grid = ATTACK_GRID
        y1 = grid["row_start"] + int(row_index) * grid["row_height"]
        y2 = y1 + grid["row_height"]
        return crop_norm(image, (ATTACK_ICON_GRID["x1"], y1, ATTACK_ICON_GRID["x2"], y2), pad=1, window_box=window_box)
    except Exception:
        return None


def _crop_icon_from_row_metadata(image: Image.Image, row: pd.Series | Dict[str, Any], fallback_index: int) -> Image.Image | None:
    try:
        vals = [row.get("_icon_x1"), row.get("_icon_y1"), row.get("_icon_x2"), row.get("_icon_y2")]
        if all(v is not None and str(v) not in ["", "nan", "None"] for v in vals):
            box = tuple(int(float(v)) for v in vals)
            return _crop_pixel_box(image, box, pad=0, expand_ratio=ICON_CROP_EXPAND_RATIO)
    except Exception:
        pass
    return crop_attack_icon(image, fallback_index, window_box=_row_window_box(row))


def correct_battle_skill_names_with_icons(
    df: pd.DataFrame,
    attack_image: Image.Image,
    skill_candidates: List[Dict[str, str]],
    *,
    name_col: str = "이름",
    threshold: float = 0.62,
    icon_threshold: float = 0.72,
    drop_unmatched: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty or not skill_candidates or name_col not in df.columns:
        return df
    out = df.copy()
    rows = []
    for idx, row in out.iterrows():
        try:
            row_index = int(float(row.get("_ocr_row_index", idx)))
        except Exception:
            row_index = int(idx)
        icon_crop = _crop_icon_from_row_metadata(attack_image, row, row_index)
        matched, text_score, icon_score, ok, reason = best_skill_name_match_with_icon(
            row.get(name_col),
            icon_crop,
            skill_candidates,
            text_threshold=threshold,
            icon_threshold=icon_threshold,
        )
        new_row = row.copy()
        if ok:
            new_row[name_col] = matched
        elif drop_unmatched:
            continue
        new_row["_name_match_text_score"] = text_score
        new_row["_name_match_icon_score"] = icon_score
        new_row["_name_match_reason"] = reason
        rows.append(new_row)
    if not rows:
        return out.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


def extract_elapsed_seconds(text: str) -> float | None:
    """전투 시간 추출 강화.

    EasyOCR이 `05:04`를 `05 4 049`처럼 읽는 케이스를 보정합니다.
    """
    text = clean_ocr_text(text)
    t = text.replace("O", "0").replace("o", "0").replace("ㅣ", "1").replace("|", "1")
    m = re.search(r"(\d{1,2})\s*[:：]\s*(\d{1,2})", t)
    if m:
        mm, ss = int(m.group(1)), int(m.group(2))
        if 0 <= mm < 60 and 0 <= ss < 60:
            return mm * 60 + ss
    m = re.search(r"(\d{1,2})\s*[6bB]\s*(\d{1,2})", t)
    if m:
        mm, ss = int(m.group(1)), int(m.group(2))
        if 0 <= mm < 60 and 0 <= ss < 60:
            return mm * 60 + ss

    tokens = re.findall(r"\d+", t)
    if not tokens:
        return None

    # 콜론이 사라져 시간이 '한 덩어리'(예: 02:37 → '0237', 05:04 → '504'/'0504')로만 읽힌 경우.
    # 토큰이 정확히 1개일 때만 적용합니다. (예: '05 : : 043'처럼 토큰 2개면 아래 쌍 탐색이
    # '05'+'04'로 05:04를 맞추므로, 여기서 '043'을 00:43으로 잘못 해석하지 않게 합니다.)
    if len(tokens) == 1 and 3 <= len(tokens[0]) <= 4:
        tok = tokens[0]
        mm, ss = int(tok[:-2]), int(tok[-2:])
        if 0 <= mm < 60 and 0 <= ss < 60:
            return mm * 60 + ss

    def mm_candidates(tok: str) -> List[int]:
        vals = []
        if len(tok) <= 2:
            vals.append(int(tok))
        else:
            vals.append(int(tok[-2:]))
            vals.append(int(tok[:2]))
        return [v for v in vals if 0 <= v < 60]

    def ss_candidates(tok: str) -> List[Tuple[int, float]]:
        vals: List[Tuple[int, float]] = []
        if len(tok) <= 2:
            vals.append((int(tok), 2.0 if len(tok) == 2 else 0.4))
        else:
            if tok.startswith("0") and len(tok) >= 2:
                vals.append((int(tok[:2]), 2.2))
            vals.append((int(tok[-2:]), 0.8))
            vals.append((int(tok[:2]), 0.7))
        return [(v, sc) for v, sc in vals if 0 <= v < 60]

    best: Tuple[float, int, int] | None = None
    for i, a in enumerate(tokens):
        for j in range(i + 1, min(len(tokens), i + 4)):
            for mm in mm_candidates(a):
                for ss, ssc in ss_candidates(tokens[j]):
                    score = ssc
                    score += 1.0 if len(a) == 2 else 0.0
                    score += 1.0 if j == i + 1 else 0.4
                    # 콜론이 1자리 숫자로 끼어들어간 흔한 케이스: 05 4 049
                    if j == i + 2 and len(tokens[i + 1]) == 1 and tokens[j].startswith("0"):
                        score += 1.4
                    if 1 <= mm <= 20:
                        score += 0.3
                    if best is None or score > best[0]:
                        best = (score, mm, ss)
    if best is not None:
        return best[1] * 60 + best[2]
    return None


def parse_summary_fixed_grid(image: Image.Image, reader: Any, scale: int = 7, auto_window: bool = True) -> Dict[str, Any]:
    _pt = _prof_now()
    window_box = detect_analyzer_window(image, reader) if auto_window else None
    _prof_add("summary_1_window_detect", _pt)
    # 제목(전투 분석기) 텍스트 OCR이 한글 인식 실패로 창을 못 찾으면, 선(Hough) fallback이
    # 엉뚱한 박스(x1≈30, y1≈190 등)를 돌려줍니다. 이때 x 뿐 아니라 y도 틀어져서 모든 종합정보
    # ROI가 위/아래로 밀려 DPS·총피해량·전투시간을 못 읽습니다.
    # → x1이 비정상적으로 작으면(감지 실패로 간주) 창 박스 '전체'를 BASE 좌표(해상도 스케일)로
    #   되돌립니다. 게임 전체화면에서 전투분석기 창은 고정 위치(가운데)에 열리므로 BASE가 정답입니다.
    sx = image.width / BASE_SCREEN_W
    sy = image.height / BASE_SCREEN_H
    base_box = (
        int(round(BASE_WINDOW_BOX[0] * sx)),
        int(round(BASE_WINDOW_BOX[1] * sy)),
        int(round(BASE_WINDOW_BOX[2] * sx)),
        int(round(BASE_WINDOW_BOX[3] * sy)),
    )
    if window_box is None:
        window_box = base_box
        print(f"[parse_summary] ⚠ 창 미감지 → BASE 좌표 사용: {window_box}")
    else:
        dx1 = abs(window_box[0] - base_box[0])
        dy1 = abs(window_box[1] - base_box[1])
        # x1이 비정상적으로 작거나(감지 실패), 상단 y가 기대 위치에서 크게 벗어나면
        # 제목 OCR(한글) 실패로 인한 잘못된 박스로 보고 창 박스 전체를 BASE로 되돌립니다.
        if window_box[0] < 100 or dx1 > image.width * 0.10 or dy1 > image.height * 0.05:
            raw_box = tuple(window_box)
            window_box = base_box
            print(f"[parse_summary] ⚠ 창 감지 부정확 → BASE 좌표로 전체 보정: {raw_box} → {window_box}")
    raw: Dict[str, str] = {}
    percent_keys = {"damage_increase_efficiency", "crit_rate", "head_attack_rate", "back_attack_rate", "damage_increase_uptime", "burst_uptime"}
    # v70.5-glyph: 종합정보 퍼센트 필드도 글리프 우선(게이트 통과 시 EasyOCR 변형 3~4회를 통째로 생략).
    # 전투시간(time)·총피해량·DPS·무력화(억/만/숫자)는 글리프 금지, 항상 EasyOCR.
    _sum_glyph_store = _get_glyph_store_v705() if OCR_GLYPH_FAST_V705 else None
    _sum_glyph_stats = {"attempt": 0, "accept": 0}
    _pt_glyph = _prof_now()
    _pt_easy_total = 0.0
    for key, box in SUMMARY_ROIS.items():
        kind = "percent" if key in percent_keys else "korean_number" if key in {"total_damage_text", "dps_text", "stagger_text"} else "generic"
        if _sum_glyph_store is not None and key in percent_keys:
            _sum_glyph_stats["attempt"] += 1
            try:
                _gcrop = crop_norm(image, box, pad=1, window_box=window_box)
                # 종합정보 셀은 넓어서 라벨/막대가 섞임 → 우측 숫자만 분리 후 글리프.
                _gcrop = _glyph_isolate_number_crop_v705(_gcrop)
                _gtext, _gconf = _sum_glyph_store.recognize(_gcrop, kind="percent")
            except Exception:
                _gtext, _gconf = "", 0.0
            if _gconf >= GLYPH_GATE_PERCENT and _glyph_pct_format_ok_v705(_gtext):
                raw[key] = _gtext
                _sum_glyph_stats["accept"] += 1
                continue  # EasyOCR 생략
        _pt_field = _prof_now()
        raw[key] = ocr_crop(reader, image, box, numeric=True, scale=scale, window_box=window_box, kind=kind)
        if _pt_field is not None:
            _pt_easy_total += (_time_v705.perf_counter() - _pt_field)
    _prof_add("summary_2_fields_total", _pt_glyph)
    if OCR_PROFILE_V705:
        _OCR_PROFILE_DATA_V705.setdefault("phases", {})["summary_3_fields_easyocr"] = \
            _OCR_PROFILE_DATA_V705.get("phases", {}).get("summary_3_fields_easyocr", 0.0) + _pt_easy_total
    _GLYPH_LAST_STATS_V705["summary"] = _sum_glyph_stats

    from modules.calculators import parse_korean_number, format_korean_number

    meta: Dict[str, Any] = {
        "raw": raw,
        "window_box": window_box,
        "window_detected": window_box is not None,
        "elapsed_seconds": extract_elapsed_seconds(raw.get("elapsed_text", "")),
        "damage_increase_efficiency": extract_first_percent(raw.get("damage_increase_efficiency", "")),
        "crit_rate": extract_first_percent(raw.get("crit_rate", "")),
        "head_attack_rate": extract_first_percent(raw.get("head_attack_rate", "")),
        "back_attack_rate": extract_first_percent(raw.get("back_attack_rate", "")),
        "damage_increase_uptime": extract_first_percent(raw.get("damage_increase_uptime", "")),
        "burst_uptime": extract_first_percent(raw.get("burst_uptime", "")),
    }
    # 총피해량/DPS: '억/만/조' 글자 오인식에 흔들리지 않도록 전체 원시 숫자를 우선 사용하고,
    # 단위는 format_korean_number로 정확히 역산해 표시 텍스트(예: 1,015.60억)를 만듭니다.
    _td_raw = full_raw_number_from_summary(raw.get("total_damage_text", ""))
    _dps_raw = full_raw_number_from_summary(raw.get("dps_text", ""))
    meta["total_damage"] = _td_raw if _td_raw is not None else parse_korean_number(extract_korean_number_text(raw.get("total_damage_text", "")))
    meta["dps"] = _dps_raw if _dps_raw is not None else parse_korean_number(extract_korean_number_text(raw.get("dps_text", "")))
    meta["total_damage_text"] = format_korean_number(meta["total_damage"]) if meta.get("total_damage") else extract_korean_number_text(raw.get("total_damage_text", ""))
    meta["dps_text"] = format_korean_number(meta["dps"]) if meta.get("dps") else extract_korean_number_text(raw.get("dps_text", ""))
    # v46: 전투시간은 '(전투 시간 …)' 라벨 템플릿으로 숫자 칸만 잘라 더 정밀하게 읽습니다.
    # 기존 ROI OCR이 '정보 (전투 시간 05:04)'처럼 라벨과 섞여 깨지는 문제를 줄입니다.
    _te = _read_elapsed_text_by_template_v46(image, window_box, reader)
    if _te:
        _ts = extract_elapsed_seconds(_te)
        if _ts is not None:
            meta["elapsed_seconds"] = _ts
            try:
                meta["raw"]["elapsed_text"] = _te
            except Exception:
                pass
    meta["elapsed_source"] = "ocr" if meta.get("elapsed_seconds") is not None else "ocr_failed"
    return meta


def make_attack_ocr_debug(image: Image.Image, reader: Any, row_count: int = 18, scale: int = 7) -> List[Dict[str, Any]]:
    grid = ATTACK_GRID
    window_box = detect_analyzer_window(image, reader)
    detected_rows = detect_attack_icon_rows(image, reader, window_box=window_box, max_rows=int(row_count)) if window_box else []
    if not detected_rows:
        detected_rows = []
        for i in range(int(row_count)):
            y1 = grid["row_start"] + i * grid["row_height"]
            y2 = y1 + grid["row_height"]
            row_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, grid["columns"][-1][3], y2), pad=0, window_box=window_box)
            icon_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, ATTACK_ICON_GRID["x2"], y2), pad=1, window_box=window_box)
            detected_rows.append({"row_index": i, "row_box": row_box, "icon_box": icon_box, "window_box": window_box, "source": "fixed_fallback"})

    percent_keys = {"back_attack_rate", "crit_rate", "cooldown_rate", "share_rate"}
    korean_number_keys = {"damage_text", "dps_text"}
    rows: List[Dict[str, Any]] = []
    for i, rinfo in enumerate(detected_rows[: int(row_count)]):
        row_crop = _crop_pixel_box(image, rinfo["row_box"], pad=0, expand_ratio=0.01)
        icon_crop = _crop_pixel_box(image, rinfo["icon_box"], pad=0, expand_ratio=ICON_CROP_EXPAND_RATIO)
        cells: List[Dict[str, Any]] = []
        for key, label, x1, x2 in grid["columns"]:
            if window_box:
                box = _column_pixel_box_for_row(image, window_box, rinfo["row_box"], x1, x2)
            else:
                px1 = int(round(x1 * image.width)); px2 = int(round(x2 * image.width))
                box = _clip_box(image, (px1, rinfo["row_box"][1], px2, rinfo["row_box"][3]))
            raw_crop = _crop_pixel_box(image, box, pad=1, expand_ratio=CELL_CROP_EXPAND_RATIO)
            kind = "percent" if key in percent_keys else "korean_number" if key in korean_number_keys else "text"
            variants = _ocr_text_variants(reader, raw_crop, numeric=(key != "name"), scale=scale, kind=kind)
            variants.sort(key=lambda x: x[3], reverse=True)
            best_name, best_crop, best_text, best_score = variants[0]
            if key in korean_number_keys:
                best_text = extract_korean_number_text(best_text)
            elif key in percent_keys:
                best_text = format_percent_from_ocr(best_text)
            cells.append({
                "key": key,
                "label": label,
                "text": best_text,
                "raw_crop": raw_crop,
                "processed_crop": best_crop,
                "preprocess": best_name,
                "score": best_score,
            })
        rows.append({
            "row_index": i,
            "row_crop": row_crop,
            "icon_crop": icon_crop,
            "cells": cells,
            "window_box": window_box,
            "row_source": rinfo.get("source", ""),
            "row_box": rinfo.get("row_box"),
            "icon_box": rinfo.get("icon_box"),
        })
    return rows

# ==============================
# v31: rune/orb/misc candidate normalization override
# ==============================
KNOWN_RUNE_NAMES_V31 = [
    "출혈", "중독", "질풍", "광분", "풍요", "집중", "단죄", "심판", "압도", "정화", "속행", "수호", "철벽",
]
RUNE_OCR_ALIASES_V31 = {
    "출혈": ["출혈", "축혈", "출현", "춘혈", "추혈", "출협"],
    "중독": ["중독", "종독", "중록", "증독"],
    "질풍": ["질풍", "질품", "질퐁"],
    "광분": ["광분", "광문", "괭분"],
    "풍요": ["풍요", "풍오", "풍유"],
    "집중": ["집중", "집증", "집주"],
    "단죄": ["단죄", "단제"],
    "심판": ["심판", "심팜"],
    "압도": ["압도", "압토"],
    "정화": ["정화", "청화"],
    "속행": ["속행", "속헹"],
    "수호": ["수호", "수오"],
    "철벽": ["철벽", "철벅"],
}


def _compact_ocr_name_v31(text: Any) -> str:
    s = clean_ocr_text(str(text or ""))
    s = s.replace(" ", "")
    s = s.replace("|", "I").replace("ㅣ", "I")
    s = re.sub(r"[^0-9A-Za-z가-힣]+", "", s)
    return s


def canonicalize_battle_name_v31(text: Any) -> str:
    """전투분석기 이름 정규화 기본 함수.

    v33에서는 특정 OCR 오인식 문자열을 특정 이름으로 강제 치환하지 않습니다.
    """
    return clean_ocr_text(str(text or "")).strip()


def _candidate_name_icon_pairs(skill_candidates: List[Any]) -> List[Dict[str, Any]]:  # type: ignore[override]
    pairs: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in skill_candidates or []:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("스킬명") or item.get("Name") or "").strip()
            icon = str(item.get("icon") or item.get("아이콘") or item.get("Icon") or "").strip()
            source = str(item.get("source") or "").strip()
        else:
            name = str(item or "").strip()
            icon = ""
            source = ""
        if name:
            key = (name, icon)
            if key not in seen:
                seen.add(key)
                pairs.append({"name": name, "icon": icon, "source": source})
    # API가 룬/기타를 주지 않아도 텍스트 후보는 항상 둡니다.
    for r in KNOWN_RUNE_NAMES_V31:
        key = (f"스킬룬 {r}", "")
        if key not in seen:
            seen.add(key)
            pairs.append({"name": f"스킬룬 {r}", "icon": "", "source": "rune_builtin"})
    for name in ["기타", "기본 공격", "맥스웰 맥시마", "맥스웰 멕시마"]:
        key = (name, "")
        if key not in seen:
            seen.add(key)
            pairs.append({"name": name, "icon": "", "source": "builtin_misc"})
    return pairs


def best_skill_name_match_with_icon(  # type: ignore[override]
    ocr_name: Any,
    row_icon: Image.Image | None,
    skill_candidates: List[Any],
    *,
    text_threshold: float = 0.55,
    icon_threshold: float = 0.78,
) -> Tuple[str, float, float, bool, str]:
    """OCR 이름을 API/룬/보주/기타 후보로 보정합니다.

    v33: 특정 OCR 문자열 강제 치환 없이 API/아이콘 후보만으로 보정합니다.
    """
    pairs = _candidate_name_icon_pairs(skill_candidates)
    original = str(ocr_name or "").strip()
    canonical = canonicalize_battle_name_v31(original)
    if not pairs:
        return canonical or original, 0.0, 0.0, bool(canonical and canonical != original), "canonical" if canonical != original else "no_candidates"

    norm_original = normalize_skill_name_for_match(canonical or original)
    scored: List[Dict[str, Any]] = []
    for pair in pairs:
        name = pair["name"]
        norm_skill = normalize_skill_name_for_match(name)
        text_score = 0.0
        if norm_original and norm_skill:
            text_score = SequenceMatcher(None, norm_original, norm_skill).ratio()
            if norm_original in norm_skill or norm_skill in norm_original:
                text_score = max(text_score, 0.88)

        icon_score = 0.0
        if row_icon is not None and pair.get("icon"):
            api_icon = _download_icon_image(pair["icon"])
            if api_icon is not None:
                icon_score = _icon_similarity(row_icon, api_icon)

        # 룬/보주/기타는 텍스트가 조금만 맞아도 후보로 남기되, 스킬/보주 아이콘이 있으면 아이콘을 우선합니다.
        combined = 0.30 * text_score + 0.70 * icon_score
        scored.append({"name": name, "text": text_score, "icon": icon_score, "combined": combined, "source": pair.get("source", "")})

    scored.sort(key=lambda x: (x["icon"], x["combined"], x["text"]), reverse=True)
    best_icon = scored[0]
    second_icon = scored[1]["icon"] if len(scored) > 1 else 0.0
    icon_margin = best_icon["icon"] - second_icon
    icon_confident = best_icon["icon"] >= icon_threshold and (icon_margin >= 0.045 or best_icon["icon"] >= min(0.95, icon_threshold + 0.09))
    if icon_confident:
        return best_icon["name"], round(best_icon["text"] * 100.0, 2), round(best_icon["icon"] * 100.0, 2), True, "icon"

    best_text = max(scored, key=lambda x: x["text"])
    if best_text["text"] >= text_threshold:
        return best_text["name"], round(best_text["text"] * 100.0, 2), round(best_text["icon"] * 100.0, 2), True, "text"

    # 후보 매칭이 실패해도 전투분석기 전용 이름은 정규화해서 보존합니다.
    if canonical and canonical != original:
        return canonical, round(best_icon.get("text", 0.0) * 100.0, 2), round(best_icon.get("icon", 0.0) * 100.0, 2), True, "canonical"
    return original, round(best_icon.get("text", 0.0) * 100.0, 2), round(best_icon.get("icon", 0.0) * 100.0, 2), False, "unmatched"


def correct_battle_skill_names_with_icons(  # type: ignore[override]
    df: pd.DataFrame,
    attack_image: Image.Image,
    skill_candidates: List[Dict[str, str]],
    *,
    name_col: str = "이름",
    threshold: float = 0.62,
    icon_threshold: float = 0.72,
    drop_unmatched: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty or name_col not in df.columns:
        return df
    out = df.copy()
    rows = []
    for idx, row in out.iterrows():
        try:
            row_index = int(float(row.get("_ocr_row_index", idx)))
        except Exception:
            row_index = int(idx)
        icon_crop = _crop_icon_from_row_metadata(attack_image, row, row_index)
        matched, text_score, icon_score, ok, reason = best_skill_name_match_with_icon(
            row.get(name_col),
            icon_crop,
            skill_candidates or [],
            text_threshold=threshold,
            icon_threshold=icon_threshold,
        )
        new_row = row.copy()
        if ok or matched:
            new_row[name_col] = matched
        elif drop_unmatched:
            continue
        new_row["_name_match_text_score"] = text_score
        new_row["_name_match_icon_score"] = icon_score
        new_row["_name_match_reason"] = reason
        rows.append(new_row)
    if not rows:
        return out.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)

# ==============================
# v34: full name OCR restored + GPU default + safer icon fallback
# ==============================
# 요약:
# - API가 룬/보주/기타 아이콘을 주지 않는 경우가 있어 프로젝트 assets/battle_icons의 로컬 아이콘도 후보로 사용합니다.
# - 이미지 OCR 속도 개선을 위해 이름 칸 OCR을 생략하고 아이콘 매칭으로 이름을 확정하는 fast mode를 추가합니다.
# - 사용자가 나중에 직접 수정한 이름과 행 아이콘을 manual cache로 저장할 수 있는 보조 함수도 제공합니다.

import shutil


def _project_root_v32() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path.cwd()


def _safe_icon_filename_v32(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z가-힣_ -]+", "_", str(name or "").strip())
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "unknown"


# local path도 아이콘 후보로 받을 수 있게 override합니다.
@lru_cache(maxsize=1024)
def _download_icon_image(url: str) -> Image.Image | None:  # type: ignore[override]
    if not url:
        return None
    try:
        value = str(url).strip()
        if value.startswith("file://"):
            value = value[7:]
        p = Path(value)
        if p.exists() and p.is_file():
            return Image.open(p).convert("RGB")

        # 상대 경로 후보
        root = _project_root_v32()
        rel = root / value
        if rel.exists() and rel.is_file():
            return Image.open(rel).convert("RGB")

        cache_dir = root / "cache" / "skill_icons"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ext = ".png"
        m = re.search(r"\.(png|jpg|jpeg|webp)(?:$|[?])", value, flags=re.I)
        if m:
            ext = "." + m.group(1).lower().replace("jpeg", "jpg")
        key = hashlib.sha1(value.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{key}{ext}"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return Image.open(cache_path).convert("RGB")
        if not value.lower().startswith("http"):
            return None
        resp = requests.get(value, timeout=6)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def _local_icon_candidates_v32() -> List[Dict[str, str]]:
    """로컬 수동 아이콘 후보 v41.

    이전 버전에서 잘못 저장된 manual 아이콘이 이름 매칭을 오염시키는 문제가 있어
    기본적으로 manual cache는 사용하지 않습니다. 필요하면 프로젝트 루트에
    `cache/use_manual_icons.flag` 파일을 만든 경우에만 읽습니다.
    """
    root = _project_root_v32()
    flag = root / "cache" / "use_manual_icons.flag"
    if not flag.exists():
        return []
    rows: List[Dict[str, str]] = []
    manual_dir = root / "cache" / "battle_icons" / "manual"
    if manual_dir.exists():
        for p in sorted(manual_dir.glob("*.png")):
            name = p.stem.replace("_", " ").strip()
            if name:
                rows.append({"name": name, "icon": str(p), "source": "local_manual_icon"})
    return rows


def canonicalize_battle_name_v31(text: Any) -> str:  # type: ignore[override]
    """전투분석기 이름 정규화 v33.

    범용성을 위해 특정 OCR 오인식 문자열을 특정 이름으로 강제 치환하지 않습니다.
    이름 확정은 API/로컬 manual 아이콘 매칭 또는 API 후보명과의 일반 유사도만 사용합니다.
    """
    return clean_ocr_text(str(text or "")).strip()


def _candidate_name_icon_pairs(skill_candidates: List[Any]) -> List[Dict[str, Any]]:  # type: ignore[override]
    pairs: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(name: Any, icon: Any = "", source: str = "") -> None:
        n = str(name or "").strip()
        i = str(icon or "").strip()
        if not n:
            return
        key = (n, i)
        if key in seen:
            return
        seen.add(key)
        pairs.append({"name": n, "icon": i, "source": source})

    # API/외부 후보. 스킬, 룬, 보주 등 API에서 실제로 받은 이름/아이콘만 사용합니다.
    for item in skill_candidates or []:
        if isinstance(item, dict):
            add(item.get("name") or item.get("스킬명") or item.get("Name"), item.get("icon") or item.get("아이콘") or item.get("Icon"), item.get("source") or "")
        else:
            add(item, "", "")

    # 사용자가 검수 후 저장한 manual 아이콘만 추가합니다.
    for item in _local_icon_candidates_v32():
        add(item.get("name"), item.get("icon"), item.get("source", "local"))

    return pairs


def best_skill_name_match_with_icon(  # type: ignore[override]
    ocr_name: Any,
    row_icon: Image.Image | None,
    skill_candidates: List[Any],
    *,
    text_threshold: float = 0.55,
    icon_threshold: float = 0.72,
) -> Tuple[str, float, float, bool, str]:
    pairs = _candidate_name_icon_pairs(skill_candidates)
    original = str(ocr_name or "").strip()
    canonical = canonicalize_battle_name_v31(original)
    if not pairs:
        return canonical or original, 0.0, 0.0, bool(canonical and canonical != original), "canonical" if canonical != original else "no_candidates"

    norm_original = normalize_skill_name_for_match(canonical or original)
    scored: List[Dict[str, Any]] = []
    for pair in pairs:
        name = pair["name"]
        norm_skill = normalize_skill_name_for_match(name)
        text_score = 0.0
        if norm_original and norm_skill:
            text_score = SequenceMatcher(None, norm_original, norm_skill).ratio()
            if norm_original in norm_skill or norm_skill in norm_original:
                text_score = max(text_score, 0.88)

        icon_score = 0.0
        if row_icon is not None and pair.get("icon"):
            api_icon = _download_icon_image(pair["icon"])
            if api_icon is not None:
                icon_score = _icon_similarity(row_icon, api_icon)
        combined = 0.25 * text_score + 0.75 * icon_score
        scored.append({"name": name, "text": text_score, "icon": icon_score, "combined": combined, "source": pair.get("source", "")})

    scored.sort(key=lambda x: (x["icon"], x["combined"], x["text"]), reverse=True)
    best_icon = scored[0]
    second_icon = scored[1]["icon"] if len(scored) > 1 else 0.0
    icon_margin = best_icon["icon"] - second_icon

    # 로컬/수동 아이콘은 같은 화면에서 잘라 둔 것이므로 점수가 조금 낮아도 신뢰합니다.
    source = str(best_icon.get("source") or "")
    local_bonus = source.startswith("local_")
    local_threshold = max(0.52, icon_threshold - 0.18) if local_bonus else icon_threshold
    icon_confident = best_icon["icon"] >= local_threshold and (icon_margin >= 0.025 or best_icon["icon"] >= min(0.94, local_threshold + 0.08))
    best_text = max(scored, key=lambda x: x["text"])

    # v34: 이름 OCR을 다시 신뢰합니다.
    # OCR이 현재 API 스킬명과 거의 일치하면, 아이콘 매칭이 약간 흔들려도 텍스트를 우선합니다.
    # 이렇게 해야 아이콘 crop이 밀려 여러 행이 길로틴/진화 같은 한 후보로 몰리는 문제를 줄일 수 있습니다.
    strong_text = best_text["text"] >= max(float(text_threshold), 0.82)
    very_strong_icon = best_icon["icon"] >= 0.94 and icon_margin >= 0.05
    if strong_text and not very_strong_icon:
        return best_text["name"], round(best_text["text"] * 100.0, 2), round(best_text["icon"] * 100.0, 2), True, "text_first"

    if icon_confident:
        return best_icon["name"], round(best_icon["text"] * 100.0, 2), round(best_icon["icon"] * 100.0, 2), True, f"icon:{source or 'api'}"

    if best_text["text"] >= text_threshold:
        return best_text["name"], round(best_text["text"] * 100.0, 2), round(best_text["icon"] * 100.0, 2), True, "text"

    if canonical and canonical != original:
        return canonical, round(best_icon.get("text", 0.0) * 100.0, 2), round(best_icon.get("icon", 0.0) * 100.0, 2), True, "canonical"
    return original, round(best_icon.get("text", 0.0) * 100.0, 2), round(best_icon.get("icon", 0.0) * 100.0, 2), False, "unmatched"


def parse_attack_fixed_grid(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int | None = None,
    scale: int = 7,
    auto_window: bool = True,
) -> pd.DataFrame:
    """공격정보 OCR v35.

    이름 칸 OCR을 항상 읽고, API/아이콘 후보는 보조 보정으로 사용합니다.
    """
    grid = ATTACK_GRID
    if row_count is None:
        row_count = int(grid["row_count"])
    window_box = detect_analyzer_window(image, reader) if auto_window else None
    detected_rows = detect_attack_icon_rows(image, reader, window_box=window_box, max_rows=int(row_count)) if window_box else []

    if not detected_rows:
        detected_rows = []
        for i in range(int(row_count)):
            y1 = grid["row_start"] + i * grid["row_height"]
            y2 = y1 + grid["row_height"]
            row_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, grid["columns"][-1][3], y2), pad=0, window_box=window_box)
            icon_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, ATTACK_ICON_GRID["x2"], y2), pad=1, window_box=window_box)
            detected_rows.append({"row_index": i, "row_box": row_box, "icon_box": icon_box, "window_box": window_box, "source": "fixed_fallback"})

    rows: List[Dict[str, Any]] = []
    for i, rinfo in enumerate(detected_rows[: int(row_count)]):
        row: Dict[str, Any] = {col: "" for col in STANDARD_COLUMNS}
        row["_ocr_row_index"] = str(i)
        row["_row_source"] = rinfo.get("source", "")
        if window_box:
            row["_window_x1"], row["_window_y1"], row["_window_x2"], row["_window_y2"] = window_box
        ix1, iy1, ix2, iy2 = rinfo["icon_box"]
        row["_icon_x1"], row["_icon_y1"], row["_icon_x2"], row["_icon_y2"] = ix1, iy1, ix2, iy2
        row["_row_x1"], row["_row_y1"], row["_row_x2"], row["_row_y2"] = rinfo["row_box"]

        for key, label, x1, x2 in grid["columns"]:
            if window_box:
                box = _column_pixel_box_for_row(image, window_box, rinfo["row_box"], x1, x2)
            else:
                px1 = int(round(x1 * image.width)); px2 = int(round(x2 * image.width))
                box = _clip_box(image, (px1, rinfo["row_box"][1], px2, rinfo["row_box"][3]))
            row[label] = _ocr_attack_cell_by_box(reader, image, box, key=key, scale=scale)

        # 이름 OCR을 생략해도 숫자 중 하나라도 있으면 행 유지
        meaningful = False
        for col in [c for c in STANDARD_COLUMNS if c != "이름"]:
            val = str(row.get(col) or "").strip()
            if val and val not in ["-", "_", "None", "nan"]:
                meaningful = True
                break
        if meaningful or str(row.get("이름") or "").strip():
            rows.append(row)

    cols = STANDARD_COLUMNS + [
        "_ocr_row_index", "_row_source",
        "_window_x1", "_window_y1", "_window_x2", "_window_y2",
        "_icon_x1", "_icon_y1", "_icon_x2", "_icon_y2",
        "_row_x1", "_row_y1", "_row_x2", "_row_y2",
    ]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        nonempty = df[STANDARD_COLUMNS].astype(str).apply(lambda r: any(x.strip() and x.strip() not in ["", "-", "_", "None", "nan"] for x in r), axis=1)
        df = df[nonempty].reset_index(drop=True)
    return df


def save_learned_battle_icons_v32(df: pd.DataFrame, attack_image: Image.Image | None) -> int:
    """검수된 이름과 행 아이콘을 로컬 manual cache에 저장합니다.

    다음 OCR부터 API에 없는 기타/보주/룬도 아이콘으로 매칭할 수 있습니다.
    """
    if df is None or df.empty or attack_image is None or "이름" not in df.columns:
        return 0
    root = _project_root_v32()
    out_dir = root / "cache" / "battle_icons" / "manual"
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, row in df.iterrows():
        name = str(row.get("이름") or "").strip()
        if not name or name in {"이름", "피해량", "DPS"}:
            continue
        icon = _crop_icon_from_row_metadata(attack_image, row, int(row.get("_ocr_row_index", idx) or idx))
        if icon is None:
            continue
        filename = _safe_icon_filename_v32(name) + ".png"
        try:
            # 아이콘 영역에서 왼쪽 정사각형 중심부만 저장해두면 다음 매칭이 더 안정적입니다.
            icon_norm = _square_resize_for_icon(icon, size=64, margin_ratio=0.04)
            icon_norm.save(out_dir / filename)
            count += 1
        except Exception:
            pass
    return count

# ==============================
# v36: 신규 전투분석기 공격정보 레이아웃 지원
# ==============================
# 신규 표: 이름 / 피해량 / 백어택 적중률 / 백어택 비중 / 치명타 적중률 / 치명타 비중 / 사용 횟수 / 쿨타임 비율
# DPS와 피해량 지분은 OCR하지 않고 app.py에서 전투시간/총피해량으로 역산합니다.
STANDARD_COLUMNS = [
    "이름",
    "피해량",
    "백어택 적중률",
    "백어택 비중",
    "헤드어택 적중률",
    "치명타 적중률",
    "치명타 비중",
    "사용 횟수",
    "쿨타임 비율",
    "초당 피해량",
    "피해량 지분",
]

# 2048x1048 예시 스크린샷의 새 표 열 위치를 기존 1920x1080 기준 창 내부 좌표로 환산한 값입니다.
# 전투분석기 창이 이동되어도 detect_analyzer_window 후 창 내부 좌표로 다시 변환됩니다.
ATTACK_GRID = {
    "row_start": 285.1 / 1080,
    "row_height": 45.1 / 1080,
    "row_count": 14,
    "columns": [
        ("name", "이름", 278.7 / 1920, 474.8 / 1920),
        ("damage_text", "피해량", 478.6 / 1920, 646.0 / 1920),
        ("back_attack_rate", "백어택 적중률", 649.8 / 1920, 816.4 / 1920),
        ("back_attack_share", "백어택 비중", 819.2 / 1920, 986.7 / 1920),
        ("crit_rate", "치명타 적중률", 990.5 / 1920, 1156.1 / 1920),
        ("crit_share", "치명타 비중", 1159.9 / 1920, 1327.4 / 1920),
        ("casts", "사용 횟수", 1331.2 / 1920, 1494.9 / 1920),
        ("cooldown_rate", "쿨타임 비율", 1496.8 / 1920, 1656.7 / 1920),
    ],
}
ATTACK_ICON_GRID = {
    "x1": 229.2 / 1920,
    "x2": 273.0 / 1920,
}


def _ocr_attack_cell_by_box(  # type: ignore[override]
    reader: Any,
    image: Image.Image,
    box: Tuple[int, int, int, int],
    *,
    key: str,
    scale: int,
) -> str:
    raw = _crop_pixel_box(image, box, pad=1, expand_ratio=CELL_CROP_EXPAND_RATIO)
    percent_keys = {"back_attack_rate", "back_attack_share", "crit_rate", "crit_share", "cooldown_rate", "share_rate", "head_attack_rate"}
    korean_number_keys = {"damage_text", "dps_text"}
    kind = "percent" if key in percent_keys else "korean_number" if key in korean_number_keys else "text"
    variants = _ocr_text_variants(reader, raw, numeric=(key != "name"), scale=scale, kind=kind)
    if not variants:
        return ""
    variants.sort(key=lambda x: x[3], reverse=True)
    text = clean_ocr_text(variants[0][2])
    if key in korean_number_keys:
        return extract_korean_number_text(text)
    if key in percent_keys:
        return format_percent_from_ocr(text)
    return text


def parse_attack_fixed_grid(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int | None = None,
    scale: int = 7,
    auto_window: bool = True,
) -> pd.DataFrame:
    """공격정보 OCR v36. 새 전투분석기 표 레이아웃을 읽습니다."""
    grid = ATTACK_GRID
    if row_count is None:
        row_count = int(grid["row_count"])
    window_box = detect_analyzer_window(image, reader) if auto_window else None
    detected_rows = detect_attack_icon_rows(image, reader, window_box=window_box, max_rows=int(row_count)) if window_box else []

    if not detected_rows:
        detected_rows = []
        for i in range(int(row_count)):
            y1 = grid["row_start"] + i * grid["row_height"]
            y2 = y1 + grid["row_height"]
            row_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, grid["columns"][-1][3], y2), pad=0, window_box=window_box)
            icon_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, ATTACK_ICON_GRID["x2"], y2), pad=1, window_box=window_box)
            detected_rows.append({"row_index": i, "row_box": row_box, "icon_box": icon_box, "window_box": window_box, "source": "fixed_fallback"})

    rows: List[Dict[str, Any]] = []
    for i, rinfo in enumerate(detected_rows[: int(row_count)]):
        row: Dict[str, Any] = {col: "" for col in STANDARD_COLUMNS}
        row["_ocr_row_index"] = str(i)
        row["_row_source"] = rinfo.get("source", "")
        if window_box:
            row["_window_x1"], row["_window_y1"], row["_window_x2"], row["_window_y2"] = window_box
        ix1, iy1, ix2, iy2 = rinfo["icon_box"]
        row["_icon_x1"], row["_icon_y1"], row["_icon_x2"], row["_icon_y2"] = ix1, iy1, ix2, iy2
        row["_row_x1"], row["_row_y1"], row["_row_x2"], row["_row_y2"] = rinfo["row_box"]

        for key, label, x1, x2 in grid["columns"]:
            if window_box:
                box = _column_pixel_box_for_row(image, window_box, rinfo["row_box"], x1, x2)
            else:
                px1 = int(round(x1 * image.width)); px2 = int(round(x2 * image.width))
                box = _clip_box(image, (px1, rinfo["row_box"][1], px2, rinfo["row_box"][3]))
            row[label] = _ocr_attack_cell_by_box(reader, image, box, key=key, scale=scale)

        meaningful = False
        for col in [c for c in STANDARD_COLUMNS if c not in {"이름", "초당 피해량", "피해량 지분"}]:
            val = str(row.get(col) or "").strip()
            if val and val not in ["-", "_", "None", "nan"]:
                meaningful = True
                break
        if meaningful or str(row.get("이름") or "").strip():
            rows.append(row)

    cols = STANDARD_COLUMNS + [
        "_ocr_row_index", "_row_source",
        "_window_x1", "_window_y1", "_window_x2", "_window_y2",
        "_icon_x1", "_icon_y1", "_icon_x2", "_icon_y2",
        "_row_x1", "_row_y1", "_row_x2", "_row_y2",
    ]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        check_cols = [c for c in STANDARD_COLUMNS if c not in {"초당 피해량", "피해량 지분"}]
        nonempty = df[check_cols].astype(str).apply(lambda r: any(x.strip() and x.strip() not in ["", "-", "_", "None", "nan"] for x in r), axis=1)
        df = df[nonempty].reset_index(drop=True)
    return df


def make_attack_ocr_debug(image: Image.Image, reader: Any, row_count: int = 18, scale: int = 7) -> List[Dict[str, Any]]:  # type: ignore[override]
    grid = ATTACK_GRID
    window_box = detect_analyzer_window(image, reader)
    detected_rows = detect_attack_icon_rows(image, reader, window_box=window_box, max_rows=int(row_count)) if window_box else []
    if not detected_rows:
        detected_rows = []
        for i in range(int(row_count)):
            y1 = grid["row_start"] + i * grid["row_height"]
            y2 = y1 + grid["row_height"]
            row_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, grid["columns"][-1][3], y2), pad=0, window_box=window_box)
            icon_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, ATTACK_ICON_GRID["x2"], y2), pad=1, window_box=window_box)
            detected_rows.append({"row_index": i, "row_box": row_box, "icon_box": icon_box, "window_box": window_box, "source": "fixed_fallback"})

    rows: List[Dict[str, Any]] = []
    for i, rinfo in enumerate(detected_rows[: int(row_count)]):
        row_crop = _crop_pixel_box(image, rinfo["row_box"], pad=0, expand_ratio=0.01)
        icon_crop = _crop_pixel_box(image, rinfo["icon_box"], pad=0, expand_ratio=ICON_CROP_EXPAND_RATIO)
        cells: List[Dict[str, Any]] = []
        for key, label, x1, x2 in grid["columns"]:
            if window_box:
                box = _column_pixel_box_for_row(image, window_box, rinfo["row_box"], x1, x2)
            else:
                px1 = int(round(x1 * image.width)); px2 = int(round(x2 * image.width))
                box = _clip_box(image, (px1, rinfo["row_box"][1], px2, rinfo["row_box"][3]))
            raw_crop = _crop_pixel_box(image, box, pad=1, expand_ratio=CELL_CROP_EXPAND_RATIO)
            percent_keys = {"back_attack_rate", "back_attack_share", "crit_rate", "crit_share", "cooldown_rate", "head_attack_rate"}
            korean_number_keys = {"damage_text", "dps_text"}
            kind = "percent" if key in percent_keys else "korean_number" if key in korean_number_keys else "text"
            variants = _ocr_text_variants(reader, raw_crop, numeric=(key != "name"), scale=scale, kind=kind)
            variants.sort(key=lambda x: x[3], reverse=True)
            best_name, best_crop, best_text, best_score = variants[0]
            if key in korean_number_keys:
                best_text = extract_korean_number_text(best_text)
            elif key in percent_keys:
                best_text = format_percent_from_ocr(best_text)
            cells.append({
                "key": key,
                "label": label,
                "text": best_text,
                "raw_crop": raw_crop,
                "processed_crop": best_crop,
                "preprocess": best_name,
                "score": best_score,
            })
        rows.append({
            "row_index": i,
            "row_crop": row_crop,
            "icon_crop": icon_crop,
            "cells": cells,
            "window_box": window_box,
            "row_source": rinfo.get("source", ""),
            "row_box": rinfo.get("row_box"),
            "icon_box": rinfo.get("icon_box"),
        })
    return rows


# ==============================
# v37: 창/표 자동 탐지 강화
# ==============================
# 이전 버전은 전투분석기 창을 감지한 뒤에도 화면 비율에 따라 창 크기를 추정해 21:9/16:9에서
# 창 위치가 흔들릴 수 있었습니다. v37은 스크린샷 전체 비율이 아니라 세로 해상도/UI 스케일과
# 실제 이미지의 긴 가로/세로 선을 이용해 창 바운딩 박스를 보정합니다.

BASE_WINDOW_TITLE_CENTER_X = (BASE_WINDOW_BOX[0] + BASE_WINDOW_BOX[2]) / 2.0
BASE_TITLE_OFFSET_X = BASE_WINDOW_TITLE_CENTER_X - BASE_WINDOW_BOX[0]


def _hough_window_lines_v37(image: Image.Image) -> tuple[list[tuple[int, int, int, int, int]], list[tuple[int, int, int, int, int]]]:
    """전투분석기 창/표의 긴 수평/수직선을 찾습니다.

    반환값은 (horizontals, verticals)이며 각 항목은 (x1, y1, x2, y2, length)입니다.
    OCR이 실패하거나 21:9에서 좌표 스케일이 어긋날 때 창 경계 보정에 사용합니다.
    """
    try:
        import cv2
        arr = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        # UI 선은 얇고 배경이 복잡하므로 Canny threshold를 낮춰 후보를 넉넉히 잡습니다.
        edges = cv2.Canny(gray, 45, 135)
        min_len = max(240, int(image.height * 0.38))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=85, minLineLength=min_len, maxLineGap=26)
        hs: list[tuple[int, int, int, int, int]] = []
        vs: list[tuple[int, int, int, int, int]] = []
        if lines is None:
            return hs, vs
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, l)
            if abs(y2 - y1) <= 3 and abs(x2 - x1) >= min_len:
                xa, xb = sorted((x1, x2))
                hs.append((xa, y1, xb, y2, xb - xa))
            elif abs(x2 - x1) <= 3 and abs(y2 - y1) >= max(260, int(image.height * 0.28)):
                ya, yb = sorted((y1, y2))
                vs.append((x1, ya, x2, yb, yb - ya))
        return hs, vs
    except Exception:
        return [], []


def _estimate_window_from_title_v37(image: Image.Image, title_cx: float, title_cy: float) -> tuple[int, int, int, int]:
    """제목 중심점으로 전투분석기 창을 추정합니다.

    핵심 변경점: 창 크기를 이미지 가로폭이 아니라 세로 해상도 기준으로 계산합니다.
    이렇게 해야 16:9/21:9에서 같은 UI 스케일을 유지할 수 있습니다.
    """
    scale = image.height / BASE_SCREEN_H
    win_w = BASE_WINDOW_W * scale
    win_h = BASE_WINDOW_H * scale
    off_x = BASE_TITLE_OFFSET_X * scale
    off_y = BASE_TITLE_OFFSET_Y * scale
    x1 = int(round(title_cx - off_x))
    y1 = int(round(title_cy - off_y))
    x2 = int(round(x1 + win_w))
    y2 = int(round(y1 + win_h))
    return _clip_box(image, (x1, y1, x2, y2), pad=0)


def _refine_window_box_with_lines_v37(image: Image.Image, predicted: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """예상 창 박스를 긴 UI 선으로 보정합니다.

    이동된 창은 크기가 고정이라 제목 기반 추정만으로도 대부분 맞지만, 21:9나 창이 하단으로 내려간
    경우 우측/하단 경계가 밀릴 수 있어 Hough line으로 가까운 경계를 스냅합니다.
    """
    px1, py1, px2, py2 = predicted
    pw, ph = px2 - px1, py2 - py1
    if pw <= 0 or ph <= 0:
        return predicted
    hs, vs = _hough_window_lines_v37(image)

    # 창 내부에 있는 긴 수평선/수직선을 후보로 제한합니다.
    h_near = []
    for x1, y1, x2, y2, length in hs:
        if length < pw * 0.42:
            continue
        if x2 < px1 - pw * 0.20 or x1 > px2 + pw * 0.20:
            continue
        if y1 < py1 - ph * 0.18 or y1 > py2 + ph * 0.18:
            continue
        h_near.append((x1, y1, x2, y2, length))

    v_near = []
    for x1, y1, x2, y2, length in vs:
        if length < ph * 0.35:
            continue
        if x1 < px1 - pw * 0.18 or x1 > px2 + pw * 0.18:
            continue
        if y2 < py1 - ph * 0.20 or y1 > py2 + ph * 0.20:
            continue
        v_near.append((x1, y1, x2, y2, length))

    nx1, ny1, nx2, ny2 = px1, py1, px2, py2

    # 위쪽 경계: 제목/탭 위쪽 긴 선이 근처에 있습니다.
    top_candidates = [y for _x1, y, _x2, _y2, _l in h_near if abs(y - py1) < ph * 0.18]
    if top_candidates:
        ny1 = int(round(min(top_candidates, key=lambda y: abs(y - py1))))

    # 아래쪽 경계: 표 하단/창 하단 긴 선. 너무 아래 HUD 선은 predicted 근처만 사용합니다.
    bottom_candidates = [y for _x1, y, _x2, _y2, _l in h_near if abs(y - py2) < ph * 0.20]
    if bottom_candidates:
        ny2 = int(round(min(bottom_candidates, key=lambda y: abs(y - py2))))

    # 좌우 경계는 세로선이 잡히면 스냅, 없으면 수평선 시작/끝으로 보정합니다.
    left_candidates = [x for x, _y1, _x2, _y2, _l in v_near if abs(x - px1) < pw * 0.16]
    right_candidates = [x for x, _y1, _x2, _y2, _l in v_near if abs(x - px2) < pw * 0.16]
    if left_candidates:
        nx1 = int(round(min(left_candidates, key=lambda x: abs(x - px1))))
    if right_candidates:
        nx2 = int(round(min(right_candidates, key=lambda x: abs(x - px2))))

    if h_near:
        left_from_h = [x1 for x1, y, _x2, _y2, l in h_near if l > pw * 0.50 and abs(y - py1) < ph * 0.30]
        right_from_h = [x2 for _x1, y, x2, _y2, l in h_near if l > pw * 0.50 and abs(y - py1) < ph * 0.30]
        if not left_candidates and left_from_h:
            nx1 = int(round(min(left_from_h, key=lambda x: abs(x - px1))))
        if not right_candidates and right_from_h:
            nx2 = int(round(min(right_from_h, key=lambda x: abs(x - px2))))

    # 보정값이 말이 안 되면 predicted 유지
    box = _clip_box(image, (nx1, ny1, nx2, ny2), pad=0)
    x1, y1, x2, y2 = box
    if (x2 - x1) < image.height * 0.95 or (y2 - y1) < image.height * 0.48:
        return predicted
    return box


def _detect_window_by_lines_only_v37(image: Image.Image) -> tuple[int, int, int, int] | None:
    """OCR 제목 감지가 실패했을 때 긴 UI 선만으로 창을 찾는 fallback입니다."""
    hs, vs = _hough_window_lines_v37(image)
    if not hs:
        return None
    scale = image.height / BASE_SCREEN_H
    expected_w = BASE_WINDOW_W * scale
    expected_h = BASE_WINDOW_H * scale
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for x1, y, x2, _y2, length in hs:
        # 창 상단/탭/헤더 후보. 창 너비의 절반 이상인 긴 선만 사용합니다.
        if length < expected_w * 0.45:
            continue
        if y > image.height * 0.70:
            continue
        # 선 시작점이 창 왼쪽이거나 창 내부 탭선일 수 있으므로, 예상 너비로 여러 x 후보를 만듭니다.
        for cand_x1 in (x1, x2 - int(expected_w), x1 - int(expected_w * 0.03)):
            cand_x2 = cand_x1 + int(expected_w)
            cand_y1 = int(y - (70 * scale)) if y > image.height * 0.10 else int(y)
            cand_y2 = cand_y1 + int(expected_h)
            box = _clip_box(image, (cand_x1, cand_y1, cand_x2, cand_y2), pad=0)
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw < expected_w * 0.75 or bh < expected_h * 0.65:
                continue
            # 후보 박스 내부에 긴 수평/수직선이 많을수록 점수 높임
            score = float(length)
            for hx1, hy, hx2, _hy2, hl in hs:
                if box[0] - 30 <= hx1 <= box[2] + 30 and box[1] - 30 <= hy <= box[3] + 30 and hl > expected_w * 0.35:
                    score += hl * 0.08
            for vx, vy1, _vx2, vy2, vl in vs:
                if box[0] - 30 <= vx <= box[2] + 30 and box[1] - 30 <= vy1 <= box[3] + 30 and vl > expected_h * 0.30:
                    score += vl * 0.10
            if best is None or score > best[0]:
                best = (score, box)
    if best is None:
        return None
    return _refine_window_box_with_lines_v37(image, best[1])


def detect_analyzer_window(image: Image.Image, reader: Any | None = None) -> Tuple[int, int, int, int] | None:  # type: ignore[override]
    """전투분석기 창 위치 자동 감지 v37.

    - 1순위: 전체 OCR에서 `전투 분석기` 제목을 찾습니다.
    - 창 크기는 이미지 가로폭이 아니라 세로 해상도 기준으로 산정해 16:9/21:9 모두 대응합니다.
    - 이후 Hough line으로 실제 UI 경계에 보정합니다.
    - OCR 실패 시 긴 UI 선만으로 fallback합니다.
    """
    title_best = None
    if reader is not None:
        try:
            # 제목 감지용은 원본 그대로 쓰면 한글이 작게 보일 수 있어 1.5배만 확대합니다.
            title_img = image.convert("RGB")
            scale_title = 1.0
            if image.height <= 1200:
                scale_title = 1.35
                title_img = title_img.resize((int(image.width * scale_title), int(image.height * scale_title)), Image.Resampling.LANCZOS)
            results = reader.readtext(np.array(title_img), detail=1, paragraph=False)
            best_score = -1.0
            for item in results:
                try:
                    bbox, text, conf = item[0], str(item[1]), float(item[2]) if len(item) > 2 else 0.5
                except Exception:
                    continue
                norm = _normalize_for_window_detection(text)
                if not (("전투" in norm and "분석" in norm) or "전투분석" in norm):
                    continue
                xs = [float(pt[0]) / scale_title for pt in bbox]
                ys = [float(pt[1]) / scale_title for pt in bbox]
                cx = (min(xs) + max(xs)) / 2.0
                cy = (min(ys) + max(ys)) / 2.0
                # 화면 어디든 가능. 다만 너무 아래 HUD 쪽 오탐은 제외합니다.
                if cy > image.height * 0.75:
                    continue
                score = conf + (0.5 if "전투분석" in norm else 0.0) + (0.25 if 4 <= len(norm) <= 10 else 0.0)
                if score > best_score:
                    title_best = (cx, cy, text, conf)
                    best_score = score
        except Exception:
            title_best = None

    if title_best is not None:
        cx, cy, _text, _conf = title_best
        predicted = _estimate_window_from_title_v37(image, cx, cy)
        refined = _refine_window_box_with_lines_v37(image, predicted)
        return refined

    return _detect_window_by_lines_only_v37(image)


def _window_rel_x_from_full_norm(x_norm: float, window_box: Tuple[int, int, int, int]) -> int:  # type: ignore[override]
    """기존 1920x1080 기준 full-norm x를 감지된 창 내부 상대 x로 변환합니다."""
    wx1, wy1, wx2, wy2 = window_box
    rel_x = (x_norm * BASE_SCREEN_W - BASE_WINDOW_BOX[0]) / BASE_WINDOW_W
    return int(round(wx1 + rel_x * (wx2 - wx1)))


def _window_rel_y_from_full_norm(y_norm: float, window_box: Tuple[int, int, int, int]) -> int:  # type: ignore[override]
    wx1, wy1, wx2, wy2 = window_box
    rel_y = (y_norm * BASE_SCREEN_H - BASE_WINDOW_BOX[1]) / BASE_WINDOW_H
    return int(round(wy1 + rel_y * (wy2 - wy1)))


def _snap_column_x_v37(image: Image.Image, window_box: Tuple[int, int, int, int], expected_x: int, row_y1: int, row_y2: int) -> int:
    """예상 컬럼 경계 x를 주변 세로 구분선에 스냅합니다."""
    try:
        import cv2
        wx1, wy1, wx2, wy2 = window_box
        search = 26
        x1 = max(wx1, expected_x - search)
        x2 = min(wx2, expected_x + search)
        y1 = max(wy1, row_y1 - 10)
        y2 = min(wy2, row_y2 + 10)
        if x2 <= x1 or y2 <= y1:
            return expected_x
        crop = np.asarray(image.crop((x1, y1, x2, y2)).convert("RGB"))
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        # 세로선은 주변보다 밝거나 푸른/회색 선이므로 x 방향 gradient와 밝기 누적을 함께 봅니다.
        grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        score = np.mean(np.abs(grad), axis=0) + np.mean(gray > 55, axis=0) * 16.0
        if len(score) == 0:
            return expected_x
        idx = int(np.argmax(score))
        if float(score[idx]) < max(4.0, float(np.mean(score) + np.std(score) * 0.8)):
            return expected_x
        snapped = x1 + idx
        # 너무 많이 튀는 스냅은 사용하지 않습니다.
        if abs(snapped - expected_x) > search - 2:
            return expected_x
        return int(snapped)
    except Exception:
        return expected_x


def _column_pixel_box_for_row(  # type: ignore[override]
    image: Image.Image,
    window_box: Tuple[int, int, int, int],
    row_box: Tuple[int, int, int, int],
    x1_norm: float,
    x2_norm: float,
    *,
    pad_x: int = 1,
    pad_y: int = 1,
) -> Tuple[int, int, int, int]:
    """행 기준 컬럼 box 생성 v37.

    창 위치를 먼저 잡고, 각 컬럼 x좌표는 헤더/표 세로 구분선 근처로 스냅해 창이 이동하거나
    16:9/21:9 비율이 달라도 안정적으로 자릅니다.
    """
    x1 = _window_rel_x_from_full_norm(x1_norm, window_box)
    x2 = _window_rel_x_from_full_norm(x2_norm, window_box)
    y1, y2 = row_box[1], row_box[3]
    # 이름 칸은 왼쪽 배경 막대 때문에 스냅이 오히려 불안할 수 있어 오른쪽 경계만 약하게 보정합니다.
    sx1 = _snap_column_x_v37(image, window_box, x1, y1, y2)
    sx2 = _snap_column_x_v37(image, window_box, x2, y1, y2)
    if sx2 - sx1 < 18:
        sx1, sx2 = x1, x2
    return _clip_box(image, (sx1, y1, sx2, y2), pad=0)


def detect_attack_icon_rows(  # type: ignore[override]
    image: Image.Image,
    reader: Any | None = None,
    *,
    window_box: Tuple[int, int, int, int] | None = None,
    max_rows: int = 24,
) -> List[Dict[str, Any]]:
    """공격정보 표의 왼쪽 아이콘을 감지해 행을 나눕니다. v37은 창 내부 기준으로만 탐색합니다."""
    if window_box is None:
        window_box = detect_analyzer_window(image, reader)
    if window_box is None:
        return []

    wx1, wy1, wx2, wy2 = window_box
    ww, wh = wx2 - wx1, wy2 - wy1
    if ww <= 0 or wh <= 0:
        return []

    arr = np.asarray(image.convert("RGB"))
    win = arr[wy1:wy2, wx1:wx2]
    if win.size == 0:
        return []

    try:
        import cv2
        hsv = cv2.cvtColor(win, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        # 아이콘은 이름칸 배경보다 채도/밝기 변화가 훨씬 큽니다. 너무 넓게 잡지 않기 위해 아이콘 x band만 봅니다.
        mask = ((sat > 48) & (val > 42)).astype(np.uint8)
        # 작은 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    except Exception:
        gray = np.asarray(image.crop(window_box).convert("L"))
        mask = (gray > 65).astype(np.uint8)

    icon_x1 = _window_rel_x_from_full_norm(ATTACK_ICON_GRID["x1"], window_box) - wx1
    icon_x2 = _window_rel_x_from_full_norm(ATTACK_ICON_GRID["x2"], window_box) - wx1
    sx1 = max(0, int(icon_x1) - 18)
    sx2 = min(ww, int(icon_x2) + 24)

    # 표 시작 y를 고정하지 않고, 헤더 아래부터 창 하단까지 넓게 본 뒤 아이콘 peak만 추출합니다.
    # 탭/헤더 오탐 방지를 위해 기준 row_start 위부터 시작합니다.
    # 게임 창 위치에 따라 첫 행이 예상보다 훨씬 위에 있을 수 있으므로 여유를 4행으로 넉넉히 잡습니다.
    base_row_start = _window_rel_y_from_full_norm(ATTACK_GRID["row_start"], window_box) - wy1
    base_row_h = max(28, int(round((ATTACK_GRID["row_height"] * BASE_SCREEN_H / BASE_WINDOW_H) * wh)))
    sy1 = max(0, int(base_row_start - base_row_h * 4))
    sy2 = min(wh, int(max(base_row_start + base_row_h * (max_rows + 3), wh * 0.95)))
    band = mask[sy1:sy2, sx1:sx2]
    if band.size == 0:
        return []

    scores = band.sum(axis=1).astype(float)
    kernel_len = max(5, min(13, int(base_row_h // 4) * 2 + 1))
    smooth = np.convolve(scores, np.ones(kernel_len) / kernel_len, mode="same")
    max_score = float(np.max(smooth)) if len(smooth) else 0.0
    if max_score <= 0:
        return []
    thr = max(6.0, max_score * 0.28)

    # peak 중심점 탐지. segment가 붙으면 expected row height로 나눕니다.
    segments: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    min_seg = max(7, int(base_row_h * 0.25))
    for i, s in enumerate(smooth):
        if s > thr and not in_seg:
            start = i
            in_seg = True
        elif s <= thr and in_seg:
            if i - start >= min_seg:
                segments.append((start + sy1, i + sy1))
            in_seg = False
    if in_seg and len(smooth) - start >= min_seg:
        segments.append((start + sy1, len(smooth) + sy1))

    if not segments:
        return []

    centers: list[float] = []
    for a, b in segments:
        length = b - a
        if length > base_row_h * 1.45:
            n = max(2, int(round(length / float(base_row_h))))
            step = length / float(n)
            for k in range(n):
                centers.append(a + step * (k + 0.5))
        else:
            centers.append((a + b) / 2.0)

    centers = sorted(centers)
    dedup: list[float] = []
    for c in centers:
        if not dedup or abs(c - dedup[-1]) > base_row_h * 0.50:
            dedup.append(c)
        else:
            dedup[-1] = (dedup[-1] + c) / 2.0

    if len(dedup) >= 2:
        diffs = [dedup[i + 1] - dedup[i] for i in range(len(dedup) - 1) if base_row_h * 0.55 <= dedup[i + 1] - dedup[i] <= base_row_h * 1.55]
        row_h = int(round(float(np.median(diffs)))) if diffs else base_row_h
    else:
        row_h = base_row_h
    row_h = int(max(28, min(58, row_h)))

    rows: list[dict[str, Any]] = []
    last_col_x = _window_rel_x_from_full_norm(ATTACK_GRID["columns"][-1][3], window_box) - wx1
    for idx, c in enumerate(dedup[: int(max_rows)]):
        top_rel = int(round(c - row_h * 0.52))
        bot_rel = int(round(top_rel + row_h))
        if top_rel < 0 or bot_rel > wh:
            continue
        row_x1 = max(0, sx1 - 6)
        row_x2 = min(ww, last_col_x + 8)
        icon_box = (
            wx1 + max(0, int(icon_x1) - 10),
            wy1 + max(0, top_rel - 1),
            wx1 + min(ww, int(icon_x2) + 10),
            wy1 + min(wh, bot_rel + 1),
        )
        row_box = (wx1 + row_x1, wy1 + top_rel, wx1 + row_x2, wy1 + bot_rel)
        rows.append({
            "row_index": idx,
            "row_center_y": wy1 + int(round(c)),
            "row_box": row_box,
            "icon_box": icon_box,
            "window_box": window_box,
            "row_height": row_h,
            "source": "icon_detect_v37",
        })
    return rows


# ==============================
# v38: 제목 기준 창 고정비율 탐지 + 백/헤드 적중/비중 분리
# ==============================
# v37의 선 보정은 일부 배경/내부 표 선을 창 경계로 잘못 스냅하면서 오히려 crop이 불안정해질 수 있었습니다.
# v38은 사용자가 표시한 것처럼 `전투 분석기` 제목 중심을 우선 찾고, 전투분석기 UI의 고정 비율로
# 전체 창을 계산합니다. 창 경계선 보정은 fallback에만 사용하고, 제목 검출 성공 시에는 과도하게 스냅하지 않습니다.

STANDARD_COLUMNS = [
    "이름",
    "피해량",
    "백어택 적중률",
    "백어택 비중",
    "헤드어택 적중률",
    "헤드어택 비중",
    "치명타 적중률",
    "치명타 비중",
    "사용 횟수",
    "쿨타임 비율",
    "초당 피해량",
    "피해량 지분",
]

# 신규 전분 실제 열 구조: 이름 / 피해량 / 방향성 적중률 / 방향성 비중 / 치명타 적중률 / 치명타 비중 / 사용 횟수 / 쿨타임 비율
# 방향성 열은 헤더가 백어택이면 백 컬럼에, 헤드어택이면 헤드 컬럼에 넣습니다.
ATTACK_GRID = {
    "row_start": 285.1 / 1080,
    "row_height": 45.1 / 1080,
    "row_count": 14,
    "columns": [
        ("name", "이름", 278.7 / 1920, 474.8 / 1920),
        ("damage_text", "피해량", 478.6 / 1920, 646.0 / 1920),
        ("directional_rate", "방향성 적중률", 649.8 / 1920, 816.4 / 1920),
        ("directional_share", "방향성 비중", 819.2 / 1920, 986.7 / 1920),
        ("crit_rate", "치명타 적중률", 990.5 / 1920, 1156.1 / 1920),
        ("crit_share", "치명타 비중", 1159.9 / 1920, 1327.4 / 1920),
        ("casts", "사용 횟수", 1331.2 / 1920, 1494.9 / 1920),
        ("cooldown_rate", "쿨타임 비율", 1496.8 / 1920, 1656.7 / 1920),
    ],
}
ATTACK_ICON_GRID = {"x1": 229.2 / 1920, "x2": 273.0 / 1920}


def _score_title_candidate_v38(text: str, conf: float, cx: float, cy: float, image: Image.Image) -> float:
    norm = _normalize_for_window_detection(text)
    if not norm:
        return -999.0
    # EasyOCR이 `전투`를 `친구`처럼 읽는 경우가 있어 `분석기`만 확실하면 후보로 인정합니다.
    has_analysis = "분석" in norm
    has_gi = "기" in norm or norm.endswith("7")
    has_battle = "전투" in norm
    if not has_analysis:
        return -999.0
    if cy > image.height * 0.78:
        return -999.0
    score = float(conf or 0.0)
    score += 1.00 if has_battle else 0.0
    score += 0.60 if "실전" in norm else 0.0          # 앱 타이틀 전체 매칭 우선
    score += 0.40 if "전투분석" in norm else 0.0       # `전투분석`(연속) = 실제 제목 신호
    score += 0.70 if "분석기" in norm else 0.35 if has_gi else 0.0
    score += 0.35 if 3 <= len(norm) <= 10 else 0.0
    # 제목은 창 가로 중앙에 고정 → 화면 중앙에 가까운 후보를 우대합니다.
    # (종합정보 이미지에서 cx≈758 의 엉뚱한 후보가 진짜 제목 cx≈942 를 이기던 문제 보정)
    center_norm = abs(cx - image.width / 2.0) / max(1.0, image.width * 0.5)
    score += 0.50 * (1.0 - min(1.0, center_norm))
    # 제목은 보통 창의 상단부에 있으므로 너무 아래쪽 후보는 약하게 감점합니다.
    score -= max(0.0, (cy / max(1, image.height) - 0.30)) * 0.8
    return score


def _find_title_by_ocr_v38(image: Image.Image, reader: Any | None) -> tuple[float, float, str, float] | None:
    if reader is None:
        return None
    try:
        title_img = image.convert("RGB")
        scale_title = 1.0
        if image.height <= 1200:
            scale_title = 1.45
            title_img = title_img.resize((int(image.width * scale_title), int(image.height * scale_title)), Image.Resampling.LANCZOS)
        results = reader.readtext(np.array(title_img), detail=1, paragraph=False)
    except Exception:
        return None
    best = None
    best_score = -999.0
    for item in results:
        try:
            bbox, text, conf = item[0], str(item[1]), float(item[2]) if len(item) > 2 else 0.5
        except Exception:
            continue
        xs = [float(pt[0]) / scale_title for pt in bbox]
        ys = [float(pt[1]) / scale_title for pt in bbox]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        score = _score_title_candidate_v38(text, conf, cx, cy, image)
        if score > best_score:
            best_score = score
            best = (cx, cy, text, conf)
    return best if best_score > 0.15 else None


def _clip_window_keep_size_v38(image: Image.Image, x1: int, y1: int, w: int, h: int) -> tuple[int, int, int, int]:
    x2, y2 = x1 + w, y1 + h
    if x1 < 0:
        x2 -= x1; x1 = 0
    if y1 < 0:
        y2 -= y1; y1 = 0
    if x2 > image.width:
        shift = x2 - image.width
        x1 = max(0, x1 - shift); x2 = image.width
    if y2 > image.height:
        shift = y2 - image.height
        y1 = max(0, y1 - shift); y2 = image.height
    return (int(x1), int(y1), int(x2), int(y2))


def _estimate_window_from_title_v38(image: Image.Image, title_cx: float, title_cy: float) -> tuple[int, int, int, int]:
    scale = image.height / BASE_SCREEN_H
    win_w = int(round(BASE_WINDOW_W * scale))
    win_h = int(round(BASE_WINDOW_H * scale))
    off_x = BASE_TITLE_OFFSET_X * scale
    off_y = BASE_TITLE_OFFSET_Y * scale
    x1 = int(round(title_cx - off_x))
    y1 = int(round(title_cy - off_y))
    return _clip_window_keep_size_v38(image, x1, y1, win_w, win_h)


def _detect_window_by_lines_only_v38(image: Image.Image) -> tuple[int, int, int, int] | None:
    """제목 OCR이 실패했을 때만 쓰는 보수적 fallback.

    가장 긴 수평선 후보를 창 상단/탭선으로 보고 UI 고정 비율을 적용합니다. v37처럼 내부 선에 과도하게
    스냅하지 않고, 최종 창 크기는 세로 해상도 기준 고정 비율로 유지합니다.
    """
    hs, _vs = _hough_window_lines_v37(image)
    if not hs:
        return None
    scale = image.height / BASE_SCREEN_H
    expected_w = int(round(BASE_WINDOW_W * scale))
    expected_h = int(round(BASE_WINDOW_H * scale))
    best = None
    best_score = -1.0
    for x1, y, x2, _y2, length in hs:
        if length < expected_w * 0.45:
            continue
        if y > image.height * 0.72:
            continue
        # 기본 탭/헤더선은 창 top보다 70~155px 아래에 있으므로 여러 후보를 본다.
        for y_offset in (0, int(70 * scale), int(105 * scale), int(150 * scale)):
            cand_y1 = int(y - y_offset)
            for cand_x1 in (x1, x2 - expected_w, int((x1 + x2 - expected_w) / 2)):
                box = _clip_window_keep_size_v38(image, cand_x1, cand_y1, expected_w, expected_h)
                bx1, by1, bx2, by2 = box
                if (bx2 - bx1) < expected_w * 0.86 or (by2 - by1) < expected_h * 0.86:
                    continue
                score = float(length) - abs((bx2 - bx1) - expected_w) * 0.2
                # 박스 안에 유사한 긴 선이 많으면 가산
                for hx1, hy, hx2, _hy2, hl in hs:
                    if bx1 - 20 <= hx1 <= bx2 + 20 and by1 <= hy <= by2 and hl > expected_w * 0.30:
                        score += hl * 0.035
                if score > best_score:
                    best_score = score
                    best = box
    return best


# ──────────────────────────────────────────────────────────────────────────
# v45: `전투 분석기` 제목 '이미지 템플릿' 매칭으로 창 위치 감지
#   - 제목 글자는 UI에서 고정 모양/위치이므로, 한글 OCR(인식 불안정) 대신
#     제목 글자 이미지를 통째로 템플릿 매칭해서 찾습니다. 가장 안정적입니다.
#   - 템플릿(data/title_template.png)은 1920x1080 기준에서 제목 중심이 (950, 97)일 때
#     창 박스가 BASE_WINDOW_BOX(214,68,1670,958)가 되도록 보정값을 미리 측정했습니다.
# ──────────────────────────────────────────────────────────────────────────
TITLE_TEMPLATE_FILENAME_V45 = "title_template.png"
# 템플릿이 잘린 원본(1920x1080)에서 측정한 제목 글자 중심
TITLE_TEMPLATE_CENTER_X_V45 = 950.0
TITLE_TEMPLATE_CENTER_Y_V45 = 97.0
# 그때의 창 박스(= BASE_WINDOW_BOX). 제목 중심에서 창 좌상단까지의 오프셋을 역산합니다.
_TITLE_OFF_X_V45 = TITLE_TEMPLATE_CENTER_X_V45 - BASE_WINDOW_BOX[0]   # 736
_TITLE_OFF_Y_V45 = TITLE_TEMPLATE_CENTER_Y_V45 - BASE_WINDOW_BOX[1]   # 29
TITLE_TEMPLATE_MATCH_THRESHOLD_V45 = 0.55


@lru_cache(maxsize=1)
def _load_title_template_v45() -> "np.ndarray | None":
    """제목 템플릿(그레이스케일)을 1회 로드해 캐시합니다."""
    try:
        import cv2
        candidates = [
            _project_root_v32() / "data" / TITLE_TEMPLATE_FILENAME_V45,
            Path(__file__).resolve().parent.parent / "data" / TITLE_TEMPLATE_FILENAME_V45,
            Path(__file__).resolve().parent / "data" / TITLE_TEMPLATE_FILENAME_V45,
        ]
        for p in candidates:
            if p.is_file():
                arr = np.asarray(Image.open(p).convert("RGB"))
                return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        print(f"[detect_window] 템플릿 로드 실패: {e}")
    return None


def _find_title_by_template_v45(image: Image.Image) -> tuple[float, float, float, float] | None:
    """제목 글자 이미지를 다중 스케일 템플릿 매칭으로 찾습니다.

    반환: (cx, cy, score, scale) 또는 None
    """
    try:
        import cv2
        tmpl = _load_title_template_v45()
        if tmpl is None:
            return None
        th, tw = tmpl.shape[:2]
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        H, W = gray.shape[:2]
        # 창은 사용자가 이동할 수 있으므로 제목을 화면 전체에서 찾습니다.
        # (제목은 창 상단이라 보통 위쪽이지만, 창을 아래로 옮기면 더 내려갈 수 있음)
        band = gray
        # 해상도 차이를 흡수하도록 스케일을 폭넓게 시도합니다(1080p≈1.0, 1440p≈1.33, 4K≈2.0).
        base_scale = H / BASE_SCREEN_H
        scales = sorted({round(base_scale * m, 3) for m in (0.85, 0.92, 1.0, 1.08, 1.16, 1.25)})
        best = (-1.0, None, None)
        for s in scales:
            tw2, th2 = int(round(tw * s)), int(round(th * s))
            if tw2 < 12 or th2 < 8 or th2 > band.shape[0] or tw2 > band.shape[1]:
                continue
            t = cv2.resize(tmpl, (tw2, th2), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            res = cv2.matchTemplate(band, t, cv2.TM_CCOEFF_NORMED)
            _minv, maxv, _minl, maxl = cv2.minMaxLoc(res)
            if maxv > best[0]:
                best = (float(maxv), (maxl[0] + tw2 / 2.0, maxl[1] + th2 / 2.0), s)
        if best[1] is None or best[0] < TITLE_TEMPLATE_MATCH_THRESHOLD_V45:
            return None
        cx, cy = best[1]
        return (cx, cy, best[0], best[2])
    except Exception as e:
        print(f"[detect_window] 템플릿 매칭 오류: {e}")
        return None


def _estimate_window_from_title_template_v45(image: Image.Image, cx: float, cy: float, scale: float) -> Tuple[int, int, int, int]:
    """템플릿으로 찾은 제목 중심(cx, cy)과 매칭 스케일로 창 박스를 역산합니다."""
    x1 = cx - _TITLE_OFF_X_V45 * scale
    y1 = cy - _TITLE_OFF_Y_V45 * scale
    w = BASE_WINDOW_W * scale
    h = BASE_WINDOW_H * scale
    return _clip_window_keep_size_v38(image, int(round(x1)), int(round(y1)), int(round(w)), int(round(h)))


# ──────────────────────────────────────────────────────────────────────────
# v46: '(전투 시간      )' 라벨 이미지 템플릿으로 전투시간 숫자만 정밀 추출
#   - 라벨은 고정 모양이라 add_icon/battle time.jpg(숫자 뺀 라벨)를 템플릿으로 매칭하면
#     위치를 정확히 찾고, 라벨과 ')' 사이의 '숫자 칸'만 잘라 OCR → 라벨 글자 잡음 제거.
#   - 템플릿 분석: 폭 163px 중 라벨 '(전투 시간'은 ~7~75px, 숫자칸은 ~76~149px, ')'는 ~150px.
# ──────────────────────────────────────────────────────────────────────────
BATTLE_TIME_TEMPLATE_FILENAME_V46 = "battle_time_template.png"
BT_NUM_X1_FRAC_V46 = 0.46   # 숫자 칸 시작(템플릿 폭 대비)
BT_NUM_X2_FRAC_V46 = 0.93   # 숫자 칸 끝
BT_MATCH_THRESHOLD_V46 = 0.45


@lru_cache(maxsize=1)
def _load_battle_time_template_v46() -> "np.ndarray | None":
    try:
        import cv2
        cands = [
            _project_root_v32() / "data" / BATTLE_TIME_TEMPLATE_FILENAME_V46,
            Path(__file__).resolve().parent.parent / "data" / BATTLE_TIME_TEMPLATE_FILENAME_V46,
            _project_root_v32() / "add_icon" / "battle time.jpg",
            Path(__file__).resolve().parent.parent / "add_icon" / "battle time.jpg",
        ]
        for p in cands:
            if p.is_file():
                return cv2.cvtColor(np.asarray(Image.open(p).convert("RGB")), cv2.COLOR_RGB2GRAY)
    except Exception as e:
        print(f"[elapsed-template] 템플릿 로드 실패: {e}")
    return None


def _read_elapsed_text_by_template_v46(image: Image.Image, window_box: Any, reader: Any) -> str | None:
    """'(전투 시간 …)' 라벨을 템플릿 매칭으로 찾아 숫자 칸만 잘라 OCR한 텍스트를 반환합니다."""
    try:
        import cv2
        tmpl = _load_battle_time_template_v46()
        if tmpl is None or reader is None:
            return None
        th, tw = tmpl.shape[:2]
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        H, W = gray.shape[:2]
        band = gray[0:max(th + 4, int(H * 0.5)), :]   # 전투시간 라벨은 창 상단부
        base = H / BASE_SCREEN_H
        best = (-1.0, None, None)
        for m in (0.85, 0.92, 1.0, 1.08, 1.16, 1.25, 1.35):
            s = base * m
            tw2, th2 = int(round(tw * s)), int(round(th * s))
            if tw2 < 20 or th2 < 8 or th2 > band.shape[0] or tw2 > band.shape[1]:
                continue
            res = cv2.matchTemplate(band, cv2.resize(tmpl, (tw2, th2)), cv2.TM_CCOEFF_NORMED)
            _minv, maxv, _minl, maxl = cv2.minMaxLoc(res)
            if maxv > best[0]:
                best = (float(maxv), (maxl[0], maxl[1]), s)
        if best[1] is None or best[0] < BT_MATCH_THRESHOLD_V46:
            return None
        (lx, ly), s = best[1], best[2]
        nx1 = max(0, int(round(lx + tw * BT_NUM_X1_FRAC_V46 * s)))
        nx2 = min(W, int(round(lx + tw * BT_NUM_X2_FRAC_V46 * s)))
        ny1 = max(0, int(round(ly - 2 * s)))
        ny2 = min(H, int(round(ly + th * s + 2 * s)))
        if nx2 - nx1 < 8 or ny2 - ny1 < 6:
            return None
        crop = image.convert("RGB").crop((nx1, ny1, nx2, ny2))
        crop = preprocess_crop(crop, scale=6)
        arr = np.array(crop)
        # 분/초 두 숫자 박스를 '왼→오(x좌표)' 순으로 정렬해 합칩니다.
        # (RapidOCR 검출 순서가 뒤바뀌면 '05 : 04'가 '04 05'로 합쳐져 04:05로 잘못 읽히던 문제 방지)
        try:
            res = reader.readtext(arr, detail=1, paragraph=False, allowlist="0123456789:.,： ")
        except TypeError:
            res = reader.readtext(arr, detail=1, paragraph=False)

        def _lx(it: Any) -> float:
            try:
                return min(float(p[0]) for p in it[0])
            except Exception:
                return 0.0

        parts = [str(it[1]).strip() for it in sorted(res or [], key=_lx) if len(it) > 1 and str(it[1]).strip()]
        txt = clean_ocr_text(" ".join(parts))
        print(f"[elapsed-template] match={best[0]:.3f} crop=({nx1},{ny1},{nx2},{ny2}) ocr={txt!r}")
        return txt
    except Exception as e:
        print(f"[elapsed-template] 오류: {e}")
        return None


def detect_analyzer_window(image: Image.Image, reader: Any | None = None) -> Tuple[int, int, int, int] | None:  # type: ignore[override]
    """전투분석기 창 위치 자동 감지 v45.

    1순위: `전투 분석기` 제목 글자 '이미지 템플릿' 매칭 (한글 OCR 불필요, 가장 안정적).
    2순위: 제목 텍스트 OCR(v38).
    3순위: 수평선(Hough) 기반 추정(v38).
    """
    tpl = _find_title_by_template_v45(image)
    if tpl is not None:
        cx, cy, score, scale = tpl
        box = _estimate_window_from_title_template_v45(image, cx, cy, scale)
        x1, y1, x2, y2 = box
        if (x2 - x1) >= image.height * 0.95 and (y2 - y1) >= image.height * 0.55:
            print(f"[detect_window] ✓ 제목 템플릿 매칭 score={score:.3f} center=({cx:.0f},{cy:.0f}) scale={scale:.2f} → box={box}")
            return box

    title = _find_title_by_ocr_v38(image, reader)
    if title is not None:
        cx, cy, _text, _conf = title
        box = _estimate_window_from_title_v38(image, cx, cy)
        x1, y1, x2, y2 = box
        if (x2 - x1) >= image.height * 0.95 and (y2 - y1) >= image.height * 0.55:
            return box
    return _detect_window_by_lines_only_v38(image)


def _detect_attack_direction_kind_v38(image: Image.Image, reader: Any, window_box: tuple[int, int, int, int] | None, scale: int = 4) -> str:
    """공격정보 표의 방향성 열이 백어택인지 헤드어택인지 헤더 OCR로 감지합니다."""
    if window_box is None:
        return "back"
    try:
        row_start = ATTACK_GRID["row_start"]
        row_h = ATTACK_GRID["row_height"]
        # 데이터 첫 행 바로 위 헤더 영역. 방향성 적중률/비중 두 칸을 같이 읽습니다.
        y1 = max(0.0, row_start - row_h * 1.12)
        y2 = max(y1 + row_h * 0.75, row_start - row_h * 0.12)
        x1 = ATTACK_GRID["columns"][2][2]
        x2 = ATTACK_GRID["columns"][3][3]
        crop = crop_norm(image, (x1, y1, x2, y2), pad=0, window_box=window_box)
        crop = preprocess_crop(crop, scale=max(3, min(scale, 5)))
        text = _ocr_text_once(reader, crop, numeric=False)
        norm = _normalize_for_window_detection(text)
        if "헤드" in norm:
            return "head"
        if "백" in norm or "어택" in norm:
            return "back"
    except Exception:
        pass
    return "back"


def _ocr_attack_cell_by_box(  # type: ignore[override]
    reader: Any,
    image: Image.Image,
    box: Tuple[int, int, int, int],
    *,
    key: str,
    scale: int,
) -> str:
    raw = _crop_pixel_box(image, box, pad=1, expand_ratio=CELL_CROP_EXPAND_RATIO)
    percent_keys = {"directional_rate", "directional_share", "back_attack_rate", "back_attack_share", "head_attack_rate", "head_attack_share", "crit_rate", "crit_share", "cooldown_rate", "share_rate"}
    korean_number_keys = {"damage_text", "dps_text"}
    kind = "percent" if key in percent_keys else "korean_number" if key in korean_number_keys else "text"
    variants = _ocr_text_variants(reader, raw, numeric=(key != "name"), scale=scale, kind=kind)
    if not variants:
        return ""
    variants.sort(key=lambda x: x[3], reverse=True)
    text = clean_ocr_text(variants[0][2])
    if key in korean_number_keys:
        return extract_korean_number_text(text)
    if key in percent_keys:
        return format_percent_from_ocr(text)
    return text


def parse_attack_fixed_grid(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int | None = None,
    scale: int = 7,
    auto_window: bool = True,
) -> pd.DataFrame:
    """공격정보 OCR v38. 제목 기반 창 탐지 + 백/헤드 방향성 자동 분기."""
    grid = ATTACK_GRID
    if row_count is None:
        row_count = int(grid["row_count"])
    window_box = detect_analyzer_window(image, reader) if auto_window else None
    direction_kind = _detect_attack_direction_kind_v38(image, reader, window_box, scale=scale) if window_box else "back"
    detected_rows = detect_attack_icon_rows(image, reader, window_box=window_box, max_rows=int(row_count)) if window_box else []

    if not detected_rows:
        detected_rows = []
        for i in range(int(row_count)):
            y1 = grid["row_start"] + i * grid["row_height"]
            y2 = y1 + grid["row_height"]
            row_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, grid["columns"][-1][3], y2), pad=0, window_box=window_box)
            icon_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, ATTACK_ICON_GRID["x2"], y2), pad=1, window_box=window_box)
            detected_rows.append({"row_index": i, "row_box": row_box, "icon_box": icon_box, "window_box": window_box, "source": "fixed_fallback"})

    rows: List[Dict[str, Any]] = []
    for i, rinfo in enumerate(detected_rows[: int(row_count)]):
        row: Dict[str, Any] = {col: "" for col in STANDARD_COLUMNS}
        row["_ocr_row_index"] = str(i)
        row["_row_source"] = rinfo.get("source", "")
        row["_direction_kind"] = direction_kind
        if window_box:
            row["_window_x1"], row["_window_y1"], row["_window_x2"], row["_window_y2"] = window_box
        ix1, iy1, ix2, iy2 = rinfo["icon_box"]
        row["_icon_x1"], row["_icon_y1"], row["_icon_x2"], row["_icon_y2"] = ix1, iy1, ix2, iy2
        row["_row_x1"], row["_row_y1"], row["_row_x2"], row["_row_y2"] = rinfo["row_box"]

        directional_rate = ""
        directional_share = ""
        for key, label, x1, x2 in grid["columns"]:
            if window_box:
                box = _column_pixel_box_for_row(image, window_box, rinfo["row_box"], x1, x2)
            else:
                px1 = int(round(x1 * image.width)); px2 = int(round(x2 * image.width))
                box = _clip_box(image, (px1, rinfo["row_box"][1], px2, rinfo["row_box"][3]))
            val = _ocr_attack_cell_by_box(reader, image, box, key=key, scale=scale)
            if key == "directional_rate":
                directional_rate = val
            elif key == "directional_share":
                directional_share = val
            else:
                row[label] = val

        if direction_kind == "head":
            row["헤드어택 적중률"] = directional_rate
            row["헤드어택 비중"] = directional_share
        else:
            row["백어택 적중률"] = directional_rate
            row["백어택 비중"] = directional_share

        meaningful = False
        for col in [c for c in STANDARD_COLUMNS if c not in {"이름", "초당 피해량", "피해량 지분"}]:
            val = str(row.get(col) or "").strip()
            if val and val not in ["-", "_", "None", "nan"]:
                meaningful = True
                break
        if meaningful or str(row.get("이름") or "").strip():
            rows.append(row)

    cols = STANDARD_COLUMNS + [
        "_ocr_row_index", "_row_source", "_direction_kind",
        "_window_x1", "_window_y1", "_window_x2", "_window_y2",
        "_icon_x1", "_icon_y1", "_icon_x2", "_icon_y2",
        "_row_x1", "_row_y1", "_row_x2", "_row_y2",
    ]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        check_cols = [c for c in STANDARD_COLUMNS if c not in {"초당 피해량", "피해량 지분"}]
        nonempty = df[check_cols].astype(str).apply(lambda r: any(x.strip() and x.strip() not in ["", "-", "_", "None", "nan"] for x in r), axis=1)
        df = df[nonempty].reset_index(drop=True)
    return df


def make_attack_ocr_debug(image: Image.Image, reader: Any, row_count: int = 18, scale: int = 7) -> List[Dict[str, Any]]:  # type: ignore[override]
    grid = ATTACK_GRID
    window_box = detect_analyzer_window(image, reader)
    direction_kind = _detect_attack_direction_kind_v38(image, reader, window_box, scale=scale) if window_box else "back"
    detected_rows = detect_attack_icon_rows(image, reader, window_box=window_box, max_rows=int(row_count)) if window_box else []
    if not detected_rows:
        detected_rows = []
        for i in range(int(row_count)):
            y1 = grid["row_start"] + i * grid["row_height"]
            y2 = y1 + grid["row_height"]
            row_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, grid["columns"][-1][3], y2), pad=0, window_box=window_box)
            icon_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, ATTACK_ICON_GRID["x2"], y2), pad=1, window_box=window_box)
            detected_rows.append({"row_index": i, "row_box": row_box, "icon_box": icon_box, "window_box": window_box, "source": "fixed_fallback"})

    rows: List[Dict[str, Any]] = []
    for i, rinfo in enumerate(detected_rows[: int(row_count)]):
        row_crop = _crop_pixel_box(image, rinfo["row_box"], pad=0, expand_ratio=0.01)
        icon_crop = _crop_pixel_box(image, rinfo["icon_box"], pad=0, expand_ratio=ICON_CROP_EXPAND_RATIO)
        cells: List[Dict[str, Any]] = []
        for key, label, x1, x2 in grid["columns"]:
            if window_box:
                box = _column_pixel_box_for_row(image, window_box, rinfo["row_box"], x1, x2)
            else:
                px1 = int(round(x1 * image.width)); px2 = int(round(x2 * image.width))
                box = _clip_box(image, (px1, rinfo["row_box"][1], px2, rinfo["row_box"][3]))
            raw_crop = _crop_pixel_box(image, box, pad=1, expand_ratio=CELL_CROP_EXPAND_RATIO)
            percent_keys = {"directional_rate", "directional_share", "back_attack_rate", "back_attack_share", "crit_rate", "crit_share", "cooldown_rate", "head_attack_rate", "head_attack_share"}
            korean_number_keys = {"damage_text", "dps_text"}
            kind = "percent" if key in percent_keys else "korean_number" if key in korean_number_keys else "text"
            variants = _ocr_text_variants(reader, raw_crop, numeric=(key != "name"), scale=scale, kind=kind)
            variants.sort(key=lambda x: x[3], reverse=True)
            best_name, best_crop, best_text, best_score = variants[0]
            if key in korean_number_keys:
                best_text = extract_korean_number_text(best_text)
            elif key in percent_keys:
                best_text = format_percent_from_ocr(best_text)
            # 디버그에서는 실제 출력 컬럼명을 함께 보여줍니다.
            out_label = label
            if key == "directional_rate":
                out_label = "헤드어택 적중률" if direction_kind == "head" else "백어택 적중률"
            elif key == "directional_share":
                out_label = "헤드어택 비중" if direction_kind == "head" else "백어택 비중"
            cells.append({
                "key": key,
                "label": out_label,
                "text": best_text,
                "raw_crop": raw_crop,
                "processed_crop": best_crop,
                "preprocess": best_name,
                "score": best_score,
            })
        rows.append({
            "row_index": i,
            "row_crop": row_crop,
            "icon_crop": icon_crop,
            "cells": cells,
            "window_box": window_box,
            "direction_kind": direction_kind,
            "row_source": rinfo.get("source", ""),
            "row_box": rinfo.get("row_box"),
            "icon_box": rinfo.get("icon_box"),
        })
    return rows


# ==============================
# v39: icon crop + source-aware name matching override
# ==============================
# v38에서 표 행/숫자 crop은 안정화됐지만, 이름 보정 단계에서 행 아이콘 crop이 이름 배경까지 포함되어
# API 아이콘 비교가 흔들릴 수 있었습니다. v39는 아이콘 crop을 항상 왼쪽 정사각형 중심으로 정규화하고,
# 스킬/룬/보주 후보의 source 우선순위를 반영해서 잘못된 룬/보주 후보로 스킬명이 덮이는 일을 줄입니다.


def _left_square_icon_part_v39(img: Image.Image) -> Image.Image:
    """전투분석기 행 아이콘 crop에서 실제 아이콘만 최대한 분리합니다.

    행 crop이 `아이콘 + 이름 파란 막대`까지 들어오면 기존 중앙 정사각형 crop은 글자/배경을 잡아
    아이콘 매칭이 망가질 수 있습니다. 전투분석기 아이콘은 항상 왼쪽에 있으므로, 폭이 높이보다
    넓은 crop은 왼쪽 정사각형을 사용합니다.
    """
    img = img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    if w > h * 1.12:
        side = min(h, w)
        # 아이콘 테두리가 좌측에 딱 붙어 있을 수 있어 왼쪽에서 조금만 여유를 둡니다.
        left = 0
        top = max(0, (h - side) // 2)
        return img.crop((left, top, min(w, left + side), min(h, top + side)))
    return img


def _square_resize_for_icon(img: Image.Image, size: int = 48, margin_ratio: float = 0.08) -> Image.Image:  # type: ignore[override]
    """전투분석기/API 아이콘 비교용 정규화 v39.

    가로로 긴 행 아이콘 crop은 왼쪽 정사각형만 사용합니다. API 아이콘은 이미 정사각형이라 영향이 거의 없습니다.
    """
    img = _left_square_icon_part_v39(img)
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = max(0, (w - side) // 2)
    top = max(0, (h - side) // 2)
    img = img.crop((left, top, left + side, top + side))
    if img.width > 8 and img.height > 8 and margin_ratio > 0:
        margin = max(1, int(min(img.width, img.height) * margin_ratio))
        if img.width - margin * 2 > 4 and img.height - margin * 2 > 4:
            img = img.crop((margin, margin, img.width - margin, img.height - margin))
    img = ImageEnhance.Contrast(img).enhance(1.30)
    img = ImageEnhance.Sharpness(img).enhance(1.20)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def _crop_icon_from_row_metadata(image: Image.Image, row: pd.Series | Dict[str, Any], fallback_index: int) -> Image.Image | None:  # type: ignore[override]
    """행 메타데이터의 아이콘 좌표를 사용하되, 실제 비교에는 왼쪽 정사각형 아이콘만 남깁니다."""
    try:
        vals = [row.get("_icon_x1"), row.get("_icon_y1"), row.get("_icon_x2"), row.get("_icon_y2")]
        if all(v is not None and str(v) not in ["", "nan", "None"] for v in vals):
            box = tuple(int(float(v)) for v in vals)
            crop = _crop_pixel_box(image, box, pad=0, expand_ratio=ICON_CROP_EXPAND_RATIO)
            return _left_square_icon_part_v39(crop)
    except Exception:
        pass
    crop = crop_attack_icon(image, fallback_index, window_box=_row_window_box(row))
    return _left_square_icon_part_v39(crop) if crop is not None else None


def _source_group_v39(source: Any) -> str:
    s = str(source or "").lower()
    if s.startswith("skill"):
        return "skill"
    if "rune" in s:
        return "rune"
    if "orb" in s or "보주" in s:
        return "orb"
    if "manual" in s or "local" in s:
        return "manual"
    return "other"


def best_skill_name_match_with_icon(  # type: ignore[override]
    ocr_name: Any,
    row_icon: Image.Image | None,
    skill_candidates: List[Any],
    *,
    text_threshold: float = 0.55,
    icon_threshold: float = 0.72,
) -> Tuple[str, float, float, bool, str]:
    """OCR 이름을 API 후보로 보정합니다. v39 source-aware 버전.

    - 스킬 행은 가능한 한 `source=skill` 후보를 우선합니다.
    - 룬/보주 후보가 스킬 아이콘보다 아주 확실히 높을 때만 스킬 후보를 덮습니다.
    - OCR 텍스트가 이미 스킬명과 강하게 맞으면 텍스트를 우선합니다.
    """
    pairs = _candidate_name_icon_pairs(skill_candidates)
    original = str(ocr_name or "").strip()
    canonical = canonicalize_battle_name_v31(original)
    if not pairs:
        return canonical or original, 0.0, 0.0, bool(canonical and canonical != original), "canonical" if canonical != original else "no_candidates"

    norm_original = normalize_skill_name_for_match(canonical or original)
    scored: List[Dict[str, Any]] = []
    for pair in pairs:
        name = pair.get("name")
        source = pair.get("source", "")
        norm_skill = normalize_skill_name_for_match(name)
        text_score = 0.0
        if norm_original and norm_skill:
            text_score = SequenceMatcher(None, norm_original, norm_skill).ratio()
            if norm_original in norm_skill or norm_skill in norm_original:
                text_score = max(text_score, 0.88)

        icon_score = 0.0
        if row_icon is not None and pair.get("icon"):
            api_icon = _download_icon_image(pair["icon"])
            if api_icon is not None:
                icon_score = _icon_similarity(row_icon, api_icon)
        combined = 0.22 * text_score + 0.78 * icon_score
        scored.append({
            "name": name,
            "text": text_score,
            "icon": icon_score,
            "combined": combined,
            "source": source,
            "group": _source_group_v39(source),
        })

    if not scored:
        return canonical or original, 0.0, 0.0, False, "unmatched"

    best_text = max(scored, key=lambda x: x["text"])
    strong_text = best_text["text"] >= max(float(text_threshold), 0.82)
    if strong_text:
        return best_text["name"], round(best_text["text"] * 100.0, 2), round(best_text["icon"] * 100.0, 2), True, f"text_first:{best_text.get('source','')}"

    sorted_icon = sorted(scored, key=lambda x: (x["icon"], x["combined"], x["text"]), reverse=True)
    best_icon = sorted_icon[0]
    second_icon = sorted_icon[1]["icon"] if len(sorted_icon) > 1 else 0.0
    icon_margin = best_icon["icon"] - second_icon

    skill_scores = [x for x in scored if x.get("group") == "skill"]
    best_skill = max(skill_scores, key=lambda x: (x["icon"], x["combined"], x["text"]), default=None)

    # 채용 스킬 아이콘이 어느 정도 맞으면, 룬/보주/일반 API 후보가 아주 확실하지 않은 이상 스킬 후보를 우선합니다.
    if best_skill is not None and best_skill["icon"] >= max(0.58, float(icon_threshold) - 0.12):
        if best_icon.get("group") != "skill":
            non_skill_really_clear = best_icon["icon"] >= max(0.92, float(icon_threshold) + 0.15) and (best_icon["icon"] - best_skill["icon"] >= 0.12)
            if not non_skill_really_clear:
                return best_skill["name"], round(best_skill["text"] * 100.0, 2), round(best_skill["icon"] * 100.0, 2), True, f"icon_skill_guard:{best_skill.get('source','')}"
        else:
            skill_margin = best_skill["icon"] - max([x["icon"] for x in skill_scores if x is not best_skill] or [0.0])
            if best_skill["icon"] >= float(icon_threshold) or skill_margin >= 0.035:
                return best_skill["name"], round(best_skill["text"] * 100.0, 2), round(best_skill["icon"] * 100.0, 2), True, f"icon_skill:{best_skill.get('source','')}"

    # 그래도 전체 best_icon이 충분히 확실하면 사용합니다.
    icon_confident = best_icon["icon"] >= float(icon_threshold) and (icon_margin >= 0.045 or best_icon["icon"] >= min(0.95, float(icon_threshold) + 0.12))
    if icon_confident:
        return best_icon["name"], round(best_icon["text"] * 100.0, 2), round(best_icon["icon"] * 100.0, 2), True, f"icon:{best_icon.get('source','')}"

    # 텍스트 보정 fallback. 단, 아주 짧은 OCR은 오보정 위험이 커서 기준을 조금 올립니다.
    short_ocr = len(re.sub(r"[^0-9A-Za-z가-힣]", "", original)) <= 2
    text_floor = max(float(text_threshold), 0.68) if short_ocr else float(text_threshold)
    if best_text["text"] >= text_floor:
        return best_text["name"], round(best_text["text"] * 100.0, 2), round(best_text["icon"] * 100.0, 2), True, f"text:{best_text.get('source','')}"

    if canonical and canonical != original:
        return canonical, round(best_icon.get("text", 0.0) * 100.0, 2), round(best_icon.get("icon", 0.0) * 100.0, 2), True, "canonical"
    return original, round(best_icon.get("text", 0.0) * 100.0, 2), round(best_icon.get("icon", 0.0) * 100.0, 2), False, "unmatched"


def correct_battle_skill_names_with_icons(  # type: ignore[override]
    df: pd.DataFrame,
    attack_image: Image.Image,
    skill_candidates: List[Dict[str, str]],
    *,
    name_col: str = "이름",
    threshold: float = 0.62,
    icon_threshold: float = 0.72,
    drop_unmatched: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty or name_col not in df.columns:
        return df
    out = df.copy()
    rows = []
    for idx, row in out.iterrows():
        try:
            row_index = int(float(row.get("_ocr_row_index", idx)))
        except Exception:
            row_index = int(idx)
        icon_crop = _crop_icon_from_row_metadata(attack_image, row, row_index)
        matched, text_score, icon_score, ok, reason = best_skill_name_match_with_icon(
            row.get(name_col),
            icon_crop,
            skill_candidates or [],
            text_threshold=threshold,
            icon_threshold=icon_threshold,
        )
        new_row = row.copy()
        if ok:
            new_row[name_col] = matched
        elif drop_unmatched:
            continue
        # ok가 아니면 OCR 원문을 그대로 둡니다. 이전처럼 항상 matched를 덮지 않습니다.
        new_row["_name_match_text_score"] = text_score
        new_row["_name_match_icon_score"] = icon_score
        new_row["_name_match_reason"] = reason
        rows.append(new_row)
    if not rows:
        return out.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


# ==============================
# v40: icon-first matching + debug top candidates
# ==============================
def _rank_icon_candidates_v40(
    row_icon: Image.Image | None,
    skill_candidates: List[Any],
    *,
    ocr_name: Any = "",
    topn: int = 8,
) -> List[Dict[str, Any]]:
    """행 아이콘과 API/룬/보주 후보 아이콘을 비교해 상위 후보를 반환합니다."""
    pairs = _candidate_name_icon_pairs(skill_candidates or [])
    original = str(ocr_name or "").strip()
    norm_original = normalize_skill_name_for_match(original)
    ranked: List[Dict[str, Any]] = []
    for pair in pairs:
        name = pair.get("name") or ""
        source = pair.get("source") or ""
        icon = pair.get("icon") or ""
        group = _source_group_v39(source)
        norm_name = normalize_skill_name_for_match(name)
        text_score = 0.0
        if norm_original and norm_name:
            text_score = SequenceMatcher(None, norm_original, norm_name).ratio()
            if norm_original in norm_name or norm_name in norm_original:
                text_score = max(text_score, 0.88)
        icon_score = 0.0
        if row_icon is not None and icon:
            api_icon = _download_icon_image(icon)
            if api_icon is not None:
                icon_score = _icon_similarity(row_icon, api_icon)
        # 스킬 후보는 이름보다 아이콘을 훨씬 더 신뢰합니다.
        combined = 0.10 * text_score + 0.90 * icon_score if group == "skill" else 0.22 * text_score + 0.78 * icon_score
        ranked.append({
            "name": str(name),
            "source": str(source),
            "group": group,
            "icon": str(icon),
            "text_score": round(float(text_score) * 100.0, 2),
            "icon_score": round(float(icon_score) * 100.0, 2),
            "combined": round(float(combined) * 100.0, 2),
        })
    ranked.sort(key=lambda x: (x["icon_score"], x["combined"], x["text_score"]), reverse=True)
    return ranked[: int(topn)]


def best_skill_name_match_with_icon(  # type: ignore[override]
    ocr_name: Any,
    row_icon: Image.Image | None,
    skill_candidates: List[Any],
    *,
    text_threshold: float = 0.55,
    icon_threshold: float = 0.72,
) -> Tuple[str, float, float, bool, str]:
    """v42: 스킬명은 아이콘으로만 결정합니다 (OCR 텍스트 완전 무시).

    RapidOCR 중국어 모델이 한글 스킬명을 인식하지 못하므로,
    스킬 후보(skill group)는 아이콘 유사도만으로 매칭합니다.
    - icon_score >= 0.40 인 스킬 후보 중 최고점을 바로 선택합니다.
    - text_score/margin/임계값 등 복잡한 조건 없이 단순 최고 아이콘 선택.
    룬/보주/기타 후보만 텍스트 fallback을 유지합니다.
    """
    original = str(ocr_name or "").strip()
    canonical = canonicalize_battle_name_v31(original)
    ranked = _rank_icon_candidates_v40(row_icon, skill_candidates or [], ocr_name=canonical or original, topn=12)
    if not ranked:
        return canonical or original, 0.0, 0.0, bool(canonical and canonical != original), "canonical" if canonical != original else "no_candidates"

    # ── 스킬 후보: 아이콘 점수만으로 선택 (OCR 텍스트 완전 무시) ──────────────
    skill_ranked = [r for r in ranked if r.get("group") == "skill"]
    if skill_ranked:
        # row_icon이 있으면 아이콘 유사도 기준으로 정렬된 best 선택
        # row_icon이 None이면 text_score 기준 best 선택 (한글 OCR 안 되므로 보통 0점이지만 후보 중 최선)
        if row_icon is not None:
            best_skill = skill_ranked[0]  # 이미 icon_score 내림차순 정렬
            skill_icon_score = float(best_skill["icon_score"]) / 100.0
            reason_suffix = f"icon_skill_only:{best_skill.get('source', '')}"
            # v46: 아이콘 매칭 신뢰도가 낮으면(점수 낮거나 1·2위가 거의 동점) 엉뚱한 스킬명을
            # 강제로 넣지 않고 '기타'(기타 딜)로 처리합니다. 직업에 상관없이 일반 적용됩니다.
            _best100 = float(best_skill["icon_score"])
            _second100 = float(skill_ranked[1]["icon_score"]) if len(skill_ranked) > 1 else 0.0
            _low_conf = (_best100 < ICON_MATCH_MIN_SCORE_V46) or (
                _best100 < ICON_MATCH_TIE_SCORE_V46 and (_best100 - _second100) < ICON_MATCH_TIE_MARGIN_V46
            )
            if _low_conf:
                return ("기타", 0.0, round(_best100, 2), True, f"icon_low_conf_etc:{best_skill.get('source', '')}")
        else:
            # 아이콘 crop 실패 시에도 스킬 후보 중 하나를 선택해 개별 행으로 유지
            best_skill = max(skill_ranked, key=lambda x: x["text_score"])
            skill_icon_score = 0.0
            reason_suffix = f"icon_skill_noicon:{best_skill.get('source', '')}"
        return (
            str(best_skill["name"]),
            0.0,
            round(skill_icon_score * 100.0, 2),
            True,
            reason_suffix,
        )

    # ── 비스킬(룬/보주/기타): 아이콘+텍스트 복합 fallback ───────────────────
    best = ranked[0]
    best_icon = float(best["icon_score"]) / 100.0
    best_text = float(best["text_score"]) / 100.0
    second_icon = float(ranked[1]["icon_score"]) / 100.0 if len(ranked) > 1 else 0.0
    margin = best_icon - second_icon

    if best_icon >= float(icon_threshold) and (margin >= 0.035 or best_icon >= min(0.94, float(icon_threshold) + 0.12)):
        return str(best["name"]), round(best_text * 100.0, 2), round(best_icon * 100.0, 2), True, f"icon_first:{best.get('source', '')}"

    text_best = max(ranked, key=lambda x: x["text_score"])
    text_score = float(text_best["text_score"]) / 100.0
    text_icon = float(text_best["icon_score"]) / 100.0
    if text_score >= float(text_threshold):
        return str(text_best["name"]), round(text_score * 100.0, 2), round(text_icon * 100.0, 2), True, f"text_fallback:{text_best.get('source', '')}"

    if canonical and canonical != original:
        return canonical, round(text_score * 100.0, 2), round(best_icon * 100.0, 2), True, "canonical"
    return original, round(text_score * 100.0, 2), round(best_icon * 100.0, 2), False, "unmatched"


def correct_battle_skill_names_with_icons(  # type: ignore[override]
    df: pd.DataFrame,
    attack_image: Image.Image,
    skill_candidates: List[Dict[str, str]],
    *,
    name_col: str = "이름",
    threshold: float = 0.62,
    icon_threshold: float = 0.72,
    drop_unmatched: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty or name_col not in df.columns:
        return df
    out = df.copy()
    rows = []
    for idx, row in out.iterrows():
        try:
            row_index = int(float(row.get("_ocr_row_index", idx)))
        except Exception:
            row_index = int(idx)
        icon_crop = _crop_icon_from_row_metadata(attack_image, row, row_index)
        top = _rank_icon_candidates_v40(icon_crop, skill_candidates or [], ocr_name=row.get(name_col), topn=5)
        matched, text_score, icon_score, ok, reason = best_skill_name_match_with_icon(
            row.get(name_col),
            icon_crop,
            skill_candidates or [],
            text_threshold=threshold,
            icon_threshold=icon_threshold,
        )
        new_row = row.copy()
        if ok:
            new_row[name_col] = matched
        elif drop_unmatched:
            continue
        new_row["_name_match_text_score"] = text_score
        new_row["_name_match_icon_score"] = icon_score
        new_row["_name_match_reason"] = reason
        new_row["_icon_match_name"] = _canonical_display_name_v86(str(top[0]["name"])) if top else ""
        new_row["_icon_match_score"] = top[0]["icon_score"] if top else 0.0
        new_row["_icon_match_source"] = str(top[0]["source"]) if top else ""
        new_row["_icon_match_top3"] = " | ".join(f"{_canonical_display_name_v86(r['name'])}:{r['icon_score']}" for r in top[:3])
        rows.append(new_row)
    if not rows:
        return out.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


_make_attack_ocr_debug_base_v40 = make_attack_ocr_debug


def make_attack_ocr_debug(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int = 18,
    scale: int = 7,
    skill_candidates: List[Dict[str, str]] | None = None,
) -> List[Dict[str, Any]]:
    """v40: 기존 OCR crop 디버그에 아이콘 매칭 상위 후보를 함께 붙입니다."""
    rows = _make_attack_ocr_debug_base_v40(image, reader, row_count=row_count, scale=scale)
    if not skill_candidates:
        return rows
    for row in rows:
        icon_crop = row.get("icon_crop")
        ocr_name = ""
        for cell in row.get("cells", []) or []:
            if cell.get("key") == "name":
                ocr_name = str(cell.get("text") or "")
                break
        top = _rank_icon_candidates_v40(icon_crop, skill_candidates or [], ocr_name=ocr_name, topn=5)
        if top:
            row["icon_match"] = {
                "name": top[0]["name"],
                "icon_score": top[0]["icon_score"],
                "text_score": top[0]["text_score"],
                "source": top[0]["source"],
                "reason": "icon_rank_v40",
                "top": top,
            }
        else:
            row["icon_match"] = {"name": "", "icon_score": 0.0, "text_score": 0.0, "source": "", "reason": "no_candidates", "top": []}
        # v44 디버그 강화: 실제 비교에 들어가는 '정규화된 아이콘 코어'와
        # 상위 후보 API 아이콘 이미지를 함께 저장해 매칭 근거를 눈으로 확인할 수 있게 합니다.
        try:
            if icon_crop is not None:
                row["icon_core_crop"] = _extract_battle_icon_core_v42(icon_crop)
        except Exception:
            pass
        cand_imgs = []
        for c in (top or [])[:4]:
            try:
                cimg = _download_icon_image(c.get("icon") or "")
            except Exception:
                cimg = None
            if cimg is not None:
                cand_imgs.append({"name": c.get("name", ""), "icon_score": c.get("icon_score", 0.0), "img": cimg})
        row["icon_candidate_images"] = cand_imgs
    return rows


# ==============================
# v42: 전분 아이콘 정밀 crop + 검수 아이콘 라이브러리
# ==============================
def _extract_battle_icon_core_v42(img: Image.Image) -> Image.Image:
    """전투분석기 행 아이콘 crop에서 실제 아이콘 사각형만 남깁니다.

    이전 버전은 `아이콘+이름 파란 막대` crop에서 왼쪽 정사각형을 잘랐는데,
    행 crop 높이 전체를 정사각형 폭으로 쓰면서 이름 글자 일부가 섞이는 문제가 있었습니다.
    v42는 왼쪽 70~82% 높이 폭만 우선 사용하고, 밝기/채도 bbox로 실제 아이콘 중심부를 다시 잡습니다.
    """
    img = img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return img

    # 1) 가로로 긴 crop이면 이름 영역을 강하게 제거합니다.
    #    예: 80x59 crop이면 실제 아이콘은 대략 40~46px 폭입니다.
    if w > h * 1.05:
        focus_w = min(w, max(16, int(round(h * 0.78))))
        focus = img.crop((0, 0, focus_w, h))
    else:
        focus = img

    # 2) 너무 어두운 배경을 제외한 실제 아이콘 bbox를 찾습니다.
    try:
        arr = np.asarray(focus.convert("RGB"), dtype=np.uint8)
        gray = arr.mean(axis=2)
        maxc = arr.max(axis=2).astype(np.int16)
        minc = arr.min(axis=2).astype(np.int16)
        sat = maxc - minc
        # 아이콘 내부는 배경보다 밝거나 채도가 높습니다. 테두리까지 어느 정도 포함합니다.
        mask = (gray > 28) & (sat > 12)
        ys, xs = np.where(mask)
        if len(xs) > 20 and len(ys) > 20:
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            pad = max(1, int(min(focus.width, focus.height) * 0.06))
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(focus.width, x2 + pad); y2 = min(focus.height, y2 + pad)
            bw, bh = x2 - x1, y2 - y1
            # bbox가 너무 작거나 한쪽으로 길면 기존 focus를 사용합니다.
            if bw >= focus.width * 0.45 and bh >= focus.height * 0.45:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                side = int(max(bw, bh))
                side = max(12, min(side, focus.width, focus.height))
                nx1 = int(round(cx - side / 2)); ny1 = int(round(cy - side / 2))
                nx1 = max(0, min(focus.width - side, nx1))
                ny1 = max(0, min(focus.height - side, ny1))
                return focus.crop((nx1, ny1, nx1 + side, ny1 + side))
    except Exception:
        pass

    # 3) fallback: 왼쪽 focus에서 중앙 정사각형
    side = min(focus.width, focus.height)
    return focus.crop((0, max(0, (focus.height - side)//2), side, max(0, (focus.height - side)//2) + side))


def _left_square_icon_part_v39(img: Image.Image) -> Image.Image:  # type: ignore[override]
    return _extract_battle_icon_core_v42(img)


def _square_resize_for_icon(img: Image.Image, size: int = 48, margin_ratio: float = 0.04) -> Image.Image:  # type: ignore[override]
    img = _extract_battle_icon_core_v42(img)
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = max(0, (w - side) // 2)
    top = max(0, (h - side) // 2)
    img = img.crop((left, top, left + side, top + side))
    if img.width > 12 and img.height > 12 and margin_ratio > 0:
        margin = max(1, int(min(img.width, img.height) * margin_ratio))
        if img.width - margin * 2 > 8 and img.height - margin * 2 > 8:
            img = img.crop((margin, margin, img.width - margin, img.height - margin))
    img = ImageEnhance.Contrast(img).enhance(1.22)
    img = ImageEnhance.Sharpness(img).enhance(1.15)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def _local_icon_candidates_v32() -> List[Dict[str, str]]:  # type: ignore[override]
    """검수된 전분 아이콘 후보만 읽습니다.

    - cache/battle_icons/manual: 이전 자동 학습 오염 가능성이 있어 기본 사용하지 않음
    - cache/battle_icons/verified: 사용자가 버튼으로 검수 저장한 아이콘만 사용
    - assets/battle_icons: 사용자가 직접 넣은 공용 룬/기타/보주 아이콘
    """
    root = _project_root_v32()
    rows: List[Dict[str, str]] = []
    for folder, source in [
        (root / "assets" / "battle_icons", "asset_battle_icon"),
        (root / "cache" / "battle_icons" / "verified", "verified_battle_icon"),
    ]:
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.png")):
            name = p.stem.replace("_", " ").strip()
            if name and name not in {"이름", "피해량", "DPS"}:
                rows.append({"name": name, "icon": str(p), "source": source})
    return rows


def save_learned_battle_icons_v32(df: pd.DataFrame, attack_image: Image.Image | None) -> int:  # type: ignore[override]
    """검수된 이름과 행 아이콘을 verified cache에 저장합니다.

    자동 저장이 아니라 사용자가 검수 버튼을 눌렀을 때만 app.py에서 호출합니다.
    """
    if df is None or df.empty or attack_image is None or "이름" not in df.columns:
        return 0
    root = _project_root_v32()
    out_dir = root / "cache" / "battle_icons" / "verified"
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, row in df.iterrows():
        name = str(row.get("이름") or "").strip()
        if not name or name in {"이름", "피해량", "DPS", "지분%", "백적중%", "치적%"}:
            continue
        # OCR 찌꺼기처럼 특수문자가 과한 이름은 저장하지 않습니다.
        clean_name = re.sub(r"\s+", " ", name).strip()
        if len(clean_name) < 2 or len(clean_name) > 24:
            continue
        icon = _crop_icon_from_row_metadata(attack_image, row, int(float(row.get("_ocr_row_index", idx) or idx)))
        if icon is None:
            continue
        try:
            icon_norm = _square_resize_for_icon(icon, size=72, margin_ratio=0.02)
            filename = _safe_icon_filename_v32(clean_name) + ".png"
            icon_norm.save(out_dir / filename)
            count += 1
        except Exception:
            pass
    # 새로 저장한 아이콘을 바로 쓸 수 있도록 캐시 비우기
    try:
        _download_icon_image.cache_clear()
    except Exception:
        pass
    return count


# ==============================
# v43: 전분 아이콘 crop 축소 + 템플릿 라이브러리 비활성화
# ==============================
# 검수/로컬 템플릿 라이브러리는 범용 사용 시 오염 가능성이 있어 기본 후보에서 완전히 제외합니다.
# 전분 행 아이콘은 이전보다 더 좁게, 해상도 비율 기반으로 crop합니다.

ICON_CROP_EXPAND_RATIO = 0.0
ICON_EDGE_TRIM_RATIO_V43 = 0.055
ICON_WIDTH_FROM_ROW_RATIO_V43 = 0.92  # v44: 가운데 정렬 정사각형 폭(행 높이 대비). 파란 띠 양쪽 배제
ICON_LEFT_PAD_RATIO_V43 = 0.04        # (v44부터 미사용: 왼쪽 정렬 → 가운데 정렬로 변경)

# v46: 아이콘 매칭 신뢰도 임계값. 이 점수 미만이거나 1·2위가 거의 동점이면 '기타'로 처리.
#  - 관측: 정타 스킬은 보통 72~93점, 오인식(엉뚱한 스킬)은 ~58~60점에 1·2위 동점에 가까움.
#  - 직업 무관 일반 규칙. 너무 빡세면(이름 누락) 값을 낮추고, 오인식이 남으면 올리세요.
ICON_MATCH_MIN_SCORE_V46 = 66.0   # 절대 점수 하한
ICON_MATCH_TIE_SCORE_V46 = 76.0   # 이 점수 미만이면서
ICON_MATCH_TIE_MARGIN_V46 = 2.5   # 1·2위 차이가 이보다 작으면 애매 → 기타


def _trim_icon_edges_v43(img: Image.Image, trim_ratio: float = ICON_EDGE_TRIM_RATIO_V43) -> Image.Image:
    img = img.convert("RGB")
    if img.width <= 12 or img.height <= 12:
        return img
    trim = max(1, int(round(min(img.width, img.height) * trim_ratio)))
    if img.width - trim * 2 > 10 and img.height - trim * 2 > 10:
        return img.crop((trim, trim, img.width - trim, img.height - trim))
    return img


def _extract_battle_icon_core_v42(img: Image.Image) -> Image.Image:  # type: ignore[override]
    """v43: 행 아이콘 crop에서 실제 아이콘 부분만 비율 기반으로 좁게 잘라냅니다.

    이전 v42는 밝기/채도 bbox로 실제 아이콘을 다시 찾으려 했는데, 선택행의 파란 배경이나
    이름칸 일부가 mask에 섞이면 엉뚱한 영역으로 중심이 밀릴 수 있었습니다.
    v43은 전투분석기 UI 비율이 고정이라는 점을 이용해서, 행 높이 기준의 왼쪽 사각형만 사용합니다.
    """
    img = img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return img

    # v44: 행 아이콘 crop(icon_box)은 실제 아이콘이 '가운데'에 오고 좌·우로 행 선택
    # 파란 하이라이트 띠가 끼는 구조입니다. 이전 v43은 왼쪽에서 사각형을 잘라(left_pad)
    # 왼쪽 파란 띠를 포함하고 아이콘 오른쪽을 잘라먹어 매칭이 어긋났습니다.
    # → crop 중앙을 기준으로 정사각형을 잡아 파란 띠를 양쪽에서 균등하게 배제합니다.
    side = int(round(min(w, h) * ICON_WIDTH_FROM_ROW_RATIO_V43))
    side = max(12, min(side, h, w))
    left = max(0, (w - side) // 2)          # ← 가운데 정렬 (was: left_pad 기반 왼쪽 정렬)
    top = max(0, (h - side) // 2)
    icon = img.crop((left, top, left + side, top + side))

    # 테두리/검은 외곽선이 비교를 방해하므로 아주 조금만 안쪽으로 자릅니다.
    return _trim_icon_edges_v43(icon)


def _left_square_icon_part_v39(img: Image.Image) -> Image.Image:  # type: ignore[override]
    return _extract_battle_icon_core_v42(img)


def _square_resize_for_icon(img: Image.Image, size: int = 48, margin_ratio: float = 0.025) -> Image.Image:  # type: ignore[override]
    img = _extract_battle_icon_core_v42(img).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = max(0, (w - side) // 2)
    top = max(0, (h - side) // 2)
    img = img.crop((left, top, left + side, top + side))
    if img.width > 12 and img.height > 12 and margin_ratio > 0:
        margin = max(1, int(min(img.width, img.height) * margin_ratio))
        if img.width - margin * 2 > 8 and img.height - margin * 2 > 8:
            img = img.crop((margin, margin, img.width - margin, img.height - margin))
    img = ImageEnhance.Contrast(img).enhance(1.18)
    img = ImageEnhance.Sharpness(img).enhance(1.10)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def _tight_icon_box_from_row_v43(image: Image.Image, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    side = int(round(h * 0.92))
    side = max(12, min(side, h, w))
    # v44: 아이콘은 셀 가운데에 있으므로 좌우 중앙 기준으로 정사각형을 잡습니다.
    left = x1 + max(0, (w - side) // 2)
    left = max(x1, min(x2 - side, left))
    top = y1 + int(round((h - side) / 2.0))
    return (
        max(0, left),
        max(0, top),
        min(image.width, left + side),
        min(image.height, top + side),
    )


def _crop_icon_from_row_metadata(image: Image.Image, row: pd.Series | Dict[str, Any], fallback_index: int) -> Image.Image | None:  # type: ignore[override]
    """v43: metadata icon_box를 확대하지 않고, 행 높이 기준으로 더 좁은 왼쪽 사각형만 crop합니다."""
    try:
        vals = [row.get("_icon_x1"), row.get("_icon_y1"), row.get("_icon_x2"), row.get("_icon_y2")]
        if all(v is not None and str(v) not in ["", "nan", "None"] for v in vals):
            box = tuple(int(float(v)) for v in vals)  # type: ignore[arg-type]
            tight = _tight_icon_box_from_row_v43(image, box)
            return _crop_pixel_box(image, tight, pad=0, expand_ratio=0.0)
    except Exception:
        pass
    # fallback도 확대하지 않습니다.
    return crop_attack_icon(image, fallback_index, window_box=_row_window_box(row))


# add_icon/ 폴더 파일명 → 전투분석기 표시 스킬명 매핑 (API에 안 나오는 스킬룬 등 수동 등록).
# 사용자가 새 아이콘을 추가하려면 파일을 넣고 아래 매핑에 한 줄 추가하면 됩니다.
# 배포 안전성을 위해 add_icon 파일명은 영어(ASCII)만 씁니다. 한글 스킬명은 파일명이 아니라
# 매핑(코드의 ADD_ICON_NAME_MAP_V45 또는 add_icon/mapping.csv 파일 '내용')으로만 둡니다.
# (파일 내용은 UTF-8이라 어떤 OS/배포에서도 안전, 파일명만 ASCII면 됩니다.)
ADD_ICON_NAME_MAP_V45 = {
    # 영어 파일명(소문자) → 전투분석기 표시 스킬명
    "poison_rune.png": "스킬룬 중독",
    "bleed_rune.png": "스킬룬 출혈",
    # 하위호환: 기존 공백 파일명도 계속 인식
    "poison rune.png": "스킬룬 중독",
    "bleed rune.png": "스킬룬 출혈",
}

# 낙원 아이템(공용 장신구에 달린 스킬) 아이콘 매핑.
# 모든 캐릭터의 전투분석기에 들어가므로 출혈룬/중독룬처럼 '공용'으로 처리합니다.
# 이미지는 add_icon/common/paradise/*.png 에 두고, 파일명(ASCII)→표시 스킬명만 여기 등록합니다.
# 새 낙원 아이템이 생기면 파일을 그 폴더에 넣고 아래에 한 줄만 추가하면 됩니다.
ADD_ICON_COMMON_SUB_MAP_V150 = {
    "acc_329.png": "섭리의 물결",
    "acc_330.png": "플루딩 플럭스",
    "acc_331.png": "성창해방",
    "acc_332.png": "흰 꽃의 화원",
    "acc_333.png": "맥스웰 맥시마",
    "acc_334.png": "성역선포",
    "acc_335.png": "너른너울",
    "acc_336.png": "등걸들녘",
}


def _read_text_robust_v151(path) -> str:
    """CSV/txt를 여러 인코딩으로 안전하게 읽습니다.

    한글 Windows/엑셀에서 CSV를 저장하면 cp949(euc-kr)로 저장되는 경우가 많습니다.
    utf-8만 시도하면 디코딩이 실패해 매핑이 통째로 무시되고, 스킬 이름이 파일명 그대로
    표시됩니다. utf-8 계열 → cp949 → euc-kr 순으로 시도해 어떤 인코딩으로 저장돼도 읽습니다.
    """
    try:
        p = Path(path)
    except Exception:
        return ""
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return p.read_text(encoding=enc)
        except Exception:
            continue
    try:
        return p.read_text(errors="ignore")
    except Exception:
        return ""


def _read_add_icon_mapping_file_v50(add_dir: Path) -> Dict[str, str]:
    """add_icon/mapping.csv(또는 .txt)를 읽어 '영어파일명 → 한글스킬명' 매핑을 만듭니다.

    형식(줄 단위, # 은 주석): `english_file.png = 한글 스킬명`  또는  `english_file.png, 한글 스킬명`
    파일명은 ASCII(영어)로, 스킬명만 한글로 적으면 배포 시 인코딩 문제가 없습니다.
    """
    mapping: Dict[str, str] = {}
    for fname in ("mapping.csv", "mapping.txt"):
        mp = add_dir / fname
        if not mp.is_file():
            continue
        try:
            for raw in _read_text_robust_v151(mp).splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                elif "," in line:
                    k, v = line.split(",", 1)
                else:
                    continue
                k, v = k.strip().lower(), v.strip()
                if k and v:
                    mapping[k] = v
        except Exception as e:  # noqa: BLE001
            print(f"[add_icon] mapping 파일 읽기 실패({mp.name}): {e}")
    return mapping


# ==============================================================================
# v80: 로컬 스킬 DB + 서브폴더(class/, common/) 아이콘 로드
# ==============================================================================

# 한글 클래스명 → lostark_skill_db_output 영어 폴더명
CLASS_NAME_TO_FOLDER_V80: Dict[str, str] = {
    "기상술사": "aeromancer",
    "아르카나": "arcana",
    "도화가": "artist",
    "바드": "bard",
    "배틀마스터": "battlemaster",
    "버서커": "berserker",
    "블레이드": "blade",
    "블래스터": "blaster",
    "브레이커": "breaker",
    "데모닉": "demonic",
    "디스트로이어": "destroyer",
    "데빌헌터": "devilhunter",
    "건슬링어": "gunslinger",
    "호크아이": "hawkeye",
    "홀리나이트": "holyknight",
    "인파이터": "infighter",
    "창술사": "lancemaster",
    "리퍼": "reaper",
    "스카우터": "scouter",
    "슬레이어": "slayer",
    "소서리스": "sorceress",
    "소울이터": "souleater",
    "서머너": "summoner",
    "발키리": "valkyrie",
    "워로드": "warlord",
    "환수사": "wildsoul",
    "스트라이커": "striker",
    "기공사": "soulmaster",
}


def _scan_icon_dir_v80(
    folder: "Path",
    mapping: Dict[str, str],
    source: str,
    out: List[Dict[str, str]],
    skip_names: "set[str] | None" = None,
) -> None:
    """폴더 내 PNG/JPG를 스캔해 out 에 추가합니다. 이중확장자(.png.png)도 자동 처리합니다."""
    skip = {s.lower() for s in (skip_names or set())}
    for p in sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")):
        low = p.name.lower()
        if low in skip:
            continue
        name = mapping.get(low)
        if name is None:
            # 이중 확장자 대응: vk_final_light.png.png → 키를 "vk_final_light.png"로 재시도
            stem_low = p.stem.lower()
            if stem_low.endswith(".png") or stem_low.endswith(".jpg"):
                name = mapping.get(stem_low)
        if not name:
            name = p.stem  # fallback: 영어 파일명 그대로
        out.append({"name": _canonical_display_name_v86(name), "icon": str(p.resolve()), "source": source})


@lru_cache(maxsize=30)
def _load_local_skill_db_candidates_v80(class_name_kr: str) -> List[Dict[str, str]]:
    """v80: lostark_skill_db_output/<class>/ 의 로컬 스킬 아이콘을 반환합니다.

    API CDN 이미지 대신 로컬 PNG 파일을 사용해 인터넷 없이도 아이콘 매칭이 가능합니다.
    all_skill_name_map.csv 로 파일명→한글스킬명을 조회합니다.
    """
    if not class_name_kr:
        return []
    folder_name = CLASS_NAME_TO_FOLDER_V80.get(class_name_kr.strip())
    if not folder_name:
        print(f"[skill_db] 미등록 클래스: {class_name_kr!r} — 로컬 DB 아이콘 없음")
        return []

    out: List[Dict[str, str]] = []
    try:
        for base in {_project_root_v32(), Path(__file__).resolve().parent.parent}:
            db_root = base / "add_icon" / "lostark_skill_db_output"
            if not db_root.is_dir():
                continue

            # all_skill_name_map.csv 에서 해당 클래스의 file_name → skill_name_kr 매핑 구축
            name_map: Dict[str, str] = {}
            csv_path = db_root / "all_skill_name_map.csv"
            if csv_path.is_file():
                try:
                    import csv as _csv_v80
                    with io.StringIO(_read_text_robust_v151(csv_path)) as fh:
                        for row in _csv_v80.DictReader(fh):
                            fn = (row.get("file_name") or "").strip()
                            sn = (row.get("skill_name_kr") or "").strip()
                            cf = (row.get("class_folder") or "").strip()
                            if fn and sn and cf == folder_name:
                                name_map[fn.lower()] = sn
                except Exception as csv_err:
                    print(f"[skill_db] CSV 파싱 실패: {csv_err}")

            # 클래스 폴더 내 PNG/JPG 스캔
            class_dir = db_root / folder_name
            if class_dir.is_dir():
                _scan_icon_dir_v80(class_dir, name_map, "skill_local_db", out)
            break
    except Exception as e:  # noqa: BLE001
        print(f"[skill_db] 로컬 스킬 DB 로드 실패({class_name_kr}): {e}")

    if out:
        print(f"[skill_db] {class_name_kr}({folder_name}) 로컬 스킬 {len(out)}개 등록")
    return out


@lru_cache(maxsize=1)
def _local_icon_candidates_v32() -> List[Dict[str, str]]:  # type: ignore[override]
    """v80: add_icon/ 루트 + class/ + common/ 서브폴더의 수동 아이콘을 스킬 후보로 등록.

    스캔 경로:
      add_icon/*.png|jpg          → 스킬룬 등 공용 (add_icon/mapping.csv)
      add_icon/class/*.png|jpg    → 아이덴티티 스킬 (add_icon/class/mapping.csv)
      add_icon/common/*.png|jpg   → 공용 UI 아이콘 (기본공격, 기타딜 등)

    lostark_skill_db_output/ 의 스킬 아이콘은 클래스별로 _load_local_skill_db_candidates_v80()
    에서 별도 로드하며, get_ocr_skill_candidates_full()에서 skill_candidates에 합쳐집니다.
    """
    out: List[Dict[str, str]] = []
    try:
        for base in {_project_root_v32(), Path(__file__).resolve().parent.parent}:
            add_dir = base / "add_icon"
            if not add_dir.is_dir():
                continue

            # 1) 루트 레벨 (스킬룬 등)
            root_mapping: Dict[str, str] = dict(ADD_ICON_NAME_MAP_V45)
            root_mapping.update(_read_add_icon_mapping_file_v50(add_dir))
            _scan_icon_dir_v80(
                add_dir, root_mapping, "skill_local_rune", out,
                skip_names={"battle time.jpg", "battle_time_template.png"},
            )

            # 2) class/ 서브폴더 (아이덴티티 스킬)
            class_dir = add_dir / "class"
            if class_dir.is_dir():
                class_mapping = _read_add_icon_mapping_file_v50(class_dir)
                _scan_icon_dir_v80(class_dir, class_mapping, "skill_local_identity", out)

            # 3) common/ 서브폴더 (기본공격, 기타딜 등 공용)
            common_dir = add_dir / "common"
            if common_dir.is_dir():
                common_mapping = _read_add_icon_mapping_file_v50(common_dir)
                _scan_icon_dir_v80(common_dir, common_mapping, "skill_local_common", out)

            break
    except Exception as e:  # noqa: BLE001
        print(f"[add_icon] 수동 아이콘 로드 실패: {e}")
    if out:
        print(f"[add_icon] 수동 아이콘 {len(out)}개 등록: " + ", ".join(c['name'] for c in out))
    return out


def save_learned_battle_icons_v32(df: pd.DataFrame, attack_image: Image.Image | None) -> int:  # type: ignore[override]
    """v43: 템플릿 라이브러리 저장 기능 비활성화."""
    return 0


# ==============================================================================
# v70.5: OCR 속도 개선 (셀 OCR 병렬화)
# ==============================================================================
# 배경:
#   전투분석기 OCR은 셀마다 _ocr_text_variants가 전처리 후보 3~4개를 각각 readtext로
#   읽고 점수가 가장 높은 결과를 고릅니다. 공격정보 표는 (행 14~20) x (열 7) x (후보 3~4)
#   = 수백 회의 readtext가 '순차'로 돌아 매우 느렸습니다. 비동기/병렬 처리는 전혀 없었습니다.
#
# 핵심 안전성:
#   한 셀 안의 후보(variant)들은 서로 독립적이고, 최종 선택은 점수 정렬로 결정됩니다.
#   즉 후보들을 동시에 읽어도 '입력→출력'이 순차 처리와 100% 동일합니다(정확도 변화 없음).
#   따라서 readtext 호출만 스레드로 동시에 돌려 벽시계 시간을 단축합니다.
#   PyTorch 추론(eval 모드)은 같은 모델을 여러 스레드에서 동시에 호출해도 안전하며,
#   readtext 내부는 호출별 지역 상태만 사용합니다.
#
# 토글:
#   OCR_PARALLEL_VARIANTS_V705 = False 로 두면 기존 순차 동작으로 즉시 복귀합니다.
import os as _os_v705
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor_v705

OCR_PARALLEL_VARIANTS_V705 = True
# 후보 수가 최대 4개이므로 워커도 4면 충분합니다. CPU 코어가 적으면 그만큼만 사용합니다.
_OCR_VARIANT_WORKERS_V705 = max(2, min(4, (_os_v705.cpu_count() or 4)))

# 셀 단위 병렬 처리에서 ThreadPoolExecutor를 매번 새로 만들지 않도록 공유 풀을 둡니다.
_OCR_VARIANT_POOL_V705: Any = None


def _get_ocr_variant_pool_v705() -> Any:
    global _OCR_VARIANT_POOL_V705
    if _OCR_VARIANT_POOL_V705 is None:
        _OCR_VARIANT_POOL_V705 = _ThreadPoolExecutor_v705(
            max_workers=_OCR_VARIANT_WORKERS_V705,
            thread_name_prefix="ocr_variant_v705",
        )
    return _OCR_VARIANT_POOL_V705


def _ocr_text_variants(  # type: ignore[override]
    reader: Any,
    raw_crop: Image.Image,
    *,
    numeric: bool,
    scale: int,
    kind: str = "generic",
) -> List[Tuple[str, Image.Image, str, float]]:
    """v70.5: 후보별 readtext를 병렬로 실행합니다(결과는 순차와 동일).

    전처리(이미지 확대/마스킹)는 가볍고 PIL/numpy라 순차로 만들고, 무거운 readtext만
    스레드로 동시에 돌립니다. 어떤 이유로든 병렬 실행이 실패하면 순차로 자동 폴백합니다.
    """
    # --- 후보 전처리 (원본 _ocr_text_variants와 동일) ---
    variants: List[Tuple[str, Image.Image]] = []
    variants.append(("normal", preprocess_crop(raw_crop, scale=scale)))
    variants.append(("normal_more", preprocess_crop(raw_crop, scale=min(10, max(scale + 1, 7)))))
    if numeric:
        variants.append(("numeric_mask", _preprocess_numeric_text_mask(raw_crop, scale=min(10, max(scale, 7)))))
        variants.append(("numeric_mask_big", _preprocess_numeric_text_mask(raw_crop, scale=min(12, max(scale + 1, 8)))))
    else:
        variants.append(("text_mask", _preprocess_text_light_mask(raw_crop, scale=min(10, max(scale, 7)))))

    def _run(item: Tuple[str, Image.Image]) -> Tuple[str, Image.Image, str, float]:
        name, img = item
        try:
            txt = _ocr_text_once(reader, img, numeric=numeric)
        except Exception:
            txt = ""
        score = _numeric_ocr_score(txt, kind=kind) if numeric else (len(txt) + (3 if re.search(r"[가-힣]", txt) else 0))
        return (name, img, txt, score)

    if OCR_PARALLEL_VARIANTS_V705 and len(variants) > 1:
        try:
            pool = _get_ocr_variant_pool_v705()
            return list(pool.map(_run, variants))
        except Exception:
            # 병렬 실패 시 순차 폴백 (출력 동일)
            pass
    return [_run(v) for v in variants]


# ==============================================================================
# v70.5: OCR 초고속 모드 (행 단위 1회 OCR)  ← 3분대 → 수십초/수초로 단축
# ==============================================================================
# 기존 공격정보 OCR은 (행 N) x (열 8) x (전처리 후보 3~4) = 수백 회의 readtext를 돌렸습니다.
# 이 모드는 '한 행 전체'를 한 번만 readtext(detail=1)로 읽어 글자 위치(x좌표)로 열을 나눕니다.
# 즉 행당 readtext 1회 → 표 전체 약 14~20회로, 호출 수가 ~20배 이상 줄어듭니다.
#
# 안전장치:
#   - OCR_FAST_ROW_PASS_V705 = False 로 두면 기존(셀별) v38 방식으로 즉시 복귀합니다.
#   - 한 행에서 글자를 거의 못 읽으면 그 행만 기존 셀별 OCR로 자동 폴백합니다.
#   - 창 위치 탐지/방향성 감지/아이콘 행 탐지/아이콘 이름보정 좌표는 기존과 동일하게 유지합니다.

OCR_FAST_ROW_PASS_V705 = True
# 행 스트립 확대 배율. 게임 글자는 2~3배면 충분하고, 과확대는 오히려 느립니다.
OCR_ROW_SCALE_V705 = 3

_PERCENT_KEYS_V705 = {"directional_rate", "directional_share", "crit_rate", "crit_share", "cooldown_rate", "share_rate"}
_KOREAN_NUM_KEYS_V705 = {"damage_text", "dps_text"}

_old_parse_attack_fixed_grid_v38_for_v705 = parse_attack_fixed_grid


def _preprocess_row_strip_v705(strip: Image.Image, scale: int) -> "np.ndarray":
    crop = strip.convert("RGB")
    if scale > 1:
        crop = crop.resize((max(1, crop.width * scale), max(1, crop.height * scale)), Image.Resampling.LANCZOS)
    crop = ImageEnhance.Brightness(crop).enhance(1.05)
    crop = ImageEnhance.Contrast(crop).enhance(1.55)
    crop = ImageEnhance.Sharpness(crop).enhance(1.55)
    return np.array(crop)


def _ocr_row_single_pass_v705(
    reader: Any,
    image: Image.Image,
    col_boxes: List[Tuple[str, str, int, int, int, int]],
    scale: int,
) -> Dict[str, str]:
    """한 행을 1회 readtext로 읽고, 글자 x중심으로 열에 배분합니다.

    col_boxes: [(key, label, px1, py1, px2, py2), ...] - 각 열의 절대 픽셀 박스.
    반환: {key: 읽은_텍스트}
    """
    if not col_boxes:
        return {}
    strip_x1 = min(b[2] for b in col_boxes)
    strip_x2 = max(b[4] for b in col_boxes)
    strip_y1 = min(b[3] for b in col_boxes)
    strip_y2 = max(b[5] for b in col_boxes)
    pad = 2
    sx1 = max(0, strip_x1 - pad); sy1 = max(0, strip_y1 - pad)
    sx2 = min(image.width, strip_x2 + pad); sy2 = min(image.height, strip_y2 + pad)
    if sx2 - sx1 < 8 or sy2 - sy1 < 6:
        return {}
    strip = image.crop((sx1, sy1, sx2, sy2))
    arr = _preprocess_row_strip_v705(strip, scale)
    try:
        results = reader.readtext(arr, detail=1, paragraph=False)
    except Exception:
        return {}

    # 열별 중심/범위 (절대 픽셀)
    centers = [((b[2] + b[4]) / 2.0, b) for b in col_boxes]
    buckets: Dict[str, List[Tuple[float, str]]] = {b[0]: [] for b in col_boxes}
    for item in results:
        try:
            bbox, text = item[0], str(item[1])
        except Exception:
            continue
        text = text.strip()
        if not text:
            continue
        xs = [float(p[0]) for p in bbox]
        # strip-확대 좌표 → 절대 이미지 픽셀
        tok_cx_abs = sx1 + (sum(xs) / len(xs)) / float(scale)
        # 포함되는 열 우선, 없으면 가장 가까운 중심
        chosen = None
        for cx, b in centers:
            if b[2] <= tok_cx_abs <= b[4]:
                chosen = b
                break
        if chosen is None:
            cx, chosen = min(centers, key=lambda c: abs(c[0] - tok_cx_abs))
        buckets[chosen[0]].append((sum(xs) / len(xs), text))

    out: Dict[str, str] = {}
    for key, toks in buckets.items():
        toks.sort(key=lambda t: t[0])
        out[key] = clean_ocr_text(" ".join(t[1] for t in toks))
    return out


def _postprocess_cell_value_v705(key: str, text: str) -> str:
    text = clean_ocr_text(text)
    if key in _KOREAN_NUM_KEYS_V705:
        return extract_korean_number_text(text)
    if key in _PERCENT_KEYS_V705:
        return format_percent_from_ocr(text)
    return text


# v70.5b: 행 단위 1회 OCR은 빠르지만, '544.21억'처럼 접미사(억/만)·소수점이 있는 큰 숫자나
# 한 자리 숫자(횟수)는 글자가 분리되며 옆 칸/엉뚱한 값으로 새는 경우가 있습니다.
# (디버그 결과: 셀별 OCR은 544.21억/7로 정확히 읽지만 행단위에서 21억/0.0으로 깨짐)
# → 이 분리에 취약한 핵심 열만 셀별 정밀 OCR로 '다시' 읽어 정확도를 끌어올립니다.
#   퍼센트(NN.NN%)는 한 덩어리라 행단위에서도 잘 읽혀 재검수 대상에서 제외합니다.
OCR_RECHECK_KEYS_V705 = {"damage_text", "casts"}


def _ocr_cell_precise_v705(reader: Any, image: Image.Image, box: Tuple[int, int, int, int], key: str, scale: int = 7) -> str:
    """단일 셀을 정밀하게 읽습니다(전처리 2종 중 점수 높은 결과 선택). 행단위 결과 보정용."""
    raw = _crop_pixel_box(image, box, pad=1, expand_ratio=CELL_CROP_EXPAND_RATIO)
    numeric = key != "name"
    kind = "percent" if key in _PERCENT_KEYS_V705 else ("korean_number" if key in _KOREAN_NUM_KEYS_V705 else "text")
    cand_imgs = [preprocess_crop(raw, scale=scale)]
    if numeric:
        cand_imgs.append(_preprocess_numeric_text_mask(raw, scale=max(7, scale)))
    else:
        cand_imgs.append(_preprocess_text_light_mask(raw, scale=max(7, scale)))
    best_t, best_s = "", -1.0
    for im in cand_imgs:
        try:
            t = _ocr_text_once(reader, im, numeric=numeric)
        except Exception:
            t = ""
        s = _numeric_ocr_score(t, kind=kind) if numeric else (len(t) + (3 if re.search(r"[가-힣]", t) else 0))
        if s > best_s:
            best_t, best_s = t, s
    return _postprocess_cell_value_v705(key, best_t)


# ==============================================================================
# v70.5-glyph: 글자 템플릿(글리프) 빠른 인식 연동
# ==============================================================================
# 실제 게임 화면에서 누적 학습한 글자 템플릿(data/glyph_templates.json)으로
# 퍼센트/횟수 칸을 EasyOCR 없이 즉시 읽습니다. matchTemplate + DP 디코딩이라
# 탐욕적 인식기와 달리 신뢰도(conf)가 정답/오답을 잘 구분하므로,
# conf >= GATE 인 칸만 채택하고 미달이면 기존 EasyOCR로 폴백합니다.
#
# 설계 원칙(정확도 손실 0):
#   - 글리프 적용 대상: 퍼센트(NN.NN%) + 횟수(count)만.
#   - 항상 EasyOCR(글리프 금지): 피해량/DPS(억/만 — 혼동 시 치명적), 스킬명(text),
#     종합정보 전투시간(time)은 이 모듈의 공격표 경로에서 글리프를 쓰지 않습니다.
#   - EasyOCR 폴백 경로(행단위 1회 OCR + 정밀 재검수)는 그대로 살아 있습니다.
#
# 토글/게이트(상단 상수):
#   OCR_GLYPH_FAST_V705 = False 로 두면 글리프를 완전히 끄고 기존 동작으로 복귀합니다.
#   GLYPH_GATE_PERCENT / GLYPH_GATE_COUNT 로 채택 임계값을 조절합니다(0.65~0.70 권장).
#     검증치: 게이트 0.65 → 약 38% 칸 채택/정밀도 97%, 0.70 → 약 28% 채택/정밀도 95%.

# --- 토글 & 게이트 ---
# v2(glyph_fast_v2: 흰색도 전처리 + 형식검증 + 100초과거부)와 함께 쓰면, 형식·범위 검증이
# 정밀도를 지켜주므로 게이트를 낮춰 커버리지를 크게 올릴 수 있습니다.
# (디버그 크롭 측정: 공격표 퍼센트 25% → ~81% 채택, 정밀도 100%)
OCR_GLYPH_FAST_V705 = True          # 글리프 빠른 인식 on/off (off면 EasyOCR 단독)
GLYPH_GATE_PERCENT = 0.48           # 퍼센트 칸 채택 신뢰도 임계값(형식검증과 병행)
GLYPH_GATE_COUNT = 0.60             # 횟수 칸 채택 신뢰도 임계값(단자리 혼동 차단 위해 0.60)
GLYPH_TEMPLATES_FILENAME = "glyph_templates.json"  # data/ 하위 파일명

# 글리프를 적용할 칸 종류 매핑 (퍼센트 / 횟수만). 그 외(피해량·DPS·이름)는 None=EasyOCR.
_GLYPH_PERCENT_KEYS = {
    "directional_rate", "directional_share",
    "back_attack_rate", "back_attack_share",
    "head_attack_rate", "head_attack_share",
    "crit_rate", "crit_share",
    "cooldown_rate", "share_rate",
}
_GLYPH_COUNT_KEYS = {"casts"}

# 안전 import: glyph_engine 이 없거나 깨져도 모듈 전체는 EasyOCR 단독으로 정상 동작.
try:
    from .glyph_engine import TemplateStore as _GlyphTemplateStore  # type: ignore
except Exception:
    try:
        from glyph_engine import TemplateStore as _GlyphTemplateStore  # type: ignore
    except Exception:
        _GlyphTemplateStore = None  # type: ignore

_GLYPH_STORE_V705: Any = None
_GLYPH_STORE_TRIED_V705 = False
# 마지막 공격표 파싱의 글리프 통계(진단/벤치마크용). {attempt, accept, rate}
_GLYPH_LAST_STATS_V705: Dict[str, Any] = {"attempt": 0, "accept": 0, "rate": 0.0}


def _glyph_kind_for_key_v705(key: str) -> str | None:
    """공격표 칸 key → 글리프 kind. 퍼센트/횟수만 글리프, 나머지는 None(EasyOCR)."""
    if key in _GLYPH_COUNT_KEYS:
        return "count"
    if key in _GLYPH_PERCENT_KEYS:
        return "percent"
    return None


def _resolve_glyph_templates_path_v705() -> "Path | None":
    """data/glyph_templates.json 위치를 견고하게 탐색합니다.

    우선순위: 환경변수 GLYPH_TEMPLATES_PATH → project_root/data → modules/data
             → 현재 작업 디렉터리/data.
    """
    import os as _os
    cands: List[Path] = []
    env = _os.environ.get("GLYPH_TEMPLATES_PATH")
    if env:
        cands.append(Path(env))
    try:
        here = Path(__file__).resolve()
        cands += [
            here.parent.parent / "data" / GLYPH_TEMPLATES_FILENAME,  # project_root/data
            here.parent / "data" / GLYPH_TEMPLATES_FILENAME,         # modules/data
        ]
    except Exception:
        pass
    cands.append(Path.cwd() / "data" / GLYPH_TEMPLATES_FILENAME)
    for c in cands:
        try:
            if c.exists():
                return c
        except Exception:
            continue
    return None


def _get_glyph_store_v705() -> Any:
    """글리프 템플릿 스토어를 1회만 로드해 캐시합니다. 실패 시 None(→EasyOCR 단독)."""
    global _GLYPH_STORE_V705, _GLYPH_STORE_TRIED_V705
    if _GLYPH_STORE_TRIED_V705:
        return _GLYPH_STORE_V705
    _GLYPH_STORE_TRIED_V705 = True
    if _GlyphTemplateStore is None:
        return None
    path = _resolve_glyph_templates_path_v705()
    if path is None:
        return None
    try:
        store = _GlyphTemplateStore().load(str(path))
        # 숫자 글자(0~9)가 충분히 학습돼 있어야 사용. 아니면 끔.
        if not store.ready():
            return None
        _GLYPH_STORE_V705 = store
    except Exception:
        _GLYPH_STORE_V705 = None
    return _GLYPH_STORE_V705


# glyph_fast_v2: 흰색도 전처리 + 본문 덩어리 분리 + 빈칸감지 + 형식검증 채택(개선 인식기).
# 없으면 기존 방식으로 안전하게 폴백합니다.
try:
    from . import glyph_fast_v2 as _glyph_fast_v2  # type: ignore
except Exception:
    try:
        import glyph_fast_v2 as _glyph_fast_v2  # type: ignore
    except Exception:
        _glyph_fast_v2 = None  # type: ignore

_GLYPH_PCT_FMT_V705 = re.compile(r"^\d{1,3}\.\d{2}%$")


def _glyph_pct_format_ok_v705(text: str) -> bool:
    """퍼센트 글리프 출력 검증: NN.NN% 형식 + 값 100 이하(게임 퍼센트는 100 이하)."""
    if not text or not _GLYPH_PCT_FMT_V705.match(text):
        return False
    try:
        return float(text.rstrip("%")) <= 100.5
    except Exception:
        return False


def _glyph_recognize_cell_v705(
    store: Any, image: Image.Image, box: Tuple[int, int, int, int], kind: str
) -> Tuple[str, float, bool]:
    """단일 공격표 셀을 글리프로 인식해 (텍스트, 신뢰도, 채택여부)를 반환.

    glyph_fast_v2(흰색도 전처리+형식검증)가 있으면 그걸 사용하고, 없으면 기존 방식+게이트로 폴백.
    accepted=True 면 EasyOCR 없이 이 값을 사용 가능(정확도는 형식·범위 검증이 보장).
    """
    try:
        crop = _crop_pixel_box(image, box, pad=1, expand_ratio=CELL_CROP_EXPAND_RATIO)
    except Exception:
        return "", 0.0, False
    if _glyph_fast_v2 is not None:
        try:
            return _glyph_fast_v2.recognize(
                store, crop, kind,
                gate_percent=GLYPH_GATE_PERCENT, gate_count=GLYPH_GATE_COUNT,
            )
        except Exception:
            pass
    # 폴백: 기존 방식(원본 크롭) + 게이트 + (퍼센트는 형식검증)
    try:
        text, conf = store.recognize(crop, kind=kind)
        text = text or ""
        conf = float(conf)
    except Exception:
        return "", 0.0, False
    if kind == "percent":
        ok = conf >= GLYPH_GATE_PERCENT and _glyph_pct_format_ok_v705(text)
    else:
        ok = bool(text) and conf >= GLYPH_GATE_COUNT
    return text, conf, ok


def _strip_ocr_unit_garbage(num_text: str) -> str:
    """억/만/조 단위가 OCR에서 숫자로 오인식된 trailing 글자를 제거합니다.

    RapidOCR 중국어 모델은 '억'을 '9', '91' 등 숫자로 오인식합니다.
    게임 화면은 소수점 2자리까지 표시하므로, 소수점 2자리 초과 trailing 숫자를 제거합니다.
    예: "426.779"  → "426.77"
        "249.2891" → "249.28"
        "395.4091" → "395.40"
        "51.0594"  → "51.05"
    쉼표가 포함된 경우도 처리: "2,083.449" → "2,083.44"
    정수형(소수점 없음)도 처리: "109" → "10" (trailing digit 제거는 단위 감지 후에만)
    """
    t = num_text.strip()
    # 소수점 있는 경우: 소수점 2자리까지만 남기고 나머지 trailing 숫자 제거
    m = re.match(r'^(\d[\d,]*\.\d{2})\d+$', t)
    if m:
        return m.group(1)
    return t


def _glyph_scan_unit_direct(store: Any, crop: "Image.Image") -> str:
    """matchTemplate으로 억/만/조 템플릿을 직접 스캔합니다.

    DP 전체 시퀀스 디코딩 없이 억/만/조 한 글자만 찾습니다.
    "2,083.44억"처럼 쉼표가 포함된 긴 값에서 DP가 혼동할 때 사용합니다.

    테두리(행 구분선) 처리:
    - 상하 UI 테두리 픽셀 때문에 _band_gray_fg의 ys가 전체 높이를 덮는 경우,
      상하 ~14% 트리밍한 crop으로 재시도합니다.
    - 만 템플릿(18px)이 억(36px) 우측 패턴에 걸려 오탐되는 것을 방지하기 위해,
      억이 밴드에 들어가지 않으면 만도 매칭하지 않습니다.
    - 억/만 점수가 가까운 경우(억>=0.15이고 만이 0.20 미만으로 높을 때) 억을 우선합니다.
    """
    try:
        import cv2 as _cv2
        import numpy as _np
        from glyph_engine import _band_gray_fg, NORM_H  # type: ignore

        MIN_SCORE = 0.35

        # 템플릿 사전 로드
        tmpl_map = {}
        for ch in ("조", "억", "만"):
            t = store.mean_template(ch)
            if t is not None:
                tmpl_map[ch] = t

        if not tmpl_map:
            return ""

        # 시도할 crop 변형: [원본, 상하 트리밍(행 구분선 제거)]
        crops_to_try = [crop]
        h = crop.height
        if h > 8:
            margin = max(1, h // 7)  # 상하 ~14% 트리밍
            trimmed = crop.crop((0, margin, crop.width, h - margin))
            if trimmed.height > 3:
                crops_to_try.append(trimmed)

        # 각 문자별 최고 점수 수집
        best_scores: Dict[str, float] = {}

        for try_crop in crops_to_try:
            bg = _band_gray_fg(try_crop)
            if bg is None:
                continue
            band, _fg = bg
            H, W = band.shape

            # 억 템플릿이 들어가지 않으면 이 밴드 전체를 스킵
            # (좁은 밴드에서 만 오탐 방지)
            억_tmpl = tmpl_map.get("억")
            if 억_tmpl is not None and (억_tmpl.shape[1] > W or 억_tmpl.shape[0] > H):
                continue

            bandf = ((_np.asarray(band, dtype=_np.float32) - band.mean()) /
                     (band.std() + 1e-6)).astype(_np.float32)

            for ch in ("조", "억", "만"):
                t = tmpl_map.get(ch)
                if t is None or t.shape[1] > W or t.shape[0] > H:
                    continue
                tf = ((_np.asarray(t, dtype=_np.float32) - t.mean()) /
                      (t.std() + 1e-6)).astype(_np.float32)
                res = _cv2.matchTemplate(bandf, tf, _cv2.TM_CCOEFF_NORMED)[0]
                score = float(_np.max(res))
                if score > best_scores.get(ch, -999.0):
                    best_scores[ch] = score

        if not best_scores:
            return ""

        # 억/만 오탐 방지: 만 템플릿(18px)은 억 문자 우측 패턴과 겹칠 수 있음.
        # 만이 threshold를 넘었지만 억도 0.15 이상이고, 두 점수 차가 0.20 미만이면 억 우선.
        만_score = best_scores.get("만", -999.0)
        억_score = best_scores.get("억", -999.0)
        if 만_score >= MIN_SCORE and 억_score < MIN_SCORE and 억_score >= 0.15:
            if 만_score - 억_score < 0.20:
                return "억"

        # 조 > 억 > 만 순서로 threshold 초과 단위 반환
        for ch in ("조", "억", "만"):
            if best_scores.get(ch, -999.0) >= MIN_SCORE:
                return ch

    except Exception:
        pass
    return ""


def _glyph_detect_damage_unit_v705(store: Any, image: Image.Image, box: Tuple[int, int, int, int]) -> str:
    """피해량 셀 크롭에서 '억'/'만'/'조' 단위를 글리프 매칭으로 감지해 반환합니다.

    RapidOCR 중국어 모델은 한글 '억'/'만'/'조'를 출력하지 못하므로,
    재검수 OCR 후에 이 함수로 단위를 보정합니다.
    반환: '조', '억', '만', 또는 '' (감지 불가).

    v705d: 3단계 방식으로 조/억/만 감지
    1. _decode_dp() seq에 단위 있으면 반환 (가장 신뢰도 높음)
    2. 없으면 직접 matchTemplate으로 스캔 (comma 등에 의한 DP 혼동 우회)
    3. 둘 다 없으면 recognize() fallback (낮은 임계값)
    """
    try:
        crop = _crop_pixel_box(image, box, pad=1, expand_ratio=CELL_CROP_EXPAND_RATIO)
        # 1단계: _decode_dp 직접 호출
        dec = store._decode_dp(crop)
        if dec is not None:
            seq, _conf = dec
            if "조" in seq:
                return "조"
            if "억" in seq:
                return "억"
            if "만" in seq:
                return "만"
        # 1b단계: 파란 배경 강조 행 대비 - 상하 트리밍 후 decode_dp 재시도
        if dec is None and crop.height > 8:
            margin = max(1, crop.height // 7)
            trimmed = crop.crop((0, margin, crop.width, crop.height - margin))
            if trimmed.height > 3:
                dec2 = store._decode_dp(trimmed)
                if dec2 is not None:
                    seq2, _conf2 = dec2
                    if "조" in seq2:
                        return "조"
                    if "억" in seq2:
                        return "억"
                    if "만" in seq2:
                        return "만"
        # 2단계: matchTemplate 직접 스캔 (쉼표/특수문자로 DP가 혼동할 때 사용,
        #        파란 배경 행은 내부에서 트리밍 변형도 시도함)
        unit = _glyph_scan_unit_direct(store, crop)
        if unit:
            return unit
        # 3단계: recognize() fallback (낮은 임계값)
        text, conf = store.recognize(crop, kind=None)
        text = text or ""
        if conf >= 0.15:
            if "조" in text:
                return "조"
            if "억" in text:
                return "억"
            if "만" in text:
                return "만"
    except Exception:
        pass
    return ""


def _glyph_isolate_number_crop_v705(pil: Image.Image, pad: int = 6, gap_merge: int = 28) -> Image.Image:
    """넓은 셀(종합정보처럼 라벨+값+배경막대가 한 크롭에 들어온 경우)에서 우측 정렬된
    숫자 덩어리만 잘라냅니다. 글리프는 크롭 전체를 좌→우로 디코딩하므로, 숫자 왼쪽의
    라벨/막대가 섞이면 앞에 쓰레기 숫자가 붙어 신뢰도가 무너집니다. 이를 막는 전처리.

    원리: 밝은 글자 전경의 '열 투영'으로 덩어리를 찾고, 큰 공백(라벨↔숫자)에서 끊은 뒤
    아주 좁은 노이즈 덩어리(폭<0.3H)를 버리고 남은 것 중 맨 오른쪽(=값)만 크롭.
    실패 시 원본을 그대로 반환(=손해 없음). 공격표처럼 이미 좁은 셀에는 쓰지 않습니다.
    """
    try:
        import numpy as _np
        g = _np.asarray(pil.convert("L"), dtype=_np.float32)
        H, W = g.shape
        if W < 12 or H < 8:
            return pil
        thr = g.mean() + (g.max() - g.mean()) * 0.25
        fg = g > thr
        on = fg.sum(axis=0) > 0
        runs: List[List[int]] = []
        s = None
        for i, v in enumerate(on):
            if v and s is None:
                s = i
            if not v and s is not None:
                runs.append([s, i]); s = None
        if s is not None:
            runs.append([s, W])
        if not runs:
            return pil
        merged: List[List[int]] = []
        for r in runs:
            if merged and r[0] - merged[-1][1] <= gap_merge:
                merged[-1][1] = r[1]
            else:
                merged.append([r[0], r[1]])
        min_w = max(10, int(0.3 * H))
        cand = [(a, b) for a, b in merged if (b - a) >= min_w and fg[:, a:b].mean() >= 0.03]
        if not cand:
            return pil
        a, b = cand[-1]
        a = max(0, a - pad); b = min(W, b + pad)
        ys = _np.where(fg[:, a:b].any(axis=1))[0]
        if ys.size:
            y1 = max(0, int(ys[0]) - pad); y2 = min(H, int(ys[-1]) + 1 + pad)
        else:
            y1, y2 = 0, H
        return pil.crop((a, y1, b, y2))
    except Exception:
        return pil


def parse_attack_fixed_grid(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int | None = None,
    scale: int = 7,
    auto_window: bool = True,
) -> pd.DataFrame:
    """v70.5b 공격정보 OCR. 행 단위 1회 OCR(가속) + 분리 취약 열 셀별 정밀 재검수(정확도).

    v70.5-glyph: 퍼센트/횟수 칸은 먼저 글리프(글자 템플릿)로 인식하고, 신뢰도가
    게이트 이상이면 그 값을 채택해 EasyOCR 재검수를 건너뜁니다(가속). 게이트 미달
    칸은 기존 EasyOCR 경로로 자동 폴백하므로 정확도 손실이 없습니다.
    """
    if not OCR_FAST_ROW_PASS_V705:
        return _old_parse_attack_fixed_grid_v38_for_v705(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)

    grid = ATTACK_GRID
    if row_count is None:
        row_count = int(grid["row_count"])
    try:
        _pt = _prof_now()
        window_box = detect_analyzer_window(image, reader) if auto_window else None
        _prof_add("attack_1_window_detect", _pt)
        print(f"[parse_attack] window_box={window_box}  image={image.width}x{image.height}")
        _pt = _prof_now()
        direction_kind = _detect_attack_direction_kind_v38(image, reader, window_box, scale=scale) if window_box else "back"
        _prof_add("attack_2_direction_detect", _pt)
        _pt = _prof_now()
        detected_rows = detect_attack_icon_rows(image, reader, window_box=window_box, max_rows=int(row_count)) if window_box else []
        _prof_add("attack_3_icon_rows_detect", _pt)
        if not detected_rows:
            detected_rows = []
            for i in range(int(row_count)):
                y1 = grid["row_start"] + i * grid["row_height"]
                y2 = y1 + grid["row_height"]
                row_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, grid["columns"][-1][3], y2), pad=0, window_box=window_box)
                icon_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, ATTACK_ICON_GRID["x2"], y2), pad=1, window_box=window_box)
                detected_rows.append({"row_index": i, "row_box": row_box, "icon_box": icon_box, "window_box": window_box, "source": "fixed_fallback"})

        # ---- 1단계: 행 단위 단일 OCR + 열 박스 수집 ----
        row_states: List[Dict[str, Any]] = []
        row_scale = max(2, min(4, int(OCR_ROW_SCALE_V705)))
        _pt_rows = _prof_now()
        for i, rinfo in enumerate(detected_rows[: int(row_count)]):
            col_boxes: List[Tuple[str, str, int, int, int, int]] = []
            for key, label, x1, x2 in grid["columns"]:
                if window_box:
                    box = _column_pixel_box_for_row(image, window_box, rinfo["row_box"], x1, x2)
                else:
                    px1 = int(round(x1 * image.width)); px2 = int(round(x2 * image.width))
                    box = _clip_box(image, (px1, rinfo["row_box"][1], px2, rinfo["row_box"][3]))
                col_boxes.append((key, label, box[0], box[1], box[2], box[3]))

            cell_texts = _ocr_row_single_pass_v705(reader, image, col_boxes, row_scale)
            # 행 전체를 거의 못 읽으면 이 행은 전부 셀별 정밀 OCR로 폴백
            row_failed = sum(1 for v in cell_texts.values() if str(v).strip()) < 2
            row_states.append({"idx": i, "rinfo": rinfo, "col_boxes": col_boxes, "cell_texts": cell_texts, "row_failed": row_failed})
        _prof_add("attack_4_rowpass_easyocr", _pt_rows)

        # 앞 5행의 스킬명/피해량 OCR 원문을 출력해 한글 인식 상태를 빠르게 점검합니다.
        for _dbg_st in row_states[:5]:
            _dbg_ct = _dbg_st["cell_texts"]
            print(f"[parse_attack] row{_dbg_st['idx']:02d}: name={_dbg_ct.get('name','')!r:30s}  damage={_dbg_ct.get('damage_text','')!r}")

        # ---- 1.5단계: 글리프 빠른 인식 (퍼센트/횟수만, 신뢰도 게이트 통과 칸만 채택) ----
        _pt_glyph = _prof_now()
        # 통과한 칸은 (row_idx, key) 로 표시해 2단계 EasyOCR 재검수를 건너뜁니다(가속).
        # store 가 None(템플릿 없음/엔진 없음)이거나 토글이 꺼져 있으면 이 단계는 통째로 생략됩니다.
        glyph_confident: set[Tuple[int, str]] = set()
        glyph_attempt = 0
        glyph_accept = 0
        glyph_store = _get_glyph_store_v705() if OCR_GLYPH_FAST_V705 else None
        if glyph_store is not None:
            for st in row_states:
                ct = st["cell_texts"]
                for key, label, bx1, by1, bx2, by2 in st["col_boxes"]:
                    gkind = _glyph_kind_for_key_v705(key)
                    if gkind is None:
                        continue  # 피해량·DPS·이름 등은 항상 EasyOCR
                    glyph_attempt += 1
                    gtext, gconf, gok = _glyph_recognize_cell_v705(glyph_store, image, (bx1, by1, bx2, by2), gkind)
                    if gok:
                        ct[key] = gtext
                        glyph_confident.add((st["idx"], key))
                        glyph_accept += 1
        _GLYPH_LAST_STATS_V705["attempt"] = glyph_attempt
        _GLYPH_LAST_STATS_V705["accept"] = glyph_accept
        _GLYPH_LAST_STATS_V705["rate"] = (glyph_accept / glyph_attempt) if glyph_attempt else 0.0
        _prof_add("attack_5_glyph", _pt_glyph)

        # ---- 2단계: 분리 취약 열(피해량/횟수)·빈 칸·실패행을 셀별 정밀 OCR로 '병렬' 재검수 ----
        _pt_recheck = _prof_now()
        recheck_tasks: List[Tuple[int, str, Tuple[int, int, int, int]]] = []
        for st in row_states:
            ct = st["cell_texts"]
            for key, label, bx1, by1, bx2, by2 in st["col_boxes"]:
                # 글리프가 게이트 이상으로 읽은 칸은 신뢰하고 EasyOCR 재검수를 생략합니다.
                if (st["idx"], key) in glyph_confident:
                    continue
                need = st["row_failed"] or (key in OCR_RECHECK_KEYS_V705) or (not str(ct.get(key, "")).strip())
                if need:
                    recheck_tasks.append((st["idx"], key, (bx1, by1, bx2, by2)))

        if recheck_tasks:
            def _do_recheck(task: Tuple[int, str, Tuple[int, int, int, int]]) -> Tuple[int, str, str]:
                ridx, key, box = task
                return (ridx, key, _ocr_cell_precise_v705(reader, image, box, key, scale=scale))
            results: List[Tuple[int, str, str]] = []
            try:
                pool = _get_ocr_variant_pool_v705()
                results = list(pool.map(_do_recheck, recheck_tasks))
            except Exception:
                results = [_do_recheck(t) for t in recheck_tasks]
            by_idx = {st["idx"]: st for st in row_states}
            for ridx, key, val in results:
                if str(val).strip():
                    by_idx[ridx]["cell_texts"][key] = val
        _prof_add("attack_6_recheck_easyocr", _pt_recheck)

        # ---- 2.5단계: 피해량/DPS 글리프 단위 보정 ----
        # RapidOCR 중국어 모델은 한글 '억'/'만'/'조'를 인식하지 못합니다.
        # 재검수 OCR 결과(숫자만)에 글리프로 감지한 단위를 붙입니다.
        # 주의: OCR이 '억'을 '9', '91', '94' 등 trailing 숫자로 오인식하므로,
        #       단위 감지 후 소수점 2자리 초과 trailing 숫자를 제거합니다.
        #       예: "426.779" + glyph("억") → "426.77억"
        if glyph_store is not None:
            for st in row_states:
                ct = st["cell_texts"]
                for key, label, bx1, by1, bx2, by2 in st["col_boxes"]:
                    if key not in _KOREAN_NUM_KEYS_V705:
                        continue
                    cur = str(ct.get(key, "")).strip()
                    if not cur or re.search(r"[억만조]", cur):
                        continue  # 이미 단위 있거나 빈 칸 → 스킵
                    unit = _glyph_detect_damage_unit_v705(
                        glyph_store, image, (bx1, by1, bx2, by2)
                    )
                    if unit:
                        # OCR이 단위 글자를 trailing 숫자로 오인식한 부분 제거
                        clean_num = _strip_ocr_unit_garbage(cur)
                        ct[key] = clean_num + unit
                    elif re.match(r'^\d[\d,]*\.\d{2}\d*$', cur):
                        # 글리프 단위 감지 실패 + 소수점 2자리 패턴 → 값 범위로 억 추정
                        # 이 게임의 실전 억 단위 피해량은 보통 0.5~9999.99억 범위
                        clean_num = _strip_ocr_unit_garbage(cur)
                        try:
                            val = float(clean_num.replace(",", ""))
                            if 0.5 <= val <= 9999.99:
                                ct[key] = clean_num + "억"
                        except Exception:
                            pass

        # ---- 3단계: 행 dict 조립 ----
        rows: List[Dict[str, Any]] = []
        for st in row_states:
            i = st["idx"]; rinfo = st["rinfo"]; cell_texts = st["cell_texts"]
            row: Dict[str, Any] = {col: "" for col in STANDARD_COLUMNS}
            row["_ocr_row_index"] = str(i)
            row["_row_source"] = rinfo.get("source", "")
            row["_direction_kind"] = direction_kind
            if window_box:
                row["_window_x1"], row["_window_y1"], row["_window_x2"], row["_window_y2"] = window_box
            ix1, iy1, ix2, iy2 = rinfo["icon_box"]
            row["_icon_x1"], row["_icon_y1"], row["_icon_x2"], row["_icon_y2"] = ix1, iy1, ix2, iy2
            row["_row_x1"], row["_row_y1"], row["_row_x2"], row["_row_y2"] = rinfo["row_box"]

            directional_rate = ""
            directional_share = ""
            for key, label, *_ in st["col_boxes"]:
                val = _postprocess_cell_value_v705(key, cell_texts.get(key, ""))
                if key == "directional_rate":
                    directional_rate = val
                elif key == "directional_share":
                    directional_share = val
                else:
                    row[label] = val

            if direction_kind == "head":
                row["헤드어택 적중률"] = directional_rate
                row["헤드어택 비중"] = directional_share
            else:
                row["백어택 적중률"] = directional_rate
                row["백어택 비중"] = directional_share

            meaningful = False
            for col in [c for c in STANDARD_COLUMNS if c not in {"이름", "초당 피해량", "피해량 지분"}]:
                v = str(row.get(col) or "").strip()
                if v and v not in ["-", "_", "None", "nan"]:
                    meaningful = True
                    break
            if meaningful or str(row.get("이름") or "").strip():
                rows.append(row)

        cols = STANDARD_COLUMNS + [
            "_ocr_row_index", "_row_source", "_direction_kind",
            "_window_x1", "_window_y1", "_window_x2", "_window_y2",
            "_icon_x1", "_icon_y1", "_icon_x2", "_icon_y2",
            "_row_x1", "_row_y1", "_row_x2", "_row_y2",
        ]
        df = pd.DataFrame(rows, columns=cols)
        if not df.empty:
            check_cols = [c for c in STANDARD_COLUMNS if c not in {"초당 피해량", "피해량 지분"}]
            nonempty = df[check_cols].astype(str).apply(lambda r: any(x.strip() and x.strip() not in ["", "-", "_", "None", "nan"] for x in r), axis=1)
            df = df[nonempty].reset_index(drop=True)

        # 표 전체가 비정상적으로 적게 잡히면 안전하게 기존 방식으로 전체 폴백
        if df.empty:
            return _old_parse_attack_fixed_grid_v38_for_v705(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)
        return df
    except Exception:
        # 어떤 예외든 기존 검증된 경로로 폴백
        return _old_parse_attack_fixed_grid_v38_for_v705(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)


# ==============================================================================
# v70.5-profile: OCR 성능 프로파일러 (느린 원인 진단용)
# ==============================================================================
# 실제 분석 경로(parse_summary_fixed_grid / parse_attack_fixed_grid)를 그대로 한 번
# 돌리면서 단계별 소요시간 + EasyOCR 호출 수/총시간 + 글리프 채택률 + 실행환경(CPU/GPU)을
# 수집합니다. 평소엔 OCR_PROFILE_V705=False 라 오버헤드가 없고, run_ocr_profile()가
# 잠깐만 켰다 끕니다.
import time as _time_v705

OCR_PROFILE_V705 = False                       # 프로파일 토글(run_ocr_profile가 제어)
_OCR_PROFILE_DATA_V705: Dict[str, Any] = {}     # {"phases": {name: sec}, "easyocr_calls": n, "easyocr_time": sec}


def _prof_now():
    """프로파일링이 켜져 있으면 현재 시각, 아니면 None(=오버헤드 0)."""
    return _time_v705.perf_counter() if OCR_PROFILE_V705 else None


def _prof_add(name: str, t0) -> None:
    """t0(_prof_now 반환)부터 지금까지를 단계 name에 누적."""
    if t0 is None:
        return
    dt = _time_v705.perf_counter() - t0
    ph = _OCR_PROFILE_DATA_V705.setdefault("phases", {})
    ph[name] = ph.get(name, 0.0) + dt
    try:
        if globals().get("_PERF_TRACE_ENABLED_V83", False):
            globals().get("_perf_trace_event_v83")("phase:" + str(name), elapsed_ms=round(dt * 1000.0, 3))
    except Exception:
        pass


def _prof_reset() -> None:
    _OCR_PROFILE_DATA_V705.clear()
    _OCR_PROFILE_DATA_V705.update({"phases": {}, "easyocr_calls": 0, "easyocr_time": 0.0})


class _ProfReader:
    """EasyOCR reader 래퍼: 모든 readtext 호출의 횟수·시간을 자동 집계합니다.

    parse_* 에 이 래퍼를 넘기면 내부 어디서 readtext를 부르든 전부 잡힙니다.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def readtext(self, *args: Any, **kwargs: Any) -> Any:
        t0 = _time_v705.perf_counter()
        try:
            return self._inner.readtext(*args, **kwargs)
        finally:
            dt = _time_v705.perf_counter() - t0
            _OCR_PROFILE_DATA_V705["easyocr_calls"] = _OCR_PROFILE_DATA_V705.get("easyocr_calls", 0) + 1
            _OCR_PROFILE_DATA_V705["easyocr_time"] = _OCR_PROFILE_DATA_V705.get("easyocr_time", 0.0) + dt
            try:
                shape = ""
                if args:
                    arr0 = args[0]
                    shape = str(getattr(arr0, "shape", ""))
                if globals().get("_PERF_TRACE_ENABLED_V83", False):
                    globals().get("_perf_trace_event_v83")(
                        "readtext", elapsed_ms=round(dt * 1000.0, 3), shape=shape, kwargs=list(kwargs.keys())
                    )
            except Exception:
                pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _profile_env() -> Dict[str, Any]:
    """실행 환경: torch CUDA 가용 여부(=GPU/CPU), cv2/글리프 가용, CPU 코어 수."""
    import os as _os
    env: Dict[str, Any] = {}
    try:
        import torch  # type: ignore
        env["torch_version"] = str(getattr(torch, "__version__", "?"))
        cuda = bool(torch.cuda.is_available())
        env["cuda_available"] = cuda
        env["device"] = "GPU" if cuda else "CPU"
        env["cuda_device_name"] = torch.cuda.get_device_name(0) if cuda else None
    except Exception as e:  # torch 미설치/오류
        env["torch_version"] = f"error: {e!r}"
        env["cuda_available"] = None
        env["device"] = "unknown"
    try:
        import cv2 as _cv2_probe  # noqa: F401
        env["cv2_available"] = True
    except Exception:
        env["cv2_available"] = False
    store = _get_glyph_store_v705()
    env["glyph_store_loaded"] = store is not None
    env["glyph_toggle_on"] = bool(OCR_GLYPH_FAST_V705)
    env["glyph_gate_percent"] = GLYPH_GATE_PERCENT
    env["glyph_gate_count"] = GLYPH_GATE_COUNT
    env["cpu_count"] = _os.cpu_count()
    return env


def run_ocr_profile(
    summary_image: Image.Image | None,
    attack_image: Image.Image | None,
    reader: Any,
    *,
    row_count: int | None = None,
    scale: int = 7,
) -> Dict[str, Any]:
    """실제 분석 경로를 1회 실행하며 성능 프로파일을 수집해 dict로 반환합니다.

    반환 예:
      {
        "env": {device: "CPU"/"GPU", cuda_available, glyph_*...},
        "total_sec": 62.1,
        "summary_total_sec": 21.3, "attack_total_sec": 40.8,
        "easyocr_calls": 47, "easyocr_total_sec": 55.9,   # ← 보통 여기 대부분이 몰림
        "phases_sec": {attack_1_window_detect: .., attack_4_rowpass_easyocr: .., ...},
        "glyph": {summary: {attempt, accept}, attack: {attempt, accept, rate}},
        "verdict": "사람이 읽는 한 줄 진단",
      }
    """
    global OCR_PROFILE_V705
    _prof_reset()
    rep: Dict[str, Any] = {"env": _profile_env()}
    pr = _ProfReader(reader)
    OCR_PROFILE_V705 = True
    t_all = _time_v705.perf_counter()
    try:
        if summary_image is not None:
            t0 = _time_v705.perf_counter()
            try:
                parse_summary_fixed_grid(summary_image, pr, scale=scale)
            except Exception as e:
                rep["summary_error"] = repr(e)
            rep["summary_total_sec"] = round(_time_v705.perf_counter() - t0, 3)
        if attack_image is not None:
            t0 = _time_v705.perf_counter()
            try:
                df = parse_attack_fixed_grid(attack_image, pr, row_count=row_count, scale=scale)
                rep["attack_rows"] = 0 if df is None else int(len(df))
            except Exception as e:
                rep["attack_error"] = repr(e)
            rep["attack_total_sec"] = round(_time_v705.perf_counter() - t0, 3)
    finally:
        OCR_PROFILE_V705 = False
    rep["total_sec"] = round(_time_v705.perf_counter() - t_all, 3)
    rep["easyocr_calls"] = int(_OCR_PROFILE_DATA_V705.get("easyocr_calls", 0))
    rep["easyocr_total_sec"] = round(float(_OCR_PROFILE_DATA_V705.get("easyocr_time", 0.0)), 3)
    rep["phases_sec"] = {k: round(v, 3) for k, v in sorted(_OCR_PROFILE_DATA_V705.get("phases", {}).items())}
    rep["glyph"] = {
        "summary": dict(_GLYPH_LAST_STATS_V705.get("summary") or {}),
        "attack": {k: _GLYPH_LAST_STATS_V705.get(k) for k in ("attempt", "accept", "rate")},
    }
    # 한 줄 진단
    total = rep["total_sec"] or 0.0
    easy = rep["easyocr_total_sec"] or 0.0
    share = (easy / total * 100.0) if total > 0 else 0.0
    dev = rep["env"].get("device")
    bits = [f"총 {total:.1f}s 중 EasyOCR가 {easy:.1f}s({share:.0f}%), 호출 {rep['easyocr_calls']}회"]
    if dev == "CPU":
        bits.append("EasyOCR가 CPU로 동작 중 → GPU(CUDA) 전환이 가장 큰 속도 개선")
    elif dev == "GPU":
        bits.append("GPU 사용 중")
    if not rep["env"].get("glyph_store_loaded"):
        bits.append("글리프 템플릿 미로드(폴백만 동작) → data/glyph_templates.json 경로 확인")
    elif not rep["env"].get("glyph_toggle_on"):
        bits.append("글리프 토글 OFF 상태")
    rep["verdict"] = " · ".join(bits)
    return rep


# ==============================================================================
# v81: class identity icons moved into lostark_skill_db_output + faster icon matching
# ============================================================================== 
# 변경 요약
# - add_icon/class/mapping.csv 기준으로 정리된 아이덴티티 아이콘은 이제
#   add_icon/lostark_skill_db_output/<class_folder>/ 에 들어갑니다.
# - 로컬 스킬 DB 스캔은 png/jpg/jpeg/webp 를 모두 지원합니다.
# - 아이콘 매칭은 전체 후보에 무거운 정밀 비교를 하지 않고,
#   빠른 feature 비교로 top-k를 고른 뒤 top-k만 정밀 비교합니다.

_ICON_EXTENSIONS_V81 = {".png", ".jpg", ".jpeg", ".webp"}


def _iter_icon_files_v81(folder: Path) -> List[Path]:
    try:
        return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _ICON_EXTENSIONS_V81])
    except Exception:
        return []


def _load_class_folder_map_v81() -> Dict[str, str]:
    """class_folder_map.csv + all_skill_name_map.csv 기준 한글 클래스명→폴더명 맵."""
    mapping: Dict[str, str] = dict(globals().get("CLASS_NAME_TO_FOLDER_V80", {}) or {})
    try:
        for base in {_project_root_v32(), Path(__file__).resolve().parent.parent}:
            db_root = base / "add_icon" / "lostark_skill_db_output"
            if not db_root.is_dir():
                continue
            for csv_name in ("class_folder_map.csv", "all_skill_name_map.csv"):
                p = db_root / csv_name
                if not p.is_file():
                    continue
                try:
                    import csv as _csv_v81
                    with io.StringIO(_read_text_robust_v151(p)) as fh:
                        for row in _csv_v81.DictReader(fh):
                            cn = str(row.get("class_name_kr") or "").strip()
                            cf = str(row.get("class_folder") or "").strip()
                            if cn and cf:
                                mapping[cn] = cf
                except Exception as e:
                    print(f"[skill_db:v81] 클래스 폴더 맵 읽기 실패({p.name}): {e}")
            break
    except Exception:
        pass
    return mapping


CLASS_NAME_TO_FOLDER_V80 = _load_class_folder_map_v81()  # type: ignore[assignment]


def _scan_icon_dir_v80(  # type: ignore[override]
    folder: "Path",
    mapping: Dict[str, str],
    source: str,
    out: List[Dict[str, str]],
    skip_names: "set[str] | None" = None,
) -> None:
    """v81: 폴더 내 png/jpg/jpeg/webp 스캔. 이중 확장자(.png.png/.png.webp)도 매핑 보정."""
    skip = {s.lower() for s in (skip_names or set())}
    for p in _iter_icon_files_v81(folder):
        low = p.name.lower()
        if low in skip:
            continue
        name = mapping.get(low)
        if name is None:
            # vk_final_light.png.png → vk_final_light.png
            stem_low = p.stem.lower()
            if stem_low.endswith(tuple(_ICON_EXTENSIONS_V81)):
                name = mapping.get(stem_low)
        if name is None:
            # vk_light_knight.png.webp → vk_light_knight.png
            for ext in _ICON_EXTENSIONS_V81:
                if low.endswith(ext):
                    base_low = low[: -len(ext)]
                    name = mapping.get(base_low)
                    if name:
                        break
        if not name:
            name = p.stem
        out.append({"name": _canonical_display_name_v86(name), "icon": str(p.resolve()), "source": source})


@lru_cache(maxsize=60)
def _load_local_skill_db_candidates_v80(class_name_kr: str) -> List[Dict[str, str]]:  # type: ignore[override]
    """v81: lostark_skill_db_output/<class>/ 로컬 아이콘 후보를 빠르게 로드.

    일반 API 스킬 아이콘과 사용자가 정리한 아이덴티티 아이콘을 같은 클래스 폴더에서 함께 읽습니다.
    """
    class_name_kr = str(class_name_kr or "").strip()
    if not class_name_kr:
        return []
    folder_name = _load_class_folder_map_v81().get(class_name_kr)
    if not folder_name:
        print(f"[skill_db:v81] 미등록 클래스: {class_name_kr!r}")
        return []

    out: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    try:
        for base in {_project_root_v32(), Path(__file__).resolve().parent.parent}:
            db_root = base / "add_icon" / "lostark_skill_db_output"
            class_dir = db_root / folder_name
            if not class_dir.is_dir():
                continue

            name_map: Dict[str, str] = {}
            # 1순위: 클래스별 skill_name_map.csv, 2순위: 전체 all_skill_name_map.csv
            for csv_path in (class_dir / "skill_name_map.csv", db_root / "all_skill_name_map.csv"):
                if not csv_path.is_file():
                    continue
                try:
                    import csv as _csv_v81
                    with io.StringIO(_read_text_robust_v151(csv_path)) as fh:
                        for row in _csv_v81.DictReader(fh):
                            cf = str(row.get("class_folder") or folder_name).strip()
                            if cf and cf != folder_name:
                                continue
                            fn = str(row.get("file_name") or "").strip()
                            sn = str(row.get("skill_name_kr") or "").strip()
                            if fn and sn:
                                name_map[fn.lower()] = sn
                except Exception as csv_err:
                    print(f"[skill_db:v81] CSV 파싱 실패({csv_path.name}): {csv_err}")

            tmp: List[Dict[str, str]] = []
            _scan_icon_dir_v80(class_dir, name_map, "skill_local_db", tmp)
            for item in tmp:
                key = (str(item.get("name") or ""), str(item.get("icon") or ""))
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
            break
    except Exception as e:
        print(f"[skill_db:v81] 로컬 스킬 DB 로드 실패({class_name_kr}): {e}")

    if out:
        print(f"[skill_db:v81] {class_name_kr}({folder_name}) 로컬 스킬/아이덴티티 {len(out)}개 등록")
    return out


@lru_cache(maxsize=1)
def _local_icon_candidates_v32() -> List[Dict[str, str]]:  # type: ignore[override]
    """v81: add_icon 루트/common 공용 아이콘만 로드.

    class/ 아이덴티티 아이콘은 lostark_skill_db_output/<class> 로 정리해서
    _load_local_skill_db_candidates_v80(class_name) 쪽에서 클래스별로만 로드합니다.
    이렇게 해야 다른 직업 아이덴티티 아이콘이 현재 캐릭터 후보에 섞이지 않습니다.
    """
    out: List[Dict[str, str]] = []
    try:
        for base in {_project_root_v32(), Path(__file__).resolve().parent.parent}:
            add_dir = base / "add_icon"
            if not add_dir.is_dir():
                continue

            root_mapping: Dict[str, str] = dict(ADD_ICON_NAME_MAP_V45)
            root_mapping.update(_read_add_icon_mapping_file_v50(add_dir))
            _scan_icon_dir_v80(
                add_dir, root_mapping, "local_rune", out,
                skip_names={"battle time.jpg", "battle_time_template.png"},
            )

            common_dir = add_dir / "common"
            if common_dir.is_dir():
                common_mapping = _read_add_icon_mapping_file_v50(common_dir)
                _scan_icon_dir_v80(common_dir, common_mapping, "local_common", out)
                # common/ 하위 카테고리 폴더(paradise 낙원 아이템 등)도 공용으로 스캔합니다.
                # 각 폴더의 mapping.csv + 코드 매핑(ADD_ICON_COMMON_SUB_MAP_V150)을 함께 적용합니다.
                try:
                    for sub in sorted([d for d in common_dir.iterdir() if d.is_dir()]):
                        sub_mapping = dict(common_mapping)
                        sub_mapping.update(ADD_ICON_COMMON_SUB_MAP_V150)
                        sub_mapping.update(_read_add_icon_mapping_file_v50(sub))
                        _scan_icon_dir_v80(sub, sub_mapping, "local_common", out)
                except Exception as _sub_e:
                    print(f"[add_icon:v81] common 하위폴더 스캔 실패: {_sub_e}")
            break
    except Exception as e:
        print(f"[add_icon:v81] 공용 아이콘 로드 실패: {e}")
    if out:
        print(f"[add_icon:v81] 공용/룬 아이콘 {len(out)}개 등록")
    return out


def _as_float_array_v81(img: Image.Image, size: int = 64, margin_ratio: float = 0.06) -> "np.ndarray":
    base = _square_resize_for_icon(img, size=size, margin_ratio=margin_ratio)
    return np.asarray(base.convert("RGB"), dtype=np.float32) / 255.0


def _norm_vec_v81(arr: "np.ndarray") -> "np.ndarray":
    v = arr.astype(np.float32).reshape(-1)
    n = float(np.linalg.norm(v)) + 1e-6
    return v / n


def _corr01_v81(a: "np.ndarray", b: "np.ndarray") -> float:
    aa = a.astype(np.float32).reshape(-1)
    bb = b.astype(np.float32).reshape(-1)
    aa = aa - float(aa.mean())
    bb = bb - float(bb.mean())
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-6
    return max(0.0, min(1.0, (float(np.dot(aa, bb)) / denom + 1.0) / 2.0))


def _icon_fast_feature_v81(img: Image.Image | None) -> Dict[str, Any] | None:
    if img is None:
        return None
    try:
        rgb = _as_float_array_v81(img, size=64, margin_ratio=0.06)
        # 색/명암/엣지 특징을 작게 만들어 전체 후보를 빠르게 줄입니다.
        small = Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8)).resize((32, 32), Image.Resampling.BILINEAR)
        small_rgb = np.asarray(small.convert("RGB"), dtype=np.float32) / 255.0
        gray = rgb.mean(axis=2)
        gray_small = np.asarray(small.convert("L"), dtype=np.float32) / 255.0
        try:
            import cv2 as _cv2_v81
            g8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
            edge = _cv2_v81.Canny(g8, 48, 120).astype(np.float32) / 255.0
            hsv = _cv2_v81.cvtColor(np.clip(rgb * 255, 0, 255).astype(np.uint8), _cv2_v81.COLOR_RGB2HSV)
            hist = _cv2_v81.calcHist([hsv], [0, 1], None, [18, 12], [0, 180, 0, 256]).astype(np.float32).reshape(-1)
            hist = hist / (float(np.linalg.norm(hist)) + 1e-6)
        except Exception:
            gy, gx = np.gradient(gray)
            edge = np.sqrt(gx * gx + gy * gy).astype(np.float32)
            edge = edge / (float(edge.max()) + 1e-6)
            hist = np.histogramdd(rgb.reshape(-1, 3), bins=(6, 6, 6), range=((0, 1), (0, 1), (0, 1)))[0].astype(np.float32).reshape(-1)
            hist = hist / (float(np.linalg.norm(hist)) + 1e-6)
        return {
            "rgb_vec": _norm_vec_v81(small_rgb),
            "gray": gray_small,
            "edge": edge,
            "hist": hist,
        }
    except Exception:
        return None


@lru_cache(maxsize=4096)
def _candidate_icon_fast_feature_v81(icon: str) -> Dict[str, Any] | None:
    return _icon_fast_feature_v81(_download_icon_image(icon))


def _fast_icon_score_v81(row_feat: Dict[str, Any] | None, cand_feat: Dict[str, Any] | None) -> float:
    if row_feat is None or cand_feat is None:
        return 0.0
    try:
        rgb_score = max(0.0, min(1.0, float(np.dot(row_feat["rgb_vec"], cand_feat["rgb_vec"]))))
        gray_score = _corr01_v81(row_feat["gray"], cand_feat["gray"])
        edge_score = _corr01_v81(row_feat["edge"], cand_feat["edge"])
        hist_score = max(0.0, min(1.0, float(np.dot(row_feat["hist"], cand_feat["hist"]))))
        return 0.32 * rgb_score + 0.34 * gray_score + 0.18 * edge_score + 0.16 * hist_score
    except Exception:
        return 0.0


def _rank_icon_candidates_v40(  # type: ignore[override]
    row_icon: Image.Image | None,
    skill_candidates: List[Any],
    *,
    ocr_name: Any = "",
    topn: int = 8,
) -> List[Dict[str, Any]]:
    """v81: fast feature로 후보를 줄이고 top-k만 정밀 아이콘 비교합니다."""
    pairs = _candidate_name_icon_pairs(skill_candidates or [])
    original = str(ocr_name or "").strip()
    norm_original = normalize_skill_name_for_match(original)
    row_feat = _icon_fast_feature_v81(row_icon) if row_icon is not None else None
    ranked: List[Dict[str, Any]] = []

    for pair in pairs:
        name = pair.get("name") or ""
        source = pair.get("source") or ""
        icon = pair.get("icon") or ""
        group = _source_group_v39(source)
        norm_name = normalize_skill_name_for_match(name)
        text_score = 0.0
        if norm_original and norm_name:
            text_score = SequenceMatcher(None, norm_original, norm_name).ratio()
            if norm_original in norm_name or norm_name in norm_original:
                text_score = max(text_score, 0.88)
        fast_score = 0.0
        if row_icon is not None and icon:
            fast_score = _fast_icon_score_v81(row_feat, _candidate_icon_fast_feature_v81(str(icon)))
        combined_fast = 0.08 * text_score + 0.92 * fast_score if group == "skill" else 0.22 * text_score + 0.78 * fast_score
        ranked.append({
            "name": str(name),
            "source": str(source),
            "group": group,
            "icon": str(icon),
            "text_score": round(float(text_score) * 100.0, 2),
            "icon_score": round(float(fast_score) * 100.0, 2),
            "fast_icon_score": round(float(fast_score) * 100.0, 2),
            "combined": round(float(combined_fast) * 100.0, 2),
        })

    if row_icon is not None and ranked:
        # 1차 fast 점수로 줄인 뒤에만 기존 정밀 비교(SSIM/히스토그램/template matching)를 수행합니다.
        try:
            import os as _os_v81
            heavy_k = int(_os_v81.environ.get("LOA_ICON_HEAVY_TOPK", "5"))
        except Exception:
            heavy_k = 5
        heavy_k = max(int(topn), min(len(ranked), max(int(topn), min(heavy_k, 40))))
        shortlist = sorted(ranked, key=lambda x: (x["icon_score"], x["combined"], x["text_score"]), reverse=True)[:heavy_k]
        shortlist_ids = {id(x) for x in shortlist}
        for item in ranked:
            if id(item) not in shortlist_ids:
                continue
            icon = item.get("icon") or ""
            if not icon:
                continue
            try:
                cand_img = _download_icon_image(str(icon))
                if cand_img is not None:
                    precise = _icon_similarity(row_icon, cand_img)
                    item["icon_score"] = round(float(precise) * 100.0, 2)
                    # 정밀 점수 반영 후 combined 재계산
                    ts = float(item.get("text_score") or 0.0) / 100.0
                    group = item.get("group")
                    comb = 0.08 * ts + 0.92 * float(precise) if group == "skill" else 0.22 * ts + 0.78 * float(precise)
                    item["combined"] = round(comb * 100.0, 2)
            except Exception:
                pass

    ranked.sort(key=lambda x: (x["icon_score"], x["combined"], x["text_score"]), reverse=True)
    return ranked[: int(topn)]

# v82: 파일명 공백/언더스코어 차이 보정 (Bleed rune.png ↔ bleed_rune.png)
def _icon_key_loose_v82(value: Any) -> str:
    s = str(value or "").strip().lower()
    return re.sub(r"[\s_\-]+", "", s)


def _scan_icon_dir_v80(  # type: ignore[override]
    folder: "Path",
    mapping: Dict[str, str],
    source: str,
    out: List[Dict[str, str]],
    skip_names: "set[str] | None" = None,
) -> None:
    """v82: png/jpg/jpeg/webp + 이중확장자 + 공백/언더스코어 파일명 차이 보정."""
    skip = {s.lower() for s in (skip_names or set())}
    loose_map = {_icon_key_loose_v82(k): v for k, v in (mapping or {}).items()}
    for p in _iter_icon_files_v81(folder):
        low = p.name.lower()
        if low in skip:
            continue
        name = mapping.get(low)
        candidate_keys = [low, p.stem.lower()]
        # vk_final_light.png.png → vk_final_light.png / vk_light_knight.png.webp → vk_light_knight.png
        for ext in _ICON_EXTENSIONS_V81:
            if low.endswith(ext):
                candidate_keys.append(low[: -len(ext)])
        for key in candidate_keys:
            if not name:
                name = mapping.get(key)
            if not name:
                name = loose_map.get(_icon_key_loose_v82(key))
        if not name:
            name = p.stem
        out.append({"name": _canonical_display_name_v86(name), "icon": str(p.resolve()), "source": source})


# ==============================================================================
# v83: 성능 타임라인 로그 + 아이콘 매칭 병목 진단/1회 랭킹 최적화
# ============================================================================== 
# 디버그 도구에서 OCR/아이콘 매칭 전체 흐름을 timestamp 단위로 남기기 위한 경량 trace.
# 일반 실행에서는 _PERF_TRACE_ENABLED_V83=False라 오버헤드가 없습니다.
_PERF_TRACE_ENABLED_V83 = False
_PERF_TRACE_EVENTS_V83: List[Dict[str, Any]] = []
_PERF_TRACE_T0_V83: float | None = None
_PERF_TRACE_LABEL_V83 = ""


def _perf_trace_wall_v83() -> str:
    try:
        from datetime import datetime as _dt_v83
        return _dt_v83.now().isoformat(timespec="milliseconds")
    except Exception:
        return ""


def _perf_trace_event_v83(stage: str, detail: str = "", **data: Any) -> None:
    if not globals().get("_PERF_TRACE_ENABLED_V83", False):
        return
    try:
        import time as _time_v83
        now = _time_v83.perf_counter()
        t0 = globals().get("_PERF_TRACE_T0_V83") or now
        item: Dict[str, Any] = {
            "idx": len(_PERF_TRACE_EVENTS_V83) + 1,
            "timestamp": _perf_trace_wall_v83(),
            "elapsed_ms": round((now - t0) * 1000.0, 3),
            "stage": str(stage),
            "detail": str(detail or ""),
        }
        for k, v in (data or {}).items():
            try:
                if isinstance(v, (str, int, float, bool)) or v is None:
                    item[str(k)] = v
                else:
                    item[str(k)] = str(v)
            except Exception:
                pass
        _PERF_TRACE_EVENTS_V83.append(item)
    except Exception:
        pass


def perf_trace_reset(label: str = "") -> None:
    """성능 타임라인 로그를 새로 시작합니다."""
    global _PERF_TRACE_ENABLED_V83, _PERF_TRACE_EVENTS_V83, _PERF_TRACE_T0_V83, _PERF_TRACE_LABEL_V83
    import time as _time_v83
    _PERF_TRACE_ENABLED_V83 = True
    _PERF_TRACE_EVENTS_V83 = []
    _PERF_TRACE_T0_V83 = _time_v83.perf_counter()
    _PERF_TRACE_LABEL_V83 = str(label or "")
    _perf_trace_event_v83("trace_start", label=_PERF_TRACE_LABEL_V83)


def perf_trace_event(stage: str, detail: str = "", **data: Any) -> None:
    _perf_trace_event_v83(stage, detail, **data)


def perf_trace_stop() -> None:
    global _PERF_TRACE_ENABLED_V83
    _perf_trace_event_v83("trace_stop")
    _PERF_TRACE_ENABLED_V83 = False


def perf_wrap_reader(reader: Any) -> Any:
    """readtext 호출 수/시간을 잡는 reader 래퍼를 반환합니다."""
    try:
        return _ProfReader(reader)
    except Exception:
        return reader


def perf_trace_snapshot() -> Dict[str, Any]:
    """현재 프로파일/타임라인 상태를 JSON 직렬화 가능한 dict로 반환합니다."""
    phases = dict(_OCR_PROFILE_DATA_V705.get("phases", {}) or {})
    total_elapsed = 0.0
    try:
        if _PERF_TRACE_T0_V83 is not None:
            import time as _time_v83
            total_elapsed = _time_v83.perf_counter() - _PERF_TRACE_T0_V83
    except Exception:
        pass
    return {
        "label": _PERF_TRACE_LABEL_V83,
        "total_elapsed_sec": round(total_elapsed, 3),
        "events": list(_PERF_TRACE_EVENTS_V83),
        "phases_sec": {str(k): round(float(v), 6) for k, v in phases.items()},
        "easyocr_calls": int(_OCR_PROFILE_DATA_V705.get("easyocr_calls", 0) or 0),
        "easyocr_total_sec": round(float(_OCR_PROFILE_DATA_V705.get("easyocr_time", 0.0) or 0.0), 6),
        "glyph": {
            "summary": dict(_GLYPH_LAST_STATS_V705.get("summary") or {}),
            "attack": {k: _GLYPH_LAST_STATS_V705.get(k) for k in ("attempt", "accept", "rate")},
        },
        "env": _profile_env(),
    }


def perf_profile_begin(label: str = "") -> Any:
    """OCR_PROFILE과 trace를 함께 켭니다. 반환값은 이전 OCR_PROFILE 상태."""
    global OCR_PROFILE_V705
    old = OCR_PROFILE_V705
    _prof_reset()
    perf_trace_reset(label=label)
    OCR_PROFILE_V705 = True
    return old


def perf_profile_end(old_profile_state: Any = False) -> Dict[str, Any]:
    """OCR_PROFILE/trace를 종료하고 snapshot을 반환합니다."""
    global OCR_PROFILE_V705
    OCR_PROFILE_V705 = bool(old_profile_state)
    perf_trace_stop()
    return perf_trace_snapshot()


# --- v83 아이콘 랭킹: feature/precise 단계별 시간 기록 ---
def _rank_icon_candidates_v40(  # type: ignore[override]
    row_icon: Image.Image | None,
    skill_candidates: List[Any],
    *,
    ocr_name: Any = "",
    topn: int = 8,
) -> List[Dict[str, Any]]:
    """v83: v81 fast feature + top-k 정밀 비교에 단계별 타이밍 로그를 추가합니다."""
    import time as _time_v83
    t_all = _time_v83.perf_counter()
    pairs = _candidate_name_icon_pairs(skill_candidates or [])
    original = str(ocr_name or "").strip()
    norm_original = normalize_skill_name_for_match(original)
    row_feat = _icon_fast_feature_v81(row_icon) if row_icon is not None else None
    ranked: List[Dict[str, Any]] = []

    t_feature = _time_v83.perf_counter()
    for pair in pairs:
        name = pair.get("name") or ""
        source = pair.get("source") or ""
        icon = pair.get("icon") or ""
        group = _source_group_v39(source)
        norm_name = normalize_skill_name_for_match(name)
        text_score = 0.0
        if norm_original and norm_name:
            text_score = SequenceMatcher(None, norm_original, norm_name).ratio()
            if norm_original in norm_name or norm_name in norm_original:
                text_score = max(text_score, 0.88)
        fast_score = 0.0
        if row_icon is not None and icon:
            fast_score = _fast_icon_score_v81(row_feat, _candidate_icon_fast_feature_v81(str(icon)))
        combined_fast = 0.08 * text_score + 0.92 * fast_score if group == "skill" else 0.22 * text_score + 0.78 * fast_score
        ranked.append({
            "name": str(name),
            "source": str(source),
            "group": group,
            "icon": str(icon),
            "text_score": round(float(text_score) * 100.0, 2),
            "icon_score": round(float(fast_score) * 100.0, 2),
            "fast_icon_score": round(float(fast_score) * 100.0, 2),
            "combined": round(float(combined_fast) * 100.0, 2),
        })
    feature_ms = (_time_v83.perf_counter() - t_feature) * 1000.0

    heavy_k = 0
    precise_ms = 0.0
    if row_icon is not None and ranked:
        try:
            import os as _os_v83
            heavy_k = int(_os_v83.environ.get("LOA_ICON_HEAVY_TOPK", "5"))
        except Exception:
            heavy_k = 5
        heavy_k = max(int(topn), min(len(ranked), max(int(topn), min(heavy_k, 40))))
        shortlist = sorted(ranked, key=lambda x: (x["icon_score"], x["combined"], x["text_score"]), reverse=True)[:heavy_k]
        shortlist_ids = {id(x) for x in shortlist}
        t_precise = _time_v83.perf_counter()
        for item in ranked:
            if id(item) not in shortlist_ids:
                continue
            icon = item.get("icon") or ""
            if not icon:
                continue
            try:
                cand_img = _download_icon_image(str(icon))
                if cand_img is not None:
                    precise = _icon_similarity(row_icon, cand_img)
                    item["icon_score"] = round(float(precise) * 100.0, 2)
                    ts = float(item.get("text_score") or 0.0) / 100.0
                    group = item.get("group")
                    comb = 0.08 * ts + 0.92 * float(precise) if group == "skill" else 0.22 * ts + 0.78 * float(precise)
                    item["combined"] = round(comb * 100.0, 2)
            except Exception:
                pass
        precise_ms = (_time_v83.perf_counter() - t_precise) * 1000.0

    ranked.sort(key=lambda x: (x["icon_score"], x["combined"], x["text_score"]), reverse=True)
    out = ranked[: int(topn)]
    total_ms = (_time_v83.perf_counter() - t_all) * 1000.0
    _perf_trace_event_v83(
        "icon_rank",
        candidates=len(pairs), topn=int(topn), heavy_k=heavy_k,
        feature_ms=round(feature_ms, 3), precise_ms=round(precise_ms, 3), total_ms=round(total_ms, 3),
        best=(out[0].get("name") if out else ""), best_score=(out[0].get("icon_score") if out else 0.0),
    )
    return out



def _canonical_display_name_v86(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return raw
    compact = re.sub(r"[\s:_\-]+", "", raw.lower())
    mapping = {
        "poisonrune": "스킬룬 중독",
        "poisonrune.png": "스킬룬 중독",
        "poison": "스킬룬 중독",
        "중독룬": "스킬룬 중독",
        "스킬룬중독": "스킬룬 중독",
        "스킬룬poison": "스킬룬 중독",
        "bleedrune": "스킬룬 출혈",
        "bleedrune.png": "스킬룬 출혈",
        "bleed": "스킬룬 출혈",
        "출혈룬": "스킬룬 출혈",
        "스킬룬출혈": "스킬룬 출혈",
        "스킬룬bleed": "스킬룬 출혈",
    }
    if compact in mapping:
        return mapping[compact]
    m = re.match(r"^스킬룬\s*[:：]?\s*(.+)$", raw)
    if m:
        return f"스킬룬 {m.group(1).strip()}"
    return raw

def _select_match_from_ranked_v83(
    original_value: Any,
    ranked: List[Dict[str, Any]],
    *,
    row_icon: Image.Image | None,
    text_threshold: float = 0.55,
    icon_threshold: float = 0.72,
) -> Tuple[str, float, float, bool, str]:
    """best_skill_name_match_with_icon와 같은 판정을 이미 계산된 ranked로 수행합니다."""
    original = str(original_value or "").strip()
    canonical = canonicalize_battle_name_v31(original)
    if not ranked:
        return canonical or original, 0.0, 0.0, bool(canonical and canonical != original), "canonical" if canonical != original else "no_candidates"

    skill_ranked = [r for r in ranked if r.get("group") == "skill"]
    if skill_ranked:
        if row_icon is not None:
            best_skill = skill_ranked[0]
            _best100 = float(best_skill.get("icon_score") or 0.0)
            _second100 = float(skill_ranked[1].get("icon_score") or 0.0) if len(skill_ranked) > 1 else 0.0
            _low_conf = (_best100 < ICON_MATCH_MIN_SCORE_V46) or (
                _best100 < ICON_MATCH_TIE_SCORE_V46 and (_best100 - _second100) < ICON_MATCH_TIE_MARGIN_V46
            )
            if _low_conf:
                return ("기타", 0.0, round(_best100, 2), True, f"icon_low_conf_etc:{best_skill.get('source', '')}")
            return str(best_skill.get("name") or original), 0.0, round(_best100, 2), True, f"icon_skill_only:{best_skill.get('source', '')}"
        best_skill = max(skill_ranked, key=lambda x: x.get("text_score", 0.0))
        return str(best_skill.get("name") or original), float(best_skill.get("text_score") or 0.0), 0.0, True, f"icon_skill_noicon:{best_skill.get('source', '')}"

    best = ranked[0]
    best_icon = float(best.get("icon_score") or 0.0) / 100.0
    best_text = float(best.get("text_score") or 0.0) / 100.0
    second_icon = float(ranked[1].get("icon_score") or 0.0) / 100.0 if len(ranked) > 1 else 0.0
    margin = best_icon - second_icon
    if best_icon >= float(icon_threshold) and (margin >= 0.035 or best_icon >= min(0.94, float(icon_threshold) + 0.12)):
        return str(best.get("name") or original), round(best_text * 100.0, 2), round(best_icon * 100.0, 2), True, f"icon_first:{best.get('source', '')}"

    text_best = max(ranked, key=lambda x: x.get("text_score", 0.0))
    text_score = float(text_best.get("text_score") or 0.0) / 100.0
    text_icon = float(text_best.get("icon_score") or 0.0) / 100.0
    if text_score >= float(text_threshold):
        return str(text_best.get("name") or original), round(text_score * 100.0, 2), round(text_icon * 100.0, 2), True, f"text_fallback:{text_best.get('source', '')}"
    if canonical and canonical != original:
        return canonical, round(text_score * 100.0, 2), round(best_icon * 100.0, 2), True, "canonical"
    return original, round(text_score * 100.0, 2), round(best_icon * 100.0, 2), False, "unmatched"


def correct_battle_skill_names_with_icons(  # type: ignore[override]
    df: pd.DataFrame,
    attack_image: Image.Image,
    skill_candidates: List[Dict[str, str]],
    *,
    name_col: str = "이름",
    threshold: float = 0.62,
    icon_threshold: float = 0.72,
    drop_unmatched: bool = False,
) -> pd.DataFrame:
    """v83: 행별 아이콘 랭킹을 1번만 수행하고, 단계별 시간을 trace에 남깁니다.

    v81 이전 경로는 top 후보 표시용 _rank와 실제 best 매칭용 _rank를 각각 호출해
    같은 행에서 후보 600개 비교를 2번 돌 수 있었습니다. 여기서는 1회 랭킹 결과를
    top 표시와 최종 판정에 같이 사용합니다.
    """
    if df is None or df.empty or name_col not in df.columns:
        return df
    import time as _time_v83
    t_total = _time_v83.perf_counter()
    out = df.copy()
    rows = []
    candidate_count = len(skill_candidates or [])
    _perf_trace_event_v83("icon_correct_start", rows=len(out), candidates=candidate_count)
    for idx, row in out.iterrows():
        row_t = _time_v83.perf_counter()
        try:
            row_index = int(float(row.get("_ocr_row_index", idx)))
        except Exception:
            row_index = int(idx)

        t_crop = _time_v83.perf_counter()
        icon_crop = _crop_icon_from_row_metadata(attack_image, row, row_index)
        crop_ms = (_time_v83.perf_counter() - t_crop) * 1000.0

        t_rank = _time_v83.perf_counter()
        top = _rank_icon_candidates_v40(icon_crop, skill_candidates or [], ocr_name=row.get(name_col), topn=5)
        rank_ms = (_time_v83.perf_counter() - t_rank) * 1000.0

        matched, text_score, icon_score, ok, reason = _select_match_from_ranked_v83(
            row.get(name_col),
            top,
            row_icon=icon_crop,
            text_threshold=threshold,
            icon_threshold=icon_threshold,
        )
        matched = _canonical_display_name_v86(matched)
        new_row = row.copy()
        if ok:
            new_row[name_col] = matched
        elif drop_unmatched:
            continue
        new_row["_name_match_text_score"] = text_score
        new_row["_name_match_icon_score"] = icon_score
        new_row["_name_match_reason"] = reason
        new_row["_icon_match_name"] = _canonical_display_name_v86(str(top[0]["name"])) if top else ""
        new_row["_icon_match_score"] = top[0]["icon_score"] if top else 0.0
        new_row["_icon_match_source"] = str(top[0]["source"]) if top else ""
        new_row["_icon_match_top3"] = " | ".join(f"{_canonical_display_name_v86(r['name'])}:{r['icon_score']}" for r in top[:3])
        rows.append(new_row)
        _perf_trace_event_v83(
            "icon_correct_row",
            row_index=row_index,
            crop_ms=round(crop_ms, 3),
            rank_ms=round(rank_ms, 3),
            row_total_ms=round((_time_v83.perf_counter() - row_t) * 1000.0, 3),
            matched=matched,
            icon_score=icon_score,
            reason=reason,
        )
    total_ms = (_time_v83.perf_counter() - t_total) * 1000.0
    _perf_trace_event_v83("icon_correct_end", rows=len(rows), total_ms=round(total_ms, 3))
    if not rows:
        return out.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)

# ==============================
# v83: 종합정보 OCR 경량화 override
# ==============================
# 종합정보 탭은 계산에 전투시간/총피해량/DPS 3개만 필요합니다.
# 이전 버전은 피해 증가 유효율, 무력화, 치명/백/헤드, 가동률까지 모두 OCR해서
# summary 단계에서 불필요한 readtext가 많이 발생했습니다.
# v83부터 summary 경로는 기본 3개 필드만 읽고, 창 감지도 OCR 없이 제목 템플릿/선 기반으로만 시도합니다.

SUMMARY_REQUIRED_KEYS_V83 = ("elapsed_text", "total_damage_text", "dps_text")
SUMMARY_LABELS_V83 = {
    "elapsed_text": "전투 시간",
    "total_damage_text": "총 피해량",
    "dps_text": "DPS",
}


def _base_window_box_v83(image: Image.Image) -> Tuple[int, int, int, int]:
    sx = image.width / BASE_SCREEN_W
    sy = image.height / BASE_SCREEN_H
    return (
        int(round(BASE_WINDOW_BOX[0] * sx)),
        int(round(BASE_WINDOW_BOX[1] * sy)),
        int(round(BASE_WINDOW_BOX[2] * sx)),
        int(round(BASE_WINDOW_BOX[3] * sy)),
    )


def _summary_window_box_fast_v83(image: Image.Image, auto_window: bool = True) -> Tuple[int, int, int, int]:
    """종합정보용 창 위치 탐지.

    reader를 넘기지 않아서 전체 화면 OCR을 하지 않습니다.
    1) 제목 이미지 템플릿 매칭
    2) 수평/수직선 감지
    3) 실패/이상치면 기본 1920x1080 비율 좌표
    """
    base_box = _base_window_box_v83(image)
    if not auto_window:
        return base_box
    try:
        box = detect_analyzer_window(image, None)
    except Exception:
        box = None
    if box is None:
        return base_box
    try:
        x1, y1, x2, y2 = [int(v) for v in box]
        bw, bh = x2 - x1, y2 - y1
        if bw < image.height * 0.90 or bh < image.height * 0.50:
            return base_box
        # 선 감지 fallback이 엉뚱한 위치를 물면 base와 크게 벌어지는 경우가 많습니다.
        if abs(x1 - base_box[0]) > image.width * 0.18 or abs(y1 - base_box[1]) > image.height * 0.12:
            return base_box
        return (x1, y1, x2, y2)
    except Exception:
        return base_box


def _ocr_summary_card_text_v83(reader: Any, raw_crop: Image.Image, *, scale: int = 6) -> Tuple[str, Image.Image, str, float]:
    """총피해량/DPS 카드 전용 저비용 OCR.

    기본은 1회 OCR만 수행합니다. 원시 숫자나 한국식 단위가 전혀 안 잡힐 때만
    numeric_mask 1회를 추가 시도합니다. 이전처럼 4개 전처리 후보를 매번 돌리지 않습니다.
    """
    proc = preprocess_crop(raw_crop, scale=max(3, min(int(scale or 6), 7)))
    try:
        text = _ocr_text_once(reader, proc, numeric=True)
    except Exception:
        text = ""
    score = 0.0
    try:
        score = _numeric_ocr_score(text, kind="korean_number")
    except Exception:
        score = float(len(re.findall(r"\d", str(text or ""))))

    # 총피해/DPS는 카드 하단의 전체 원시 숫자(쉼표 포함 긴 정수)를 읽는 게 가장 정확합니다.
    # 그 숫자가 안 잡히고 단위 숫자도 안 잡혔을 때만 mask 후보를 한 번 더 사용합니다.
    has_raw_integer = full_raw_number_from_summary(text) is not None
    has_korean_like = bool(re.search(r"\d[\d,]*(?:\.\d+)?\s*(?:조|억|만)", str(text or "")))
    if not has_raw_integer and not has_korean_like:
        try:
            alt_proc = _preprocess_numeric_text_mask(raw_crop, scale=max(5, min(int(scale or 6), 8)))
            alt_text = _ocr_text_once(reader, alt_proc, numeric=True)
            alt_score = _numeric_ocr_score(alt_text, kind="korean_number")
            if alt_score > score:
                return clean_ocr_text(alt_text), alt_proc, "numeric_mask_fallback", float(alt_score)
        except Exception:
            pass
    return clean_ocr_text(text), proc, "normal_fast", float(score)


def _read_elapsed_fast_v83(image: Image.Image, window_box: Tuple[int, int, int, int], reader: Any, *, scale: int = 6) -> Tuple[str, Image.Image | None, Image.Image | None, str, float]:
    """전투시간 전용 OCR.

    1순위는 기존 battle_time 템플릿으로 숫자 칸만 읽기.
    실패 시 elapsed_text ROI를 1회만 읽습니다.
    """
    try:
        txt = _read_elapsed_text_by_template_v46(image, window_box, reader)
        if txt:
            sec = extract_elapsed_seconds(txt)
            if sec is not None:
                return clean_ocr_text(txt), None, None, "battle_time_template", 100.0
    except Exception:
        pass

    raw_crop = crop_norm(image, SUMMARY_ROIS["elapsed_text"], pad=1, window_box=window_box)
    proc = preprocess_crop(raw_crop, scale=max(4, min(int(scale or 6), 7)))
    try:
        text = _ocr_text_once(reader, proc, numeric=True)
    except Exception:
        text = ""
    score = 50.0 if extract_elapsed_seconds(text) is not None else float(len(re.findall(r"\d", str(text or ""))))
    return clean_ocr_text(text), raw_crop, proc, "elapsed_roi_fast", score


def _format_elapsed_v83(seconds: Any) -> str:
    try:
        if seconds is None:
            return ""
        sec = int(float(seconds))
        return f"{sec // 60:02d}:{sec % 60:02d}"
    except Exception:
        return ""


def _format_summary_debug_text_v83(key: str, raw_text: str, parsed_value: Any) -> str:
    try:
        from modules.calculators import format_korean_number
    except Exception:
        format_korean_number = None  # type: ignore[assignment]
    if key == "elapsed_text":
        fmt = _format_elapsed_v83(parsed_value)
        return fmt or clean_ocr_text(raw_text)
    if key in {"total_damage_text", "dps_text"}:
        if parsed_value:
            try:
                return format_korean_number(parsed_value) if format_korean_number else str(parsed_value)
            except Exception:
                return str(parsed_value)
        return extract_korean_number_text(raw_text)
    return clean_ocr_text(raw_text)


def parse_summary_fixed_grid(image: Image.Image, reader: Any, scale: int = 6, auto_window: bool = True) -> Dict[str, Any]:  # type: ignore[override]
    """v83 종합정보 OCR: 전투시간/총피해량/DPS만 읽습니다."""
    _pt = _prof_now()
    window_box = _summary_window_box_fast_v83(image, auto_window=auto_window)
    _prof_add("summary_1_window_detect_fast_no_ocr", _pt)

    raw: Dict[str, str] = {}
    debug_extra: Dict[str, Any] = {}

    # 1) 전투시간: 템플릿 숫자칸 우선
    _pt_elapsed = _prof_now()
    elapsed_raw, elapsed_raw_crop, elapsed_proc_crop, elapsed_method, elapsed_score = _read_elapsed_fast_v83(image, window_box, reader, scale=scale)
    raw["elapsed_text"] = elapsed_raw
    debug_extra["elapsed"] = {"method": elapsed_method, "score": elapsed_score}
    _prof_add("summary_2_elapsed_only", _pt_elapsed)

    # 2) 총피해량 / DPS: 카드 2개만 OCR
    card_debug: Dict[str, Dict[str, Any]] = {}
    _pt_cards = _prof_now()
    for key in ("total_damage_text", "dps_text"):
        raw_crop = crop_norm(image, SUMMARY_ROIS[key], pad=1, window_box=window_box)
        text, proc, method, score = _ocr_summary_card_text_v83(reader, raw_crop, scale=scale)
        raw[key] = text
        card_debug[key] = {"raw_crop": raw_crop, "processed_crop": proc, "method": method, "score": score}
    _prof_add("summary_3_damage_dps_only", _pt_cards)

    try:
        from modules.calculators import parse_korean_number, format_korean_number
    except Exception:
        from .calculators import parse_korean_number, format_korean_number  # type: ignore

    total_raw_num = full_raw_number_from_summary(raw.get("total_damage_text", ""))
    dps_raw_num = full_raw_number_from_summary(raw.get("dps_text", ""))

    total_value = total_raw_num if total_raw_num is not None else parse_korean_number(extract_korean_number_text(raw.get("total_damage_text", "")))
    dps_value = dps_raw_num if dps_raw_num is not None else parse_korean_number(extract_korean_number_text(raw.get("dps_text", "")))
    elapsed_seconds = extract_elapsed_seconds(raw.get("elapsed_text", ""))

    # 안 읽는 필드는 None으로 둡니다. 이전처럼 raw에 빈 OCR 결과를 넣지 않습니다.
    meta: Dict[str, Any] = {
        "raw": raw,
        "window_box": window_box,
        "window_detected": window_box is not None,
        "summary_fast_v83": True,
        "summary_read_keys": list(SUMMARY_REQUIRED_KEYS_V83),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_source": "ocr" if elapsed_seconds is not None else "ocr_failed",
        "total_damage": total_value,
        "dps": dps_value,
        "total_damage_text": format_korean_number(total_value) if total_value else extract_korean_number_text(raw.get("total_damage_text", "")),
        "dps_text": format_korean_number(dps_value) if dps_value else extract_korean_number_text(raw.get("dps_text", "")),
        "damage_increase_efficiency": None,
        "crit_rate": None,
        "head_attack_rate": None,
        "back_attack_rate": None,
        "damage_increase_uptime": None,
        "burst_uptime": None,
        "_debug_v83": {"elapsed": debug_extra.get("elapsed"), "cards": {k: {kk: vv for kk, vv in v.items() if kk not in {"raw_crop", "processed_crop"}} for k, v in card_debug.items()}},
    }

    # 화면 디버그 표에서 05::043 / 4,641.269 464...처럼 보이는 대신 최종 해석값도 바로 보이게 raw_display 제공.
    meta["raw_display"] = {
        "elapsed_text": _format_summary_debug_text_v83("elapsed_text", raw.get("elapsed_text", ""), elapsed_seconds),
        "total_damage_text": _format_summary_debug_text_v83("total_damage_text", raw.get("total_damage_text", ""), total_value),
        "dps_text": _format_summary_debug_text_v83("dps_text", raw.get("dps_text", ""), dps_value),
    }
    return meta


def make_summary_ocr_debug(image: Image.Image, reader: Any, scale: int = 6) -> List[Dict[str, Any]]:  # type: ignore[override]
    """v83 종합정보 디버그: 필요한 3개 필드만 crop/OCR합니다."""
    window_box = _summary_window_box_fast_v83(image, auto_window=True)
    rows: List[Dict[str, Any]] = []

    # 전투시간
    elapsed_raw, elapsed_raw_crop, elapsed_proc_crop, elapsed_method, elapsed_score = _read_elapsed_fast_v83(image, window_box, reader, scale=scale)
    elapsed_seconds = extract_elapsed_seconds(elapsed_raw)
    if elapsed_raw_crop is None:
        # 템플릿 경로는 내부 crop을 반환하지 않으므로, 디버그 표시용으로 기존 ROI crop을 보여줍니다.
        elapsed_raw_crop = crop_norm(image, SUMMARY_ROIS["elapsed_text"], pad=1, window_box=window_box)
        elapsed_proc_crop = preprocess_crop(elapsed_raw_crop, scale=max(4, min(int(scale or 6), 7)))
    rows.append({
        "key": "elapsed_text",
        "label": SUMMARY_LABELS_V83["elapsed_text"],
        "text": _format_summary_debug_text_v83("elapsed_text", elapsed_raw, elapsed_seconds),
        "raw_text": elapsed_raw,
        "raw_crop": elapsed_raw_crop,
        "processed_crop": elapsed_proc_crop,
        "window_box": window_box,
        "preprocess": elapsed_method,
        "score": elapsed_score,
    })

    # 총피해량 / DPS
    try:
        from modules.calculators import parse_korean_number
    except Exception:
        from .calculators import parse_korean_number  # type: ignore
    for key in ("total_damage_text", "dps_text"):
        raw_crop = crop_norm(image, SUMMARY_ROIS[key], pad=1, window_box=window_box)
        text, proc, method, score = _ocr_summary_card_text_v83(reader, raw_crop, scale=scale)
        raw_num = full_raw_number_from_summary(text)
        parsed = raw_num if raw_num is not None else parse_korean_number(extract_korean_number_text(text))
        rows.append({
            "key": key,
            "label": SUMMARY_LABELS_V83[key],
            "text": _format_summary_debug_text_v83(key, text, parsed),
            "raw_text": text,
            "raw_crop": raw_crop,
            "processed_crop": proc,
            "window_box": window_box,
            "preprocess": method,
            "score": score,
        })
    return rows


def apply_summary_overrides(summary: Dict[str, Any], meta: Dict[str, Any] | None) -> Dict[str, Any]:  # type: ignore[override]
    """v83: 종합정보에서 읽은 전투시간/총피해량/DPS만 결과 요약에 반영합니다."""
    if not meta:
        return summary
    out = dict(summary)
    if meta.get("total_damage"):
        out["total_damage"] = meta["total_damage"]
    if meta.get("dps"):
        out["dps"] = meta["dps"]
    # 치명/백/헤드 등은 이제 종합정보에서 읽지 않습니다. 공격정보 표의 스킬별 값 또는 수동 입력을 사용합니다.
    return out


# ==============================================================================
# v90: 빠른 공격정보 OCR + 공용 룬 아이콘 보존
# ==============================================================================
# 병목 로그 기준: attack_6_recheck_easyocr가 19초 이상, readtext 150회 발생.
# 원인: 행 OCR이 비어 보이는 칸마다 전부 셀별 OCR을 다시 돌렸기 때문입니다.
# v90은 이름/DPS/지분/빈 퍼센트칸 재검수를 생략하고,
# 실제 계산에 꼭 필요한 피해량 + 실패한 횟수만 저배율 단일 OCR로 보정합니다.


def _v90_text_has_damage_like(text: Any) -> bool:
    s = clean_ocr_text(str(text or ""))
    if not re.search(r"\d", s):
        return False
    # 전투분석기 피해량은 대부분 123.45억/만 또는 원시 쉼표 숫자입니다.
    # 단위가 이미 있거나, 소수점/쉼표가 있으면 행 단위 OCR 결과를 우선 신뢰합니다.
    if re.search(r"\d[\d,]*(?:\.\d+)?\s*(?:조|억|만)", s):
        return True
    if re.search(r"\d+\.\d+", s):
        return True
    if re.search(r"\d{1,3}(?:,\d{3})+", s):
        return True
    return False


def _v90_text_has_count_like(text: Any) -> bool:
    s = clean_ocr_text(str(text or ""))
    if not s:
        return False
    if re.search(r"\d", s):
        return True
    # 헤더/찌꺼기 값은 False
    return False


def _ocr_cell_fast_v90(reader: Any, image: Image.Image, box: Tuple[int, int, int, int], key: str, scale: int = 4) -> str:
    """v90 저비용 셀 OCR. 기존 정밀 OCR 2후보 대신 1후보만 사용합니다."""
    raw = _crop_pixel_box(image, box, pad=1, expand_ratio=CELL_CROP_EXPAND_RATIO)
    numeric = key != "name"
    # CPU/RapidOCR에서 폭이 큰 셀을 7배 이상 키우면 매우 느려집니다. 4배로 제한합니다.
    use_scale = max(2, min(int(scale or 4), 4))
    try:
        if numeric:
            proc = _preprocess_numeric_text_mask(raw, scale=use_scale)
        else:
            proc = preprocess_crop(raw, scale=use_scale)
        t = _ocr_text_once(reader, proc, numeric=numeric)
    except Exception:
        t = ""
    val = _postprocess_cell_value_v705(key, t)
    # 피해량은 값이 비면 정확도가 더 중요하므로 기존 정밀 OCR을 저배율로 1회 폴백합니다.
    if key == "damage_text" and not _v90_text_has_damage_like(val):
        try:
            val = _ocr_cell_precise_v705(reader, image, box, key, scale=5)
        except Exception:
            pass
    return val


# 기존 선택 함수 백업 후, 공용 룬/기타 아이콘이 확실한 경우 스킬 후보보다 먼저 채택합니다.
_select_match_from_ranked_v83_prev_v90 = globals().get("_select_match_from_ranked_v83")


def _v90_is_common_candidate(item: Dict[str, Any] | None) -> bool:
    if not item:
        return False
    name = _canonical_display_name_v86(item.get("name"))
    source = str(item.get("source") or "").lower()
    group = str(item.get("group") or "").lower()
    if name.startswith("스킬룬") or name in {"기본 공격", "기타"}:
        return True
    if "rune" in source or "local_common" in source or "local_rune" in source:
        return True
    if group in {"rune", "manual"}:
        return True
    return False


def _select_match_from_ranked_v83(  # type: ignore[override]
    original_value: Any,
    ranked: List[Dict[str, Any]],
    *,
    row_icon: Image.Image | None,
    text_threshold: float = 0.55,
    icon_threshold: float = 0.72,
) -> Tuple[str, float, float, bool, str]:
    original = str(original_value or "").strip()
    canonical = canonicalize_battle_name_v31(original)
    if not ranked:
        return canonical or original, 0.0, 0.0, bool(canonical and canonical != original), "canonical" if canonical != original else "no_candidates"

    ranked = list(ranked)
    ranked.sort(key=lambda x: (float(x.get("icon_score") or 0.0), float(x.get("combined") or 0.0), float(x.get("text_score") or 0.0)), reverse=True)
    best = ranked[0]
    best_icon100 = float(best.get("icon_score") or 0.0)
    best_text100 = float(best.get("text_score") or 0.0)

    # 스킬룬/기본공격/기타 같은 공용 아이콘은 API 스킬 후보가 있어도, 아이콘이 확실하면 그대로 표시합니다.
    if row_icon is not None and _v90_is_common_candidate(best):
        best_skill_icon100 = 0.0
        for r in ranked:
            if str(r.get("group") or "") == "skill":
                best_skill_icon100 = float(r.get("icon_score") or 0.0)
                break
        # 공용 아이콘 점수가 충분히 높고, 스킬 후보보다 명확히 높으면 공용 후보 채택.
        if best_icon100 >= max(82.0, float(icon_threshold) * 100.0) and (best_icon100 - best_skill_icon100 >= 4.0 or best_icon100 >= 91.0):
            return _canonical_display_name_v86(best.get("name")), round(best_text100, 2), round(best_icon100, 2), True, f"icon_common:{best.get('source', '')}"

    # 나머지는 기존 v83 판정 로직으로 처리합니다.
    if _select_match_from_ranked_v83_prev_v90:
        return _select_match_from_ranked_v83_prev_v90(
            original_value,
            ranked,
            row_icon=row_icon,
            text_threshold=text_threshold,
            icon_threshold=icon_threshold,
        )
    return original, 0.0, best_icon100, False, "unmatched"


_old_parse_attack_fixed_grid_v90_fallback = globals().get("parse_attack_fixed_grid")


def parse_attack_fixed_grid(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int | None = None,
    scale: int = 7,
    auto_window: bool = True,
) -> pd.DataFrame:
    """v90 공격정보 OCR.

    - 행 단위 OCR 1회는 유지.
    - 재검수 OCR은 피해량이 불확실한 칸 + 횟수가 불확실한 칸만 수행.
    - 이름은 아이콘 매칭으로 확정하므로 OCR 재검수하지 않음.
    - DPS/지분은 종합정보 전투시간/총피해량으로 역산하므로 OCR 재검수하지 않음.
    """
    if not globals().get("OCR_FAST_ROW_PASS_V705", True):
        return _old_parse_attack_fixed_grid_v38_for_v705(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)

    grid = ATTACK_GRID
    if row_count is None:
        row_count = int(grid["row_count"])
    try:
        _pt = _prof_now()
        window_box = detect_analyzer_window(image, reader) if auto_window else None
        _prof_add("attack_1_window_detect", _pt)

        _pt = _prof_now()
        direction_kind = _detect_attack_direction_kind_v38(image, reader, window_box, scale=scale) if window_box else "back"
        _prof_add("attack_2_direction_detect", _pt)

        _pt = _prof_now()
        detected_rows = detect_attack_icon_rows(image, reader, window_box=window_box, max_rows=int(row_count)) if window_box else []
        _prof_add("attack_3_icon_rows_detect", _pt)
        if not detected_rows:
            detected_rows = []
            for i in range(int(row_count)):
                y1 = grid["row_start"] + i * grid["row_height"]
                y2 = y1 + grid["row_height"]
                row_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, grid["columns"][-1][3], y2), pad=0, window_box=window_box)
                icon_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, ATTACK_ICON_GRID["x2"], y2), pad=1, window_box=window_box)
                detected_rows.append({"row_index": i, "row_box": row_box, "icon_box": icon_box, "window_box": window_box, "source": "fixed_fallback"})

        row_states: List[Dict[str, Any]] = []
        row_scale = max(2, min(4, int(OCR_ROW_SCALE_V705)))
        _pt_rows = _prof_now()
        for i, rinfo in enumerate(detected_rows[: int(row_count)]):
            col_boxes: List[Tuple[str, str, int, int, int, int]] = []
            for key, label, x1, x2 in grid["columns"]:
                if window_box:
                    box = _column_pixel_box_for_row(image, window_box, rinfo["row_box"], x1, x2)
                else:
                    px1 = int(round(x1 * image.width)); px2 = int(round(x2 * image.width))
                    box = _clip_box(image, (px1, rinfo["row_box"][1], px2, rinfo["row_box"][3]))
                col_boxes.append((key, label, box[0], box[1], box[2], box[3]))
            cell_texts = _ocr_row_single_pass_v705(reader, image, col_boxes, row_scale)
            row_failed = sum(1 for v in cell_texts.values() if str(v).strip()) < 2
            row_states.append({"idx": i, "rinfo": rinfo, "col_boxes": col_boxes, "cell_texts": cell_texts, "row_failed": row_failed})
        _prof_add("attack_4_rowpass_easyocr", _pt_rows)

        # 글리프 빠른 인식: 퍼센트/횟수 칸 우선 채택
        _pt_glyph = _prof_now()
        glyph_confident: set[Tuple[int, str]] = set()
        glyph_attempt = 0
        glyph_accept = 0
        glyph_store = _get_glyph_store_v705() if OCR_GLYPH_FAST_V705 else None
        if glyph_store is not None:
            for st in row_states:
                ct = st["cell_texts"]
                for key, label, bx1, by1, bx2, by2 in st["col_boxes"]:
                    gkind = _glyph_kind_for_key_v705(key)
                    if gkind is None:
                        continue
                    glyph_attempt += 1
                    gtext, gconf, gok = _glyph_recognize_cell_v705(glyph_store, image, (bx1, by1, bx2, by2), gkind)
                    if gok:
                        ct[key] = gtext
                        glyph_confident.add((st["idx"], key))
                        glyph_accept += 1
        _GLYPH_LAST_STATS_V705["attempt"] = glyph_attempt
        _GLYPH_LAST_STATS_V705["accept"] = glyph_accept
        _GLYPH_LAST_STATS_V705["rate"] = (glyph_accept / glyph_attempt) if glyph_attempt else 0.0
        _prof_add("attack_5_glyph", _pt_glyph)

        # v90 핵심: 무작정 빈 칸 전체 재검수 금지.
        _pt_recheck = _prof_now()
        recheck_tasks: List[Tuple[int, str, Tuple[int, int, int, int]]] = []
        for st in row_states:
            ct = st["cell_texts"]
            for key, label, bx1, by1, bx2, by2 in st["col_boxes"]:
                if (st["idx"], key) in glyph_confident:
                    continue
                cur = str(ct.get(key, "")).strip()
                need = False
                if key == "damage_text":
                    try:
                        import os as _os_v107a
                        _force_damage_recheck = str(_os_v107a.environ.get("LOA_FORCE_DAMAGE_CELL_OCR", "0")).lower() in {"1", "true", "yes", "on"}
                    except Exception:
                        _force_damage_recheck = False
                    need = _force_damage_recheck or (not _v90_text_has_damage_like(cur))
                elif key == "casts":
                    try:
                        import os as _os_v107b
                        _force_casts_recheck = str(_os_v107b.environ.get("LOA_FORCE_CASTS_CELL_OCR", "0")).lower() in {"1", "true", "yes", "on"}
                    except Exception:
                        _force_casts_recheck = False
                    need = _force_casts_recheck or (not _v90_text_has_count_like(cur))
                # row_failed여도 이름/퍼센트/DPS/지분 전체 재OCR은 하지 않습니다.
                if need:
                    recheck_tasks.append((st["idx"], key, (bx1, by1, bx2, by2)))

        # v93 FAST 기본값: 공격정보 재검수 OCR은 병목이 매우 커서 기본 비활성화합니다.
        # 필요하면 환경변수 LOA_ATTACK_RECHECK=1 로 켤 수 있습니다.
        try:
            import os as _os_v93
            _recheck_on_v93 = str(_os_v93.environ.get("LOA_ATTACK_RECHECK", "0")).lower() in {"1", "true", "yes", "on"}
        except Exception:
            _recheck_on_v93 = False
        if recheck_tasks and _recheck_on_v93:
            for ridx, key, box in recheck_tasks:
                val = _ocr_cell_fast_v90(reader, image, box, key, scale=scale)
                if str(val).strip():
                    for st in row_states:
                        if st["idx"] == ridx:
                            st["cell_texts"][key] = val
                            break
        _prof_add("attack_6_recheck_easyocr", _pt_recheck)

        # 피해량/DPS 단위 보정
        if glyph_store is not None:
            for st in row_states:
                ct = st["cell_texts"]
                for key, label, bx1, by1, bx2, by2 in st["col_boxes"]:
                    if key not in _KOREAN_NUM_KEYS_V705:
                        continue
                    cur = str(ct.get(key, "")).strip()
                    if not cur or re.search(r"[억만조]", cur):
                        continue
                    unit = _glyph_detect_damage_unit_v705(glyph_store, image, (bx1, by1, bx2, by2))
                    if unit:
                        ct[key] = _strip_ocr_unit_garbage(cur) + unit
                    elif re.match(r"^\d[\d,]*\.\d{2}\d*$", cur):
                        clean_num = _strip_ocr_unit_garbage(cur)
                        try:
                            val = float(clean_num.replace(",", ""))
                            if 0.5 <= val <= 9999.99:
                                ct[key] = clean_num + "억"
                        except Exception:
                            pass

        rows: List[Dict[str, Any]] = []
        for st in row_states:
            i = st["idx"]; rinfo = st["rinfo"]; cell_texts = st["cell_texts"]
            row: Dict[str, Any] = {col: "" for col in STANDARD_COLUMNS}
            row["_ocr_row_index"] = str(i)
            row["_row_source"] = rinfo.get("source", "")
            row["_direction_kind"] = direction_kind
            if window_box:
                row["_window_x1"], row["_window_y1"], row["_window_x2"], row["_window_y2"] = window_box
            ix1, iy1, ix2, iy2 = rinfo["icon_box"]
            row["_icon_x1"], row["_icon_y1"], row["_icon_x2"], row["_icon_y2"] = ix1, iy1, ix2, iy2
            row["_row_x1"], row["_row_y1"], row["_row_x2"], row["_row_y2"] = rinfo["row_box"]

            directional_rate = ""
            directional_share = ""
            for key, label, *_ in st["col_boxes"]:
                val = _postprocess_cell_value_v705(key, cell_texts.get(key, ""))
                if key == "directional_rate":
                    directional_rate = val
                elif key == "directional_share":
                    directional_share = val
                else:
                    row[label] = val

            if direction_kind == "head":
                row["헤드어택 적중률"] = directional_rate
                row["헤드어택 비중"] = directional_share
            else:
                row["백어택 적중률"] = directional_rate
                row["백어택 비중"] = directional_share

            meaningful = False
            for col in [c for c in STANDARD_COLUMNS if c not in {"이름", "초당 피해량", "피해량 지분"}]:
                v = str(row.get(col) or "").strip()
                if v and v not in ["-", "_", "None", "nan"]:
                    meaningful = True
                    break
            if meaningful or str(row.get("이름") or "").strip():
                rows.append(row)

        cols = STANDARD_COLUMNS + [
            "_ocr_row_index", "_row_source", "_direction_kind",
            "_window_x1", "_window_y1", "_window_x2", "_window_y2",
            "_icon_x1", "_icon_y1", "_icon_x2", "_icon_y2",
            "_row_x1", "_row_y1", "_row_x2", "_row_y2",
        ]
        df = pd.DataFrame(rows, columns=cols)
        if not df.empty:
            check_cols = [c for c in STANDARD_COLUMNS if c not in {"초당 피해량", "피해량 지분"}]
            nonempty = df[check_cols].astype(str).apply(lambda r: any(x.strip() and x.strip() not in ["", "-", "_", "None", "nan"] for x in r), axis=1)
            df = df[nonempty].reset_index(drop=True)
        if df.empty:
            return _old_parse_attack_fixed_grid_v38_for_v705(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)
        return df
    except Exception as e:
        print(f"[v90 parse_attack fallback] {e}")
        return _old_parse_attack_fixed_grid_v38_for_v705(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)


# ==============================
# v93 note
# ==============================
# v93: 3초대 목표용 FAST 기본값. 공격정보 재검수 OCR 기본 OFF, 아이콘 정밀 후보 기본 5개.
#      LOA_ATTACK_RECHECK=1 / LOA_ICON_HEAVY_TOPK=N 환경변수로 되돌릴 수 있습니다.


# ==============================================================================
# v94: 종합정보 Damage/DPS guide 템플릿 앵커
# ==============================================================================
# 사용자가 제공한 Damage_guide.png / dps_guide.png 는 카드 전체의 빈 배경보다
# 상단 라벨("피해량", "초당 피해량") 부분이 매우 안정적인 앵커입니다.
# 이를 이용해 카드 위치를 정확히 잡고, 카드 하단의 원시 숫자 줄만 작게 OCR합니다.
# 기존 방식은 카드 전체를 6배 확대해 OCR해서 2회에 약 0.5초가 걸렸고,
# 큰 표시값(4,641.26억)과 원시값(464,126,236,306)이 같이 읽히는 문제가 있었습니다.

SUMMARY_GUIDE_FILES_V94 = {
    "total_damage_text": "Damage_guide.png",
    "dps_text": "dps_guide.png",
}
# 1920x1080 기준 카드 박스. 실제 위치는 window_box 기준 scale로 보정합니다.
SUMMARY_GUIDE_CARD_BASE_V94 = {
    "total_damage_text": (229.0, 247.0, 585.0, 400.0),
    "dps_text": (592.0, 247.0, 948.0, 400.0),
}
# guide 이미지 내부에서 라벨만 잘라 matchTemplate 합니다. 카드 전체는 다른 카드와 배경이 비슷해서 오탐이 큽니다.
GUIDE_LABEL_BOX_V94 = (80, 5, 250, 38)
GUIDE_LABEL_OFFSET_V94 = (GUIDE_LABEL_BOX_V94[0], GUIDE_LABEL_BOX_V94[1])
GUIDE_CARD_SIZE_V94 = (356.0, 153.0)
# 카드 하단 원시 숫자 줄 crop. 상단 한국식 단위 표시값을 읽지 않고 숫자/쉼표만 읽기 위함.
GUIDE_RAW_LINE_FRAC_V94 = (45 / 356.0, 92 / 153.0, 322 / 356.0, 130 / 153.0)
GUIDE_MATCH_THRESHOLD_V94 = 0.82


@lru_cache(maxsize=4)
def _load_summary_guide_label_v94(key: str) -> "np.ndarray | None":
    try:
        import cv2 as _cv2
        fname = SUMMARY_GUIDE_FILES_V94.get(key)
        if not fname:
            return None
        cands = [
            _project_root_v32() / "data" / fname,
            Path(__file__).resolve().parent.parent / "data" / fname,
            Path.cwd() / "data" / fname,
        ]
        for p in cands:
            if p.is_file():
                im = Image.open(p).convert("RGB")
                label = im.crop(GUIDE_LABEL_BOX_V94)
                return _cv2.cvtColor(np.asarray(label), _cv2.COLOR_RGB2GRAY)
    except Exception as e:
        print(f"[summary-guide:v94] 라벨 템플릿 로드 실패 {key}: {e}")
    return None


def _expected_summary_card_box_v94(image: Image.Image, window_box: Tuple[int, int, int, int], key: str) -> Tuple[int, int, int, int, float]:
    """window_box 기준으로 damage/dps 카드의 예상 위치를 계산합니다."""
    x1, y1, x2, y2 = [float(v) for v in window_box]
    ww = max(1.0, x2 - x1)
    scale = ww / float(BASE_WINDOW_W)
    bx1, by1, bx2, by2 = SUMMARY_GUIDE_CARD_BASE_V94[key]
    rel_x = bx1 - BASE_WINDOW_BOX[0]
    rel_y = by1 - BASE_WINDOW_BOX[1]
    w = (bx2 - bx1) * scale
    h = (by2 - by1) * scale
    cx1 = int(round(x1 + rel_x * scale))
    cy1 = int(round(y1 + rel_y * scale))
    return _clip_box(image, (cx1, cy1, int(round(cx1 + w)), int(round(cy1 + h)))) + (float(scale),)  # type: ignore[return-value]


def _find_summary_card_by_guide_v94(image: Image.Image, window_box: Tuple[int, int, int, int], key: str) -> Tuple[int, int, int, int, float, float] | None:
    """guide 라벨 템플릿으로 카드 위치를 보정합니다. 반환: card_box + score + scale."""
    try:
        import cv2 as _cv2
        tmpl0 = _load_summary_guide_label_v94(key)
        if tmpl0 is None:
            return None
        ex1, ey1, ex2, ey2, base_scale = _expected_summary_card_box_v94(image, window_box, key)
        gray = _cv2.cvtColor(np.asarray(image.convert("RGB")), _cv2.COLOR_RGB2GRAY)
        best: Tuple[float, Tuple[int, int] | None, float, int, int] = (-1.0, None, base_scale, tmpl0.shape[1], tmpl0.shape[0])
        # 예상 라벨 좌표 주변만 검색합니다. 배경이 비슷한 카드가 많으므로 전체 화면 검색은 하지 않습니다.
        for sm in (0.94, 1.0, 1.06):
            s = max(0.2, float(base_scale) * sm)
            tw = max(8, int(round(tmpl0.shape[1] * s)))
            th = max(6, int(round(tmpl0.shape[0] * s)))
            if tw >= image.width or th >= image.height:
                continue
            tmpl = _cv2.resize(tmpl0, (tw, th), interpolation=_cv2.INTER_AREA if s < 1 else _cv2.INTER_LINEAR)
            lx_exp = int(round(ex1 + GUIDE_LABEL_OFFSET_V94[0] * s))
            ly_exp = int(round(ey1 + GUIDE_LABEL_OFFSET_V94[1] * s))
            mx = max(32, int(round(80 * s)))
            my = max(24, int(round(45 * s)))
            sx1 = max(0, lx_exp - mx)
            sy1 = max(0, ly_exp - my)
            sx2 = min(image.width, lx_exp + mx + tw)
            sy2 = min(image.height, ly_exp + my + th)
            if sx2 - sx1 < tw or sy2 - sy1 < th:
                continue
            band = gray[sy1:sy2, sx1:sx2]
            res = _cv2.matchTemplate(band, tmpl, _cv2.TM_CCOEFF_NORMED)
            _minv, maxv, _minl, maxl = _cv2.minMaxLoc(res)
            if float(maxv) > best[0]:
                best = (float(maxv), (sx1 + int(maxl[0]), sy1 + int(maxl[1])), s, tw, th)
        score, loc, s, _tw, _th = best
        if loc is None or score < GUIDE_MATCH_THRESHOLD_V94:
            return None
        lx, ly = loc
        card_x = int(round(lx - GUIDE_LABEL_OFFSET_V94[0] * s))
        card_y = int(round(ly - GUIDE_LABEL_OFFSET_V94[1] * s))
        card_w = int(round(GUIDE_CARD_SIZE_V94[0] * s))
        card_h = int(round(GUIDE_CARD_SIZE_V94[1] * s))
        x1, y1, x2, y2 = _clip_box(image, (card_x, card_y, card_x + card_w, card_y + card_h))
        return x1, y1, x2, y2, score, s
    except Exception as e:
        print(f"[summary-guide:v94] 카드 매칭 실패 {key}: {e}")
        return None


def _crop_raw_line_from_card_v94(image: Image.Image, card_box: Tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = [int(v) for v in card_box]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    fx1, fy1, fx2, fy2 = GUIDE_RAW_LINE_FRAC_V94
    box = (
        int(round(x1 + w * fx1)),
        int(round(y1 + h * fy1)),
        int(round(x1 + w * fx2)),
        int(round(y1 + h * fy2)),
    )
    return image.convert("RGB").crop(_clip_box(image, box))


def _ocr_summary_raw_number_line_v94(reader: Any, raw_crop: Image.Image, *, scale: int = 4) -> Tuple[str, Image.Image, str, float]:
    """카드 하단 원시 숫자 줄 전용 OCR. 숫자/쉼표만 허용해 빠르고 안정적으로 읽습니다."""
    use_scale = max(3, min(int(scale or 4), 4))
    proc = _preprocess_numeric_text_mask(raw_crop, scale=use_scale)
    try:
        # raw integer line은 숫자/쉼표만 있으므로 allowlist가 있는 어댑터에서는 탐색 공간이 작아집니다.
        try:
            res = reader.readtext(np.asarray(proc), detail=0, paragraph=True, allowlist="0123456789, ")
            if isinstance(res, list):
                text = " ".join(str(x) for x in res)
            else:
                text = str(res or "")
        except TypeError:
            text = _ocr_text_once(reader, proc, numeric=True)
    except Exception:
        text = ""
    text = clean_ocr_text(text)
    score = 0.0
    try:
        score = _numeric_ocr_score(text, kind="integer")
    except Exception:
        score = float(len(re.findall(r"\d", str(text or ""))))
    return text, proc, "guide_raw_number_line", float(score)


_parse_summary_fixed_grid_prev_v94 = parse_summary_fixed_grid
_make_summary_ocr_debug_prev_v94 = make_summary_ocr_debug


def parse_summary_fixed_grid(image: Image.Image, reader: Any, scale: int = 6, auto_window: bool = True) -> Dict[str, Any]:  # type: ignore[override]
    """v94 종합정보 OCR: guide 템플릿으로 Damage/DPS 카드 하단 원시 숫자만 읽습니다."""
    _pt = _prof_now()
    window_box = _summary_window_box_fast_v83(image, auto_window=auto_window)
    _prof_add("summary_1_window_detect_fast_no_ocr", _pt)

    raw: Dict[str, str] = {}
    card_debug: Dict[str, Dict[str, Any]] = {}

    _pt_elapsed = _prof_now()
    elapsed_raw, elapsed_raw_crop, elapsed_proc_crop, elapsed_method, elapsed_score = _read_elapsed_fast_v83(image, window_box, reader, scale=scale)
    raw["elapsed_text"] = elapsed_raw
    _prof_add("summary_2_elapsed_only", _pt_elapsed)

    _pt_cards = _prof_now()
    for key in ("total_damage_text", "dps_text"):
        guide = _find_summary_card_by_guide_v94(image, window_box, key)
        if guide is not None:
            gx1, gy1, gx2, gy2, gscore, gscale = guide
            raw_crop = _crop_raw_line_from_card_v94(image, (gx1, gy1, gx2, gy2))
            text, proc, method, score = _ocr_summary_raw_number_line_v94(reader, raw_crop, scale=4)
            raw[key] = text
            card_debug[key] = {
                "raw_crop": raw_crop,
                "processed_crop": proc,
                "method": f"{method}:guide_score={gscore:.3f}",
                "score": score,
                "guide_card_box": (gx1, gy1, gx2, gy2),
                "guide_score": gscore,
                "guide_scale": gscale,
            }
        else:
            # guide 실패 시 v83 카드 전체 OCR로 안전 폴백
            raw_crop = crop_norm(image, SUMMARY_ROIS[key], pad=1, window_box=window_box)
            text, proc, method, score = _ocr_summary_card_text_v83(reader, raw_crop, scale=scale)
            raw[key] = text
            card_debug[key] = {"raw_crop": raw_crop, "processed_crop": proc, "method": f"fallback_{method}", "score": score}
    _prof_add("summary_3_damage_dps_guide_raw", _pt_cards)

    try:
        from modules.calculators import parse_korean_number, format_korean_number
    except Exception:
        from .calculators import parse_korean_number, format_korean_number  # type: ignore

    total_raw_num = full_raw_number_from_summary(raw.get("total_damage_text", ""))
    dps_raw_num = full_raw_number_from_summary(raw.get("dps_text", ""))
    total_value = total_raw_num if total_raw_num is not None else parse_korean_number(extract_korean_number_text(raw.get("total_damage_text", "")))
    dps_value = dps_raw_num if dps_raw_num is not None else parse_korean_number(extract_korean_number_text(raw.get("dps_text", "")))
    elapsed_seconds = extract_elapsed_seconds(raw.get("elapsed_text", ""))

    meta: Dict[str, Any] = {
        "raw": raw,
        "window_box": window_box,
        "window_detected": window_box is not None,
        "summary_fast_v94": True,
        "summary_read_keys": list(SUMMARY_REQUIRED_KEYS_V83),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_source": "ocr" if elapsed_seconds is not None else "ocr_failed",
        "total_damage": total_value,
        "dps": dps_value,
        "total_damage_text": format_korean_number(total_value) if total_value else extract_korean_number_text(raw.get("total_damage_text", "")),
        "dps_text": format_korean_number(dps_value) if dps_value else extract_korean_number_text(raw.get("dps_text", "")),
        "damage_increase_efficiency": None,
        "crit_rate": None,
        "head_attack_rate": None,
        "back_attack_rate": None,
        "damage_increase_uptime": None,
        "burst_uptime": None,
        "_debug_v94": {"cards": {k: {kk: vv for kk, vv in v.items() if kk not in {"raw_crop", "processed_crop"}} for k, v in card_debug.items()}},
    }
    meta["raw_display"] = {
        "elapsed_text": _format_summary_debug_text_v83("elapsed_text", raw.get("elapsed_text", ""), elapsed_seconds),
        "total_damage_text": _format_summary_debug_text_v83("total_damage_text", raw.get("total_damage_text", ""), total_value),
        "dps_text": _format_summary_debug_text_v83("dps_text", raw.get("dps_text", ""), dps_value),
    }
    return meta


def make_summary_ocr_debug(image: Image.Image, reader: Any, scale: int = 6) -> List[Dict[str, Any]]:  # type: ignore[override]
    """v94 종합정보 디버그: guide 템플릿으로 잡은 원시 숫자 줄 crop을 표시합니다."""
    window_box = _summary_window_box_fast_v83(image, auto_window=True)
    rows: List[Dict[str, Any]] = []

    elapsed_raw, elapsed_raw_crop, elapsed_proc_crop, elapsed_method, elapsed_score = _read_elapsed_fast_v83(image, window_box, reader, scale=scale)
    elapsed_seconds = extract_elapsed_seconds(elapsed_raw)
    if elapsed_raw_crop is None:
        elapsed_raw_crop = crop_norm(image, SUMMARY_ROIS["elapsed_text"], pad=1, window_box=window_box)
        elapsed_proc_crop = preprocess_crop(elapsed_raw_crop, scale=max(4, min(int(scale or 6), 7)))
    rows.append({
        "key": "elapsed_text",
        "label": SUMMARY_LABELS_V83["elapsed_text"],
        "text": _format_summary_debug_text_v83("elapsed_text", elapsed_raw, elapsed_seconds),
        "raw_text": elapsed_raw,
        "raw_crop": elapsed_raw_crop,
        "processed_crop": elapsed_proc_crop,
        "window_box": window_box,
        "preprocess": elapsed_method,
        "score": elapsed_score,
    })

    try:
        from modules.calculators import parse_korean_number
    except Exception:
        from .calculators import parse_korean_number  # type: ignore
    for key in ("total_damage_text", "dps_text"):
        guide = _find_summary_card_by_guide_v94(image, window_box, key)
        if guide is not None:
            gx1, gy1, gx2, gy2, gscore, _gscale = guide
            raw_crop = _crop_raw_line_from_card_v94(image, (gx1, gy1, gx2, gy2))
            text, proc, method, score = _ocr_summary_raw_number_line_v94(reader, raw_crop, scale=4)
            method = f"{method}:guide_score={gscore:.3f}"
        else:
            raw_crop = crop_norm(image, SUMMARY_ROIS[key], pad=1, window_box=window_box)
            text, proc, method, score = _ocr_summary_card_text_v83(reader, raw_crop, scale=scale)
            method = f"fallback_{method}"
        raw_num = full_raw_number_from_summary(text)
        parsed = raw_num if raw_num is not None else parse_korean_number(extract_korean_number_text(text))
        rows.append({
            "key": key,
            "label": SUMMARY_LABELS_V83[key],
            "text": _format_summary_debug_text_v83(key, text, parsed),
            "raw_text": text,
            "raw_crop": raw_crop,
            "processed_crop": proc,
            "window_box": window_box,
            "preprocess": method,
            "score": score,
        })
    return rows


# ==============================================================================
# v95: 공격정보 숫자 글리프 확장 + 더 타이트한 스킬 아이콘 crop
# ==============================================================================
# 목적:
# 1) v93/v94 FAST 모드에서 damage_text/dps_text가 비면, 무거운 OCR을 다시 켜지 않고
#    data/glyph_templates.json의 0~9/.,억/만/조 템플릿으로 먼저 채웁니다.
# 2) 아이콘 crop이 행 배경/이름칸까지 넓게 섞이는 경우를 줄이기 위해, 행 높이 기준의
#    작은 정사각형 + 채도/밝기 bbox 보정으로 실제 아이콘 중심부만 비교합니다.

import os as _os_v95

ICON_TIGHT_GUIDE_ENABLED_V95 = str(_os_v95.environ.get("LOA_ICON_TIGHT_GUIDE", "1")).lower() in {"1", "true", "yes", "on"}
ICON_TIGHT_SIDE_RATIO_V95 = float(_os_v95.environ.get("LOA_ICON_TIGHT_SIDE_RATIO", "0.82") or "0.82")
GLYPH_KOREAN_GATE_V95 = float(_os_v95.environ.get("LOA_GLYPH_KOREAN_GATE", "0.34") or "0.34")

_old_glyph_kind_for_key_v95 = globals().get("_glyph_kind_for_key_v705")
_old_glyph_recognize_cell_v95 = globals().get("_glyph_recognize_cell_v705")
_old_tight_icon_box_from_row_v95 = globals().get("_tight_icon_box_from_row_v43")
_old_extract_battle_icon_core_v95 = globals().get("_extract_battle_icon_core_v42")


def _glyph_kind_for_key_v705(key: str) -> str | None:  # type: ignore[override]
    """v95: damage/DPS의 억·만 숫자도 글리프 우선 인식 대상에 포함합니다."""
    try:
        if key in globals().get("_KOREAN_NUM_KEYS_V705", {"damage_text", "dps_text"}):
            return "korean"
    except Exception:
        pass
    if callable(_old_glyph_kind_for_key_v95):
        return _old_glyph_kind_for_key_v95(key)  # type: ignore[misc]
    try:
        if key in _GLYPH_COUNT_KEYS:
            return "count"
        if key in _GLYPH_PERCENT_KEYS:
            return "percent"
    except Exception:
        pass
    return None


def _korean_num_glyph_format_ok_v95(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    # 피해량/DPS는 대부분 2,083.44억 / 8,200.00만 형태. 단위가 없으면 채택하지 않습니다.
    if not re.search(r"[억만조]", t):
        return False
    if not re.search(r"\d", t):
        return False
    # 헤더/라벨이 들어온 경우 차단.
    if re.search(r"피해량|DPS|지분|이름", t, re.I):
        return False
    digits = re.sub(r"\D", "", t)
    return len(digits) >= 2


def _glyph_recognize_cell_v705(  # type: ignore[override]
    store: Any, image: Image.Image, box: Tuple[int, int, int, int], kind: str
) -> Tuple[str, float, bool]:
    """v95: percent/count는 기존 fast_v2, korean은 glyph_engine 직접 사용."""
    if kind != "korean":
        if callable(_old_glyph_recognize_cell_v95):
            return _old_glyph_recognize_cell_v95(store, image, box, kind)  # type: ignore[misc]
        return "", 0.0, False
    try:
        crop = _crop_pixel_box(image, box, pad=1, expand_ratio=0.015)
    except Exception:
        return "", 0.0, False

    # 흰 글자만 남긴 crop을 우선 사용합니다. 실패하면 원본 crop으로 한 번 더 시도합니다.
    candidates: List[Image.Image] = []
    try:
        if _glyph_fast_v2 is not None and hasattr(_glyph_fast_v2, "whiteness_isolate"):
            iso = _glyph_fast_v2.whiteness_isolate(
                crop,
                pad=3,
                min_w_factor=0.16,
                empty_ratio=0.0035,
            )
            if iso is not None:
                candidates.append(iso)
    except Exception:
        pass
    candidates.append(crop)

    best_text, best_conf = "", 0.0
    for cand in candidates:
        try:
            text, conf = store.recognize(cand, kind="korean")
            text = clean_ocr_text(text or "")
            conf = float(conf or 0.0)
            if conf > best_conf:
                best_text, best_conf = text, conf
        except Exception:
            continue
    ok = best_conf >= GLYPH_KOREAN_GATE_V95 and _korean_num_glyph_format_ok_v95(best_text)
    return best_text, float(best_conf), bool(ok)


def _tight_icon_box_from_row_v43(  # type: ignore[override]
    image: Image.Image, box: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """v95: 실제 아이콘 크기에 맞춘 타이트 crop.

    기존 icon_box가 조금 넓거나 행 배경이 섞여도, 아이콘 셀 안에서 채도/밝기가 있는
    정사각형 중심부를 잡습니다. 실패하면 이전 v43/v44 crop으로 안전하게 폴백합니다.
    """
    if not ICON_TIGHT_GUIDE_ENABLED_V95:
        if callable(_old_tight_icon_box_from_row_v95):
            return _old_tight_icon_box_from_row_v95(image, box)  # type: ignore[misc]
    try:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1, x2, y2 = _clip_box(image, (x1, y1, x2, y2))
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        # 전투분석기 아이콘은 행 높이보다 약간 작습니다. 0.82h를 기본으로 두면
        # 테두리/선택행 배경을 줄이면서 중앙 도형은 유지됩니다.
        side_base = int(round(h * ICON_TIGHT_SIDE_RATIO_V95))
        side_base = max(14, min(side_base, h, max(w, h)))

        # 아이콘 셀 주변만 작게 탐색. 이름 글자가 섞이는 것을 막기 위해 우측 검색 폭을 제한합니다.
        pad_x = max(2, int(round(h * 0.18)))
        pad_y = max(1, int(round(h * 0.10)))
        sx1 = max(0, x1 - pad_x)
        sy1 = max(0, y1 - pad_y)
        sx2 = min(image.width, x2 + pad_x)
        sy2 = min(image.height, y2 + pad_y)
        search = image.crop((sx1, sy1, sx2, sy2)).convert("RGB")
        arr = np.asarray(search, dtype=np.uint8)
        if arr.size:
            maxc = arr.max(axis=2).astype(np.int16)
            minc = arr.min(axis=2).astype(np.int16)
            sat = maxc - minc
            gray = arr.mean(axis=2)
            # 색 있는 스킬 아이콘/룬 아이콘을 우선 잡고, 어두운 행 배경과 흰 글자 라벨은 배제.
            mask = ((sat > 22) & (gray > 26)) | ((gray > 62) & (sat > 10))
            # 우측 이름 글자 오염 방지: 예상 아이콘 셀 부근만 사용.
            max_w = min(arr.shape[1], int(round(max(w, h) * 1.25)))
            if max_w > 8:
                mask[:, max_w:] = False
            ys, xs = np.where(mask)
            if len(xs) > 25 and len(ys) > 25:
                bx1, bx2 = int(xs.min()), int(xs.max()) + 1
                by1, by2 = int(ys.min()), int(ys.max()) + 1
                bw, bh = bx2 - bx1, by2 - by1
                # bbox가 아이콘 크기와 비슷할 때만 사용. 너무 작으면 노이즈/글자일 가능성이 큼.
                if bw >= h * 0.35 and bh >= h * 0.35:
                    cx = sx1 + (bx1 + bx2) / 2.0
                    cy = sy1 + (by1 + by2) / 2.0
                    side = int(round(max(side_base, min(max(bw, bh) * 1.04, h * 0.96))))
                    side = max(14, min(side, image.width, image.height))
                    nx1 = int(round(cx - side / 2.0))
                    ny1 = int(round(cy - side / 2.0))
                    nx1 = max(0, min(image.width - side, nx1))
                    ny1 = max(0, min(image.height - side, ny1))
                    return (nx1, ny1, nx1 + side, ny1 + side)

        # mask 실패 fallback: 기존 박스 중심에서 0.82h 정사각형.
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        nx1 = int(round(cx - side_base / 2.0))
        ny1 = int(round(cy - side_base / 2.0))
        nx1 = max(0, min(image.width - side_base, nx1))
        ny1 = max(0, min(image.height - side_base, ny1))
        return (nx1, ny1, nx1 + side_base, ny1 + side_base)
    except Exception:
        if callable(_old_tight_icon_box_from_row_v95):
            return _old_tight_icon_box_from_row_v95(image, box)  # type: ignore[misc]
        return box


def _extract_battle_icon_core_v42(img: Image.Image) -> Image.Image:  # type: ignore[override]
    """v95: 이미 tight crop된 row icon과 DB 아이콘 모두 중앙 86~92% 위주로 비교."""
    try:
        img = img.convert("RGB")
        w, h = img.size
        if w <= 0 or h <= 0:
            return img
        side = min(w, h)
        left = max(0, (w - side) // 2)
        top = max(0, (h - side) // 2)
        core = img.crop((left, top, left + side, top + side))
        # row icon이 아니라 API 아이콘에도 같은 중앙 가중 crop을 적용해 테두리/워터마크 영향 감소.
        trim = max(1, int(round(side * 0.045)))
        if side - trim * 2 > 12:
            core = core.crop((trim, trim, side - trim, side - trim))
        return core
    except Exception:
        if callable(_old_extract_battle_icon_core_v95):
            return _old_extract_battle_icon_core_v95(img)  # type: ignore[misc]
        return img



# ==============================================================================
# v96: 정확도 복구 FAST 패치
# ==============================================================================
# v95의 두 가지 변경이 속도는 빨랐지만 정확도를 크게 망가뜨렸습니다.
# 1) 공격정보 피해량/DPS를 글리프가 낮은 신뢰도로 덮어써서 2,083.44억 -> 3.41억처럼 깨짐
# 2) 아이콘 tight crop이 아이콘 위쪽을 잘라 허리케인 소드 94점 -> 65점대로 하락
# 따라서 FAST 모드는 유지하되, 위험한 두 경로만 되돌립니다.

_parse_summary_fixed_grid_prev_v96 = globals().get("parse_summary_fixed_grid")
_glyph_kind_for_key_prev_v96 = globals().get("_glyph_kind_for_key_v705")
_crop_icon_from_row_metadata_prev_v96 = globals().get("_crop_icon_from_row_metadata")


def _glyph_kind_for_key_v705(key: str) -> str | None:  # type: ignore[override]
    # 피해량/DPS는 현재 glyph_templates가 실전 캡처와 충분히 맞지 않아 오인식률이 높습니다.
    # rowpass OCR이 이 칸은 이미 잘 읽고 있으므로 글리프 덮어쓰기를 금지합니다.
    if str(key) in {"damage_text", "dps_text"}:
        return None
    if callable(_glyph_kind_for_key_prev_v96):
        return _glyph_kind_for_key_prev_v96(key)  # type: ignore[misc]
    return None


def _crop_icon_from_row_metadata(image: Image.Image, row: "pd.Series | Dict[str, Any]", fallback_index: int) -> Image.Image | None:  # type: ignore[override]
    """v96: v95 tight crop 해제.

    디버그 기준 원래 64x47 아이콘 crop은 허리케인 소드 94점으로 맞았고,
    v95 tight crop은 위쪽 이펙트를 잘라 65점대로 떨어졌습니다.
    그래서 메타데이터 icon_box를 그대로 사용합니다.
    """
    try:
        vals = [row.get("_icon_x1"), row.get("_icon_y1"), row.get("_icon_x2"), row.get("_icon_y2")]
        if all(v is not None and str(v) not in ["", "nan", "None"] for v in vals):
            box = tuple(int(float(v)) for v in vals)  # type: ignore[arg-type]
            return _crop_pixel_box(image, box, pad=0, expand_ratio=0.0)
    except Exception:
        pass
    try:
        return crop_attack_icon(image, fallback_index, window_box=_row_window_box(row))
    except Exception:
        if callable(_crop_icon_from_row_metadata_prev_v96):
            return _crop_icon_from_row_metadata_prev_v96(image, row, fallback_index)  # type: ignore[misc]
        return None


def parse_summary_fixed_grid(image: Image.Image, reader: Any, scale: int = 7, auto_window: bool = True) -> Dict[str, Any]:  # type: ignore[override]
    """v96: 종합정보 OCR sanity check.

    guide raw line OCR이 낮은 점수로 총피해를 너무 작게 읽으면,
    DPS × 전투시간으로 총피해를 복구합니다. 실제 전투분석기에서 총피해와 DPS는
    같은 카드 원시 숫자 기반이라 이 검증이 매우 안정적입니다.
    """
    if callable(_parse_summary_fixed_grid_prev_v96):
        meta = _parse_summary_fixed_grid_prev_v96(image, reader, scale=scale, auto_window=auto_window)  # type: ignore[misc]
    else:
        meta = {}
    try:
        total = meta.get("total_damage")
        dps = meta.get("dps")
        elapsed = meta.get("elapsed_seconds")
        total_f = float(total) if total is not None else None
        dps_f = float(dps) if dps is not None else None
        elapsed_f = float(elapsed) if elapsed is not None else None
        if dps_f and dps_f > 0 and elapsed_f and elapsed_f > 0:
            expected = dps_f * elapsed_f
            # 예: OCR 총피해 4,041만, DPS×시간 4,794억이면 총피해 OCR 실패로 판정.
            if total_f is None or total_f <= 0 or total_f < expected * 0.35 or total_f > expected * 3.0:
                meta["total_damage"] = expected
                try:
                    from modules.calculators import format_korean_number as _fmt_v96
                except Exception:
                    from .calculators import format_korean_number as _fmt_v96  # type: ignore
                meta["total_damage_text"] = _fmt_v96(expected)
                meta["total_damage_source"] = "dps_x_elapsed_fallback_v96"
                raw_disp = meta.get("raw_display")
                if isinstance(raw_disp, dict):
                    raw_disp["total_damage_text"] = f"{_fmt_v96(expected)} (DPS×전투시간 보정)"
    except Exception:
        pass
    return meta



# ==============================================================================
# v97: 해상도 독립 비율 기반 정사각형 아이콘 crop + 기존 crop fallback
# ==============================================================================
# 고정 픽셀(64x64)로 자르지 않고, OCR 행 높이/아이콘 셀 높이를 기준으로
# 실제 화면에서 보이는 아이콘을 정사각형으로 잘라 64x64 정규화 매칭에 넘깁니다.
# 아이콘 셀은 전투분석기에서 보통 왼쪽 정렬되어 있으므로, 기존 넓은 icon_box의
# 왼쪽부터 행 높이만큼 정사각형으로 자르는 방식을 1차로 사용합니다.
# 단, 정사각형 crop 점수가 낮으면 v96의 기존 넓은 crop으로 자동 fallback합니다.

_crop_icon_from_row_metadata_prev_v97 = globals().get("_crop_icon_from_row_metadata")
_correct_battle_skill_names_with_icons_prev_v97 = globals().get("correct_battle_skill_names_with_icons")


def _env_float_v97(name: str, default: float) -> float:
    try:
        import os as _os_v97
        return float(_os_v97.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _env_bool_v97(name: str, default: bool = True) -> bool:
    try:
        import os as _os_v97
        v = str(_os_v97.environ.get(name, "1" if default else "0")).strip().lower()
        return v not in {"0", "false", "no", "off", "n"}
    except Exception:
        return bool(default)


def _row_int_box_v97(row: "pd.Series | Dict[str, Any]", prefix: str) -> tuple[int, int, int, int] | None:
    try:
        vals = [row.get(f"_{prefix}_x1"), row.get(f"_{prefix}_y1"), row.get(f"_{prefix}_x2"), row.get(f"_{prefix}_y2")]
        if all(v is not None and str(v) not in ["", "nan", "None"] for v in vals):
            x1, y1, x2, y2 = [int(round(float(v))) for v in vals]
            if x2 > x1 and y2 > y1:
                return (x1, y1, x2, y2)
    except Exception:
        pass
    return None


def _clip_square_box_v97(image: Image.Image, x1: int, y1: int, side: int) -> tuple[int, int, int, int] | None:
    try:
        side = int(round(side))
        if side <= 8:
            return None
        side = max(10, min(side, int(image.width), int(image.height)))
        x1 = int(round(max(0, min(int(image.width) - side, x1))))
        y1 = int(round(max(0, min(int(image.height) - side, y1))))
        return (x1, y1, x1 + side, y1 + side)
    except Exception:
        return None


def _square_icon_box_from_row_metadata_v97(
    image: Image.Image,
    row: "pd.Series | Dict[str, Any]",
    fallback_index: int,
) -> tuple[int, int, int, int] | None:
    """행 높이 기준 정사각형 아이콘 박스를 계산합니다.

    핵심:
    - 픽셀 고정값 사용 안 함.
    - 기존 icon_box의 height 또는 row_box height를 기준으로 side 계산.
    - 로아 전투분석기 아이콘은 icon_box 안에서 왼쪽 정렬이므로 x1은 기존 icon_box 왼쪽을 우선 사용.
    """
    try:
        icon_box = _row_int_box_v97(row, "icon")
        row_box = _row_int_box_v97(row, "row")
        if icon_box is None:
            # fallback crop에서 다시 넓은 crop을 얻을 수는 있지만, 박스 좌표가 없으면 정사각형 계산은 생략합니다.
            return None
        ix1, iy1, ix2, iy2 = icon_box
        iw = max(1, ix2 - ix1)
        ih = max(1, iy2 - iy1)
        row_h = ih
        if row_box is not None:
            row_h = max(ih, row_box[3] - row_box[1])

        # v150: 아이콘은 정사각형이므로 1:1로 자릅니다.
        # 한 변/왼쪽 x를 '검출된 icon_box 너비(iw, 행마다 흔들림)' 대신 창(window) 그리드
        # 비율로 계산합니다. 픽셀 고정값이 아니라 창 크기에 비례하므로 해상도가 달라져도 안전하고,
        # 모든 행이 동일한 크기의 정사각형으로 잘려 기존의 38~56px 편차 문제를 없앱니다.
        grid_w = None
        grid_x1 = None
        wb = _row_window_box(row)
        if wb is not None:
            try:
                gx1 = _window_rel_x_from_full_norm(ATTACK_ICON_GRID["x1"], wb)
                gx2 = _window_rel_x_from_full_norm(ATTACK_ICON_GRID["x2"], wb)
                if gx2 - gx1 > 4:
                    grid_w = int(round(gx2 - gx1))
                    grid_x1 = int(round(gx1))
            except Exception:
                grid_w = None

        ratio = _env_float_v97("LOA_ICON_SQUARE_SIDE_RATIO", 1.00)
        # 한 변 = 그리드 아이콘 칸 폭(안정, 해상도 무관). 창 정보가 없으면 행 높이로 폴백합니다.
        if grid_w:
            side = int(round(grid_w * float(ratio)))
        else:
            side = int(round(row_h * float(ratio)))
        side = max(12, side)

        # 왼쪽 앵커도 그리드 기준(안정). 없으면 검출된 icon_box 왼쪽을 사용합니다.
        nx1 = grid_x1 if grid_x1 is not None else ix1
        # v151: 세로 중심은 icon_box 기준으로 잡습니다. 일반 행은 row_box 중심과 ≤1px 차이라
        # 사실상 동일하고, 마지막 행은 iterator에서 icon_box를 아래로 내려둔 값을 그대로 반영해
        # 하단이 살짝 잘린 아이콘이 크롭 중앙에 오게 됩니다.
        cy = (iy1 + iy2) / 2.0
        ny1 = int(round(cy - side / 2.0))
        return _clip_square_box_v97(image, nx1, ny1, side)
    except Exception:
        return None


def _crop_square_icon_from_row_metadata_v97(
    image: Image.Image,
    row: "pd.Series | Dict[str, Any]",
    fallback_index: int,
) -> Image.Image | None:
    try:
        box = _square_icon_box_from_row_metadata_v97(image, row, fallback_index)
        if box is not None:
            return _crop_pixel_box(image, box, pad=0, expand_ratio=0.0)
    except Exception:
        pass
    return None


def _crop_icon_from_row_metadata(image: Image.Image, row: "pd.Series | Dict[str, Any]", fallback_index: int) -> Image.Image | None:  # type: ignore[override]
    """v97 기본 crop은 비율 기반 정사각형.

    실제 매칭에서는 correct_battle_skill_names_with_icons가 정사각형 점수가 낮을 때
    v96 넓은 crop으로 fallback합니다. 이 함수는 디버그/일반 호출에서 정사각형 crop이 보이게 합니다.
    """
    if _env_bool_v97("LOA_ICON_SQUARE_CROP", True):
        square = _crop_square_icon_from_row_metadata_v97(image, row, fallback_index)
        if square is not None:
            return square
    if callable(_crop_icon_from_row_metadata_prev_v97):
        return _crop_icon_from_row_metadata_prev_v97(image, row, fallback_index)  # type: ignore[misc]
    return None


def _choose_icon_match_v97(
    original_name: Any,
    square_crop: Image.Image | None,
    legacy_crop: Image.Image | None,
    skill_candidates: list[dict[str, str]],
    *,
    text_threshold: float,
    icon_threshold: float,
) -> tuple[Image.Image | None, list[dict[str, Any]], str, float, float, bool, str, str, float, float]:
    """정사각형 crop을 먼저 쓰고, 점수가 낮으면 기존 crop으로 fallback합니다."""
    top_square: list[dict[str, Any]] = []
    matched_sq = str(original_name or "")
    text_sq = 0.0
    score_sq = 0.0
    ok_sq = False
    reason_sq = "square_not_run"
    if square_crop is not None:
        top_square = _rank_icon_candidates_v40(square_crop, skill_candidates or [], ocr_name=original_name, topn=5)
        matched_sq, text_sq, score_sq, ok_sq, reason_sq = _select_match_from_ranked_v83(
            original_name,
            top_square,
            row_icon=square_crop,
            text_threshold=text_threshold,
            icon_threshold=icon_threshold,
        )

    use_square = bool(square_crop is not None and ok_sq and float(score_sq or 0.0) >= float(icon_threshold) * 100.0)
    top_legacy: list[dict[str, Any]] = []
    matched_lg = str(original_name or "")
    text_lg = 0.0
    score_lg = 0.0
    ok_lg = False
    reason_lg = "legacy_not_run"

    # 정사각형 점수가 기준보다 낮거나 low_conf면 기존 crop으로 검증합니다.
    need_legacy = (not use_square) or str(reason_sq).startswith("icon_low_conf")
    if need_legacy and legacy_crop is not None:
        top_legacy = _rank_icon_candidates_v40(legacy_crop, skill_candidates or [], ocr_name=original_name, topn=5)
        matched_lg, text_lg, score_lg, ok_lg, reason_lg = _select_match_from_ranked_v83(
            original_name,
            top_legacy,
            row_icon=legacy_crop,
            text_threshold=text_threshold,
            icon_threshold=icon_threshold,
        )
        fallback_margin = _env_float_v97("LOA_ICON_SQUARE_FALLBACK_MARGIN", 4.0)
        if ok_lg and (not use_square or float(score_lg or 0.0) >= float(score_sq or 0.0) + fallback_margin):
            return legacy_crop, top_legacy, matched_lg, text_lg, score_lg, ok_lg, f"square_fallback_legacy:{reason_lg}", "legacy", float(score_sq or 0.0), float(score_lg or 0.0)

    return square_crop or legacy_crop, top_square or top_legacy, matched_sq, text_sq, score_sq, ok_sq, f"square:{reason_sq}", "square", float(score_sq or 0.0), float(score_lg or 0.0)


def correct_battle_skill_names_with_icons(  # type: ignore[override]
    df: pd.DataFrame,
    attack_image: Image.Image,
    skill_candidates: List[Dict[str, str]],
    *,
    name_col: str = "이름",
    threshold: float = 0.62,
    icon_threshold: float = 0.72,
    drop_unmatched: bool = False,
) -> pd.DataFrame:
    """v97: 정사각형 아이콘 crop 우선 + 기존 넓은 crop fallback.

    해상도마다 픽셀 크기가 달라도 row/icon metadata 기준으로 정사각형을 잡습니다.
    정사각형 crop이 실패하거나 점수가 낮으면 v96의 기존 icon_box crop으로 자동 비교합니다.
    """
    if df is None or df.empty or name_col not in df.columns:
        return df
    import time as _time_v97
    t_total = _time_v97.perf_counter()
    out = df.copy()
    rows = []
    candidate_count = len(skill_candidates or [])
    _perf_trace_event_v83("icon_correct_start", rows=len(out), candidates=candidate_count, mode="v97_square_fallback")
    for idx, row in out.iterrows():
        row_t = _time_v97.perf_counter()
        try:
            row_index = int(float(row.get("_ocr_row_index", idx)))
        except Exception:
            row_index = int(idx)

        t_crop = _time_v97.perf_counter()
        square_crop = _crop_square_icon_from_row_metadata_v97(attack_image, row, row_index) if _env_bool_v97("LOA_ICON_SQUARE_CROP", True) else None
        legacy_crop = None
        if callable(_crop_icon_from_row_metadata_prev_v97):
            legacy_crop = _crop_icon_from_row_metadata_prev_v97(attack_image, row, row_index)  # type: ignore[misc]
        elif square_crop is None:
            legacy_crop = crop_attack_icon(attack_image, row_index, window_box=_row_window_box(row))
        crop_ms = (_time_v97.perf_counter() - t_crop) * 1000.0

        t_rank = _time_v97.perf_counter()
        chosen_crop, top, matched, text_score, icon_score, ok, reason, crop_mode, square_score, legacy_score = _choose_icon_match_v97(
            row.get(name_col),
            square_crop,
            legacy_crop,
            skill_candidates or [],
            text_threshold=threshold,
            icon_threshold=icon_threshold,
        )
        rank_ms = (_time_v97.perf_counter() - t_rank) * 1000.0

        matched = _canonical_display_name_v86(matched)
        new_row = row.copy()
        if ok:
            new_row[name_col] = matched
        elif drop_unmatched:
            continue
        new_row["_name_match_text_score"] = text_score
        new_row["_name_match_icon_score"] = icon_score
        new_row["_name_match_reason"] = reason
        new_row["_icon_match_name"] = _canonical_display_name_v86(str(top[0]["name"])) if top else ""
        new_row["_icon_match_score"] = top[0]["icon_score"] if top else 0.0
        new_row["_icon_match_source"] = str(top[0]["source"]) if top else ""
        new_row["_icon_match_top3"] = " | ".join(f"{_canonical_display_name_v86(r['name'])}:{r['icon_score']}" for r in top[:3])
        new_row["_icon_crop_mode"] = crop_mode
        new_row["_icon_square_score"] = round(square_score, 2)
        new_row["_icon_legacy_score"] = round(legacy_score, 2)
        rows.append(new_row)
        _perf_trace_event_v83(
            "icon_correct_row",
            row_index=row_index,
            crop_ms=round(crop_ms, 3),
            rank_ms=round(rank_ms, 3),
            row_total_ms=round((_time_v97.perf_counter() - row_t) * 1000.0, 3),
            matched=matched,
            icon_score=icon_score,
            crop_mode=crop_mode,
            square_score=round(square_score, 2),
            legacy_score=round(legacy_score, 2),
            reason=reason,
        )
    total_ms = (_time_v97.perf_counter() - t_total) * 1000.0
    _perf_trace_event_v83("icon_correct_end", rows=len(rows), total_ms=round(total_ms, 3), mode="v97_square_fallback")
    if not rows:
        return out.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


# ==============================================================================
# v101: 숫자 글리프 전용 실험 모드
# ==============================================================================
# 목적: data/glyph_templates.json으로 실제 전투분석기 숫자 칸을 어디까지 읽을 수 있는지 확인합니다.
# - 공격정보 표: 행 단위 OCR을 돌리지 않고, 숫자 칸을 전부 glyph_templates.json으로만 읽습니다.
# - 스킬명: 기존처럼 아이콘 매칭 단계에서 채웁니다. 이름 OCR은 하지 않습니다.
# - 창/행 위치: 제목 템플릿/아이콘 행 감지/비율 crop은 그대로 사용합니다.
# - 결과가 부정확해도 덮어써서 보여줍니다. 정확도 검증용 실험 모드입니다.

import os as _os_v101

_GLYPH_ONLY_NUMBERS_V101 = str(_os_v101.environ.get("LOA_GLYPH_ONLY_NUMBERS", "0")).strip().lower() not in {"0", "false", "no", "off"}
_parse_attack_fixed_grid_prev_v101 = globals().get("parse_attack_fixed_grid")
_parse_summary_fixed_grid_prev_v101 = globals().get("parse_summary_fixed_grid")
_make_attack_ocr_debug_prev_v101 = globals().get("make_attack_ocr_debug")
_make_summary_ocr_debug_prev_v101 = globals().get("make_summary_ocr_debug")


def _glyph_force_crop_v101(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    try:
        return _crop_pixel_box(image, box, pad=1, expand_ratio=CELL_CROP_EXPAND_RATIO)
    except Exception:
        return image.crop(_clip_box(image, box)).convert("RGB")


def _glyph_kind_force_v101(key: str) -> str | None:
    if key in {"damage_text", "dps_text"}:
        return "korean"
    if key in {"directional_rate", "directional_share", "crit_rate", "crit_share", "cooldown_rate", "share_rate", "back_attack_rate", "back_attack_share", "head_attack_rate", "head_attack_share"}:
        return "percent"
    if key in {"casts"}:
        return "count"
    return None


def _glyph_format_force_v101(key: str, text: str) -> str:
    t = clean_ocr_text(str(text or ""))
    if not t:
        return ""
    if key in {"damage_text", "dps_text"}:
        # glyph_engine이 '조' suffix를 아직 약하게 다루는 경우가 있어 일단 원문을 보존합니다.
        return extract_korean_number_text(t) or t
    if key in {"directional_rate", "directional_share", "crit_rate", "crit_share", "cooldown_rate", "share_rate", "back_attack_rate", "back_attack_share", "head_attack_rate", "head_attack_share"}:
        return format_percent_from_ocr(t)
    if key == "casts":
        return re.sub(r"[^0-9]", "", t) or t
    return t


def _glyph_recognize_force_v101(store: Any, image: Image.Image, box: Tuple[int, int, int, int], key: str) -> Tuple[str, float, str]:
    """단일 셀을 글리프로 강제 인식합니다. accepted/gate 없이 최고 후보를 그대로 반환합니다."""
    kind = _glyph_kind_force_v101(key)
    if kind is None or store is None:
        return "", 0.0, "not_glyph_key"
    raw = _glyph_force_crop_v101(image, box)
    candidates: List[Tuple[str, Image.Image]] = []
    try:
        if _glyph_fast_v2 is not None and hasattr(_glyph_fast_v2, "whiteness_isolate"):
            iso = _glyph_fast_v2.whiteness_isolate(
                raw,
                pad=3,
                min_w_factor=0.12 if kind == "korean" else 0.08,
                empty_ratio=0.0025,
            )
            if iso is not None:
                candidates.append(("whiteness_isolate", iso))
    except Exception:
        pass
    # 숫자 덩어리만 다시 잘라보는 후보. 넓은 칸에서 라벨/빈 배경 영향 줄이기.
    try:
        if key in {"damage_text", "dps_text"}:
            candidates.append(("number_isolate", _glyph_isolate_number_crop_v705(raw, pad=4, gap_merge=26)))
    except Exception:
        pass
    candidates.append(("raw", raw))

    best_text, best_conf, best_method = "", 0.0, ""
    for method, crop in candidates:
        try:
            text, conf = store.recognize(crop, kind=kind)
            text = clean_ocr_text(text or "")
            conf = float(conf or 0.0)
            if conf > best_conf or (text and not best_text):
                best_text, best_conf, best_method = text, conf, method
        except Exception:
            continue
    return _glyph_format_force_v101(key, best_text), float(best_conf), best_method


def _base_window_box_scaled_v101(image: Image.Image) -> Tuple[int, int, int, int]:
    sx = image.width / BASE_SCREEN_W
    sy = image.height / BASE_SCREEN_H
    return (
        int(round(BASE_WINDOW_BOX[0] * sx)),
        int(round(BASE_WINDOW_BOX[1] * sy)),
        int(round(BASE_WINDOW_BOX[2] * sx)),
        int(round(BASE_WINDOW_BOX[3] * sy)),
    )


def _detect_window_no_ocr_v101(image: Image.Image, reader: Any | None = None) -> Tuple[int, int, int, int]:
    """제목 템플릿 기반 창 감지. 실패 시 BASE 비율 좌표. OCR title fallback은 쓰지 않습니다."""
    try:
        tpl = _find_title_by_template_v45(image)
        if tpl is not None:
            cx, cy, score, scale = tpl
            box = _estimate_window_from_title_template_v45(image, cx, cy, scale)
            x1, y1, x2, y2 = box
            if (x2 - x1) >= image.height * 0.95 and (y2 - y1) >= image.height * 0.55:
                return box
    except Exception:
        pass
    return _base_window_box_scaled_v101(image)


def _detect_rows_no_ocr_v101(image: Image.Image, window_box: Tuple[int, int, int, int], row_count: int) -> List[Dict[str, Any]]:
    try:
        rows = detect_attack_icon_rows(image, None, window_box=window_box, max_rows=int(row_count))
        if rows:
            return rows
    except Exception:
        pass
    grid = ATTACK_GRID
    out: List[Dict[str, Any]] = []
    for i in range(int(row_count)):
        y1 = grid["row_start"] + i * grid["row_height"]
        y2 = y1 + grid["row_height"]
        row_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, grid["columns"][-1][3], y2), pad=0, window_box=window_box)
        icon_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, ATTACK_ICON_GRID["x2"], y2), pad=1, window_box=window_box)
        out.append({"row_index": i, "row_box": row_box, "icon_box": icon_box, "window_box": window_box, "source": "glyph_only_fixed_fallback"})
    return out


def parse_attack_fixed_grid(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int | None = None,
    scale: int = 7,
    auto_window: bool = True,
) -> pd.DataFrame:
    if not _GLYPH_ONLY_NUMBERS_V101:
        if callable(_parse_attack_fixed_grid_prev_v101):
            return _parse_attack_fixed_grid_prev_v101(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)  # type: ignore[misc]
        return pd.DataFrame()

    grid = ATTACK_GRID
    if row_count is None:
        row_count = int(grid.get("row_count", 14) or 14)
    _pt = _prof_now()
    window_box = _detect_window_no_ocr_v101(image, reader) if auto_window else _base_window_box_scaled_v101(image)
    _prof_add("attack_1_window_detect_glyph_only_no_ocr", _pt)
    direction_kind = "back"  # 헤더 OCR을 쓰지 않는 실험 모드. 필요시 추후 헤더 템플릿화.

    _pt_rows = _prof_now()
    detected_rows = _detect_rows_no_ocr_v101(image, window_box, int(row_count))
    _prof_add("attack_2_icon_rows_detect_glyph_only", _pt_rows)

    _pt_glyph = _prof_now()
    store = _get_glyph_store_v705()
    rows: List[Dict[str, Any]] = []
    glyph_attempt = 0
    glyph_accept = 0
    for i, rinfo in enumerate(detected_rows[: int(row_count)]):
        row: Dict[str, Any] = {col: "" for col in STANDARD_COLUMNS}
        row["_ocr_row_index"] = str(i)
        row["_row_source"] = str(rinfo.get("source", "")) + ":glyph_only_v101"
        row["_direction_kind"] = direction_kind
        if window_box:
            row["_window_x1"], row["_window_y1"], row["_window_x2"], row["_window_y2"] = window_box
        ix1, iy1, ix2, iy2 = rinfo["icon_box"]
        row["_icon_x1"], row["_icon_y1"], row["_icon_x2"], row["_icon_y2"] = ix1, iy1, ix2, iy2
        row["_row_x1"], row["_row_y1"], row["_row_x2"], row["_row_y2"] = rinfo["row_box"]
        row["_glyph_only_v101"] = "1"

        directional_rate = ""
        directional_share = ""
        for key, label, x1, x2 in grid["columns"]:
            if window_box:
                box = _column_pixel_box_for_row(image, window_box, rinfo["row_box"], x1, x2)
            else:
                px1 = int(round(x1 * image.width)); px2 = int(round(x2 * image.width))
                box = _clip_box(image, (px1, rinfo["row_box"][1], px2, rinfo["row_box"][3]))
            if key == "name":
                continue
            kind = _glyph_kind_force_v101(key)
            if kind is None:
                continue
            glyph_attempt += 1
            text, conf, method = _glyph_recognize_force_v101(store, image, box, key)
            if text:
                glyph_accept += 1
            row[f"_glyph_{key}_conf"] = round(float(conf or 0.0), 4)
            row[f"_glyph_{key}_method"] = method
            val = _postprocess_cell_value_v705(key, text)
            if key == "directional_rate":
                directional_rate = val
            elif key == "directional_share":
                directional_share = val
            else:
                row[label] = val
        if direction_kind == "head":
            row["헤드어택 적중률"] = directional_rate
            row["헤드어택 비중"] = directional_share
        else:
            row["백어택 적중률"] = directional_rate
            row["백어택 비중"] = directional_share
        # 행이 완전 공백이라도 icon matching 테스트를 위해 유지합니다.
        rows.append(row)

    _GLYPH_LAST_STATS_V705["attempt"] = glyph_attempt
    _GLYPH_LAST_STATS_V705["accept"] = glyph_accept
    _GLYPH_LAST_STATS_V705["rate"] = (glyph_accept / glyph_attempt) if glyph_attempt else 0.0
    _prof_add("attack_3_glyph_only_numbers", _pt_glyph)

    cols = STANDARD_COLUMNS + [
        "_ocr_row_index", "_row_source", "_direction_kind",
        "_window_x1", "_window_y1", "_window_x2", "_window_y2",
        "_icon_x1", "_icon_y1", "_icon_x2", "_icon_y2",
        "_row_x1", "_row_y1", "_row_x2", "_row_y2",
        "_glyph_only_v101",
    ]
    # 동적으로 추가된 glyph conf/method 컬럼도 보존합니다.
    extra_cols = sorted({k for r in rows for k in r.keys() if k.startswith("_glyph_") and k not in cols})
    return pd.DataFrame(rows, columns=cols + extra_cols)


def _find_elapsed_crop_by_template_v101(image: Image.Image) -> Image.Image | None:
    try:
        import cv2
        tmpl = _load_battle_time_template_v46()
        if tmpl is None:
            return None
        th, tw = tmpl.shape[:2]
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        H, W = gray.shape[:2]
        band = gray[0:max(th + 4, int(H * 0.5)), :]
        base = H / BASE_SCREEN_H
        best = (-1.0, None, None)
        for m in (0.85, 0.92, 1.0, 1.08, 1.16, 1.25, 1.35):
            s = base * m
            tw2, th2 = int(round(tw * s)), int(round(th * s))
            if tw2 < 20 or th2 < 8 or th2 > band.shape[0] or tw2 > band.shape[1]:
                continue
            res = cv2.matchTemplate(band, cv2.resize(tmpl, (tw2, th2)), cv2.TM_CCOEFF_NORMED)
            _minv, maxv, _minl, maxl = cv2.minMaxLoc(res)
            if maxv > best[0]:
                best = (float(maxv), (maxl[0], maxl[1]), s)
        if best[1] is None or best[0] < BT_MATCH_THRESHOLD_V46:
            return None
        (lx, ly), s = best[1], best[2]
        nx1 = max(0, int(round(lx + tw * BT_NUM_X1_FRAC_V46 * s)))
        nx2 = min(W, int(round(lx + tw * BT_NUM_X2_FRAC_V46 * s)))
        ny1 = max(0, int(round(ly - 2 * s)))
        ny2 = min(H, int(round(ly + th * s + 2 * s)))
        if nx2 - nx1 < 8 or ny2 - ny1 < 6:
            return None
        return image.convert("RGB").crop((nx1, ny1, nx2, ny2))
    except Exception:
        return None


def _glyph_read_summary_raw_line_v101(store: Any, image: Image.Image, window_box: Tuple[int, int, int, int], key: str) -> Tuple[str, float, str]:
    # guide 카드 하단 원시 정수 줄을 glyph로 읽습니다. kind=count면 숫자/쉼표만 반환됩니다.
    try:
        guide = _find_summary_card_by_guide_v94(image, window_box, key)
        if guide is not None:
            gx1, gy1, gx2, gy2, gscore, _gscale = guide
            crop = _crop_raw_line_from_card_v94(image, (gx1, gy1, gx2, gy2))
            text, conf = store.recognize(crop, kind="count")
            return clean_ocr_text(text or ""), float(conf or 0.0), f"guide_raw_line_glyph:{gscore:.3f}"
    except Exception:
        pass
    try:
        crop = crop_norm(image, SUMMARY_ROIS[key], pad=1, window_box=window_box)
        crop = _glyph_isolate_number_crop_v705(crop, pad=4, gap_merge=32)
        text, conf = store.recognize(crop, kind="count")
        return clean_ocr_text(text or ""), float(conf or 0.0), "summary_roi_number_isolate_glyph"
    except Exception:
        return "", 0.0, "glyph_failed"


def parse_summary_fixed_grid(image: Image.Image, reader: Any, scale: int = 7, auto_window: bool = True) -> Dict[str, Any]:  # type: ignore[override]
    if not _GLYPH_ONLY_NUMBERS_V101:
        if callable(_parse_summary_fixed_grid_prev_v101):
            return _parse_summary_fixed_grid_prev_v101(image, reader, scale=scale, auto_window=auto_window)  # type: ignore[misc]
        return {}
    _pt = _prof_now()
    window_box = _detect_window_no_ocr_v101(image, reader) if auto_window else _base_window_box_scaled_v101(image)
    _prof_add("summary_1_window_detect_glyph_only_no_ocr", _pt)
    store = _get_glyph_store_v705()
    raw: Dict[str, str] = {}
    debug: Dict[str, Any] = {}

    # 전투시간 glyph
    _pt_elapsed = _prof_now()
    elapsed_text = ""
    elapsed_conf = 0.0
    elapsed_method = ""
    try:
        ecrop = _find_elapsed_crop_by_template_v101(image)
        if ecrop is None:
            ecrop = crop_norm(image, SUMMARY_ROIS["elapsed_text"], pad=1, window_box=window_box)
            elapsed_method = "summary_elapsed_roi_glyph"
        else:
            elapsed_method = "battle_time_template_glyph"
        elapsed_text, elapsed_conf = store.recognize(ecrop, kind="time") if store is not None else ("", 0.0)
        elapsed_text = clean_ocr_text(elapsed_text or "")
    except Exception:
        elapsed_text, elapsed_conf = "", 0.0
    raw["elapsed_text"] = elapsed_text
    debug["elapsed"] = {"method": elapsed_method, "conf": elapsed_conf}
    _prof_add("summary_2_elapsed_glyph_only", _pt_elapsed)

    # 총피해량/DPS 원시 숫자 줄 glyph
    _pt_cards = _prof_now()
    for key in ("total_damage_text", "dps_text"):
        text, conf, method = _glyph_read_summary_raw_line_v101(store, image, window_box, key) if store is not None else ("", 0.0, "glyph_store_missing")
        raw[key] = text
        debug[key] = {"method": method, "conf": conf}
    _prof_add("summary_3_damage_dps_glyph_only", _pt_cards)

    try:
        from modules.calculators import format_korean_number
    except Exception:
        from .calculators import format_korean_number  # type: ignore

    total_value = full_raw_number_from_summary(raw.get("total_damage_text", ""))
    dps_value = full_raw_number_from_summary(raw.get("dps_text", ""))
    elapsed_seconds = extract_elapsed_seconds(raw.get("elapsed_text", ""))
    meta: Dict[str, Any] = {
        "raw": raw,
        "window_box": window_box,
        "window_detected": True,
        "summary_glyph_only_v101": True,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_source": "glyph_only" if elapsed_seconds is not None else "glyph_failed",
        "total_damage": total_value,
        "dps": dps_value,
        "total_damage_text": format_korean_number(total_value) if total_value else raw.get("total_damage_text", ""),
        "dps_text": format_korean_number(dps_value) if dps_value else raw.get("dps_text", ""),
        "damage_increase_efficiency": None,
        "crit_rate": None,
        "head_attack_rate": None,
        "back_attack_rate": None,
        "damage_increase_uptime": None,
        "burst_uptime": None,
        "raw_display": {
            "elapsed_text": _format_elapsed_v83(elapsed_seconds) or raw.get("elapsed_text", ""),
            "total_damage_text": format_korean_number(total_value) if total_value else raw.get("total_damage_text", ""),
            "dps_text": format_korean_number(dps_value) if dps_value else raw.get("dps_text", ""),
        },
        "_debug_v101": debug,
    }
    return meta


def make_attack_ocr_debug(image: Image.Image, reader: Any, row_count: int = 18, scale: int = 7) -> List[Dict[str, Any]]:  # type: ignore[override]
    """v101: 통합 디버그에도 glyph-only 인식값/신뢰도를 보여줍니다."""
    if not _GLYPH_ONLY_NUMBERS_V101:
        if callable(_make_attack_ocr_debug_prev_v101):
            return _make_attack_ocr_debug_prev_v101(image, reader, row_count=row_count, scale=scale)  # type: ignore[misc]
        return []
    grid = ATTACK_GRID
    window_box = _detect_window_no_ocr_v101(image, reader)
    detected_rows = _detect_rows_no_ocr_v101(image, window_box, int(row_count))
    store = _get_glyph_store_v705()
    rows: List[Dict[str, Any]] = []
    for i, rinfo in enumerate(detected_rows[: int(row_count)]):
        row_crop = _crop_pixel_box(image, rinfo["row_box"], pad=0, expand_ratio=0.01)
        icon_crop = _crop_pixel_box(image, rinfo["icon_box"], pad=0, expand_ratio=ICON_CROP_EXPAND_RATIO)
        cells: List[Dict[str, Any]] = []
        for key, label, x1, x2 in grid["columns"]:
            if window_box:
                box = _column_pixel_box_for_row(image, window_box, rinfo["row_box"], x1, x2)
            else:
                px1 = int(round(x1 * image.width)); px2 = int(round(x2 * image.width))
                box = _clip_box(image, (px1, rinfo["row_box"][1], px2, rinfo["row_box"][3]))
            raw_crop = _glyph_force_crop_v101(image, box)
            if key == "name":
                text, conf, method = "", 0.0, "name_by_icon_match"
            else:
                text, conf, method = _glyph_recognize_force_v101(store, image, box, key)
            cells.append({
                "key": key,
                "label": label,
                "text": text,
                "raw_crop": raw_crop,
                "processed_crop": raw_crop,
                "preprocess": method,
                "score": conf,
            })
        rows.append({
            "index": i,
            "row_crop": row_crop,
            "icon_crop": icon_crop,
            "cells": cells,
            "row_box": rinfo["row_box"],
            "icon_box": rinfo["icon_box"],
            "window_box": window_box,
            "source": str(rinfo.get("source", "")) + ":glyph_only_v101",
        })
    return rows


def make_summary_ocr_debug(image: Image.Image, reader: Any, scale: int = 6) -> List[Dict[str, Any]]:  # type: ignore[override]
    if not _GLYPH_ONLY_NUMBERS_V101:
        if callable(_make_summary_ocr_debug_prev_v101):
            return _make_summary_ocr_debug_prev_v101(image, reader, scale=scale)  # type: ignore[misc]
        return []
    window_box = _detect_window_no_ocr_v101(image, reader)
    store = _get_glyph_store_v705()
    rows: List[Dict[str, Any]] = []
    # elapsed
    ecrop = _find_elapsed_crop_by_template_v101() if False else None
    try:
        ecrop = _find_elapsed_crop_by_template_v101(image) or crop_norm(image, SUMMARY_ROIS["elapsed_text"], pad=1, window_box=window_box)
        etxt, econf = store.recognize(ecrop, kind="time") if store is not None else ("", 0.0)
    except Exception:
        ecrop, etxt, econf = crop_norm(image, SUMMARY_ROIS["elapsed_text"], pad=1, window_box=window_box), "", 0.0
    rows.append({"key": "elapsed_text", "label": "전투 시간", "text": etxt, "raw_text": etxt, "raw_crop": ecrop, "processed_crop": ecrop, "window_box": window_box, "preprocess": "glyph_only_time", "score": econf})
    for key, label in (("total_damage_text", "총 피해량"), ("dps_text", "DPS")):
        crop = None
        try:
            guide = _find_summary_card_by_guide_v94(image, window_box, key)
            if guide is not None:
                crop = _crop_raw_line_from_card_v94(image, guide[:4])
            else:
                crop = crop_norm(image, SUMMARY_ROIS[key], pad=1, window_box=window_box)
            text, conf = store.recognize(crop, kind="count") if store is not None else ("", 0.0)
        except Exception:
            text, conf = "", 0.0
            if crop is None:
                crop = crop_norm(image, SUMMARY_ROIS[key], pad=1, window_box=window_box)
        rows.append({"key": key, "label": label, "text": text, "raw_text": text, "raw_crop": crop, "processed_crop": crop, "window_box": window_box, "preprocess": "glyph_only_raw_integer", "score": conf})
    return rows


# ==============================================================================
# v102: common/rune first selection + canonical 64x64 icon note
# ==============================================================================
def _select_match_from_ranked_v83(  # type: ignore[override]
    original_value: Any,
    ranked: List[Dict[str, Any]],
    *,
    row_icon: Image.Image | None,
    text_threshold: float = 0.55,
    icon_threshold: float = 0.72,
) -> Tuple[str, float, float, bool, str]:
    """v102: 스킬 후보가 하나라도 있으면 공용 룬 후보를 무시하던 문제를 수정합니다.

    이전 로직은 skill_ranked가 존재하면 무조건 스킬 후보만 먼저 봤습니다. 그 결과
    `스킬룬 중독`이 77점으로 1등이어도, 낮은 점수의 클래스 스킬 후보만 보고 `기타`로
    떨어질 수 있었습니다. 이제는 best overall이 룬/기본공격/기타일 때 우선 판정합니다.
    """
    original = str(original_value or "").strip()
    canonical = canonicalize_battle_name_v31(original)
    if not ranked:
        return canonical or original, 0.0, 0.0, bool(canonical and canonical != original), "canonical" if canonical != original else "no_candidates"

    def _score100(item: Dict[str, Any]) -> float:
        try:
            return float(item.get("icon_score") or 0.0)
        except Exception:
            return 0.0

    def _text100(item: Dict[str, Any]) -> float:
        try:
            return float(item.get("text_score") or 0.0)
        except Exception:
            return 0.0

    def _src(item: Dict[str, Any]) -> str:
        return str(item.get("source") or "")

    def _name(item: Dict[str, Any]) -> str:
        return _canonical_display_name_v86(str(item.get("name") or original))

    threshold100 = float(icon_threshold) * 100.0
    best = ranked[0]
    best_name = _name(best)
    best_score = _score100(best)
    best_text = _text100(best)
    best_src = _src(best)
    best_group = str(best.get("group") or "")
    second_score = _score100(ranked[1]) if len(ranked) > 1 else 0.0
    margin = best_score - second_score

    is_common = (
        best_group != "skill"
        or "rune" in best_src.lower()
        or "local_common" in best_src.lower()
        or best_name.startswith("스킬룬")
        or best_name in {"기본 공격", "기타"}
    )

    # '수라결 기본 공격'(브레이커 수라의 길 전용)처럼 클래스 고유 기본공격은,
    # 공용 basic_attack 아이콘이 가로채면 안 됩니다. OCR로 읽힌 이름이 정확히 공용 '기본 공격'이
    # 아니라 앞에 클래스 고유 접두어가 붙어 있으면, 공용 우선 매칭을 건너뛰고 스킬 후보를 우선합니다.
    _is_class_basic = ("기본 공격" in original) and (original.strip() not in ("기본 공격", "기본공격"))

    # 룬/기본공격은 스킬 후보보다 먼저 인정합니다. `기타`는 너무 쉽게 먹지 않도록 기준을 높게 둡니다.
    if row_icon is not None and is_common and not _is_class_basic:
        if best_name != "기타" and best_score >= max(70.0, threshold100 - 6.0):
            return best_name, round(best_text, 2), round(best_score, 2), True, f"icon_common_first:{best_src}"
        if best_name == "기타" and best_score >= max(88.0, threshold100 + 8.0):
            return best_name, round(best_text, 2), round(best_score, 2), True, f"icon_common_etc_high:{best_src}"

    # 일반 스킬 판정. 이제 skill 후보가 있어도 전체 1등 common이 확실하면 위에서 먼저 빠집니다.
    skill_ranked = [r for r in ranked if r.get("group") == "skill"]
    if skill_ranked:
        if row_icon is not None:
            best_skill = skill_ranked[0]
            skill_score = _score100(best_skill)
            skill_second = _score100(skill_ranked[1]) if len(skill_ranked) > 1 else 0.0
            low_conf = (skill_score < ICON_MATCH_MIN_SCORE_V46) or (
                skill_score < ICON_MATCH_TIE_SCORE_V46 and (skill_score - skill_second) < ICON_MATCH_TIE_MARGIN_V46
            )
            if not low_conf:
                return _name(best_skill), 0.0, round(skill_score, 2), True, f"icon_skill_only:{_src(best_skill)}"

            # 스킬은 낮지만 common 룬이 꽤 높으면 룬으로 채택합니다. v101 row8 Poison rune 케이스.
            common_ranked = [r for r in ranked if r not in skill_ranked]
            if common_ranked:
                common_best = common_ranked[0]
                common_name = _name(common_best)
                common_score = _score100(common_best)
                if common_name != "기타" and common_score >= max(70.0, threshold100 - 6.0):
                    return common_name, _text100(common_best), round(common_score, 2), True, f"icon_common_over_low_skill:{_src(common_best)}"

            return "기타", 0.0, round(skill_score, 2), True, f"icon_low_conf_etc:{_src(best_skill)}"
        best_skill = max(skill_ranked, key=lambda x: x.get("text_score", 0.0))
        return _name(best_skill), _text100(best_skill), 0.0, True, f"icon_skill_noicon:{_src(best_skill)}"

    # 스킬 후보가 없는 경우의 기존 general 판정.
    best_icon = best_score / 100.0
    best_text_ratio = best_text / 100.0
    second_icon = second_score / 100.0
    if best_icon >= float(icon_threshold) and ((best_icon - second_icon) >= 0.035 or best_icon >= min(0.94, float(icon_threshold) + 0.12)):
        return best_name, round(best_text, 2), round(best_score, 2), True, f"icon_first:{best_src}"

    text_best = max(ranked, key=lambda x: x.get("text_score", 0.0))
    text_score = _text100(text_best) / 100.0
    text_icon = _score100(text_best) / 100.0
    if text_score >= float(text_threshold):
        return _name(text_best), round(text_score * 100.0, 2), round(text_icon * 100.0, 2), True, f"text_fallback:{_src(text_best)}"
    if canonical and canonical != original:
        return canonical, round(text_score * 100.0, 2), round(best_icon * 100.0, 2), True, "canonical"
    return original, round(text_score * 100.0, 2), round(best_icon * 100.0, 2), False, "unmatched"


# ==============================================================================
# v107: 정확도 우선 모드
# - 아이콘 행 감지가 일부 행만 잡으면 전체 행을 고정 그리드로 사용합니다.
# - 해상도 픽셀 고정이 아니라 전투분석기 window_box 기준 비율/행높이로 계산합니다.
# - run.bat에서 LOA_FORCE_FIXED_ATTACK_ROWS=1이면 무조건 고정 행을 사용합니다.
# ==============================================================================
try:
    _detect_attack_icon_rows_prev_v107 = detect_attack_icon_rows  # type: ignore[name-defined]
except Exception:
    _detect_attack_icon_rows_prev_v107 = None


def _fixed_attack_rows_from_window_v107(image: Image.Image, window_box: tuple[int, int, int, int] | None, max_rows: int) -> list[dict[str, Any]]:
    grid = ATTACK_GRID
    rows: list[dict[str, Any]] = []
    try:
        n = int(max_rows or grid.get("row_count", 18) or 18)
    except Exception:
        n = 18
    for i in range(n):
        y1 = grid["row_start"] + i * grid["row_height"]
        y2 = y1 + grid["row_height"]
        row_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, grid["columns"][-1][3], y2), pad=0, window_box=window_box)
        icon_box = _norm_box_to_pixels(image, (ATTACK_ICON_GRID["x1"], y1, ATTACK_ICON_GRID["x2"], y2), pad=1, window_box=window_box)
        rows.append({"row_index": i, "row_box": row_box, "icon_box": icon_box, "window_box": window_box, "source": "fixed_rows_v107"})
    return rows


def detect_attack_icon_rows(  # type: ignore[override]
    image: Image.Image,
    reader: Any | None = None,
    *,
    window_box: tuple[int, int, int, int] | None = None,
    max_rows: int = 18,
) -> list[dict[str, Any]]:
    """v107: 아이콘 contour 감지가 행을 놓치면 고정 행 비율로 전환합니다."""
    try:
        import os as _os_v107
        force_fixed = str(_os_v107.environ.get("LOA_FORCE_FIXED_ATTACK_ROWS", "0")).lower() in {"1", "true", "yes", "on"}
        min_rows = int(_os_v107.environ.get("LOA_MIN_DETECTED_ATTACK_ROWS", "12"))
    except Exception:
        force_fixed = False
        min_rows = 12
    if force_fixed:
        return _fixed_attack_rows_from_window_v107(image, window_box, max_rows)
    rows: list[dict[str, Any]] = []
    if callable(_detect_attack_icon_rows_prev_v107):
        try:
            rows = _detect_attack_icon_rows_prev_v107(image, reader, window_box=window_box, max_rows=max_rows)  # type: ignore[misc]
        except Exception:
            rows = []
    # 이전 감지가 5행처럼 너무 적게 잡히면 전체 표가 망가지므로 고정 행을 사용합니다.
    try:
        if len(rows) < min(min_rows, int(max_rows or min_rows)):
            return _fixed_attack_rows_from_window_v107(image, window_box, max_rows)
    except Exception:
        return _fixed_attack_rows_from_window_v107(image, window_box, max_rows)
    return rows




# ==============================================================================
# v108: 공격정보 탭 고정 그리드 재정의 (창 기준 비율 crop)
# ==============================================================================
# 사용자 피드백:
# - 공격정보 탭은 창 전체를 찾은 뒤, 표를 창 비율로 고정 분할하는 편이 더 안정적입니다.
# - 스킬 아이콘은 이름 글자를 포함하지 않고 정사각형으로만 crop 합니다.
# - 공격정보 표는 이름 OCR을 하지 않고, 아이콘 매칭 단계에서 이름을 채웁니다.
# - '-' 표시는 0으로 정규화합니다.

ATTACK_FIXED_GRID_V108 = {
    # detected analyzer window 기준 비율
    'row_start': 0.190,
    'row_height': 0.0492,
    'row_count': 14,
    'table_x1': 0.010,
    'table_x2': 0.844,
    'icon_x1': 0.009,
    'icon_x2': 0.037,
    'name_x1': 0.038,
    'name_x2': 0.155,
    'columns': [
        ('damage_text', '피해량', 0.157, 0.252),
        ('directional_rate', '백어택 적중률', 0.254, 0.350),
        ('directional_share', '백어택 비중', 0.353, 0.449),
        ('crit_rate', '치명타 적중률', 0.450, 0.548),
        ('crit_share', '치명타 비중', 0.550, 0.644),
        ('casts', '사용 횟수', 0.646, 0.744),
        ('cooldown_rate', '쿨타임 비율', 0.748, 0.844),
    ],
}


def _window_rel_box_v108(window_box, rel_x1, rel_y1, rel_x2, rel_y2):
    wx1, wy1, wx2, wy2 = window_box
    ww = max(1, int(wx2 - wx1))
    wh = max(1, int(wy2 - wy1))
    return (
        int(round(wx1 + ww * rel_x1)),
        int(round(wy1 + wh * rel_y1)),
        int(round(wx1 + ww * rel_x2)),
        int(round(wy1 + wh * rel_y2)),
    )


def _normalize_dash_to_zero_v108(key: str, text: str) -> str:
    t = clean_ocr_text(str(text or ''))
    if t in {'', '-', '—', '–', 'ㅡ', '一'}:
        if key in {'damage_text'}:
            return '0'
        if key in {'casts'}:
            return '0'
        if key in {'directional_rate', 'directional_share', 'crit_rate', 'crit_share', 'cooldown_rate'}:
            return '0.0'
        return '0'
    return t


def _ocr_fixed_cell_v108(reader: Any, image: Image.Image, box: tuple[int, int, int, int], key: str, scale: int = 7) -> str:
    try:
        txt = _ocr_cell_precise_v705(reader, image, box, key, scale=scale)
    except Exception:
        raw = _crop_pixel_box(image, box, pad=1, expand_ratio=0.02)
        numeric = key != 'name'
        txt = ''
        best = -1.0
        for _tag, proc, cand, score in _ocr_text_variants(reader, raw, numeric=numeric, scale=max(6, scale), kind='korean_number' if key == 'damage_text' else ('percent' if key in {'directional_rate', 'directional_share', 'crit_rate', 'crit_share', 'cooldown_rate'} else 'count')):
            if score > best:
                txt = cand
                best = score
        try:
            txt = _postprocess_cell_value_v705(key, txt)
        except Exception:
            txt = clean_ocr_text(txt)
    return _normalize_dash_to_zero_v108(key, txt)


def _iter_fixed_grid_rows_v108(image: Image.Image, window_box: tuple[int, int, int, int], row_count: int):
    grid = ATTACK_FIXED_GRID_V108
    for i in range(int(row_count or grid['row_count'])):
        rel_y1 = grid['row_start'] + i * grid['row_height']
        rel_y2 = rel_y1 + grid['row_height']
        row_box = _window_rel_box_v108(window_box, grid['table_x1'], rel_y1, grid['table_x2'], rel_y2)
        icon_box = _window_rel_box_v108(window_box, grid['icon_x1'], rel_y1, grid['icon_x2'], rel_y2)
        yield i, _clip_box(image, row_box, pad=0), _clip_box(image, icon_box, pad=0)


def parse_attack_fixed_grid(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int | None = None,
    scale: int = 7,
    auto_window: bool = True,
) -> pd.DataFrame:
    """v108: 공격정보 탭을 창 기준 고정 그리드로 자릅니다.

    - 이름 텍스트 OCR은 하지 않습니다(아이콘 매칭 단계에서 이름 채움).
    - 스킬 아이콘은 행 높이 기준 정사각형 영역만 잘라 후속 아이콘 매칭에 넘깁니다.
    - 숫자/퍼센트 칸은 셀별 정밀 OCR로 읽습니다.
    - '-'는 0으로 정규화합니다.
    """
    grid = ATTACK_FIXED_GRID_V108
    if row_count is None:
        row_count = int(grid.get('row_count', 14) or 14)
    try:
        window_box = _detect_window_no_ocr_v101(image, reader) if auto_window else _base_window_box_scaled_v101(image)
    except Exception:
        try:
            window_box = detect_analyzer_window(image, reader)
        except Exception:
            window_box = None
    if window_box is None:
        try:
            if callable(_parse_attack_fixed_grid_prev_v101):
                return _parse_attack_fixed_grid_prev_v101(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)  # type: ignore[misc]
        except Exception:
            pass
        return pd.DataFrame()

    rows = []
    for i, row_box, icon_box in _iter_fixed_grid_rows_v108(image, window_box, int(row_count)):
        row: Dict[str, Any] = {col: '' for col in STANDARD_COLUMNS}
        row['_ocr_row_index'] = str(i)
        row['_row_source'] = 'fixed_window_grid_v108'
        row['_direction_kind'] = 'back'
        row['_window_x1'], row['_window_y1'], row['_window_x2'], row['_window_y2'] = window_box
        row['_icon_x1'], row['_icon_y1'], row['_icon_x2'], row['_icon_y2'] = icon_box
        row['_row_x1'], row['_row_y1'], row['_row_x2'], row['_row_y2'] = row_box
        row['_grid_mode_v108'] = '1'
        # 기본값: 표에 없는 칸은 0 또는 공란으로 둡니다.
        row['헤드어택 적중률'] = '0.0'
        row['초당 피해량'] = ''
        row['피해량 지분'] = ''
        for key, label, rx1, rx2 in grid['columns']:
            cell_box = _window_rel_box_v108(window_box, rx1, grid['row_start'] + i * grid['row_height'], rx2, grid['row_start'] + (i + 1) * grid['row_height'])
            val = _ocr_fixed_cell_v108(reader, image, _clip_box(image, cell_box, pad=1), key, scale=max(7, scale))
            if key == 'damage_text':
                row['피해량'] = val
            elif key == 'directional_rate':
                row['백어택 적중률'] = val
            elif key == 'directional_share':
                row['백어택 비중'] = val
            elif key == 'crit_rate':
                row['치명타 적중률'] = val
            elif key == 'crit_share':
                row['치명타 비중'] = val
            elif key == 'casts':
                row['사용 횟수'] = val
            elif key == 'cooldown_rate':
                row['쿨타임 비율'] = val
        rows.append(row)
    cols = STANDARD_COLUMNS + [
        '_ocr_row_index', '_row_source', '_direction_kind',
        '_window_x1', '_window_y1', '_window_x2', '_window_y2',
        '_icon_x1', '_icon_y1', '_icon_x2', '_icon_y2',
        '_row_x1', '_row_y1', '_row_x2', '_row_y2',
        '_grid_mode_v108',
    ]
    return pd.DataFrame(rows, columns=cols)


def make_attack_ocr_debug(image: Image.Image, reader: Any, row_count: int = 18, scale: int = 7) -> List[Dict[str, Any]]:  # type: ignore[override]
    """v108: 고정 그리드 crop을 그대로 보여주는 디버그."""
    try:
        window_box = _detect_window_no_ocr_v101(image, reader)
    except Exception:
        try:
            window_box = detect_analyzer_window(image, reader)
        except Exception:
            window_box = None
    if window_box is None:
        if callable(_make_attack_ocr_debug_prev_v101):
            return _make_attack_ocr_debug_prev_v101(image, reader, row_count=row_count, scale=scale)  # type: ignore[misc]
        return []

    out: List[Dict[str, Any]] = []
    rc = min(int(row_count or ATTACK_FIXED_GRID_V108['row_count']), int(ATTACK_FIXED_GRID_V108['row_count']))
    for i, row_box, icon_box in _iter_fixed_grid_rows_v108(image, window_box, rc):
        row_crop = _crop_pixel_box(image, row_box, pad=0)
        icon_crop = _crop_pixel_box(image, icon_box, pad=0)
        cells = []
        for key, label, rx1, rx2 in ATTACK_FIXED_GRID_V108['columns']:
            cell_box = _window_rel_box_v108(window_box, rx1, ATTACK_FIXED_GRID_V108['row_start'] + i * ATTACK_FIXED_GRID_V108['row_height'], rx2, ATTACK_FIXED_GRID_V108['row_start'] + (i + 1) * ATTACK_FIXED_GRID_V108['row_height'])
            raw_crop = _crop_pixel_box(image, _clip_box(image, cell_box, pad=1), pad=0)
            text = _ocr_fixed_cell_v108(reader, image, _clip_box(image, cell_box, pad=1), key, scale=max(7, scale))
            cells.append({
                'key': key,
                'label': label,
                'text': text,
                'preprocess': 'fixed_window_grid_v108',
                'score': None,
                'window_box': _clip_box(image, cell_box, pad=1),
                'raw_crop': raw_crop,
                'processed_crop': None,
            })
        out.append({
            'row_index': i,
            'window_box': row_box,
            'row_crop': row_crop,
            'icon_crop': icon_crop,
            'cells': cells,
            'icon_match': {'name': '', 'reason': 'name_by_icon_match_after_parse_v108', 'icon_score': None, 'top': []},
        })
    return out



# ==============================================================================
# v109: 실제 디버그 이미지 기준 공격정보 그리드 재보정
# ==============================================================================
# v108 문제 원인:
# - 전투분석기 창 감지 결과가 사용자가 그린 가이드보다 오른쪽/아래쪽까지 넓게 잡혔습니다.
# - 그런데 v108의 열 비율은 그림 기준으로 잡혀서 첫 행이 헤더로 잡히고, 열도 왼쪽으로 밀렸습니다.
# - v109는 실제 통합 디버그 original_attack 기준으로, "감지된 window_box"에 대한 상대 좌표를 다시 맞춥니다.

ATTACK_FIXED_GRID_V109 = {
    'row_start': 0.2392,     # 첫 데이터 행. v108의 0.190은 헤더 행이었습니다.
    'row_height': 0.0492,
    'row_count': 14,
    'table_x1': 0.010,
    'table_x2': 0.982,
    'icon_x1': 0.010,
    'icon_x2': 0.036,
    'columns': [
        ('damage_text', '피해량', 0.176, 0.294),
        ('directional_rate', '백어택 적중률', 0.294, 0.411),
        ('directional_share', '백어택 비중', 0.412, 0.528),
        ('crit_rate', '치명타 적중률', 0.528, 0.643),
        ('crit_share', '치명타 비중', 0.644, 0.760),
        ('casts', '사용 횟수', 0.761, 0.876),
        ('cooldown_rate', '쿨타임 비율', 0.878, 0.982),
    ],
}


def _iter_fixed_grid_rows_v109(image: Image.Image, window_box: tuple[int, int, int, int], row_count: int):
    grid = ATTACK_FIXED_GRID_V109
    max_rows = min(int(row_count or grid['row_count']), int(grid['row_count']))
    for i in range(max_rows):
        rel_y1 = grid['row_start'] + i * grid['row_height']
        rel_y2 = rel_y1 + grid['row_height']
        row_box = _window_rel_box_v108(window_box, grid['table_x1'], rel_y1, grid['table_x2'], rel_y2)
        icon_box = _window_rel_box_v108(window_box, grid['icon_x1'], rel_y1, grid['icon_x2'], rel_y2)
        yield i, _clip_box(image, row_box, pad=0), _clip_box(image, icon_box, pad=0)


def parse_attack_fixed_grid(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int | None = None,
    scale: int = 7,
    auto_window: bool = True,
) -> pd.DataFrame:
    grid = ATTACK_FIXED_GRID_V109
    if row_count is None:
        row_count = int(grid.get('row_count', 14) or 14)
    try:
        window_box = _detect_window_no_ocr_v101(image, reader) if auto_window else _base_window_box_scaled_v101(image)
    except Exception:
        try:
            window_box = detect_analyzer_window(image, reader)
        except Exception:
            window_box = None
    if window_box is None:
        try:
            if callable(_parse_attack_fixed_grid_prev_v101):
                return _parse_attack_fixed_grid_prev_v101(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)  # type: ignore[misc]
        except Exception:
            pass
        return pd.DataFrame()

    rows = []
    for i, row_box, icon_box in _iter_fixed_grid_rows_v109(image, window_box, int(row_count)):
        row: Dict[str, Any] = {col: '' for col in STANDARD_COLUMNS}
        row['_ocr_row_index'] = str(i)
        row['_row_source'] = 'fixed_window_grid_v109_recalibrated'
        row['_direction_kind'] = 'back'
        row['_window_x1'], row['_window_y1'], row['_window_x2'], row['_window_y2'] = window_box
        row['_icon_x1'], row['_icon_y1'], row['_icon_x2'], row['_icon_y2'] = icon_box
        row['_row_x1'], row['_row_y1'], row['_row_x2'], row['_row_y2'] = row_box
        row['_grid_mode_v109'] = '1'
        row['헤드어택 적중률'] = '0.0'
        row['헤드어택 비중'] = '0.0'
        row['초당 피해량'] = ''
        row['피해량 지분'] = ''
        for key, label, rx1, rx2 in grid['columns']:
            cell_box = _window_rel_box_v108(window_box, rx1, grid['row_start'] + i * grid['row_height'], rx2, grid['row_start'] + (i + 1) * grid['row_height'])
            val = _ocr_fixed_cell_v108(reader, image, _clip_box(image, cell_box, pad=1), key, scale=max(7, scale))
            if key == 'damage_text':
                row['피해량'] = val
            elif key == 'directional_rate':
                row['백어택 적중률'] = val
            elif key == 'directional_share':
                row['백어택 비중'] = val
            elif key == 'crit_rate':
                row['치명타 적중률'] = val
            elif key == 'crit_share':
                row['치명타 비중'] = val
            elif key == 'casts':
                row['사용 횟수'] = val
            elif key == 'cooldown_rate':
                row['쿨타임 비율'] = val
        rows.append(row)
    cols = STANDARD_COLUMNS + [
        '_ocr_row_index', '_row_source', '_direction_kind',
        '_window_x1', '_window_y1', '_window_x2', '_window_y2',
        '_icon_x1', '_icon_y1', '_icon_x2', '_icon_y2',
        '_row_x1', '_row_y1', '_row_x2', '_row_y2',
        '_grid_mode_v109',
    ]
    return pd.DataFrame(rows, columns=cols)


def make_attack_ocr_debug(image: Image.Image, reader: Any, row_count: int = 18, scale: int = 7) -> List[Dict[str, Any]]:  # type: ignore[override]
    try:
        window_box = _detect_window_no_ocr_v101(image, reader)
    except Exception:
        try:
            window_box = detect_analyzer_window(image, reader)
        except Exception:
            window_box = None
    if window_box is None:
        if callable(_make_attack_ocr_debug_prev_v101):
            return _make_attack_ocr_debug_prev_v101(image, reader, row_count=row_count, scale=scale)  # type: ignore[misc]
        return []

    out: List[Dict[str, Any]] = []
    rc = min(int(row_count or ATTACK_FIXED_GRID_V109['row_count']), int(ATTACK_FIXED_GRID_V109['row_count']))
    for i, row_box, icon_box in _iter_fixed_grid_rows_v109(image, window_box, rc):
        row_crop = _crop_pixel_box(image, row_box, pad=0)
        icon_crop = _crop_pixel_box(image, icon_box, pad=0)
        cells = []
        for key, label, rx1, rx2 in ATTACK_FIXED_GRID_V109['columns']:
            cell_box = _window_rel_box_v108(window_box, rx1, ATTACK_FIXED_GRID_V109['row_start'] + i * ATTACK_FIXED_GRID_V109['row_height'], rx2, ATTACK_FIXED_GRID_V109['row_start'] + (i + 1) * ATTACK_FIXED_GRID_V109['row_height'])
            clipped = _clip_box(image, cell_box, pad=1)
            raw_crop = _crop_pixel_box(image, clipped, pad=0)
            text = _ocr_fixed_cell_v108(reader, image, clipped, key, scale=max(7, scale))
            cells.append({
                'key': key,
                'label': label,
                'text': text,
                'preprocess': 'fixed_window_grid_v109_recalibrated',
                'score': None,
                'window_box': clipped,
                'raw_crop': raw_crop,
                'processed_crop': None,
            })
        out.append({
            'row_index': i,
            'window_box': row_box,
            'row_crop': row_crop,
            'icon_crop': icon_crop,
            'cells': cells,
            'icon_match': {'name': '', 'reason': 'name_by_icon_match_after_parse_v109', 'icon_score': None, 'top': []},
        })
    return out



# ==============================================================================
# v110: title_template 기준 창 박스 + 실제 원본(실전_2.jpg) 기준 공격정보 grid 재보정
# ==============================================================================
# 확인 결과 title_template.png는 1920x1080 원본에서 (858,78)-(1042,116)에 거의 정확히 매칭되고,
# 중심 (950,97) 기준 창 박스는 기존 BASE_WINDOW_BOX=(214,68,1670,958)과 일치합니다.
# 따라서 문제는 제목 템플릿이 아니라, window_box 내부에서 공격정보 표 row/column 상대좌표가 틀어진 것입니다.
# v110은 사용자가 올린 원본 화면의 실제 셀 경계 기준으로 다시 계산했습니다.

ATTACK_FIXED_GRID_V110 = {
    # window_box=(214,68,1670,958) 기준 상대 좌표
    'row_start': 0.2506,       # full y 291
    'row_height': 0.04835,     # 약 43px / 890px
    'row_count': 14,
    'table_x1': 0.0096,        # full x 228
    'table_x2': 0.9920,        # full x 1658 부근, 스크롤바 제외
    'icon_x1': 0.0096,         # full x 228
    'icon_x2': 0.0378,         # full x 269
    'columns': [
        ('damage_text', '피해량', 0.1813, 0.2988),          # full x 478-649
        ('directional_rate', '백어택 적중률', 0.2988, 0.4162),
        ('directional_share', '백어택 비중', 0.4162, 0.5337),
        ('crit_rate', '치명타 적중률', 0.5337, 0.6525),
        ('crit_share', '치명타 비중', 0.6525, 0.7685),
        ('casts', '사용 횟수', 0.7685, 0.8860),
        ('cooldown_rate', '쿨타임 비율', 0.8860, 0.9920),
    ],
}


def _iter_fixed_grid_rows_v110(image: Image.Image, window_box: tuple[int, int, int, int], row_count: int):
    grid = ATTACK_FIXED_GRID_V110
    max_rows = min(int(row_count or grid['row_count']), int(grid['row_count']))
    for i in range(max_rows):
        rel_y1 = grid['row_start'] + i * grid['row_height']
        rel_y2 = rel_y1 + grid['row_height']
        row_box = _window_rel_box_v108(window_box, grid['table_x1'], rel_y1, grid['table_x2'], rel_y2)
        icon_box = _window_rel_box_v108(window_box, grid['icon_x1'], rel_y1, grid['icon_x2'], rel_y2)
        yield i, _clip_box(image, row_box, pad=0), _clip_box(image, icon_box, pad=0)


def parse_attack_fixed_grid(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int | None = None,
    scale: int = 7,
    auto_window: bool = True,
) -> pd.DataFrame:
    """v110: title_template로 전투분석기 창을 찾고, 창 내부를 실제 원본 기준 grid로 분할합니다."""
    grid = ATTACK_FIXED_GRID_V110
    if row_count is None:
        row_count = int(grid.get('row_count', 14) or 14)
    try:
        window_box = _detect_window_no_ocr_v101(image, reader) if auto_window else _base_window_box_scaled_v101(image)
    except Exception:
        try:
            window_box = detect_analyzer_window(image, reader)
        except Exception:
            window_box = None
    if window_box is None:
        try:
            if callable(_parse_attack_fixed_grid_prev_v101):
                return _parse_attack_fixed_grid_prev_v101(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)  # type: ignore[misc]
        except Exception:
            pass
        return pd.DataFrame()

    rows = []
    for i, row_box, icon_box in _iter_fixed_grid_rows_v110(image, window_box, int(row_count)):
        row: Dict[str, Any] = {col: '' for col in STANDARD_COLUMNS}
        row['_ocr_row_index'] = str(i)
        row['_row_source'] = 'title_window_fixed_grid_v110'
        row['_direction_kind'] = 'back'
        row['_window_x1'], row['_window_y1'], row['_window_x2'], row['_window_y2'] = window_box
        row['_icon_x1'], row['_icon_y1'], row['_icon_x2'], row['_icon_y2'] = icon_box
        row['_row_x1'], row['_row_y1'], row['_row_x2'], row['_row_y2'] = row_box
        row['_grid_mode_v110'] = '1'
        # 공격정보 탭에는 헤드/DPS/지분 칸이 없으므로 기본값은 비움 또는 0.
        row['헤드어택 적중률'] = '0.0'
        row['헤드어택 비중'] = '0.0'
        row['초당 피해량'] = ''
        row['피해량 지분'] = ''
        for key, label, rx1, rx2 in grid['columns']:
            rel_y1 = grid['row_start'] + i * grid['row_height']
            rel_y2 = rel_y1 + grid['row_height']
            cell_box = _window_rel_box_v108(window_box, rx1, rel_y1, rx2, rel_y2)
            val = _ocr_fixed_cell_v108(reader, image, _clip_box(image, cell_box, pad=1), key, scale=max(7, scale))
            if key == 'damage_text':
                row['피해량'] = val
            elif key == 'directional_rate':
                row['백어택 적중률'] = val
            elif key == 'directional_share':
                row['백어택 비중'] = val
            elif key == 'crit_rate':
                row['치명타 적중률'] = val
            elif key == 'crit_share':
                row['치명타 비중'] = val
            elif key == 'casts':
                row['사용 횟수'] = val
            elif key == 'cooldown_rate':
                row['쿨타임 비율'] = val
        rows.append(row)
    cols = STANDARD_COLUMNS + [
        '_ocr_row_index', '_row_source', '_direction_kind',
        '_window_x1', '_window_y1', '_window_x2', '_window_y2',
        '_icon_x1', '_icon_y1', '_icon_x2', '_icon_y2',
        '_row_x1', '_row_y1', '_row_x2', '_row_y2',
        '_grid_mode_v110',
    ]
    return pd.DataFrame(rows, columns=cols)


def make_attack_ocr_debug(image: Image.Image, reader: Any, row_count: int = 18, scale: int = 7) -> List[Dict[str, Any]]:  # type: ignore[override]
    """v110: 실제 사용 grid와 동일한 crop을 디버그 ZIP에 저장합니다."""
    try:
        window_box = _detect_window_no_ocr_v101(image, reader)
    except Exception:
        try:
            window_box = detect_analyzer_window(image, reader)
        except Exception:
            window_box = None
    if window_box is None:
        if callable(_make_attack_ocr_debug_prev_v101):
            return _make_attack_ocr_debug_prev_v101(image, reader, row_count=row_count, scale=scale)  # type: ignore[misc]
        return []

    out: List[Dict[str, Any]] = []
    rc = min(int(row_count or ATTACK_FIXED_GRID_V110['row_count']), int(ATTACK_FIXED_GRID_V110['row_count']))
    for i, row_box, icon_box in _iter_fixed_grid_rows_v110(image, window_box, rc):
        row_crop = _crop_pixel_box(image, row_box, pad=0)
        icon_crop = _crop_pixel_box(image, icon_box, pad=0)
        cells = []
        for key, label, rx1, rx2 in ATTACK_FIXED_GRID_V110['columns']:
            rel_y1 = ATTACK_FIXED_GRID_V110['row_start'] + i * ATTACK_FIXED_GRID_V110['row_height']
            rel_y2 = rel_y1 + ATTACK_FIXED_GRID_V110['row_height']
            cell_box = _window_rel_box_v108(window_box, rx1, rel_y1, rx2, rel_y2)
            pix = _clip_box(image, cell_box, pad=1)
            raw_crop = _crop_pixel_box(image, pix, pad=0)
            text = _ocr_fixed_cell_v108(reader, image, pix, key, scale=max(7, scale))
            cells.append({
                'key': key,
                'label': label,
                'text': text,
                'preprocess': 'title_window_fixed_grid_v110',
                'score': None,
                'window_box': pix,
                'raw_crop': raw_crop,
                'processed_crop': None,
            })
        out.append({
            'row_index': i,
            'window_box': row_box,
            'row_crop': row_crop,
            'icon_crop': icon_crop,
            'cells': cells,
            'icon_match': {'name': '', 'reason': 'name_by_icon_match_after_parse_v110', 'icon_score': None, 'top': []},
        })
    return out



# ==============================================================================
# v111: 아이콘 행 감지 기반 그리드 + summary raw-line OCR 전처리 완화
# ==============================================================================
# v110 고정 row_start/row_height는 상단 행은 맞지만 스크롤/행 간격 차이 때문에 아래쪽에서
# 반 칸씩 섞이는 문제가 있었습니다. v111은 창 기준 x-column은 유지하되, y 위치는 왼쪽
# 스킬 아이콘을 CV로 감지한 중심/경계로 잡습니다.

ATTACK_FIXED_GRID_V111 = dict(ATTACK_FIXED_GRID_V110)
ATTACK_FIXED_GRID_V111['row_source'] = 'icon_detected_y_grid_v111'


def _detect_icon_boxes_for_grid_v111(image: Image.Image, window_box: tuple[int, int, int, int], max_rows: int = 14):
    """왼쪽 스킬 아이콘의 채도/명도 영역을 찾아 행 y좌표를 얻습니다."""
    try:
        import cv2
        arr = np.asarray(image.convert('RGB'))
        wx1, wy1, wx2, wy2 = [int(v) for v in window_box]
        ww = max(1, wx2 - wx1); wh = max(1, wy2 - wy1)
        # 아이콘 column 주변. window 비율을 쓰지만 여유를 조금 줍니다.
        sx1 = max(0, int(round(wx1 + ww * 0.006)))
        sx2 = min(image.width, int(round(wx1 + ww * 0.047)))
        sy1 = max(0, int(round(wy1 + wh * 0.235)))
        sy2 = min(image.height, int(round(wy1 + wh * 0.94)))
        strip = arr[sy1:sy2, sx1:sx2]
        if strip.size == 0:
            return []
        hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
        # 아이콘은 배경/글자보다 채도가 높습니다. 회색 UI 라인 제거용 조건.
        mask = (((hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 38))).astype('uint8') * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        min_y = wy1 + wh * 0.245  # 헤더/빈칸 제거
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            X1, Y1, X2, Y2 = sx1 + x, sy1 + y, sx1 + x + w, sy1 + y + h
            if w < 18 or h < 18:
                continue
            if Y1 < min_y:
                continue
            if X2 - X1 > 60 or Y2 - Y1 > 60:
                continue
            boxes.append((X1, Y1, X2, Y2))
        boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2.0)
        # 가까운 contour 중복 제거/병합
        merged = []
        for b in boxes:
            cx = (b[0] + b[2]) / 2.0; cy = (b[1] + b[3]) / 2.0
            if merged and abs(cy - ((merged[-1][1] + merged[-1][3]) / 2.0)) < 16:
                p = merged[-1]
                merged[-1] = (min(p[0], b[0]), min(p[1], b[1]), max(p[2], b[2]), max(p[3], b[3]))
            else:
                merged.append(b)
        # 너무 아래 잘린 마지막 조각은 제외하되, row_count까지 유지
        good = []
        for b in merged:
            if (b[3] - b[1]) < 22 and len(good) >= max_rows:
                continue
            good.append(b)
        return good[:max_rows]
    except Exception:
        return []


def _row_bands_from_icon_boxes_v111(boxes, window_box, image: Image.Image, max_rows: int):
    if not boxes:
        return []
    centers = [((b[1] + b[3]) / 2.0) for b in boxes]
    # 중앙 간격 기반. 이상치가 있어도 median으로 안정화.
    diffs = [centers[i+1] - centers[i] for i in range(len(centers)-1) if 28 <= centers[i+1] - centers[i] <= 60]
    row_h = int(round(float(np.median(diffs)) if diffs else 45.0))
    row_h = max(36, min(52, row_h))
    bands = []
    for i, b in enumerate(boxes[:max_rows]):
        cy = (b[1] + b[3]) / 2.0
        y1 = int(round(cy - row_h / 2.0))
        y2 = int(round(cy + row_h / 2.0))
        # 아이콘은 실제 contour 중심 기준 정사각형. 이름 글자 제외.
        bw, bh = b[2] - b[0], b[3] - b[1]
        side = int(round(max(bw, bh, 34) + 4))
        icx = int(round((b[0] + b[2]) / 2.0)); icy = int(round((b[1] + b[3]) / 2.0))
        icon_box = (icx - side//2, icy - side//2, icx - side//2 + side, icy - side//2 + side)
        bands.append((_clip_box(image, (0, y1, image.width, y2), pad=0), _clip_box(image, icon_box, pad=0)))
    return bands


def _iter_fixed_grid_rows_v111(image: Image.Image, window_box: tuple[int, int, int, int], row_count: int):
    grid = ATTACK_FIXED_GRID_V111
    max_rows = min(int(row_count or grid.get('row_count', 14)), int(grid.get('row_count', 14)))
    wx1, wy1, wx2, wy2 = [int(v) for v in window_box]
    ww = max(1, wx2 - wx1)
    icon_boxes = _detect_icon_boxes_for_grid_v111(image, window_box, max_rows=max_rows)
    bands = _row_bands_from_icon_boxes_v111(icon_boxes, window_box, image, max_rows)
    # 감지가 실패하면 v110 고정 그리드 폴백
    if len(bands) < max(5, min(max_rows, 10)):
        for i, row_box, icon_box in _iter_fixed_grid_rows_v110(image, window_box, max_rows):
            yield i, row_box, icon_box, 'v110_fixed_fallback'
        return
    _bands = bands[:max_rows]
    _n = len(_bands)
    # v151: 마지막 행은 캡처 하단이 살짝 잘려 행 밴드가 실제 아이콘보다 위로 잡히는 문제가
    # 모든 전투분석기에서 공통으로 나타납니다. 아이콘 박스만 아래로 ≈0.18×아이콘변 내려
    # 크롭 정렬을 맞춥니다(텍스트 셀 위치는 그대로 유지 → 숫자 OCR 영향 없음).
    try:
        _gx1 = _window_rel_x_from_full_norm(ATTACK_ICON_GRID['x1'], window_box)
        _gx2 = _window_rel_x_from_full_norm(ATTACK_ICON_GRID['x2'], window_box)
        _last_dy = int(round(max(1, _gx2 - _gx1) * 0.18))
    except Exception:
        _last_dy = 0
    for i, (yband, icon_box) in enumerate(_bands):
        _x0, y1, _x2, y2 = yband
        row_box = (int(round(wx1 + ww * grid['table_x1'])), y1, int(round(wx1 + ww * grid['table_x2'])), y2)
        if i == _n - 1 and _last_dy:
            ix1, iy1, ix2, iy2 = icon_box
            icon_box = (ix1, iy1 + _last_dy, ix2, iy2 + _last_dy)
        yield i, _clip_box(image, row_box, pad=0), icon_box, 'icon_detected_y_grid_v111'


def _ocr_summary_raw_number_line_v94(reader: Any, raw_crop: Image.Image, *, scale: int = 4) -> Tuple[str, Image.Image, str, float]:  # type: ignore[override]
    """v111: summary 원시 숫자 줄은 이진 mask가 글자를 깨뜨려 OCR을 악화시켰습니다.
    사람이 읽을 수 있는 normal/contrast 전처리를 우선 사용하고, mask는 마지막 폴백으로만 둡니다.
    """
    use_scale = max(2, min(int(scale or 4), 4))
    variants = []
    base = raw_crop.convert('RGB')
    for tag, sc, contrast, sharp in [
        ('raw_upscale', use_scale, 1.25, 1.15),
        ('raw_upscale_big', min(5, use_scale + 1), 1.35, 1.25),
    ]:
        im = base.resize((max(1, base.width * sc), max(1, base.height * sc)), Image.Resampling.LANCZOS)
        im = ImageEnhance.Contrast(im).enhance(contrast)
        im = ImageEnhance.Sharpness(im).enhance(sharp)
        variants.append((tag, im))
    try:
        variants.append(('mask_fallback', _preprocess_numeric_text_mask(raw_crop, scale=use_scale)))
    except Exception:
        pass
    best_text, best_proc, best_method, best_score = '', variants[0][1], variants[0][0], -1.0
    for tag, proc in variants:
        try:
            try:
                res = reader.readtext(np.asarray(proc), detail=0, paragraph=True, allowlist='0123456789, ')
                text = ' '.join(str(x) for x in res) if isinstance(res, list) else str(res or '')
            except TypeError:
                text = _ocr_text_once(reader, proc, numeric=True)
        except Exception:
            text = ''
        text = clean_ocr_text(text)
        try:
            score = _numeric_ocr_score(text, kind='integer')
        except Exception:
            score = float(len(re.findall(r'\d', str(text or ''))))
        # comma가 포함된 긴 정수는 우대합니다.
        if re.search(r'\d,\d', text):
            score += 1.5
        if score > best_score:
            best_text, best_proc, best_method, best_score = text, proc, tag, score
    return best_text, best_proc, f'guide_raw_number_line_v111_{best_method}', float(best_score)


def parse_attack_fixed_grid(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int | None = None,
    scale: int = 7,
    auto_window: bool = True,
) -> pd.DataFrame:
    grid = ATTACK_FIXED_GRID_V111
    if row_count is None:
        row_count = int(grid.get('row_count', 14) or 14)
    try:
        window_box = _detect_window_no_ocr_v101(image, reader) if auto_window else _base_window_box_scaled_v101(image)
    except Exception:
        try:
            window_box = detect_analyzer_window(image, reader)
        except Exception:
            window_box = None
    if window_box is None:
        return pd.DataFrame()
    rows = []
    for i, row_box, icon_box, source in _iter_fixed_grid_rows_v111(image, window_box, int(row_count)):
        row: Dict[str, Any] = {col: '' for col in STANDARD_COLUMNS}
        row['_ocr_row_index'] = str(i)
        row['_row_source'] = source
        row['_direction_kind'] = 'back'
        row['_window_x1'], row['_window_y1'], row['_window_x2'], row['_window_y2'] = window_box
        row['_icon_x1'], row['_icon_y1'], row['_icon_x2'], row['_icon_y2'] = icon_box
        row['_row_x1'], row['_row_y1'], row['_row_x2'], row['_row_y2'] = row_box
        row['_grid_mode_v111'] = '1'
        row['헤드어택 적중률'] = '0.0'
        row['헤드어택 비중'] = '0.0'
        row['초당 피해량'] = ''
        row['피해량 지분'] = ''
        for key, label, rx1, rx2 in grid['columns']:
            cell_box = (int(round(window_box[0] + (window_box[2]-window_box[0]) * rx1)), row_box[1], int(round(window_box[0] + (window_box[2]-window_box[0]) * rx2)), row_box[3])
            val = _ocr_fixed_cell_v108(reader, image, _clip_box(image, cell_box, pad=1), key, scale=max(7, scale))
            if key == 'damage_text': row['피해량'] = val
            elif key == 'directional_rate': row['백어택 적중률'] = val
            elif key == 'directional_share': row['백어택 비중'] = val
            elif key == 'crit_rate': row['치명타 적중률'] = val
            elif key == 'crit_share': row['치명타 비중'] = val
            elif key == 'casts': row['사용 횟수'] = val
            elif key == 'cooldown_rate': row['쿨타임 비율'] = val
        rows.append(row)
    cols = STANDARD_COLUMNS + [
        '_ocr_row_index', '_row_source', '_direction_kind',
        '_window_x1', '_window_y1', '_window_x2', '_window_y2',
        '_icon_x1', '_icon_y1', '_icon_x2', '_icon_y2',
        '_row_x1', '_row_y1', '_row_x2', '_row_y2', '_grid_mode_v111',
    ]
    return pd.DataFrame(rows, columns=cols)


def _square_icon_box_from_grid_v150(image: Image.Image, window_box, center_box) -> "tuple[int, int, int, int] | None":
    """창 그리드 비율로 1:1 정사각형 아이콘 박스를 계산합니다(해상도 무관, 행마다 동일 크기).

    세로 중심은 center_box(아이콘 박스) 기준으로 잡습니다 → 마지막 행처럼 icon_box를 아래로
    내려둔 경우 그 보정이 그대로 반영됩니다. 매처(_square_icon_box_from_row_metadata_v97)와
    동일 규칙이라 디버그로 보이는 crop이 실제 매칭 crop과 같아집니다.
    """
    try:
        gx1 = _window_rel_x_from_full_norm(ATTACK_ICON_GRID["x1"], window_box)
        gx2 = _window_rel_x_from_full_norm(ATTACK_ICON_GRID["x2"], window_box)
        side = int(round(gx2 - gx1))
        if side <= 4:
            side = int(round(center_box[3] - center_box[1]))
        cy = (center_box[1] + center_box[3]) / 2.0
        return _clip_square_box_v97(image, int(round(gx1)), int(round(cy - side / 2.0)), side)
    except Exception:
        return None


def make_attack_ocr_debug(image: Image.Image, reader: Any, row_count: int = 18, scale: int = 7) -> List[Dict[str, Any]]:  # type: ignore[override]
    try:
        window_box = _detect_window_no_ocr_v101(image, reader)
    except Exception:
        try:
            window_box = detect_analyzer_window(image, reader)
        except Exception:
            window_box = None
    if window_box is None:
        return []
    out: List[Dict[str, Any]] = []
    for i, row_box, icon_box, source in _iter_fixed_grid_rows_v111(image, window_box, int(row_count or ATTACK_FIXED_GRID_V111['row_count'])):
        row_crop = _crop_pixel_box(image, row_box, pad=0)
        # v150/v151: 매칭과 동일한 정사각형(1:1) 아이콘 crop. 세로 중심은 (마지막 행이 내려간)
        # icon_box 기준으로 잡아 디버그 crop이 실제 매칭 crop과 일치하게 합니다.
        _sq_box = _square_icon_box_from_grid_v150(image, window_box, icon_box)
        icon_crop = _crop_pixel_box(image, _sq_box or icon_box, pad=0)
        cells = []
        for key, label, rx1, rx2 in ATTACK_FIXED_GRID_V111['columns']:
            cell_box = (int(round(window_box[0] + (window_box[2]-window_box[0]) * rx1)), row_box[1], int(round(window_box[0] + (window_box[2]-window_box[0]) * rx2)), row_box[3])
            pix = _clip_box(image, cell_box, pad=1)
            raw_crop = _crop_pixel_box(image, pix, pad=0)
            text = _ocr_fixed_cell_v108(reader, image, pix, key, scale=max(7, scale))
            cells.append({'key': key, 'label': label, 'text': text, 'preprocess': source, 'score': None, 'window_box': pix, 'raw_crop': raw_crop, 'processed_crop': None})
        out.append({'row_index': i, 'window_box': row_box, 'row_crop': row_crop, 'icon_crop': icon_crop, 'cells': cells, 'icon_match': {'name': '', 'reason': 'name_by_icon_match_after_parse_v111', 'icon_score': None, 'top': []}})
    return out



# ==============================================================================
# v112: v111 정확도 기반 FAST 모드
# ==============================================================================
# v111은 셀마다 정밀 OCR을 돌려 정확하지만 readtext 호출이 너무 많았습니다.
# v112는 행 전체를 1회 OCR(row-pass)로 먼저 읽고, 핵심 칸만 필요 시 정밀 재검수합니다.
# - row-pass: 각 행 1회 OCR로 퍼센트/쿨타임/횟수 대부분 처리
# - damage_text: 단위가 중요하므로 정밀 재검수 유지
# - casts: row-pass 실패 시에만 정밀 재검수

import os as _os_v112

ATTACK_FAST_GRID_V112 = str(_os_v112.environ.get('LOA_ATTACK_FAST_GRID', '1')).strip().lower() not in {'0', 'false', 'no', 'off'}
ATTACK_DAMAGE_RECHECK_V112 = str(_os_v112.environ.get('LOA_ATTACK_DAMAGE_RECHECK', '1')).strip().lower() not in {'0', 'false', 'no', 'off'}
ATTACK_CASTS_RECHECK_V112 = str(_os_v112.environ.get('LOA_ATTACK_CASTS_RECHECK', '1')).strip().lower() not in {'0', 'false', 'no', 'off'}
ROW_PASS_SCALE_V112 = int(_os_v112.environ.get('LOA_ATTACK_ROW_PASS_SCALE', '2') or 2)
CRITICAL_CELL_SCALE_V112 = int(_os_v112.environ.get('LOA_ATTACK_CRITICAL_CELL_SCALE', '5') or 5)

_parse_attack_fixed_grid_prev_v112 = globals().get('parse_attack_fixed_grid')


def _fixed_cell_box_v112(image: Image.Image, window_box: tuple[int, int, int, int], row_box: tuple[int, int, int, int], rx1: float, rx2: float) -> tuple[int, int, int, int]:
    wx1, wy1, wx2, wy2 = [int(v) for v in window_box]
    ww = max(1, wx2 - wx1)
    return _clip_box(image, (int(round(wx1 + ww * rx1)), row_box[1], int(round(wx1 + ww * rx2)), row_box[3]), pad=1)


def _valid_numberish_v112(key: str, val: str) -> bool:
    t = clean_ocr_text(str(val or ''))
    if not t:
        return False
    if t in {'0', '0.0'}:
        return True
    if key == 'damage_text':
        return bool(re.search(r'\d', t) and re.search(r'억|만|조|,|\.', t))
    if key == 'casts':
        return bool(re.fullmatch(r'\d+(?:\.0)?', t))
    if key in {'directional_rate','directional_share','crit_rate','crit_share','cooldown_rate'}:
        return bool(re.search(r'\d', t))
    return bool(t)


def parse_attack_fixed_grid(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int | None = None,
    scale: int = 7,
    auto_window: bool = True,
) -> pd.DataFrame:
    if not ATTACK_FAST_GRID_V112:
        if callable(_parse_attack_fixed_grid_prev_v112):
            return _parse_attack_fixed_grid_prev_v112(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)  # type: ignore[misc]
        return pd.DataFrame()

    grid = ATTACK_FIXED_GRID_V111
    if row_count is None:
        row_count = int(grid.get('row_count', 14) or 14)
    try:
        window_box = _detect_window_no_ocr_v101(image, reader) if auto_window else _base_window_box_scaled_v101(image)
    except Exception:
        try:
            window_box = detect_analyzer_window(image, reader)
        except Exception:
            window_box = None
    if window_box is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for i, row_box, icon_box, source in _iter_fixed_grid_rows_v111(image, window_box, int(row_count)):
        row: Dict[str, Any] = {col: '' for col in STANDARD_COLUMNS}
        row['_ocr_row_index'] = str(i)
        row['_row_source'] = source + ':fast_rowpass_v112'
        row['_direction_kind'] = 'back'
        row['_window_x1'], row['_window_y1'], row['_window_x2'], row['_window_y2'] = window_box
        row['_icon_x1'], row['_icon_y1'], row['_icon_x2'], row['_icon_y2'] = icon_box
        row['_row_x1'], row['_row_y1'], row['_row_x2'], row['_row_y2'] = row_box
        row['_grid_mode_v111'] = '1'
        row['_fast_grid_v112'] = '1'
        row['헤드어택 적중률'] = '0.0'
        row['헤드어택 비중'] = '0.0'
        row['초당 피해량'] = ''
        row['피해량 지분'] = ''

        col_boxes: list[tuple[str, str, int, int, int, int]] = []
        box_map: dict[str, tuple[int, int, int, int]] = {}
        for key, label, rx1, rx2 in grid['columns']:
            pix = _fixed_cell_box_v112(image, window_box, row_box, rx1, rx2)
            box_map[key] = pix
            col_boxes.append((key, label, pix[0], pix[1], pix[2], pix[3]))

        rowpass: dict[str, str] = {}
        try:
            rowpass = _ocr_row_single_pass_v705(reader, image, col_boxes, scale=max(2, ROW_PASS_SCALE_V112))
        except Exception:
            rowpass = {}
        row['_rowpass_used_v112'] = '1' if rowpass else '0'

        for key, label, _rx1, _rx2 in grid['columns']:
            raw_text = clean_ocr_text(rowpass.get(key, ''))
            try:
                val = _postprocess_cell_value_v705(key, raw_text)
            except Exception:
                val = raw_text
            val = _normalize_dash_to_zero_v108(key, val)

            # 피해량은 OCR 오차가 계산 전체에 치명적이어서 정밀 재검수를 유지합니다.
            if key == 'damage_text' and ATTACK_DAMAGE_RECHECK_V112:
                try:
                    precise = _ocr_fixed_cell_v108(reader, image, box_map[key], key, scale=max(5, CRITICAL_CELL_SCALE_V112))
                    if _valid_numberish_v112(key, precise):
                        val = precise
                        row['_damage_recheck_v112'] = '1'
                except Exception:
                    pass
            # 횟수는 row-pass가 비거나 숫자가 아니면만 재검수합니다.
            elif key == 'casts' and ATTACK_CASTS_RECHECK_V112 and not _valid_numberish_v112(key, val):
                try:
                    precise = _ocr_fixed_cell_v108(reader, image, box_map[key], key, scale=max(5, CRITICAL_CELL_SCALE_V112))
                    if _valid_numberish_v112(key, precise):
                        val = precise
                        row['_casts_recheck_v112'] = '1'
                except Exception:
                    pass

            if key == 'damage_text': row['피해량'] = val
            elif key == 'directional_rate': row['백어택 적중률'] = val
            elif key == 'directional_share': row['백어택 비중'] = val
            elif key == 'crit_rate': row['치명타 적중률'] = val
            elif key == 'crit_share': row['치명타 비중'] = val
            elif key == 'casts': row['사용 횟수'] = val
            elif key == 'cooldown_rate': row['쿨타임 비율'] = val
        rows.append(row)

    cols = STANDARD_COLUMNS + [
        '_ocr_row_index', '_row_source', '_direction_kind',
        '_window_x1', '_window_y1', '_window_x2', '_window_y2',
        '_icon_x1', '_icon_y1', '_icon_x2', '_icon_y2',
        '_row_x1', '_row_y1', '_row_x2', '_row_y2',
        '_grid_mode_v111', '_fast_grid_v112', '_rowpass_used_v112', '_damage_recheck_v112', '_casts_recheck_v112',
    ]
    # 없는 컬럼도 안전하게 생성
    for r in rows:
        for c in cols:
            r.setdefault(c, '')
    return pd.DataFrame(rows, columns=cols)



# ==============================================================================
# v113: FAST grid smart damage recheck + shifted icon crop variants
# ==============================================================================
# 목표:
# - v111/v112의 안정적인 행/열/아이콘 그리드를 유지합니다.
# - 피해량 재검수는 row-pass가 실패한 경우에만 수행해 OCR 호출 수를 줄입니다.
# - 아이콘 점수가 낮거나 1~2등이 근소하면, icon_box를 오른쪽/상하로 조금 이동한 crop도 비교합니다.
#   백렬권처럼 아이콘 crop이 왼쪽으로 밀려 잘린 케이스를 복구하기 위함입니다.

import os as _os_v113

_ATTACK_DAMAGE_RECHECK_MODE_V113 = str(_os_v113.environ.get('LOA_ATTACK_DAMAGE_RECHECK', 'smart')).strip().lower()
_ICON_VARIANT_RETRY_V113 = str(_os_v113.environ.get('LOA_ICON_VARIANT_RETRY', '1')).strip().lower() not in {'0','false','no','off'}
_ICON_LOWCONF_EXTRA_THRESHOLD_V113 = float(_os_v113.environ.get('LOA_ICON_LOWCONF_EXTRA_THRESHOLD', '68') or 68)
_parse_attack_fixed_grid_prev_v113 = globals().get('parse_attack_fixed_grid')
_correct_battle_skill_names_with_icons_prev_v113 = globals().get('correct_battle_skill_names_with_icons')


def _damage_recheck_needed_v113(rowpass_value: str) -> bool:
    """row-pass 피해량이 이미 544.21억/8,200.25만 형태면 정밀 OCR을 생략합니다."""
    t = clean_ocr_text(str(rowpass_value or ''))
    if not t or t in {'0','0.0','-'}:
        return True
    if re.search(r'\d', t) and re.search(r'조|억|만', t):
        return False
    # 395.4091 처럼 억이 9/91로 붙어 들어온 경우는 앱 단위보정 단계가 복구할 수 있으므로 통과시킵니다.
    if re.fullmatch(r'[\d,]+(?:\.\d+)?(?:9|91|94)?', t) and ('.' in t or ',' in t):
        return False
    return True


def parse_attack_fixed_grid(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int | None = None,
    scale: int = 7,
    auto_window: bool = True,
) -> pd.DataFrame:
    """v113: v112 fast grid에서 피해량 재검수를 smart 모드로 줄인 버전."""
    # v112 고속 모드가 꺼져 있으면 이전 체인을 그대로 사용합니다.
    try:
        if not ATTACK_FAST_GRID_V112:
            if callable(_parse_attack_fixed_grid_prev_v113):
                return _parse_attack_fixed_grid_prev_v113(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)  # type: ignore[misc]
            return pd.DataFrame()
    except Exception:
        pass

    grid = ATTACK_FIXED_GRID_V111
    if row_count is None:
        row_count = int(grid.get('row_count', 14) or 14)
    try:
        window_box = _detect_window_no_ocr_v101(image, reader) if auto_window else _base_window_box_scaled_v101(image)
    except Exception:
        try:
            window_box = detect_analyzer_window(image, reader)
        except Exception:
            window_box = None
    if window_box is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for i, row_box, icon_box, source in _iter_fixed_grid_rows_v111(image, window_box, int(row_count)):
        row: Dict[str, Any] = {col: '' for col in STANDARD_COLUMNS}
        row['_ocr_row_index'] = str(i)
        row['_row_source'] = source + ':fast_rowpass_v113'
        row['_direction_kind'] = 'back'
        row['_window_x1'], row['_window_y1'], row['_window_x2'], row['_window_y2'] = window_box
        row['_icon_x1'], row['_icon_y1'], row['_icon_x2'], row['_icon_y2'] = icon_box
        row['_row_x1'], row['_row_y1'], row['_row_x2'], row['_row_y2'] = row_box
        row['_grid_mode_v111'] = '1'
        row['_fast_grid_v112'] = '1'
        row['_fast_grid_v113'] = '1'
        row['헤드어택 적중률'] = '0.0'
        row['헤드어택 비중'] = '0.0'
        row['초당 피해량'] = ''
        row['피해량 지분'] = ''

        col_boxes: list[tuple[str, str, int, int, int, int]] = []
        box_map: dict[str, tuple[int, int, int, int]] = {}
        for key, label, rx1, rx2 in grid['columns']:
            pix = _fixed_cell_box_v112(image, window_box, row_box, rx1, rx2)
            box_map[key] = pix
            col_boxes.append((key, label, pix[0], pix[1], pix[2], pix[3]))

        rowpass: dict[str, str] = {}
        try:
            rowpass = _ocr_row_single_pass_v705(reader, image, col_boxes, scale=max(2, ROW_PASS_SCALE_V112))
        except Exception:
            rowpass = {}
        row['_rowpass_used_v112'] = '1' if rowpass else '0'

        for key, label, _rx1, _rx2 in grid['columns']:
            raw_text = clean_ocr_text(rowpass.get(key, ''))
            try:
                val = _postprocess_cell_value_v705(key, raw_text)
            except Exception:
                val = raw_text
            val = _normalize_dash_to_zero_v108(key, val)

            if key == 'damage_text':
                do_recheck = False
                if _ATTACK_DAMAGE_RECHECK_MODE_V113 in {'1','true','yes','on','always'}:
                    do_recheck = True
                elif _ATTACK_DAMAGE_RECHECK_MODE_V113 in {'smart','auto'}:
                    do_recheck = _damage_recheck_needed_v113(val)
                if do_recheck:
                    try:
                        precise = _ocr_fixed_cell_v108(reader, image, box_map[key], key, scale=max(4, CRITICAL_CELL_SCALE_V112))
                        if _valid_numberish_v112(key, precise):
                            val = precise
                            row['_damage_recheck_v112'] = '1'
                        else:
                            row['_damage_recheck_v112'] = '0'
                    except Exception:
                        row['_damage_recheck_v112'] = 'error'
                else:
                    row['_damage_recheck_v112'] = 'skip_valid_rowpass'
            elif key == 'casts' and ATTACK_CASTS_RECHECK_V112 and not _valid_numberish_v112(key, val):
                try:
                    precise = _ocr_fixed_cell_v108(reader, image, box_map[key], key, scale=max(4, CRITICAL_CELL_SCALE_V112))
                    if _valid_numberish_v112(key, precise):
                        val = precise
                        row['_casts_recheck_v112'] = '1'
                except Exception:
                    pass

            if key == 'damage_text': row['피해량'] = val
            elif key == 'directional_rate': row['백어택 적중률'] = val
            elif key == 'directional_share': row['백어택 비중'] = val
            elif key == 'crit_rate': row['치명타 적중률'] = val
            elif key == 'crit_share': row['치명타 비중'] = val
            elif key == 'casts': row['사용 횟수'] = val
            elif key == 'cooldown_rate': row['쿨타임 비율'] = val
        rows.append(row)

    cols = STANDARD_COLUMNS + [
        '_ocr_row_index', '_row_source', '_direction_kind',
        '_window_x1', '_window_y1', '_window_x2', '_window_y2',
        '_icon_x1', '_icon_y1', '_icon_x2', '_icon_y2',
        '_row_x1', '_row_y1', '_row_x2', '_row_y2',
        '_grid_mode_v111', '_fast_grid_v112', '_fast_grid_v113', '_rowpass_used_v112', '_damage_recheck_v112', '_casts_recheck_v112',
    ]
    for r in rows:
        for c in cols:
            r.setdefault(c, '')
    return pd.DataFrame(rows, columns=cols)


def _icon_variant_crops_v113(image: Image.Image, row: 'pd.Series | Dict[str, Any]', row_index: int) -> list[tuple[str, Image.Image]]:
    """아이콘 crop이 반 칸 정도 밀린 행을 복구하기 위한 소량 variant."""
    variants: list[tuple[str, Image.Image]] = []
    seen: set[tuple[int,int]] = set()
    try:
        base_box = _row_int_box_v97(row, 'icon')
        row_box = _row_int_box_v97(row, 'row')
        if base_box is None:
            return variants
        ix1, iy1, ix2, iy2 = base_box
        side0 = max(16, min(max(iy2 - iy1, ix2 - ix1), max(iy2 - iy1, ix2 - ix1) + 8))
        if row_box is not None:
            row_h = max(16, row_box[3] - row_box[1])
            side0 = max(side0, int(row_h * 0.86))
        cx0 = (ix1 + ix2) / 2.0
        cy0 = (iy1 + iy2) / 2.0
        if row_box is not None:
            cy0 = (row_box[1] + row_box[3]) / 2.0
        # 실제 디버그에서 아이콘이 왼쪽으로 잘리는 경우가 있어 오른쪽 이동 variant를 더 많이 둡니다.
        # v151: 마지막 행 등에서 행 밴드가 실제 아이콘보다 위/아래로 밀리는 경우를 잡기 위해
        # 세로 이동(sy) variant를 더 큰 폭(±0.15~0.24)까지 확장합니다. 저신뢰 행에서만 재시도되므로
        # 이미 잘 맞는 행에는 영향이 없습니다.
        shifts = [
            (0.00, 0.00, 1.00),
            (0.10, 0.00, 1.00),
            (0.18, 0.00, 1.04),
            (0.25, 0.00, 1.08),
            (0.12, -0.08, 1.04),
            (0.12, 0.08, 1.04),
            (-0.06, 0.00, 1.04),
            (0.00, 0.15, 1.00),
            (0.00, 0.24, 1.00),
            (0.00, -0.15, 1.00),
            (0.00, -0.24, 1.00),
            (0.00, 0.15, 1.14),
            (0.00, -0.15, 1.14),
            (0.10, 0.16, 1.00),
        ]
        for sx, sy, sm in shifts:
            side = int(round(side0 * sm))
            cx = cx0 + side0 * sx
            cy = cy0 + side0 * sy
            x1 = int(round(cx - side / 2.0))
            y1 = int(round(cy - side / 2.0))
            box = _clip_square_box_v97(image, x1, y1, side)
            sig = (box[0], box[1])
            if sig in seen:
                continue
            seen.add(sig)
            crop = _crop_pixel_box(image, box, pad=0, expand_ratio=0.0)
            variants.append((f'variant_sx{sx:+.2f}_sy{sy:+.2f}_s{sm:.2f}', crop))
    except Exception:
        pass
    return variants


def _rank_best_crop_variant_v113(original_name: Any, crops: list[tuple[str, Image.Image]], skill_candidates: list[dict[str, str]], *, text_threshold: float, icon_threshold: float) -> tuple[str, Image.Image | None, list[dict[str, Any]], str, float, float, bool, str]:
    best = ('', None, [], str(original_name or ''), 0.0, 0.0, False, 'no_variant')
    for mode, crop in crops:
        top = _rank_icon_candidates_v40(crop, skill_candidates or [], ocr_name=original_name, topn=5)
        matched, text_score, icon_score, ok, reason = _select_match_from_ranked_v83(
            original_name, top, row_icon=crop, text_threshold=text_threshold, icon_threshold=icon_threshold
        )
        # ok가 아니어도 가장 높은 점수/후보를 기록합니다.
        score = float(icon_score or (top[0].get('icon_score') if top else 0.0) or 0.0)
        if score > float(best[5] or 0.0):
            best = (mode, crop, top, matched, float(text_score or 0.0), score, bool(ok), reason)
    return best


def correct_battle_skill_names_with_icons(  # type: ignore[override]
    df: pd.DataFrame,
    attack_image: Image.Image,
    skill_candidates: List[Dict[str, str]],
    *,
    name_col: str = '이름',
    threshold: float = 0.62,
    icon_threshold: float = 0.72,
    drop_unmatched: bool = False,
) -> pd.DataFrame:
    """v113: low confidence/동점권 행에서 아이콘 crop variant를 재시도합니다."""
    if df is None or df.empty or name_col not in df.columns:
        return df
    import time as _time_v113
    t_total = _time_v113.perf_counter()
    rows = []
    _perf_trace_event_v83('icon_correct_start', rows=len(df), candidates=len(skill_candidates or []), mode='v113_variant_retry')
    for idx, row in df.copy().iterrows():
        try:
            row_index = int(float(row.get('_ocr_row_index', idx)))
        except Exception:
            row_index = int(idx)
        row_t = _time_v113.perf_counter()
        square_crop = _crop_square_icon_from_row_metadata_v97(attack_image, row, row_index) if _env_bool_v97('LOA_ICON_SQUARE_CROP', True) else None
        legacy_crop = None
        if callable(_crop_icon_from_row_metadata_prev_v97):
            legacy_crop = _crop_icon_from_row_metadata_prev_v97(attack_image, row, row_index)  # type: ignore[misc]
        elif square_crop is None:
            legacy_crop = crop_attack_icon(attack_image, row_index, window_box=_row_window_box(row))
        chosen_crop, top, matched, text_score, icon_score, ok, reason, crop_mode, square_score, legacy_score = _choose_icon_match_v97(
            row.get(name_col), square_crop, legacy_crop, skill_candidates or [], text_threshold=threshold, icon_threshold=icon_threshold
        )
        variant_score = 0.0
        # top1이 기준 미달이거나(신뢰 부족) top2와 근소하면 variant crop을 재시도합니다.
        # v151: 근소 동점이어도 primary가 이미 기준을 넘긴 확정 매칭(ok=True)이면 재시도하지
        # 않습니다. 재시도가 확정된 스킬을 '기타' 등으로 강등하는 문제를 막습니다.
        need_variant = _ICON_VARIANT_RETRY_V113 and skill_candidates and (
            float(icon_score or 0.0) < max(72.0, float(icon_threshold) * 100.0)
            or (not ok and len(top or []) >= 2 and abs(float(top[0].get('icon_score') or 0.0) - float(top[1].get('icon_score') or 0.0)) < 1.5)
        )
        if need_variant:
            vcrops = _icon_variant_crops_v113(attack_image, row, row_index)
            mode_v, crop_v, top_v, matched_v, text_v, score_v, ok_v, reason_v = _rank_best_crop_variant_v113(
                row.get(name_col), vcrops, skill_candidates or [], text_threshold=threshold, icon_threshold=icon_threshold
            )
            variant_score = float(score_v or 0.0)
            if top_v and (ok_v or score_v >= float(icon_score or 0.0) + 1.0):
                chosen_crop, top, matched, text_score, icon_score, ok, reason, crop_mode = crop_v, top_v, matched_v, text_v, score_v, ok_v, f'variant:{mode_v}:{reason_v}', f'variant:{mode_v}'
        matched = _canonical_display_name_v86(matched)
        new_row = row.copy()
        if ok:
            new_row[name_col] = matched
        elif drop_unmatched:
            continue
        new_row['_name_match_text_score'] = text_score
        new_row['_name_match_icon_score'] = icon_score
        new_row['_name_match_reason'] = reason
        new_row['_icon_match_name'] = _canonical_display_name_v86(str(top[0]['name'])) if top else ''
        new_row['_icon_match_score'] = top[0]['icon_score'] if top else 0.0
        new_row['_icon_match_source'] = str(top[0]['source']) if top else ''
        new_row['_icon_match_top3'] = ' | '.join(f"{_canonical_display_name_v86(r['name'])}:{r['icon_score']}" for r in (top or [])[:3])
        new_row['_icon_crop_mode'] = crop_mode
        new_row['_icon_square_score'] = round(float(square_score or 0.0), 2)
        new_row['_icon_legacy_score'] = round(float(legacy_score or 0.0), 2)
        new_row['_icon_variant_score_v113'] = round(float(variant_score or 0.0), 2)
        rows.append(new_row)
        _perf_trace_event_v83('icon_correct_row', row_index=row_index, row_total_ms=round((_time_v113.perf_counter()-row_t)*1000.0,3), matched=matched, icon_score=round(float(icon_score or 0.0),2), crop_mode=crop_mode, reason=reason, variant_score=round(float(variant_score or 0.0),2))
    _perf_trace_event_v83('icon_correct_end', rows=len(rows), total_ms=round((_time_v113.perf_counter()-t_total)*1000.0,3), mode='v113_variant_retry')
    return pd.DataFrame(rows).reset_index(drop=True) if rows else df.iloc[0:0].copy()



# ==============================================================================
# v117: 실제 속도 최적화 적용
# ==============================================================================
# v116은 진단 중심이었고, 병목은 다음 3곳이었습니다.
# 1) 공격정보 damage 재검수 OCR 2회/행
# 2) 아이콘 정밀비교 heavy_k + low-confidence variant 다중 재시도
# 3) 종합정보 total/dps raw line OCR 후보 과다
# v117은 정확도 손실을 최소화하며 호출 수를 줄입니다.

import os as _os_v117

# 직접 streamlit run app.py로 실행해도 빠른 기본값을 사용합니다.
_os_v117.environ.setdefault('LOA_ATTACK_ROW_PASS_SCALE', '2')
_os_v117.environ.setdefault('LOA_ATTACK_DAMAGE_RECHECK', 'smart')
_os_v117.environ.setdefault('LOA_ATTACK_CRITICAL_CELL_SCALE', '3')
_os_v117.environ.setdefault('LOA_ICON_HEAVY_TOPK', '5')
# v151: 저신뢰 행 세로 밀림(마지막 기본공격 등) 복구를 위해 변형 재시도를 기본 ON으로.
_os_v117.environ.setdefault('LOA_ICON_VARIANT_RETRY', '1')
_os_v117.environ.setdefault('LOA_SUMMARY_RAW_SCALE', '3')
_os_v117.environ.setdefault('LOA_FAST_SINGLE_CELL_OCR', '1')

try:
    _ICON_VARIANT_RETRY_V113 = str(_os_v117.environ.get('LOA_ICON_VARIANT_RETRY', '0')).strip().lower() not in {'0','false','no','off'}
except Exception:
    _ICON_VARIANT_RETRY_V113 = False


def _preprocess_numeric_readable_v117(raw_crop: Image.Image, scale: int = 3) -> Image.Image:
    """사람도 읽을 수 있는 수준의 빠른 숫자 전처리.

    기존 numeric_mask는 글자가 깨져 보이는 경우가 있어 OCR도 흔들렸습니다.
    v117에서는 raw 확대+대비+선명도만 적용합니다.
    """
    crop = raw_crop.convert('RGB')
    s = max(2, min(int(scale or 3), 4))
    if s > 1:
        crop = crop.resize((max(1, crop.width * s), max(1, crop.height * s)), Image.Resampling.LANCZOS)
    crop = ImageEnhance.Brightness(crop).enhance(1.05)
    crop = ImageEnhance.Contrast(crop).enhance(1.45)
    crop = ImageEnhance.Sharpness(crop).enhance(1.35)
    return crop


def _ocr_text_once_allow_v117(reader: Any, crop: Image.Image, allowlist: str) -> str:
    arr = np.asarray(crop)
    try:
        res = reader.readtext(arr, detail=0, paragraph=False, allowlist=allowlist)
    except TypeError:
        try:
            res = reader.readtext(arr, detail=0, paragraph=False)
        except Exception:
            res = []
    except Exception:
        res = []
    if isinstance(res, list):
        return clean_ocr_text(' '.join(str(x).strip() for x in res if str(x).strip()))
    return clean_ocr_text(str(res or ''))


# v117 summary: 카드 하단 원시 숫자 줄은 단일 readable 전처리 1회만 OCR합니다.
def _ocr_summary_raw_number_line_v94(reader: Any, raw_crop: Image.Image, *, scale: int = 4) -> Tuple[str, Image.Image, str, float]:  # type: ignore[override]
    use_scale = int(_os_v117.environ.get('LOA_SUMMARY_RAW_SCALE', '3') or 3)
    proc = _preprocess_numeric_readable_v117(raw_crop, scale=use_scale)
    text = _ocr_text_once_allow_v117(reader, proc, '0123456789, ')
    score = 0.0
    try:
        score = _numeric_ocr_score(text, kind='integer')
    except Exception:
        score = float(len(re.findall(r'\d', str(text or ''))))
    return text, proc, 'guide_raw_number_line_fast_v117', float(score)


# v117 attack: damage/casts 정밀 재검수는 단일 OCR만 돌립니다.
def _ocr_fixed_cell_v108(reader: Any, image: Image.Image, box: tuple[int, int, int, int], key: str, scale: int = 7) -> str:  # type: ignore[override]
    raw = _crop_pixel_box(image, box, pad=1, expand_ratio=0.015)
    s = int(_os_v117.environ.get('LOA_ATTACK_CRITICAL_CELL_SCALE', '3') or 3)
    if key == 'damage_text':
        proc = _preprocess_numeric_readable_v117(raw, scale=s)
        txt = _ocr_text_once_allow_v117(reader, proc, '0123456789.,억만조 ')
        try:
            txt = _postprocess_cell_value_v705(key, txt)
        except Exception:
            txt = extract_korean_number_text(txt)
        return _normalize_dash_to_zero_v108(key, txt)
    if key == 'casts':
        proc = _preprocess_numeric_readable_v117(raw, scale=max(2, min(s, 3)))
        txt = _ocr_text_once_allow_v117(reader, proc, '0123456789- ')
        try:
            txt = _postprocess_cell_value_v705(key, txt)
        except Exception:
            txt = clean_ocr_text(txt)
        return _normalize_dash_to_zero_v108(key, txt)
    if key in {'directional_rate','directional_share','crit_rate','crit_share','cooldown_rate','share_rate'}:
        proc = _preprocess_numeric_readable_v117(raw, scale=max(2, min(s, 3)))
        txt = _ocr_text_once_allow_v117(reader, proc, '0123456789.%,- ')
        try:
            txt = _postprocess_cell_value_v705(key, txt)
        except Exception:
            txt = format_percent_from_ocr(txt)
        return _normalize_dash_to_zero_v108(key, txt)
    # 이름은 여기서 읽지 않습니다.
    return ''

# v151: 변형 crop 재시도를 기본 ON으로 되돌립니다.
# 저신뢰(점수<72 또는 1·2위 동점권) 행에서만 발동하므로 보통 0~1개 행에만 적용되어 비용이 작고,
# 마지막 행(기본 공격)처럼 행 밴드가 실제 아이콘보다 밀린 경우 세로 이동 변형으로 정확히 복구합니다.
# 성능이 문제면 환경변수 LOA_ICON_VARIANT_RETRY=0 으로 끌 수 있습니다.
try:
    _ICON_VARIANT_RETRY_V113 = str(_os_v117.environ.get('LOA_ICON_VARIANT_RETRY', '1')).strip().lower() not in {'0','false','no','off'}
except Exception:
    pass

# ==============================================================================
# v118: fast fallback OCR for missed back/share/cast/cooldown columns
# ==============================================================================
# v117 reduced cell OCR calls too aggressively.  In the user debug zip the debug
# cell crops read correctly, but the main parser displayed 0.0 for 백적중/백비중/
# 치비중/횟수/쿨%.  Cause: row-pass OCR missed several narrow columns and v117 only
# rechecked damage/casts.  v118 keeps the fast row-pass path, but rechecks only
# cells that are invalid/zero AND visually not a dash.  This restores values while
# avoiding OCR on obvious '-' cells.

import os as _os_v118
import time as _time_v118

_os_v118.environ.setdefault('LOA_ATTACK_ROW_PASS_SCALE', '3')
_os_v118.environ.setdefault('LOA_ATTACK_DAMAGE_RECHECK', 'smart')
_os_v118.environ.setdefault('LOA_ATTACK_CRITICAL_CELL_SCALE', '3')
_os_v118.environ.setdefault('LOA_ATTACK_PERCENT_FALLBACK', '1')
_os_v118.environ.setdefault('LOA_ATTACK_CAST_COOLDOWN_FALLBACK', '1')
_os_v118.environ.setdefault('LOA_ICON_HEAVY_TOPK', '5')
_os_v118.environ.setdefault('LOA_ICON_VARIANT_RETRY', '0')
_os_v118.environ.setdefault('LOA_SUMMARY_RAW_SCALE', '3')

_parse_attack_fixed_grid_prev_v118 = globals().get('parse_attack_fixed_grid')


def _num_float_v118(value: Any) -> float | None:
    try:
        if value is None:
            return None
        s = str(value).strip().replace('%', '').replace(',', '')
        if s in {'', '-', 'None', 'nan', 'NaN'}:
            return None
        return float(re.sub(r'[^0-9.\-]', '', s) or 'nan')
    except Exception:
        return None


def _is_invalid_or_zero_v118(key: str, val: Any) -> bool:
    s = clean_ocr_text(str(val or ''))
    if not s or s in {'-', 'None', 'nan', 'NaN'}:
        return True
    n = _num_float_v118(s)
    if n is None:
        return True
    # v117 failure mode: real values became 0.0 in narrow columns.
    if key in {'directional_rate', 'directional_share', 'crit_share', 'cooldown_rate'} and abs(n) < 1e-9:
        return True
    if key == 'casts' and abs(n) < 1e-9:
        return True
    return False


def _cell_visual_stats_v118(image: Image.Image, box: tuple[int, int, int, int]) -> dict[str, float]:
    try:
        raw = _crop_pixel_box(image, box, pad=0, expand_ratio=0.0).convert('L')
        arr = np.asarray(raw, dtype=np.uint8)
        if arr.size == 0:
            return {'bright': 0, 'ratio': 0.0, 'w': 0, 'h': 0}
        # Lost Ark combat table text/dash is bright on dark background.  We use a
        # conservative high threshold so blue row bars/background do not count.
        mask = arr > 172
        bright = int(mask.sum())
        ratio = float(bright) / float(mask.size)
        return {'bright': float(bright), 'ratio': ratio, 'w': float(arr.shape[1]), 'h': float(arr.shape[0])}
    except Exception:
        return {'bright': 999.0, 'ratio': 1.0, 'w': 0, 'h': 0}


def _cell_looks_dash_or_empty_v118(image: Image.Image, box: tuple[int, int, int, int], key: str = '') -> bool:
    stt = _cell_visual_stats_v118(image, box)
    bright = stt.get('bright', 999.0)
    ratio = stt.get('ratio', 1.0)
    # Percent strings usually have hundreds of bright pixels after crop; '-' has
    # very few. Counts like '1' can be small, so casts are never skipped by this.
    if key == 'casts':
        return False
    return bright < 42 or ratio < 0.0022


def _ocr_fixed_cell_fast_v118(reader: Any, image: Image.Image, box: tuple[int, int, int, int], key: str) -> str:
    """One fast readable OCR for fallback cells."""
    try:
        return _ocr_fixed_cell_v108(reader, image, box, key, scale=int(_os_v118.environ.get('LOA_ATTACK_CRITICAL_CELL_SCALE', '3') or 3))
    except Exception:
        return ''


def _damage_recheck_needed_v118(raw_val: str, raw_rowpass: str = '') -> bool:
    # Prefer skipping damage recheck when row-pass already captured a plausible
    # Korean number or the common '억 misread as 9/91/94' pattern.  This avoids the
    # expensive 14x damage-cell OCR when row-pass is enough.
    for t in [raw_val, raw_rowpass]:
        s = clean_ocr_text(str(t or ''))
        if not s or s in {'0', '0.0', '-'}:
            continue
        if re.search(r'\d', s) and re.search(r'조|억|만', s):
            return False
        if re.fullmatch(r'[\d,]+(?:\.\d+)?(?:9|91|94)?', s) and ('.' in s or ',' in s):
            return False
    return True


def parse_attack_fixed_grid(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int | None = None,
    scale: int = 7,
    auto_window: bool = True,
) -> pd.DataFrame:
    """v118: row-pass + selective fallback for missed columns."""
    try:
        if not ATTACK_FAST_GRID_V112:
            if callable(_parse_attack_fixed_grid_prev_v118):
                return _parse_attack_fixed_grid_prev_v118(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)  # type: ignore[misc]
            return pd.DataFrame()
    except Exception:
        pass

    grid = ATTACK_FIXED_GRID_V111
    if row_count is None:
        row_count = int(grid.get('row_count', 14) or 14)
    try:
        window_box = _detect_window_no_ocr_v101(image, reader) if auto_window else _base_window_box_scaled_v101(image)
    except Exception:
        try:
            window_box = detect_analyzer_window(image, reader)
        except Exception:
            window_box = None
    if window_box is None:
        return pd.DataFrame()

    percent_fallback = str(_os_v118.environ.get('LOA_ATTACK_PERCENT_FALLBACK', '1')).strip().lower() not in {'0','false','no','off'}
    cast_cd_fallback = str(_os_v118.environ.get('LOA_ATTACK_CAST_COOLDOWN_FALLBACK', '1')).strip().lower() not in {'0','false','no','off'}
    rowpass_scale = int(_os_v118.environ.get('LOA_ATTACK_ROW_PASS_SCALE', ROW_PASS_SCALE_V112) or ROW_PASS_SCALE_V112)
    damage_mode = str(_os_v118.environ.get('LOA_ATTACK_DAMAGE_RECHECK', 'smart')).strip().lower()

    rows: list[dict[str, Any]] = []
    for i, row_box, icon_box, source in _iter_fixed_grid_rows_v111(image, window_box, int(row_count)):
        row: Dict[str, Any] = {col: '' for col in STANDARD_COLUMNS}
        row['_ocr_row_index'] = str(i)
        row['_row_source'] = source + ':fast_rowpass_v118'
        row['_direction_kind'] = 'back'
        row['_window_x1'], row['_window_y1'], row['_window_x2'], row['_window_y2'] = window_box
        row['_icon_x1'], row['_icon_y1'], row['_icon_x2'], row['_icon_y2'] = icon_box
        row['_row_x1'], row['_row_y1'], row['_row_x2'], row['_row_y2'] = row_box
        row['_grid_mode_v111'] = '1'
        row['_fast_grid_v112'] = '1'
        row['_fast_grid_v113'] = '1'
        row['_fast_grid_v118'] = '1'
        row['헤드어택 적중률'] = '0.0'
        row['헤드어택 비중'] = '0.0'
        row['초당 피해량'] = ''
        row['피해량 지분'] = ''

        col_boxes: list[tuple[str, str, int, int, int, int]] = []
        box_map: dict[str, tuple[int, int, int, int]] = {}
        for key, label, rx1, rx2 in grid['columns']:
            pix = _fixed_cell_box_v112(image, window_box, row_box, rx1, rx2)
            box_map[key] = pix
            col_boxes.append((key, label, pix[0], pix[1], pix[2], pix[3]))

        rowpass: dict[str, str] = {}
        try:
            rowpass = _ocr_row_single_pass_v705(reader, image, col_boxes, scale=max(2, rowpass_scale))
        except Exception:
            rowpass = {}
        row['_rowpass_used_v112'] = '1' if rowpass else '0'
        row['_rowpass_raw_v118'] = ' | '.join(f'{k}={v}' for k, v in rowpass.items() if str(v or '').strip())
        rechecked: list[str] = []
        skipped_dash: list[str] = []

        for key, label, _rx1, _rx2 in grid['columns']:
            raw_rowpass = clean_ocr_text(rowpass.get(key, ''))
            try:
                val = _postprocess_cell_value_v705(key, raw_rowpass)
            except Exception:
                val = raw_rowpass
            val = _normalize_dash_to_zero_v108(key, val)

            if key == 'damage_text':
                do_recheck = False
                if damage_mode in {'1','true','yes','on','always'}:
                    do_recheck = True
                elif damage_mode in {'smart','auto'}:
                    do_recheck = _damage_recheck_needed_v118(str(val), raw_rowpass)
                if do_recheck:
                    precise = _ocr_fixed_cell_fast_v118(reader, image, box_map[key], key)
                    if _valid_numberish_v112(key, precise):
                        val = precise
                        row['_damage_recheck_v112'] = '1'
                        rechecked.append(key)
                    else:
                        row['_damage_recheck_v112'] = '0'
                else:
                    row['_damage_recheck_v112'] = 'skip_valid_rowpass_v118'

            elif key in {'directional_rate', 'directional_share', 'crit_share', 'cooldown_rate'}:
                if percent_fallback and _is_invalid_or_zero_v118(key, val):
                    if _cell_looks_dash_or_empty_v118(image, box_map[key], key):
                        val = '0.0'
                        skipped_dash.append(key)
                    else:
                        precise = _ocr_fixed_cell_fast_v118(reader, image, box_map[key], key)
                        if _valid_numberish_v112(key, precise):
                            val = precise
                            rechecked.append(key)

            elif key == 'casts':
                if cast_cd_fallback and _is_invalid_or_zero_v118(key, val):
                    precise = _ocr_fixed_cell_fast_v118(reader, image, box_map[key], key)
                    if _valid_numberish_v112(key, precise):
                        val = precise
                        rechecked.append(key)
                    else:
                        val = '0'

            elif key == 'crit_rate':
                # crit_rate was already good in v117, but keep a cheap fallback for rows where row-pass misses it.
                if percent_fallback and _is_invalid_or_zero_v118(key, val):
                    if _cell_looks_dash_or_empty_v118(image, box_map[key], key):
                        val = '0.0'
                        skipped_dash.append(key)
                    else:
                        precise = _ocr_fixed_cell_fast_v118(reader, image, box_map[key], key)
                        if _valid_numberish_v112(key, precise):
                            val = precise
                            rechecked.append(key)

            if key == 'damage_text': row['피해량'] = val
            elif key == 'directional_rate': row['백어택 적중률'] = val
            elif key == 'directional_share': row['백어택 비중'] = val
            elif key == 'crit_rate': row['치명타 적중률'] = val
            elif key == 'crit_share': row['치명타 비중'] = val
            elif key == 'casts': row['사용 횟수'] = val
            elif key == 'cooldown_rate': row['쿨타임 비율'] = val

        row['_v118_recheck_keys'] = ','.join(rechecked)
        row['_v118_dash_skip_keys'] = ','.join(skipped_dash)
        rows.append(row)

    cols = STANDARD_COLUMNS + [
        '_ocr_row_index', '_row_source', '_direction_kind',
        '_window_x1', '_window_y1', '_window_x2', '_window_y2',
        '_icon_x1', '_icon_y1', '_icon_x2', '_icon_y2',
        '_row_x1', '_row_y1', '_row_x2', '_row_y2',
        '_grid_mode_v111', '_fast_grid_v112', '_fast_grid_v113', '_fast_grid_v118',
        '_rowpass_used_v112', '_damage_recheck_v112', '_casts_recheck_v112',
        '_rowpass_raw_v118', '_v118_recheck_keys', '_v118_dash_skip_keys',
    ]
    for r in rows:
        for c in cols:
            r.setdefault(c, '')
    return pd.DataFrame(rows, columns=cols)


# ==============================================================================
# v122: analysis-only share mode
# ==============================================================================
# 분석용으로 확정: 실전 보정에는 적중률(백적중/치적)보다 실제 피해 비중
# (백비중/헤드비중/치비중)이 직접 필요합니다. 따라서 기본 빠른 모드에서는
# 적중률 칸의 셀 OCR fallback을 생략하고, 피해량/비중/횟수/쿨%만 보정합니다.

import os as _os_v122
_os_v122.environ.setdefault('LOA_ANALYSIS_SHARE_ONLY', '1')
_os_v122.environ.setdefault('LOA_ATTACK_PERCENT_FALLBACK', '1')
_os_v122.environ.setdefault('LOA_ATTACK_CAST_COOLDOWN_FALLBACK', '1')
_os_v122.environ.setdefault('LOA_ATTACK_ROW_PASS_SCALE', '3')
_os_v122.environ.setdefault('LOA_ATTACK_DAMAGE_RECHECK', 'smart')
_os_v122.environ.setdefault('LOA_ATTACK_CRITICAL_CELL_SCALE', '3')

_parse_attack_fixed_grid_prev_v122 = globals().get('parse_attack_fixed_grid')

def _analysis_share_only_enabled_v122() -> bool:
    try:
        return str(_os_v122.environ.get('LOA_ANALYSIS_SHARE_ONLY', '1')).strip().lower() not in {'0','false','no','off'}
    except Exception:
        return True


def parse_attack_fixed_grid(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int | None = None,
    scale: int = 7,
    auto_window: bool = True,
) -> pd.DataFrame:
    """v122: analysis-only fast parser.

    기본 계산은 백비중/헤드비중/치비중을 사용하므로 백적중/치적 셀 재OCR을
    생략합니다. row-pass에서 우연히 읽힌 값은 보존할 수 있지만, 실패해도
    다시 OCR하지 않습니다. 이로써 v118에서 속도를 다시 올렸던 적중률 fallback
    호출을 제거합니다.
    """
    if not _analysis_share_only_enabled_v122():
        if callable(_parse_attack_fixed_grid_prev_v122):
            return _parse_attack_fixed_grid_prev_v122(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)  # type: ignore[misc]
        return pd.DataFrame()

    try:
        if not ATTACK_FAST_GRID_V112:
            if callable(_parse_attack_fixed_grid_prev_v122):
                return _parse_attack_fixed_grid_prev_v122(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)  # type: ignore[misc]
            return pd.DataFrame()
    except Exception:
        pass

    grid = ATTACK_FIXED_GRID_V111
    if row_count is None:
        row_count = int(grid.get('row_count', 14) or 14)
    try:
        window_box = _detect_window_no_ocr_v101(image, reader) if auto_window else _base_window_box_scaled_v101(image)
    except Exception:
        try:
            window_box = detect_analyzer_window(image, reader)
        except Exception:
            window_box = None
    if window_box is None:
        return pd.DataFrame()

    percent_fallback = str(_os_v122.environ.get('LOA_ATTACK_PERCENT_FALLBACK', '1')).strip().lower() not in {'0','false','no','off'}
    cast_cd_fallback = str(_os_v122.environ.get('LOA_ATTACK_CAST_COOLDOWN_FALLBACK', '1')).strip().lower() not in {'0','false','no','off'}
    rowpass_scale = int(_os_v122.environ.get('LOA_ATTACK_ROW_PASS_SCALE', ROW_PASS_SCALE_V112) or ROW_PASS_SCALE_V112)
    damage_mode = str(_os_v122.environ.get('LOA_ATTACK_DAMAGE_RECHECK', 'smart')).strip().lower()

    rows: list[dict[str, Any]] = []
    # 분석용 필수 fallback 대상만 남깁니다.
    # directional_rate / crit_rate는 row-pass에 잡히면 표시만 하고, 별도 OCR하지 않습니다.
    share_fallback_keys = {'directional_share', 'crit_share', 'cooldown_rate'}

    for i, row_box, icon_box, source in _iter_fixed_grid_rows_v111(image, window_box, int(row_count)):
        row: Dict[str, Any] = {col: '' for col in STANDARD_COLUMNS}
        row['_ocr_row_index'] = str(i)
        row['_row_source'] = source + ':analysis_share_fast_v122'
        row['_direction_kind'] = 'back'
        row['_window_x1'], row['_window_y1'], row['_window_x2'], row['_window_y2'] = window_box
        row['_icon_x1'], row['_icon_y1'], row['_icon_x2'], row['_icon_y2'] = icon_box
        row['_row_x1'], row['_row_y1'], row['_row_x2'], row['_row_y2'] = row_box
        row['_grid_mode_v111'] = '1'
        row['_fast_grid_v112'] = '1'
        row['_fast_grid_v113'] = '1'
        row['_fast_grid_v118'] = '1'
        row['_analysis_share_fast_v122'] = '1'
        row['헤드어택 적중률'] = '0.0'
        row['헤드어택 비중'] = '0.0'
        row['초당 피해량'] = ''
        row['피해량 지분'] = ''

        col_boxes: list[tuple[str, str, int, int, int, int]] = []
        box_map: dict[str, tuple[int, int, int, int]] = {}
        for key, label, rx1, rx2 in grid['columns']:
            pix = _fixed_cell_box_v112(image, window_box, row_box, rx1, rx2)
            box_map[key] = pix
            col_boxes.append((key, label, pix[0], pix[1], pix[2], pix[3]))

        rowpass: dict[str, str] = {}
        try:
            rowpass = _ocr_row_single_pass_v705(reader, image, col_boxes, scale=max(2, rowpass_scale))
        except Exception:
            rowpass = {}
        row['_rowpass_used_v112'] = '1' if rowpass else '0'
        row['_rowpass_raw_v122'] = ' | '.join(f'{k}={v}' for k, v in rowpass.items() if str(v or '').strip())
        rechecked: list[str] = []
        skipped_dash: list[str] = []
        skipped_rate: list[str] = []

        for key, label, _rx1, _rx2 in grid['columns']:
            raw_rowpass = clean_ocr_text(rowpass.get(key, ''))
            try:
                val = _postprocess_cell_value_v705(key, raw_rowpass)
            except Exception:
                val = raw_rowpass
            val = _normalize_dash_to_zero_v108(key, val)

            if key == 'damage_text':
                do_recheck = False
                if damage_mode in {'1','true','yes','on','always'}:
                    do_recheck = True
                elif damage_mode in {'smart','auto'}:
                    do_recheck = _damage_recheck_needed_v118(str(val), raw_rowpass)
                if do_recheck:
                    precise = _ocr_fixed_cell_fast_v118(reader, image, box_map[key], key)
                    if _valid_numberish_v112(key, precise):
                        val = precise
                        row['_damage_recheck_v112'] = '1'
                        rechecked.append(key)
                    else:
                        row['_damage_recheck_v112'] = '0'
                else:
                    row['_damage_recheck_v112'] = 'skip_valid_rowpass_v122'

            elif key in share_fallback_keys:
                if percent_fallback and _is_invalid_or_zero_v118(key, val):
                    if _cell_looks_dash_or_empty_v118(image, box_map[key], key):
                        val = '0.0'
                        skipped_dash.append(key)
                    else:
                        precise = _ocr_fixed_cell_fast_v118(reader, image, box_map[key], key)
                        if _valid_numberish_v112(key, precise):
                            val = precise
                            rechecked.append(key)

            elif key == 'casts':
                if cast_cd_fallback and _is_invalid_or_zero_v118(key, val):
                    precise = _ocr_fixed_cell_fast_v118(reader, image, box_map[key], key)
                    if _valid_numberish_v112(key, precise):
                        val = precise
                        rechecked.append(key)
                    else:
                        val = '0'

            elif key in {'directional_rate', 'crit_rate'}:
                # 분석용 빠른 모드: 적중률은 계산 필수값이 아니므로 추가 OCR하지 않습니다.
                # row-pass가 읽은 값만 표시하고, 없으면 0 처리합니다.
                if _is_invalid_or_zero_v118(key, val):
                    val = '0.0'
                    skipped_rate.append(key)

            if key == 'damage_text': row['피해량'] = val
            elif key == 'directional_rate': row['백어택 적중률'] = val
            elif key == 'directional_share': row['백어택 비중'] = val
            elif key == 'crit_rate': row['치명타 적중률'] = val
            elif key == 'crit_share': row['치명타 비중'] = val
            elif key == 'casts': row['사용 횟수'] = val
            elif key == 'cooldown_rate': row['쿨타임 비율'] = val

        row['_v118_recheck_keys'] = ','.join(rechecked)
        row['_v118_dash_skip_keys'] = ','.join(skipped_dash)
        row['_v122_skipped_rate_keys'] = ','.join(skipped_rate)
        rows.append(row)

    cols = STANDARD_COLUMNS + [
        '_ocr_row_index', '_row_source', '_direction_kind',
        '_window_x1', '_window_y1', '_window_x2', '_window_y2',
        '_icon_x1', '_icon_y1', '_icon_x2', '_icon_y2',
        '_row_x1', '_row_y1', '_row_x2', '_row_y2',
        '_grid_mode_v111', '_fast_grid_v112', '_fast_grid_v113', '_fast_grid_v118', '_analysis_share_fast_v122',
        '_rowpass_used_v112', '_damage_recheck_v112', '_casts_recheck_v112',
        '_rowpass_raw_v122', '_v118_recheck_keys', '_v118_dash_skip_keys', '_v122_skipped_rate_keys',
    ]
    for r in rows:
        for c in cols:
            r.setdefault(c, '')
    return pd.DataFrame(rows, columns=cols)

# ==============================================================================
# v123: column-pass OCR + low CPU mode
# ==============================================================================
# v122 디버그 기준 병목은 행당 5개 셀 재OCR(약 70회 readtext)이었습니다.
# v123은 필요한 열(피해량/백비중/치비중/횟수/쿨%)을 열 단위로 한 번씩 읽고,
# y좌표로 행에 배분합니다. 즉 공격정보 OCR 호출 수를 약 70회 -> 약 5회로 줄입니다.
# 정확도 모드가 필요하면 LOA_ATTACK_COLUMN_PASS=0 으로 이전 파서를 사용할 수 있습니다.

import os as _os_v123
import time as _time_v123

_os_v123.environ.setdefault('LOA_ATTACK_COLUMN_PASS', '1')
_os_v123.environ.setdefault('LOA_ATTACK_COLUMN_PASS_SCALE', '2')
_os_v123.environ.setdefault('LOA_ATTACK_COLUMN_CELL_FALLBACK', '0')
_os_v123.environ.setdefault('LOA_WEB_LOW_CPU', '1')

_parse_attack_fixed_grid_prev_v123 = globals().get('parse_attack_fixed_grid')


def _v123_enabled() -> bool:
    return str(_os_v123.environ.get('LOA_ATTACK_COLUMN_PASS', '1')).strip().lower() not in {'0','false','no','off'}


def _allowlist_for_column_v123(key: str) -> str | None:
    if key == 'damage_text':
        return '0123456789.,억만조兆億萬万 '
    if key == 'casts':
        return '0123456789-— '
    return '0123456789.,%-— '


def _column_preprocess_v123(crop: Image.Image, scale: int) -> Image.Image:
    scale = max(1, min(int(scale or 2), 4))
    im = crop.convert('RGB')
    if scale != 1:
        im = im.resize((max(1, im.width * scale), max(1, im.height * scale)), Image.Resampling.LANCZOS)
    try:
        im = ImageEnhance.Contrast(im).enhance(1.25)
        im = ImageEnhance.Sharpness(im).enhance(1.10)
    except Exception:
        pass
    return im


def _bbox_center_v123(box: Any) -> tuple[float, float]:
    try:
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        return sum(xs) / max(1, len(xs)), sum(ys) / max(1, len(ys))
    except Exception:
        return 0.0, 0.0


def _assign_texts_to_rows_v123(
    results: list[Any],
    *,
    crop_x1: int,
    crop_y1: int,
    scale: int,
    row_items: list[tuple[int, tuple[int, int, int, int], tuple[int, int, int, int], str]],
) -> dict[int, str]:
    buckets: dict[int, list[tuple[float, str]]] = {int(i): [] for i, *_ in row_items}
    row_centers = []
    for i, row_box, _icon_box, _source in row_items:
        _x1, y1, _x2, y2 = row_box
        row_centers.append((int(i), (y1 + y2) / 2.0, max(12.0, (y2 - y1) * 0.75)))
    for item in results or []:
        try:
            box, text = item[0], str(item[1])
        except Exception:
            continue
        text = clean_ocr_text(text)
        if not text:
            continue
        cx, cy = _bbox_center_v123(box)
        full_y = crop_y1 + (cy / max(1, scale))
        full_x = crop_x1 + (cx / max(1, scale))
        best_i = None
        best_d = 10**9
        best_tol = 0.0
        for i, rc, tol in row_centers:
            d = abs(full_y - rc)
            if d < best_d:
                best_i, best_d, best_tol = i, d, tol
        if best_i is not None and best_d <= best_tol:
            buckets.setdefault(int(best_i), []).append((full_x, text))
    out = {}
    for i, vals in buckets.items():
        vals.sort(key=lambda x: x[0])
        out[i] = ''.join(v for _x, v in vals).strip()
    return out


def _ocr_required_columns_v123(
    reader: Any,
    image: Image.Image,
    window_box: tuple[int, int, int, int],
    row_items: list[tuple[int, tuple[int, int, int, int], tuple[int, int, int, int], str]],
    grid: dict[str, Any],
    keys: list[str],
    *,
    scale: int = 2,
) -> tuple[dict[str, dict[int, str]], dict[str, Any]]:
    values: dict[str, dict[int, str]] = {}
    stats: dict[str, Any] = {'column_calls': 0, 'columns': {}, 'scale': scale}
    if not row_items:
        return values, stats
    all_y1 = min(int(r[1][1]) for r in row_items)
    all_y2 = max(int(r[1][3]) for r in row_items)
    # 행 사이 경계선이 잘리지 않도록 세로 여유를 줍니다.
    y1 = max(0, all_y1 - 3)
    y2 = min(image.height, all_y2 + 3)
    rx_map = {str(k): (float(rx1), float(rx2), str(label)) for k, label, rx1, rx2 in grid.get('columns', [])}
    for key in keys:
        if key not in rx_map:
            continue
        rx1, rx2, _label = rx_map[key]
        # 첫 행 기준 셀 x좌표만 재사용합니다. y는 전체 데이터 행 범위입니다.
        first_row_box = row_items[0][1]
        cx1, _cy1, cx2, _cy2 = _fixed_cell_box_v112(image, window_box, first_row_box, rx1, rx2)
        pad_x = 2 if key != 'damage_text' else 4
        x1 = max(0, int(cx1) - pad_x)
        x2 = min(image.width, int(cx2) + pad_x)
        crop = image.crop((x1, y1, x2, y2))
        proc = _column_preprocess_v123(crop, scale)
        allow = _allowlist_for_column_v123(key)
        t0 = _time_v123.perf_counter()
        try:
            res = reader.readtext(np.asarray(proc), detail=1, paragraph=False, allowlist=allow)
        except TypeError:
            res = reader.readtext(np.asarray(proc), detail=1, paragraph=False)
        except Exception:
            res = []
        dt = (_time_v123.perf_counter() - t0) * 1000.0
        stats['column_calls'] += 1
        stats['columns'][key] = {
            'elapsed_ms': round(dt, 3),
            'crop_shape': [int(proc.height), int(proc.width), 3],
            'raw_count': len(res or []),
        }
        values[key] = _assign_texts_to_rows_v123(res or [], crop_x1=x1, crop_y1=y1, scale=scale, row_items=row_items)
    return values, stats


def _normalize_column_value_v123(key: str, raw: str) -> str:
    raw = clean_ocr_text(raw or '')
    try:
        val = _postprocess_cell_value_v705(key, raw)
    except Exception:
        val = raw
    try:
        val = _normalize_dash_to_zero_v108(key, val)
    except Exception:
        pass
    if key == 'casts':
        if not _valid_numberish_v112(key, str(val)):
            return '0'
    elif key != 'damage_text':
        if not _valid_numberish_v112(key, str(val)):
            return '0.0'
    return str(val or '').strip()


def parse_attack_fixed_grid(  # type: ignore[override]
    image: Image.Image,
    reader: Any,
    row_count: int | None = None,
    scale: int = 7,
    auto_window: bool = True,
) -> pd.DataFrame:
    """v123: 분석용 column-pass OCR.

    필요한 숫자 열만 세로로 한 번씩 OCR합니다. 셀별 OCR을 거의 없애서 CPU 점유율과
    readtext 호출 수를 낮춥니다. 문제가 생기면 LOA_ATTACK_COLUMN_PASS=0 으로 v122로 되돌릴 수 있습니다.
    """
    if not _v123_enabled():
        if callable(_parse_attack_fixed_grid_prev_v123):
            return _parse_attack_fixed_grid_prev_v123(image, reader, row_count=row_count, scale=scale, auto_window=auto_window)  # type: ignore[misc]
        return pd.DataFrame()

    grid = ATTACK_FIXED_GRID_V111
    if row_count is None:
        row_count = int(grid.get('row_count', 14) or 14)
    try:
        window_box = _detect_window_no_ocr_v101(image, reader) if auto_window else _base_window_box_scaled_v101(image)
    except Exception:
        try:
            window_box = detect_analyzer_window(image, reader)
        except Exception:
            window_box = None
    if window_box is None:
        return pd.DataFrame()

    row_items = list(_iter_fixed_grid_rows_v111(image, window_box, int(row_count)))
    # 실제 데이터 행만 사용합니다. v111 감지 자체가 최대 14개로 제한되어 있습니다.
    required_keys = ['damage_text', 'directional_share', 'crit_share', 'casts', 'cooldown_rate']
    try:
        column_scale = int(_os_v123.environ.get('LOA_ATTACK_COLUMN_PASS_SCALE', '2') or 2)
    except Exception:
        column_scale = 2
    column_values, column_stats = _ocr_required_columns_v123(reader, image, window_box, row_items, grid, required_keys, scale=max(1, column_scale))

    rows: list[dict[str, Any]] = []
    for i, row_box, icon_box, source in row_items:
        row: Dict[str, Any] = {col: '' for col in STANDARD_COLUMNS}
        row['_ocr_row_index'] = str(i)
        row['_row_source'] = source + ':column_pass_v123'
        row['_direction_kind'] = 'back'
        row['_window_x1'], row['_window_y1'], row['_window_x2'], row['_window_y2'] = window_box
        row['_icon_x1'], row['_icon_y1'], row['_icon_x2'], row['_icon_y2'] = icon_box
        row['_row_x1'], row['_row_y1'], row['_row_x2'], row['_row_y2'] = row_box
        row['_grid_mode_v111'] = '1'
        row['_fast_grid_v112'] = '1'
        row['_fast_grid_v113'] = '1'
        row['_fast_grid_v118'] = '1'
        row['_analysis_share_fast_v122'] = '1'
        row['_column_pass_v123'] = '1'
        row['_column_pass_scale_v123'] = str(column_scale)
        row['_column_pass_stats_v123'] = str(column_stats)
        row['헤드어택 적중률'] = '0.0'
        row['헤드어택 비중'] = '0.0'
        row['백어택 적중률'] = '0.0'
        row['치명타 적중률'] = '0.0'
        row['초당 피해량'] = ''
        row['피해량 지분'] = ''

        dmg = _normalize_column_value_v123('damage_text', column_values.get('damage_text', {}).get(int(i), ''))
        back_share = _normalize_column_value_v123('directional_share', column_values.get('directional_share', {}).get(int(i), ''))
        crit_share = _normalize_column_value_v123('crit_share', column_values.get('crit_share', {}).get(int(i), ''))
        casts = _normalize_column_value_v123('casts', column_values.get('casts', {}).get(int(i), ''))
        cd = _normalize_column_value_v123('cooldown_rate', column_values.get('cooldown_rate', {}).get(int(i), ''))

        # column-pass가 아주 드물게 한 칸을 놓치면, 옵션으로 해당 칸만 fallback할 수 있습니다.
        fallback_enabled = str(_os_v123.environ.get('LOA_ATTACK_COLUMN_CELL_FALLBACK', '0')).strip().lower() not in {'0','false','no','off'}
        rechecked: list[str] = []
        if fallback_enabled:
            rx_map = {str(k): (float(rx1), float(rx2)) for k, _label, rx1, rx2 in grid.get('columns', [])}
            for key, cur in [('damage_text', dmg), ('directional_share', back_share), ('crit_share', crit_share), ('casts', casts), ('cooldown_rate', cd)]:
                if _valid_numberish_v112(key, str(cur)):
                    continue
                try:
                    rx1, rx2 = rx_map[key]
                    box = _fixed_cell_box_v112(image, window_box, row_box, rx1, rx2)
                    val = _ocr_fixed_cell_fast_v118(reader, image, box, key)
                    val = _normalize_column_value_v123(key, val)
                    if _valid_numberish_v112(key, val):
                        if key == 'damage_text': dmg = val
                        elif key == 'directional_share': back_share = val
                        elif key == 'crit_share': crit_share = val
                        elif key == 'casts': casts = val
                        elif key == 'cooldown_rate': cd = val
                        rechecked.append(key)
                except Exception:
                    pass
        row['피해량'] = dmg
        row['백어택 비중'] = back_share
        row['치명타 비중'] = crit_share
        row['사용 횟수'] = casts
        row['쿨타임 비율'] = cd
        row['_v123_column_raw_damage'] = column_values.get('damage_text', {}).get(int(i), '')
        row['_v123_column_raw_back_share'] = column_values.get('directional_share', {}).get(int(i), '')
        row['_v123_column_raw_crit_share'] = column_values.get('crit_share', {}).get(int(i), '')
        row['_v123_column_raw_casts'] = column_values.get('casts', {}).get(int(i), '')
        row['_v123_column_raw_cooldown'] = column_values.get('cooldown_rate', {}).get(int(i), '')
        row['_v123_cell_fallback_keys'] = ','.join(rechecked)
        rows.append(row)

    cols = STANDARD_COLUMNS + [
        '_ocr_row_index', '_row_source', '_direction_kind',
        '_window_x1', '_window_y1', '_window_x2', '_window_y2',
        '_icon_x1', '_icon_y1', '_icon_x2', '_icon_y2',
        '_row_x1', '_row_y1', '_row_x2', '_row_y2',
        '_grid_mode_v111', '_fast_grid_v112', '_fast_grid_v113', '_fast_grid_v118', '_analysis_share_fast_v122',
        '_column_pass_v123', '_column_pass_scale_v123', '_column_pass_stats_v123',
        '_v123_column_raw_damage', '_v123_column_raw_back_share', '_v123_column_raw_crit_share', '_v123_column_raw_casts', '_v123_column_raw_cooldown', '_v123_cell_fallback_keys',
    ]
    for r in rows:
        for c in cols:
            r.setdefault(c, '')
    return pd.DataFrame(rows, columns=cols)
