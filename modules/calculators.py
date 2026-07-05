from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def parse_percent(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("%", "").replace(",", "")
    if text in ["", "-", "None", "nan"]:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def clamp_percent(value: Any, min_value: float = 0.0, max_value: float = 100.0) -> float | None:
    parsed = parse_percent(value)
    if parsed is None:
        return None
    return round(max(min_value, min(max_value, float(parsed))), 2)


def parse_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().replace(",", "")
    if text in ["", "-", "None", "nan"]:
        return None
    nums = re.findall(r"-?\d+", text)
    if not nums:
        return None
    return int(nums[0])


def parse_korean_number(value: Any) -> float | None:
    """'2,083.44억', '8,200.25만', '464,126,236,306' 같은 값을 숫자로 변환."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(" ", "").replace(",", "")
    if text in ["", "-", "None", "nan"]:
        return None
    multiplier = 1.0
    if "억" in text:
        multiplier = 100_000_000.0
        text = text.replace("억", "")
    elif "만" in text:
        multiplier = 10_000.0
        text = text.replace("만", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text or text in [".", "-"]:
        return None
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def format_korean_number(num: float | int | None) -> str:
    if num is None or (isinstance(num, float) and math.isnan(num)):
        return "-"
    num = float(num)
    abs_num = abs(num)
    if abs_num >= 1_000_000_000_000:
        return f"{num / 1_000_000_000_000:,.2f}조"
    if abs_num >= 100_000_000:
        return f"{num / 100_000_000:,.2f}억"
    if abs_num >= 10_000:
        return f"{num / 10_000:,.2f}만"
    return f"{num:,.0f}"


def normalize_battle_df(df: pd.DataFrame) -> pd.DataFrame:
    """전투분석기 표를 계산하기 쉬운 숫자 컬럼으로 정리.

    v36부터 신규 전투분석기의 `백어택 비중`, `치명타 비중`을 함께 받습니다.
    `초당 피해량`과 `피해량 지분`은 OCR에서 직접 읽지 않아도 app.py에서
    총피해/전투시간 기준으로 역산해서 채워 넣을 수 있습니다.
    """
    base_cols = [
        "name", "damage_text", "dps_text", "damage", "dps",
        "back_attack_rate", "back_attack_share", "head_attack_rate", "head_attack_share",
        "crit_rate", "crit_share", "casts", "cooldown_rate", "share_rate",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=base_cols)
    out = df.copy()
    rename_map = {
        "이름": "name",
        "스킬명": "name",
        "피해량": "damage_text",
        "초당 피해량": "dps_text",
        "DPS": "dps_text",
        "백어택 적중률": "back_attack_rate",
        "백어택 비중": "back_attack_share",
        "백어택 피해 비중": "back_attack_share",
        "헤드어택 적중률": "head_attack_rate",
        "헤드어택 비중": "head_attack_share",
        "헤드어택 피해 비중": "head_attack_share",
        "치명타 적중률": "crit_rate",
        "치명타 비중": "crit_share",
        "치명타 피해 비중": "crit_share",
        "사용 횟수": "casts",
        "쿨타임 비율": "cooldown_rate",
        "피해량 지분": "share_rate",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    for col in ["name", "damage_text", "dps_text", "back_attack_rate", "back_attack_share", "head_attack_rate", "head_attack_share", "crit_rate", "crit_share", "casts", "cooldown_rate", "share_rate"]:
        if col not in out.columns:
            out[col] = None
    out["damage"] = out["damage_text"].apply(parse_korean_number)
    out["dps"] = out["dps_text"].apply(parse_korean_number)
    out["back_attack_rate"] = out["back_attack_rate"].apply(clamp_percent)
    out["back_attack_share"] = out["back_attack_share"].apply(clamp_percent)
    out["head_attack_rate"] = out["head_attack_rate"].apply(clamp_percent)
    out["head_attack_share"] = out["head_attack_share"].apply(clamp_percent)
    out["crit_rate"] = out["crit_rate"].apply(clamp_percent)
    out["crit_share"] = out["crit_share"].apply(clamp_percent)
    out["cooldown_rate"] = out["cooldown_rate"].apply(clamp_percent)
    out["share_rate"] = out["share_rate"].apply(clamp_percent)
    out["casts"] = out["casts"].apply(parse_int)
    out["name"] = out["name"].fillna("").astype(str).str.strip()
    out = out[out["name"].ne("")].reset_index(drop=True)
    return out


def weighted_average(df: pd.DataFrame, value_col: str, weight_col: str = "damage") -> float | None:
    valid = df[[value_col, weight_col]].dropna()
    valid = valid[valid[weight_col] > 0]
    if valid.empty:
        return None
    return float(np.average(valid[value_col], weights=valid[weight_col]))


def summarize_battle(df: pd.DataFrame, elapsed_seconds: float | None = None) -> Dict[str, Any]:
    norm = normalize_battle_df(df)
    total_damage = norm["damage"].dropna().sum() if not norm.empty else 0.0
    skill_count = int(len(norm))
    top = norm.sort_values("damage", ascending=False).head(10) if not norm.empty else norm
    dps = None
    if elapsed_seconds and elapsed_seconds > 0 and total_damage > 0:
        dps = total_damage / elapsed_seconds
    return {
        "df": norm,
        "total_damage": float(total_damage),
        "dps": dps,
        "skill_count": skill_count,
        "weighted_back_attack_rate": weighted_average(norm, "back_attack_rate"),
        "weighted_back_attack_share": weighted_average(norm, "back_attack_share"),
        "weighted_head_attack_rate": weighted_average(norm, "head_attack_rate"),
        "weighted_head_attack_share": weighted_average(norm, "head_attack_share"),
        "weighted_crit_rate": weighted_average(norm, "crit_rate"),
        "weighted_crit_share": weighted_average(norm, "crit_share"),
        "weighted_cooldown_rate": weighted_average(norm, "cooldown_rate"),
        "top_skills": top,
    }


def compute_efficiency(real: Dict[str, Any], dummy: Dict[str, Any] | None = None, manual_dummy_dps: float | None = None) -> Dict[str, Any]:
    real_dps = real.get("dps")
    base_dps = manual_dummy_dps
    if dummy and dummy.get("dps"):
        base_dps = dummy.get("dps")
    efficiency = None
    if real_dps and base_dps and base_dps > 0:
        efficiency = real_dps / base_dps * 100.0
    return {
        "real_dps": real_dps,
        "base_dps": base_dps,
        "efficiency_percent": efficiency,
    }


def correction_multiplier_for_crit(observed_crit_percent: float | None, target_crit_percent: float, crit_damage_percent: float) -> float | None:
    if observed_crit_percent is None:
        return None
    obs = max(0.0, min(100.0, observed_crit_percent)) / 100.0
    tgt = max(0.0, min(100.0, target_crit_percent)) / 100.0
    crit_mult = crit_damage_percent / 100.0
    observed_expected = 1.0 + obs * (crit_mult - 1.0)
    target_expected = 1.0 + tgt * (crit_mult - 1.0)
    if observed_expected <= 0:
        return None
    return target_expected / observed_expected


def correction_multiplier_for_directional(
    observed_percent: float | None,
    target_percent: float,
    bonus_percent: float,
) -> float | None:
    if observed_percent is None:
        return None
    obs = max(0.0, min(100.0, observed_percent)) / 100.0
    tgt = max(0.0, min(100.0, target_percent)) / 100.0
    bonus = bonus_percent / 100.0
    observed_expected = 1.0 + obs * bonus
    target_expected = 1.0 + tgt * bonus
    if observed_expected <= 0:
        return None
    return target_expected / observed_expected


def corrected_dps(
    dps: float | None,
    observed_crit_percent: float | None,
    observed_back_percent: float | None,
    target_crit_percent: float,
    target_back_percent: float,
    crit_damage_percent: float,
    back_bonus_percent: float,
) -> Dict[str, Any]:
    crit_mul = correction_multiplier_for_crit(observed_crit_percent, target_crit_percent, crit_damage_percent)
    back_mul = correction_multiplier_for_directional(observed_back_percent, target_back_percent, back_bonus_percent)
    total_mul = 1.0
    used = []
    if crit_mul:
        total_mul *= crit_mul
        used.append("치명")
    if back_mul:
        total_mul *= back_mul
        used.append("백어택")
    corrected = dps * total_mul if dps else None
    return {
        "crit_multiplier": crit_mul,
        "back_multiplier": back_mul,
        "total_multiplier": total_mul if used else None,
        "corrected_dps": corrected,
        "used": used,
    }


def binomial_pmf(n: int, k: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def binomial_cdf(n: int, k: int, p: float) -> float:
    return sum(binomial_pmf(n, i, p) for i in range(0, k + 1))


def binomial_metrics(n: int | None, k: int | None, p: float = 0.75) -> Dict[str, Any] | None:
    if n is None or k is None or n <= 0:
        return None
    k = max(0, min(k, n))
    expected = n * p
    variance = n * p * (1 - p)
    sd = math.sqrt(variance) if variance > 0 else 0.0
    z = (k - expected) / sd if sd > 0 else None
    percentile = binomial_cdf(n, k, p) * 100.0
    pmf = binomial_pmf(n, k, p) * 100.0
    return {
        "n": n,
        "k": k,
        "p": p,
        "expected": expected,
        "sd": sd,
        "z": z,
        "percentile": percentile,
        "exact_probability_percent": pmf,
        "diff_from_expected": k - expected,
    }


def infer_hurricane_brutal_counts(df: pd.DataFrame) -> Tuple[Optional[int], Optional[int]]:
    norm = normalize_battle_df(df)
    hurricane = None
    brutal = None

    def clean_casts(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            return int(float(value))
        except Exception:
            return None

    for _, row in norm.iterrows():
        name = str(row.get("name", ""))
        casts = clean_casts(row.get("casts"))
        if "허리케인" in name:
            hurricane = casts
        if "브루탈" in name:
            brutal = casts
    return hurricane, brutal


def grade_efficiency(efficiency_percent: float | None) -> str:
    if efficiency_percent is None:
        return "기준 없음"
    if efficiency_percent >= 90:
        return "매우 높음"
    if efficiency_percent >= 80:
        return "높음"
    if efficiency_percent >= 70:
        return "보통 이상"
    if efficiency_percent >= 60:
        return "보통"
    return "낮음"
