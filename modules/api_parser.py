from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def _safe_data(bundle: Dict[str, Any], key: str) -> Any:
    value = bundle.get(key)
    if value is None:
        return None
    # LostArkApiClient.ApiResult 객체 또는 serializable dict 둘 다 지원
    if hasattr(value, "data"):
        return value.data if value.ok else None
    if isinstance(value, dict):
        return value.get("data") if value.get("ok", True) else None
    return value


def _flatten_tooltip(value: Any) -> str:
    """Lost Ark Tooltip JSON/HTML 텍스트를 대충 사람이 읽는 텍스트로 평탄화합니다."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
        try:
            parsed = json.loads(text)
            return _flatten_tooltip(parsed)
        except Exception:  # noqa: BLE001
            return _clean_text(text)
    if isinstance(value, dict):
        parts: List[str] = []
        for v in value.values():
            flat = _flatten_tooltip(v)
            if flat:
                parts.append(flat)
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(_flatten_tooltip(v) for v in value if v is not None)
    return _clean_text(str(value))


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_profile(bundle: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    profiles = _safe_data(bundle, "profiles") or _safe_data(bundle, "summary") or {}
    summary = {
        "캐릭터명": profiles.get("CharacterName"),
        "서버": profiles.get("ServerName"),
        "클래스": profiles.get("CharacterClassName"),
        "전투레벨": profiles.get("CharacterLevel"),
        "아이템레벨": profiles.get("ItemAvgLevel"),
        "이미지": profiles.get("CharacterImage"),
        "칭호": profiles.get("Title"),
        "길드": profiles.get("GuildName"),
    }
    stats = []
    for s in profiles.get("Stats", []) or []:
        stats.append({"스탯": s.get("Type"), "값": s.get("Value"), "툴팁": " / ".join(s.get("Tooltip", []) or [])})
    return summary, pd.DataFrame(stats)


def parse_combat_skills(bundle: Dict[str, Any]) -> pd.DataFrame:
    skills = _safe_data(bundle, "combat_skills") or []
    rows: List[Dict[str, Any]] = []
    for s in skills:
        tripod_names = []
        for tripod in s.get("Tripods", []) or []:
            if tripod.get("IsSelected"):
                tripod_names.append(f"{tripod.get('Name')} Lv.{tripod.get('Level')}" if tripod.get("Level") else tripod.get("Name"))
        rune = s.get("Rune") or {}
        rows.append({
            "스킬명": s.get("Name"),
            "레벨": s.get("Level"),
            "타입": s.get("Type") or s.get("SkillType"),
            "룬": rune.get("Name") if isinstance(rune, dict) else rune,
            "트포": " / ".join([x for x in tripod_names if x]),
            "아이콘": s.get("Icon"),
        })
    return pd.DataFrame(rows)


def parse_engravings(bundle: Dict[str, Any]) -> pd.DataFrame:
    engravings = _safe_data(bundle, "engravings") or {}
    rows: List[Dict[str, Any]] = []

    # 구형/일반 각인
    for e in engravings.get("Effects", []) or []:
        rows.append({"구분": "각인", "이름": e.get("Name"), "레벨/설명": e.get("Description")})

    # 아크패시브 각인 모델
    for e in engravings.get("ArkPassiveEffects", []) or []:
        rows.append({
            "구분": "아크패시브 각인",
            "이름": e.get("Name"),
            "레벨/설명": e.get("Grade") or e.get("Level") or e.get("Description"),
        })
    return pd.DataFrame(rows)


def parse_equipment(bundle: Dict[str, Any]) -> pd.DataFrame:
    equipment = _safe_data(bundle, "equipment") or []
    rows: List[Dict[str, Any]] = []
    for item in equipment:
        tooltip_text = _flatten_tooltip(item.get("Tooltip"))
        rows.append({
            "부위": item.get("Type"),
            "이름": item.get("Name"),
            "등급": item.get("Grade"),
            "품질": item.get("Quality"),
            "툴팁요약": tooltip_text[:500],
            "아이콘": item.get("Icon"),
        })
    return pd.DataFrame(rows)


def parse_gems(bundle: Dict[str, Any]) -> pd.DataFrame:
    gems = _safe_data(bundle, "gems") or {}
    rows: List[Dict[str, Any]] = []
    for gem in gems.get("Gems", []) or []:
        rows.append({
            "보석명": gem.get("Name"),
            "레벨": gem.get("Level"),
            "등급": gem.get("Grade"),
            "효과": _flatten_tooltip(gem.get("Tooltip"))[:300],
        })
    return pd.DataFrame(rows)


def parse_cards(bundle: Dict[str, Any]) -> pd.DataFrame:
    cards = _safe_data(bundle, "cards") or {}
    rows: List[Dict[str, Any]] = []
    for card in cards.get("Cards", []) or []:
        rows.append({
            "카드명": card.get("Name"),
            "각성": card.get("AwakeCount"),
            "등급": card.get("Grade"),
        })
    effects = []
    for e in cards.get("Effects", []) or []:
        effects.append(_flatten_tooltip(e))
    if effects:
        rows.append({"카드명": "세트효과", "각성": "", "등급": " / ".join(effects)[:500]})
    return pd.DataFrame(rows)


def parse_arkpassive(bundle: Dict[str, Any]) -> pd.DataFrame:
    ark = _safe_data(bundle, "arkpassive") or {}
    rows: List[Dict[str, Any]] = []
    # 구조가 시기별로 바뀔 수 있어 재귀적으로 Name/Level/Grade/Description 비슷한 필드만 수집
    def walk(obj: Any, path: str = ""):
        if isinstance(obj, dict):
            name = obj.get("Name") or obj.get("name")
            desc = obj.get("Description") or obj.get("desc") or obj.get("Value")
            level = obj.get("Level") or obj.get("level") or obj.get("Grade")
            if name:
                rows.append({"경로": path, "이름": name, "레벨/등급": level, "설명": _flatten_tooltip(desc)[:300]})
            for k, v in obj.items():
                walk(v, f"{path}/{k}" if path else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
    walk(ark)
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["경로", "이름", "레벨/등급", "설명"])


def parse_arkgrid(bundle: Dict[str, Any]) -> pd.DataFrame:
    ark = _safe_data(bundle, "arkgrid") or {}
    rows: List[Dict[str, Any]] = []
    def walk(obj: Any, path: str = ""):
        if isinstance(obj, dict):
            name = obj.get("Name") or obj.get("name")
            desc = obj.get("Description") or obj.get("desc") or obj.get("Value")
            level = obj.get("Level") or obj.get("level") or obj.get("Grade")
            if name:
                rows.append({"경로": path, "이름": name, "레벨/등급": level, "설명": _flatten_tooltip(desc)[:300]})
            for k, v in obj.items():
                walk(v, f"{path}/{k}" if path else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
    walk(ark)
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["경로", "이름", "레벨/등급", "설명"])


def summarize_all(bundle: Dict[str, Any]) -> Dict[str, Any]:
    profile_summary, stats = parse_profile(bundle)

    # 치명/치피/진피/피해군은 API 툴팁 구조가 옵션별로 달라질 수 있어서
    # 별도 휴리스틱 파서에서 추정값과 출처표를 함께 제공합니다.
    try:
        from modules.api_skill_estimator import estimate_skill_crit_tables
        crit_tables = estimate_skill_crit_tables(bundle)
    except Exception as e:  # noqa: BLE001 - 화면에서 검수 가능하도록 실패를 삼킵니다.
        crit_tables = {
            "skill_crit_estimates": pd.DataFrame(),
            "crit_sources": pd.DataFrame(),
            "damage_sources": pd.DataFrame(),
            "base_crit_percent": 0.0,
            "base_crit_raw": f"치명/치피/피해군 추정 실패: {e}",
            "global_static_crit_rate_percent": 0.0,
            "global_static_crit_damage_percent": 0.0,
            "avg_evolution_damage_percent": 0.0,
            "avg_final_multiplier": 1.0,
        }

    return {
        "profile_summary": profile_summary,
        "stats": stats,
        "equipment": parse_equipment(bundle),
        "skills": parse_combat_skills(bundle),
        "engravings": parse_engravings(bundle),
        "cards": parse_cards(bundle),
        "gems": parse_gems(bundle),
        "arkpassive": parse_arkpassive(bundle),
        "arkgrid": parse_arkgrid(bundle),
        **crit_tables,
    }

# ==============================
# v14 overrides
# ==============================
# 선택 트라이포드로 스킬 조작 타입이 변경되는 경우를 표시합니다.
# 예: 볼케이노 이럽션 + 블러드 이럽션 트포 → 기본 홀딩이지만 실제 타입은 일반.

def _selected_tripod_text_for_parser(skill: Dict[str, Any]) -> str:
    parts: List[str] = []
    for tripod in skill.get("Tripods", []) or []:
        if tripod.get("IsSelected"):
            parts.append(_flatten_tooltip(tripod))
    return _clean_text(" ".join(parts))


def _effective_type_for_parser(skill: Dict[str, Any]) -> tuple[str, str]:
    base = str(skill.get("Type") or skill.get("SkillType") or "").strip()
    text = _selected_tripod_text_for_parser(skill)
    patterns = [
        ("일반", [r"일반\s*(?:조작|스킬)?(?:으로|로)?\s*변경", r"일반\s*스킬로\s*변경"]),
        ("홀딩", [r"홀딩\s*(?:조작|스킬)?(?:으로|로)?\s*변경"]),
        ("차지", [r"차지\s*(?:조작|스킬)?(?:으로|로)?\s*변경", r"오버\s*차지"]),
        ("콤보", [r"콤보\s*(?:조작|스킬)?(?:으로|로)?\s*변경"]),
        ("지점", [r"지점\s*(?:조작|스킬)?(?:으로|로)?\s*변경"]),
        ("캐스팅", [r"캐스팅\s*(?:조작|스킬)?(?:으로|로)?\s*변경"]),
        ("토글", [r"토글\s*(?:조작|스킬)?(?:으로|로)?\s*변경"]),
    ]
    for label, pats in patterns:
        for pat in pats:
            if re.search(pat, text):
                return label, f"선택 트포 설명에서 '{label} 변경' 감지"
    return base or "", ""


def parse_combat_skills(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    skills = _safe_data(bundle, "combat_skills") or []
    rows: List[Dict[str, Any]] = []
    for s in skills:
        tripod_names = []
        for tripod in s.get("Tripods", []) or []:
            if tripod.get("IsSelected"):
                tripod_names.append(f"{tripod.get('Name')} Lv.{tripod.get('Level')}" if tripod.get("Level") else tripod.get("Name"))
        rune = s.get("Rune") or {}
        base_type = s.get("Type") or s.get("SkillType")
        effective_type, reason = _effective_type_for_parser(s)
        rows.append({
            "스킬명": s.get("Name"),
            "레벨": s.get("Level"),
            "기본타입": base_type,
            "타입": effective_type or base_type,
            "타입변경근거": reason,
            "룬": rune.get("Name") if isinstance(rune, dict) else rune,
            "트포": " / ".join([x for x in tripod_names if x]),
            "아이콘": s.get("Icon"),
        })
    return pd.DataFrame(rows)

# ==============================
# v17 parser override
# ==============================
# Level 1 스킬이라도 도약/초각성/아크패시브 원문에 스킬명이 직접 등장하면 채용 스킬로 표시합니다.

def _v17_flatten_for_parser(value):
    try:
        return _flatten(value)
    except Exception:
        return str(value or "")


def _v17_skill_level_value_for_parser(skill: Dict[str, Any]) -> int:
    for key in ["Level", "level", "SkillLevel", "skillLevel"]:
        if key in skill:
            try:
                return int(float(str(skill.get(key)).replace(",", "")))
            except Exception:
                pass
    text = _v17_flatten_for_parser(skill)
    m = re.search(r"스킬\s*레벨\s*(\d+)", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return 0


def _v17_referenced_skill_names_for_parser(bundle: Dict[str, Any]) -> set[str]:
    raw_parts = []
    for key in ["arkpassive", "arkgrid", "equipment"]:
        raw_parts.append(_v17_flatten_for_parser(_safe_data(bundle, key) or ""))
    raw = re.sub(r"<[^>]+>", " ", " ".join(raw_parts))
    raw = re.sub(r"\s+", " ", raw)
    refs = set()
    for s in (_safe_data(bundle, "combat_skills") or []):
        name = str(s.get("Name") or "").strip()
        if not name:
            continue
        if name in raw and _v17_skill_level_value_for_parser(s) <= 1:
            idx = raw.find(name)
            win = raw[max(0, idx - 120): idx + len(name) + 220] if idx >= 0 else raw
            if any(k in win for k in ["도약", "초각성", "피해량", "치명타 피해", "치명타 적중률", "숙련된 힘", "아크그리드"]):
                refs.add(name)
    return refs


def parse_combat_skills(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    skills = _safe_data(bundle, "combat_skills") or []
    refs = _v17_referenced_skill_names_for_parser(bundle)
    rows: List[Dict[str, Any]] = []
    for s in skills:
        name = s.get("Name")
        if not name:
            continue
        level = _v17_skill_level_value_for_parser(s)
        if level == 1 and name not in refs:
            continue
        tripod_names = []
        for tripod in s.get("Tripods", []) or []:
            if tripod.get("IsSelected"):
                tripod_names.append(f"{tripod.get('Name')} Lv.{tripod.get('Level')}" if tripod.get("Level") else tripod.get("Name"))
        rune = s.get("Rune") or {}
        base_type = s.get("Type") or s.get("SkillType")
        effective_type, reason = _effective_type_for_parser(s)
        if name in refs and level <= 1:
            reason = (reason + " / " if reason else "") + "도약/아크패시브 설명에서 스킬명 감지"
        rows.append({
            "스킬명": name,
            "레벨": s.get("Level"),
            "기본타입": base_type,
            "타입": effective_type or base_type,
            "타입변경근거": reason,
            "룬": rune.get("Name") if isinstance(rune, dict) else rune,
            "트포": " / ".join([x for x in tripod_names if x]),
            "아이콘": s.get("Icon"),
        })
    return pd.DataFrame(rows)
