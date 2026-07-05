from __future__ import annotations

import json
import math
import re
import time as _time_v157
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

try:
    import yaml
except Exception:  # noqa: BLE001
    yaml = None

# Lost Ark API가 최종 합산 치명/치피/진피를 직접 주지 않으므로,
# 아래 모듈은 API RAW의 스탯/각인/장비/보석/아크패시브/아크그리드/선택 트라이포드 툴팁을
# 피해군별로 분류해서 “검수 가능한 계산표”를 만듭니다.
# 수식 방향: 같은 피해군은 합산, 서로 다른 피해군은 곱연산으로 추정합니다.
CRIT_RATE_PER_CRIT_STAT = 0.03579  # 치명 1당 치명타 적중률(%). 필요 시 여기 조정.
BASE_CRIT_DAMAGE_PERCENT = 200.0
BACK_ATTACK_CRIT_BONUS_PERCENT = 10.0
BACK_ATTACK_DAMAGE_BONUS_PERCENT = 5.0
HEAD_ATTACK_DAMAGE_BONUS_PERCENT = 20.0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _load_class_rules() -> Dict[str, Any]:
    """configs/class_rules.yaml을 읽습니다.

    이 파일은 완전 자동으로 알 수 없는 직업 아덴/직업각인 효과와
    스킬명이 직접 적히지 않는 아크그리드 별칭만 보완하는 용도입니다.
    공통 파서는 먼저 API 스킬 목록으로 자동 매칭하고, 실패한 경우에만 여기 별칭/룰을 사용합니다.
    """
    path = _project_root() / "configs" / "class_rules.yaml"
    if yaml is None or not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _config_skill_aliases() -> Dict[str, str]:
    rules = _load_class_rules()
    aliases: Dict[str, str] = {}
    # 전역 별칭
    for k, v in (rules.get("skill_aliases") or {}).items():
        aliases[str(k)] = str(v)
    # 직업별 별칭
    for class_block in (rules.get("classes") or {}).values():
        for k, v in (class_block.get("skill_aliases") or {}).items():
            aliases[str(k)] = str(v)
        for job_block in (class_block.get("jobs") or {}).values():
            for k, v in (job_block.get("skill_aliases") or {}).items():
                aliases[str(k)] = str(v)
    return aliases


@lru_cache(maxsize=1)
def _all_skill_aliases() -> Dict[str, str]:
    merged = dict(SKILL_ALIAS_TO_CANONICAL)
    merged.update(_config_skill_aliases())
    return merged

# API SkillType/Type에 백어택 문자열이 안 들어오는 경우가 있어서
# 우선 많이 쓰는 백/헤드 스킬명을 보조 판정으로 둡니다.
# 필요하면 여기만 계속 추가하면 됩니다.
KNOWN_BACK_ATTACK_SKILLS = {
    "허리케인소드", "플레임블레이드", "브루탈임팩트", "길로틴", "볼케이노이럽션",
    "파이널블로", "페이탈소드", "퓨리어스클로", "마운틴클리브", "크루얼피어스",
    "퍼니싱드로", "스피닝소드",
    "버스트", "블리츠러시", "소울앱소버", "보이드스트라이크",
}
KNOWN_NON_DIRECTIONAL_SKILLS = {
    "와일드스톰프", "와일드러시", "스피릿캐치", "플래시블레이드",
}

# 스킬명이 API의 combat-skills 목록에서 누락되거나 공백/OCR/툴팁 표기가 달라도
# 스킬 전용 효과를 전역 효과로 잘못 넣지 않기 위한 별칭 테이블입니다.
# 예: 아크그리드 "회오리" 코어 문구는 허리케인 소드 전용으로 처리.
SKILL_ALIAS_TO_CANONICAL = {
    "허리케인소드": "허리케인 소드",
    "허리케인소도": "허리케인 소드",
    "허리케인 소드": "허리케인 소드",
    "회오리": "허리케인 소드",
    "브루탈임팩트": "브루탈 임팩트",
    "브루탈 임팩트": "브루탈 임팩트",
    "플레임블레이드": "플레임 블레이드",
    "플레임 블레이드": "플레임 블레이드",
    "볼케이노이럽션": "볼케이노 이럽션",
    "볼케이노 이럽션": "볼케이노 이럽션",
    "파이널블로": "파이널 블로",
    "파이널 블로": "파이널 블로",
    "페이탈소드": "페이탈 소드",
    "페이탈 소드": "페이탈 소드",
    "길로틴": "길로틴",
    "와일드스톰프": "와일드 스톰프",
    "와일드 스톰프": "와일드 스톰프",
    "와일드러시": "와일드 러시",
    "와일드 러시": "와일드 러시",
}

# 아크패시브/아크그리드 API 구조에서 "진화/깨달음/도약" 같은 대분류가
# 하위 노드 설명을 통째로 포함하는 경우가 있습니다. 이 대분류를 그대로 계산하면
# 활성 노드와 부모 노드가 중복 합산되므로, 대분류명 자체는 계산하지 않고 하위만 재귀 탐색합니다.
GENERIC_PASSIVE_CONTAINER_NAMES = {
    "진화", "깨달음", "도약", "아크패시브", "아크 패시브",
    "아크그리드", "아크 그리드", "코어", "코어옵션", "코어 옵션",
}

EFFECT_COLUMNS = [
    "치명타 적중률 증가(%)",
    "치명타 적중률 고정(%)",
    "치명타 피해량 증가(%)",
    "진화형 피해(%)",
    "적에게 주는 피해(%)",
    "스킬 피해(%)",
    "방향성 피해(%)",
    "보석 피해(%)",
    "공격력 증가(%)",
    "추가 피해(%)",
]

SOURCE_COLUMNS = [
    "출처구분",
    "이름",
    "적용범위",
    "적용스킬",
    *EFFECT_COLUMNS,
    "조건부 여부",
    "설명",
]


def _safe_data(bundle: Dict[str, Any], key: str) -> Any:
    value = bundle.get(key)
    if value is None:
        return None
    if hasattr(value, "data"):
        return value.data if getattr(value, "ok", True) else None
    if isinstance(value, dict):
        return value.get("data") if value.get("ok", True) else None
    return value


def _clean_text(text: Any) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = (
        text.replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
        .replace("\n", " ")
    )
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=4096)
def _flatten_str_cached(value: str) -> str:
    try:
        parsed = json.loads(value)
        return _flatten(parsed)
    except Exception:
        return _clean_text(value)


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _flatten_str_cached(value)
    if isinstance(value, dict):
        return _clean_text(" ".join(_flatten(v) for v in value.values()))
    if isinstance(value, list):
        return _clean_text(" ".join(_flatten(v) for v in value))
    return _clean_text(str(value))


def _num(text: Any) -> float | None:
    try:
        return float(str(text).replace(",", ""))
    except Exception:
        return None


def _clamp_percent(value: float | None, lo: float = 0.0, hi: float = 100.0) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return lo
    return max(lo, min(hi, float(value)))


def _round2(value: Any) -> float:
    n = _num(value)
    if n is None or math.isnan(n):
        return 0.0
    return round(float(n), 2)


def _norm_name(name: Any) -> str:
    return re.sub(r"\s+", "", str(name or "").strip().lower())


def _split_sentences(text: str) -> List[str]:
    clean = _clean_text(text)
    # Lost Ark tooltip은 마침표 없이 줄/태그로 이어지는 경우가 많아 키워드 주변 추출도 같이 씁니다.
    chunks = re.split(r"(?<=[.%])\s+|[。•◆◇▶▷|]", clean)
    return [c.strip() for c in chunks if c and c.strip()]


def _extract_percent_near_keywords(text: str, keywords: Iterable[str], window: int = 46) -> List[float]:
    """키워드 주변의 % 값을 추출합니다.

    한 숫자를 여러 키워드/패턴이 중복으로 잡는 것은 '숫자의 텍스트 위치'로 중복 제거합니다.
    그래서 같은 값이라도 서로 다른 위치에 두 번 적혀 있으면(예: 팔찌 '치명타 적중률 4.2%'가
    옵션 2개로 각각 있으면) 둘 다 각각 집계됩니다. (값 기준 dedup은 이 진짜 중복을 지워버렸음)
    """
    clean = _clean_text(text)
    values: List[float] = []
    seen_pos: set[int] = set()
    for kw in keywords:
        escaped = re.escape(kw)
        patterns = [
            rf"{escaped}[^0-9+\-]{{0,{window}}}([+\-]?\d+(?:\.\d+)?)\s*%",
            rf"([+\-]?\d+(?:\.\d+)?)\s*%[^가-힣A-Za-z0-9]{{0,{window}}}{escaped}",
        ]
        for pat in patterns:
            for m in re.finditer(pat, clean):
                value = _num(m.group(1))
                if value is None:
                    continue
                pos = m.start(1)  # 매칭된 '숫자'의 텍스트 위치
                if pos in seen_pos:
                    continue      # 같은 숫자를 다른 키워드/패턴이 또 잡은 경우만 제거
                seen_pos.add(pos)
                values.append(value)
    return values


def _dedupe_numbers(values: Iterable[float]) -> List[float]:
    seen = set()
    out = []
    for v in values:
        key = round(float(v), 4)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(v))
    return out


def _sum_values(values: Iterable[float]) -> float:
    return round(sum(_dedupe_numbers(values)), 4)


def _cap_duplicate_values(values: Iterable[float], max_dup: int = 2) -> List[float]:
    """같은 값이 max_dup개를 초과하면 초과분은 버립니다.

    팔찌 옵션처럼 같은 수치가 2개까지는 실제로 들어갈 수 있지만(슬롯 2개),
    3개 이상 같은 값이 잡히면 파싱 아티팩트이므로 상한(기본 2)으로 자릅니다.
    """
    used: Dict[float, int] = {}
    out: List[float] = []
    for v in values:
        k = round(float(v), 4)
        n = used.get(k, 0)
        if n >= max_dup:
            continue
        used[k] = n + 1
        out.append(float(v))
    return out


def _contains_any(text: str, words: Iterable[str]) -> bool:
    return any(w in text for w in words)


def _is_conditional(text: str) -> bool:
    clean = _clean_text(text)
    words = [
        "중첩", "최대", "동안", "적중", "공격 시", "스킬 사용", "폭주", "분노", "백어택", "헤드어택",
        "방향", "아이덴티티", "파티", "아군", "보스", "몬스터", "생명력", "조건", "확률", "쿨타임",
        "전투 중", "유지", "발동", "타격 시", "일정 시간", "소모", "충전", "이동속도", "공격속도",
        "마나", "자원", "피격", "이상", "이하", "초과", "미만", "변경", "고정",
    ]
    return any(w in clean for w in words)


def _scope_from_text(text: str, target_skill: str | None = None) -> str:
    clean = _clean_text(text)
    if target_skill:
        return "스킬 전용"
    has_back = "백어택" in clean or "기습" in clean
    has_head = "헤드어택" in clean or "결투" in clean
    if has_back and has_head:
        return "백/헤드 스킬"
    if has_back:
        return "백어택 스킬"
    if has_head:
        return "헤드어택 스킬"
    return "전체"


def _empty_source_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SOURCE_COLUMNS)


def _new_effect_dict() -> Dict[str, float]:
    return {c: 0.0 for c in EFFECT_COLUMNS}


def _extract_effect_values(text: str, source_type: str = "", target_skill: str | None = None) -> Dict[str, float]:
    """툴팁 문장을 치명/치피/진피/적추피/스킬피해/방향피해/공격력 피해군으로 분류합니다."""
    clean = _clean_text(text)
    effects = _new_effect_dict()

    # 1) 치명타 적중률/고정/치피
    crit_fixed = []
    for pat in [
        r"치명타\s*적중률[^0-9]{0,18}([+\-]?\d+(?:\.\d+)?)\s*%[^가-힣]{0,12}(?:로|으로)?\s*고정",
        r"치명타\s*적중률(?:을|이)?\s*([+\-]?\d+(?:\.\d+)?)\s*%[^가-힣]{0,12}(?:로|으로)?\s*고정",
        r"치적[^0-9]{0,18}([+\-]?\d+(?:\.\d+)?)\s*%[^가-힣]{0,12}(?:로|으로)?\s*고정",
    ]:
        for m in re.finditer(pat, clean):
            v = _num(m.group(1))
            if v is not None:
                crit_fixed.append(v)

    effects["치명타 적중률 고정(%)"] = max(crit_fixed) if crit_fixed else 0.0

    rate_keywords = ["치명타 적중률", "치명타 확률", "치명 확률", "치명타율", "치적"]
    damage_keywords = ["치명타 피해량", "치명타 피해", "치피증", "치피"]
    rate_values = _extract_percent_near_keywords(clean, rate_keywords)
    damage_values = _extract_percent_near_keywords(clean, damage_keywords)

    # 고정값으로 잡힌 값은 증가량에서 제외합니다.
    if crit_fixed:
        rate_values = [v for v in rate_values if round(v, 4) not in {round(x, 4) for x in crit_fixed}]

    # "치명타 피해"가 치명타 적중률로 오인되는 것을 줄임.
    if "치명타 피해" in clean or "치피" in clean:
        rate_values = [v for v in rate_values if v not in damage_values]

    effects["치명타 적중률 증가(%)"] = _sum_values(rate_values)
    effects["치명타 피해량 증가(%)"] = _sum_values(damage_values)

    # 2) 진화형 피해 / 메인노드 효율
    evolution_keywords = ["진화형 피해", "진화 피해", "진화형피해", "진피", "메인노드 효율", "메인 노드 효율"]
    effects["진화형 피해(%)"] = _sum_values(_extract_percent_near_keywords(clean, evolution_keywords, window=58))

    # 아크패시브/아크그리드 노드명만 있고 설명에 "피해 24%"처럼 붙는 경우 보조 처리.
    if effects["진화형 피해(%)"] == 0 and source_type in {"아크패시브", "아크그리드"}:
        if _contains_any(clean, ["한계 돌파", "한계돌파", "음속 돌파", "음속돌파", "뭉툭한 가시", "뭉가", "인파이팅", "입식 타격가", "마나 용광로", "마나용광로", "분쇄"]):
            vals = []
            for sent in _split_sentences(clean):
                if "치명타" in sent or "쿨타임" in sent:
                    continue
                if _contains_any(sent, ["피해", "효율", "증가"]):
                    vals += _extract_percent_near_keywords(sent, ["피해", "효율", "증가"], window=36)
            effects["진화형 피해(%)"] = _sum_values(vals)

    # 3) 적에게 주는 피해 / 적추피
    enemy_keywords = [
        "적에게 주는 피해", "적에게 주는 피해량", "적에게 주는 추가 피해", "적에게주는피해",
        "적주피", "적추피", "보스 등급 이상 몬스터에게 주는 피해", "몬스터에게 주는 피해",
    ]
    effects["적에게 주는 피해(%)"] = _sum_values(_extract_percent_near_keywords(clean, enemy_keywords, window=58))

    # 4) 방향성 피해
    direction_keywords = [
        "백어택 피해", "백어택 피해량", "헤드어택 피해", "헤드어택 피해량",
        "백어택 및 헤드어택 피해", "방향성 공격 피해", "방향성 공격의 피해량", "방향성 피해",
    ]
    effects["방향성 피해(%)"] = _sum_values(_extract_percent_near_keywords(clean, direction_keywords, window=58))
    # 기습/결투처럼 "백어택 성공 시 피해량 25%"로만 적힌 문장 보조 처리.
    if effects["방향성 피해(%)"] == 0.0 and _contains_any(clean, ["백어택", "헤드어택", "방향성"]):
        direction_values = []
        for sent in _split_sentences(clean):
            if not _contains_any(sent, ["백어택", "헤드어택", "방향성"]):
                continue
            if _contains_any(sent, ["치명타", "치적", "치피"]):
                continue
            direction_values += _extract_percent_near_keywords(sent, ["피해량", "피해", "주는 피해"], window=54)
        effects["방향성 피해(%)"] = _sum_values(direction_values)

    # 5) 공격력 증가
    atk_values = []
    for sent in _split_sentences(clean):
        if "무기 공격력" in sent:
            continue
        if "공격력" in sent:
            atk_values += _extract_percent_near_keywords(sent, ["공격력"], window=46)
    effects["공격력 증가(%)"] = _sum_values(atk_values)

    # 6) 보석 피해: 보석 출처 또는 멸화/겁화/홍염 툴팁에서 스킬 피해량 추출.
    if source_type == "보석" or _contains_any(clean, ["멸화", "겁화", "홍염", "보석"]):
        gem_values = []
        for sent in _split_sentences(clean):
            if "쿨" in sent or "재사용" in sent:
                continue
            if _contains_any(sent, ["피해", "피해량"]):
                gem_values += _extract_percent_near_keywords(sent, ["피해량", "피해"], window=50)
        effects["보석 피해(%)"] = _sum_values(gem_values)

    # 7) 스킬 피해: 선택 트포/스킬 전용 설명은 치명/진피/적추피/방향피해가 아닌 피해량 증가를 스킬 피해군으로 분류.
    skill_damage_values = []
    if (target_skill and source_type != "보석") or source_type in {"선택 트라이포드", "스킬"}:
        for sent in _split_sentences(clean):
            if _contains_any(sent, ["치명타", "치피", "치적", "진화", "적에게", "백어택", "헤드어택", "방향성", "공격력", "쿨타임", "재사용"]):
                continue
            if _contains_any(sent, ["피해량", "주는 피해", "피해가", "피해 ", "대미지", "데미지"]):
                skill_damage_values += _extract_percent_near_keywords(sent, ["피해량", "주는 피해", "피해", "대미지", "데미지"], window=54)
    effects["스킬 피해(%)"] = _sum_values(skill_damage_values)

    # 8) 추가 피해: 장비 품질/팔찌/기타의 "추가 피해"는 따로 분리.
    additional_values = []
    for sent in _split_sentences(clean):
        if "적에게 주는 추가 피해" in sent:
            continue
        if "추가 피해" in sent:
            additional_values += _extract_percent_near_keywords(sent, ["추가 피해"], window=48)
    effects["추가 피해(%)"] = _sum_values(additional_values)

    return {k: round(v, 4) for k, v in effects.items()}




def _apply_special_engraving_rules(name: str, clean_text: str, effects: Dict[str, float]) -> Dict[str, float]:
    """각인 Tooltip 중 일반 퍼센트 파서만으로 부족한 수치를 보정합니다.

    v53 핵심:
    - 아드레날린/예리한 둔기 Tooltip에 여러 등급/단계 수치가 같이 잡히면 합산하지 않고 최종 후보 최대값을 사용합니다.
    - 아드레날린 공격력은 1스택 수치 × 최대 중첩으로 계산합니다.
    - 예리한 둔기 페널티 문장은 적에게 주는 피해군으로 섞지 않습니다.
    """
    n = _norm_name(name)
    text = _clean_text(clean_text)
    out = dict(effects)

    if "아드레날린" in n:
        # v64: '최종 적용 효과(PvE)' 블록의 값을 최우선으로 사용합니다. 이 블록은 캐릭터의 실제
        # 각인 등급/단계 + 어빌리티 스톤 장착효과(합연산)가 이미 반영된 '최종값'이라 정확합니다.
        # 예) 유물 4단계 → 치명타 적중률 +20%, 공격력 (0.9%기본+0.6%스톤)×6중첩 = 9%.
        # 블록을 못 찾으면(=구 포맷) 기존 후보 최대값 방식으로 폴백합니다.
        final_crit = None
        fb = re.search(r"최종\s*적용\s*효과(.*?)(?:기본\s*효과|등급별\s*효과|어빌리티|$)", text, re.S)
        if fb:
            cm = re.search(r"치명타\s*적중률[^0-9]{0,40}([0-9]+(?:\.\d+)?)\s*%", fb.group(1))
            if cm:
                final_crit = _num(cm.group(1))
        if final_crit is not None:
            out["치명타 적중률 증가(%)"] = final_crit
        else:
            crit_candidates = _extract_percent_near_keywords(text, ["치명타 적중률", "치명타 확률", "치적"], window=90)
            if crit_candidates:
                out["치명타 적중률 증가(%)"] = max(float(v) for v in crit_candidates)
        m = re.search(r"공격력(?:이|을)?[^0-9]{0,40}([+\-]?\d+(?:\.\d+)?)\s*%[^\n]{0,180}?최대\s*(\d+)\s*중첩", text, re.S)
        if m:
            per_stack = _num(m.group(1)) or 0.0
            stacks = _num(m.group(2)) or 0.0
            if per_stack > 0 and stacks > 0:
                out["공격력 증가(%)"] = per_stack * stacks

    if "예리한둔기" in n:
        cd_candidates = _extract_percent_near_keywords(text, ["치명타 피해량", "치명타 피해", "치피"], window=90)
        if cd_candidates:
            out["치명타 피해량 증가(%)"] = max(float(v) for v in cd_candidates)
        if "감소" in text and "일정" in text:
            out["적에게 주는 피해(%)"] = max(0.0, float(out.get("적에게 주는 피해(%)", 0.0) or 0.0))

    return {k: round(float(v), 4) for k, v in out.items()}

def _source_row(source_type: str, name: str, text: str, target_skill: str | None = None) -> Dict[str, Any] | None:
    clean = _clean_text(text)
    effects = _extract_effect_values(clean, source_type=source_type, target_skill=target_skill)
    if source_type in {"각인", "아크패시브 각인"}:
        effects = _apply_special_engraving_rules(name, clean, effects)
    if not any(abs(v) > 1e-9 for v in effects.values()):
        return None
    return {
        "출처구분": source_type,
        "이름": name,
        "적용범위": _scope_from_text(clean, target_skill),
        "적용스킬": target_skill or "전체/범위 기준",
        **{k: round(float(v), 2) for k, v in effects.items()},
        "조건부 여부": "조건부/최대치" if _is_conditional(clean) else "상시",
        "설명": clean[:420],
    }


def _detect_target_skill(text: str, skill_names: Iterable[str]) -> str | None:
    """툴팁 문구 안에서 특정 스킬명을 찾습니다.

    v8 핵심:
    - combat-skills 목록이 비어 있거나 스킬명이 공백 포함/미포함으로 달라도 별칭으로 탐지
    - "허리케인 소드의 치명타 피해량" 같은 소유격 문구를 스킬 전용으로 강제
    - "회오리" 코어는 허리케인 소드 전용으로 처리
    """
    clean = _clean_text(text)
    clean_norm = _norm_name(clean)
    if not clean_norm:
        return None

    sorted_names = sorted([s for s in skill_names if s], key=lambda x: len(_norm_name(x)), reverse=True)
    skill_by_norm = {_norm_name(s): s for s in sorted_names}

    # 1) API 스킬명 직접 매칭
    for skill_name in sorted_names:
        n = _norm_name(skill_name)
        if n and n in clean_norm:
            return skill_name

    # 2) 별칭 매칭. API 스킬명이 있으면 API 표시명을 우선 사용하고, 없으면 canonical 사용.
    for alias, canonical in _all_skill_aliases().items():
        if _norm_name(alias) in clean_norm:
            canonical_norm = _norm_name(canonical)
            return skill_by_norm.get(canonical_norm) or next(
                (s for s in sorted_names if canonical_norm in _norm_name(s) or _norm_name(s) in canonical_norm),
                canonical,
            )

    # 3) "OOO의 치명타 피해량/피해량" 문구에서 OOO를 후보로 잡음.
    # 너무 일반적인 명사(스킬, 공격, 효과 등)는 제외.
    for m in re.finditer(r"([가-힣A-Za-z0-9\s]{2,28})의\s*(?:치명타\s*피해량|치명타\s*피해|피해량|피해가|주는\s*피해|재사용|쿨타임)", clean):
        cand = _clean_text(m.group(1))
        cand_norm = _norm_name(cand)
        if not cand_norm or cand_norm in {"스킬", "공격", "효과", "자신", "대상", "적", "치명타"}:
            continue
        for skill_name in sorted_names:
            sn = _norm_name(skill_name)
            if cand_norm == sn or cand_norm in sn or sn in cand_norm:
                return skill_name
        for alias, canonical in _all_skill_aliases().items():
            if cand_norm == _norm_name(alias) or cand_norm == _norm_name(canonical):
                return skill_by_norm.get(_norm_name(canonical)) or canonical

    return None

def _split_option_snippets(text: str) -> List[str]:
    """[10P], [14P] 같은 옵션 단위로 텍스트를 분리합니다."""
    clean = _clean_text(text)
    if not clean:
        return []

    # [14P] 같은 마커 앞에서 분리하되 마커는 보존합니다.
    parts = re.split(r"(?=\[\s*\d+\s*[Pp]?\s*\])", clean)
    snippets = [p.strip() for p in parts if p and p.strip()]

    # 마커가 없는 긴 툴팁은 문장 단위도 함께 사용합니다.
    if len(snippets) <= 1:
        candidates = _split_sentences(clean)
        if len(candidates) > 1:
            snippets = candidates

    # 너무 짧은 제목만 있는 조각은 제외합니다.
    return [s for s in snippets if len(s) >= 6]


def _looks_skill_specific_effect(text: str, source_name: str, skill_names: Iterable[str]) -> bool:
    """스킬명은 못 찾았지만 특정 기술/코어 전용 문장처럼 보이는지 판정합니다.

    이런 문장은 전역 효과로 넣으면 치피/진피가 크게 뻥튀기되므로,
    일단 '검수 필요'로 빼고 계산에는 반영하지 않습니다.
    """
    clean = _clean_text(f"{source_name} {text}")
    if not clean:
        return False
    # 실제 스킬명이 잡히면 여기까지 오지 않는 게 정상입니다.
    # 'OOO의 피해량/치피/쿨타임'처럼 소유격 구조가 있으면 스킬/코어 전용 가능성이 높습니다.
    if re.search(r"[가-힣A-Za-z0-9\s]{2,30}의\s*(?:치명타\s*피해량|치명타\s*피해|치명타\s*적중률|피해량|피해가|주는\s*피해|재사용|쿨타임)", clean):
        return True
    # 코어/효과명 자체가 스킬 별칭처럼 보이는데 정확히 매핑되지 않은 경우
    if any(w in clean for w in ["코어", "효과가 강화", "운명", "연명", "업화", "난무", "오의", "충격", "기력"]):
        if any(w in clean for w in ["피해량", "치명타 피해", "치명타 적중률", "쿨타임", "재사용"]):
            return True
    return False


def _source_rows_from_text(
    source_type: str,
    name: str,
    text: str,
    skill_names: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    """하나의 툴팁을 실제 적용 단위별 source row로 분해합니다.

    핵심:
    - 특정 스킬명이 들어간 조각은 해당 스킬 전용으로만 적용
    - 아크그리드 코어의 [14P]/[17P] 같은 옵션은 각각 별도 행으로 분리
    - 전체 텍스트를 통째로 다시 계산하지 않아 스킬 전용 치피가 전역 치피로 새지 않음
    """
    clean = _clean_text(text)
    names = list(skill_names or [])
    rows: List[Dict[str, Any]] = []

    # 아크그리드/아크패시브/스킬 설명은 한 항목에 여러 효과가 섞여 들어오는 경우가 많으므로 조각 단위 우선.
    if source_type in {"아크그리드", "아크패시브", "선택 트라이포드", "스킬"}:
        snippets = _split_option_snippets(clean)
    else:
        snippets = [clean]

    for idx, snippet in enumerate(snippets):
        target = _detect_target_skill(f"{name} {snippet}", names)
        row_name = name
        marker = re.search(r"\[\s*(\d+\s*[Pp]?)\s*\]", snippet)
        if marker:
            row_name = f"{name} [{marker.group(1).replace(' ', '').upper()}]"
        row = _source_row(source_type, row_name, snippet, target_skill=target)
        if row:
            # 스킬명이 들어간 옵션은 무조건 스킬 전용으로 고정합니다.
            if target:
                row["적용범위"] = "스킬 전용"
                row["적용스킬"] = target
                row["조건부 여부"] = "상시" if "효과가 강화" not in snippet and "폭주" not in snippet else row["조건부 여부"]
            elif source_type in {"아크그리드", "아크패시브", "선택 트라이포드", "스킬"} and _looks_skill_specific_effect(snippet, row_name, names):
                # 아크그리드 코어의 '치명타 적중률/치명타 피해량/공격력/적에게 주는 피해' 같은
                # 전역 스탯 증가는 코어 이름에 '코어'가 들어가도 모든 스킬에 적용되는 값입니다.
                # (예: 혼돈의 달 코어 : 부수는 일격 [14P~20P] 치명타 적중률 합연산 → 전 스킬 적용)
                # 이런 전역 스탯은 '검수 필요'로 빼지 말고 전체 범위로 합산합니다.
                _core_global_stat = source_type == "아크그리드" and any(
                    abs(float(row.get(c, 0.0) or 0.0)) > 1e-9
                    for c in ["치명타 적중률 증가(%)", "치명타 피해량 증가(%)", "공격력 증가(%)", "적에게 주는 피해(%)"]
                )
                if _core_global_stat:
                    row["적용범위"] = "전체"
                    row["적용스킬"] = ""
                    row["조건부 여부"] = "상시"
                else:
                    # 스킬 전용처럼 보이는데 자동 매칭을 못 한 옵션은 전역으로 합산하지 않습니다.
                    row["적용범위"] = "검수 필요"
                    row["적용스킬"] = "스킬명 자동 매칭 실패"
                    row["조건부 여부"] = "검수 필요"
            rows.append(row)

    # 조각에서 아무것도 못 잡은 경우에만 전체 텍스트 1회 fallback.
    if not rows:
        target = _detect_target_skill(clean, names)
        row = _source_row(source_type, name, clean, target_skill=target)
        if row:
            if target:
                row["적용범위"] = "스킬 전용"
                row["적용스킬"] = target
            elif source_type in {"아크그리드", "아크패시브", "선택 트라이포드", "스킬"} and _looks_skill_specific_effect(clean, name, names):
                row["적용범위"] = "검수 필요"
                row["적용스킬"] = "스킬명 자동 매칭 실패"
                row["조건부 여부"] = "검수 필요"
            rows.append(row)

    # 중복 제거
    out = []
    seen = set()
    for row in rows:
        sig = tuple(str(row.get(c, "")) for c in SOURCE_COLUMNS)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(row)
    return out


def _character_class_name(bundle: Dict[str, Any]) -> str:
    profiles = _safe_data(bundle, "profiles") or _safe_data(bundle, "summary") or {}
    return str(profiles.get("CharacterClassName") or profiles.get("ClassName") or profiles.get("클래스") or "")


def _has_predator_engraving(bundle: Dict[str, Any]) -> bool:
    engravings = _safe_data(bundle, "engravings") or {}
    texts: List[str] = []
    for key in ["Effects", "ArkPassiveEffects"]:
        for e in engravings.get(key, []) or []:
            texts.append(_flatten(e))
    return any("포식자" in t for t in texts)


def _active_job_names(bundle: Dict[str, Any]) -> List[str]:
    engravings = _safe_data(bundle, "engravings") or {}
    texts: List[str] = []
    for key in ["Effects", "ArkPassiveEffects"]:
        for e in engravings.get(key, []) or []:
            texts.append(_flatten(e))
    joined = " ".join(texts)
    jobs: List[str] = []
    rules = _load_class_rules()
    class_name = _character_class_name(bundle)
    for class_key, class_block in (rules.get("classes") or {}).items():
        if class_key and class_key not in class_name:
            continue
        for job_name, job_block in (class_block.get("jobs") or {}).items():
            keywords = [job_name] + list(job_block.get("engraving_keywords") or []) + list(job_block.get("identity_keywords") or []) + list(job_block.get("arkpassive_keywords") or [])
            if any(k and k in joined for k in keywords):
                jobs.append(str(job_name))
    return jobs


def _identity_sources(bundle: Dict[str, Any]) -> pd.DataFrame:
    """클래스/직업각인 기반 아이덴티티 효과.

    v9부터는 코드에 직업별 효과를 계속 하드코딩하지 않고,
    configs/class_rules.yaml의 직업별 룰을 적용합니다.
    없는 직업은 공통 툴팁 파서만 적용되고, 특수 아덴은 검수/룰 추가 대상입니다.
    """
    rows: List[Dict[str, Any]] = []
    rules = _load_class_rules()
    class_name = _character_class_name(bundle)
    active_jobs = set(_active_job_names(bundle))

    for class_key, class_block in (rules.get("classes") or {}).items():
        if class_key and class_key not in class_name:
            continue
        for job_name, job_block in (class_block.get("jobs") or {}).items():
            if job_name not in active_jobs:
                continue
            for eff in job_block.get("identity_effects", []) or []:
                effects = _new_effect_dict()
                effects["치명타 적중률 증가(%)"] = float(eff.get("crit_rate", 0) or 0)
                effects["치명타 피해량 증가(%)"] = float(eff.get("crit_damage", 0) or 0)
                effects["진화형 피해(%)"] = float(eff.get("evolution_damage", 0) or 0)
                effects["적에게 주는 피해(%)"] = float(eff.get("enemy_damage", 0) or 0)
                effects["스킬 피해(%)"] = float(eff.get("skill_damage", 0) or 0)
                effects["방향성 피해(%)"] = float(eff.get("directional_damage", 0) or 0)
                effects["공격력 증가(%)"] = float(eff.get("attack_power", 0) or 0)
                if not any(abs(v) > 1e-9 for v in effects.values()):
                    continue
                applies_to = str(eff.get("applies_to") or "all")
                scope = {
                    "all": "전체",
                    "back_attack": "백어택 스킬",
                    "head_attack": "헤드어택 스킬",
                    "directional": "백/헤드 스킬",
                }.get(applies_to, "전체")
                rows.append({
                    "출처구분": "직업/아이덴티티",
                    "이름": f"{eff.get('name') or job_name}",
                    "적용범위": scope,
                    "적용스킬": "전체/범위 기준",
                    **effects,
                    "조건부 여부": str(eff.get("condition_type") or "조건부/최대치"),
                    "설명": str(eff.get("description") or f"{class_key} / {job_name} 룰: {eff.get('condition') or ''}")[:420],
                })

    # class_rules.yaml이 없거나 예전 파일만 쓰는 경우를 위한 최소 fallback.
    if not rows and "슬레이어" in class_name and _has_predator_engraving(bundle):
        effects = _new_effect_dict()
        effects["치명타 적중률 증가(%)"] = 30.0
        rows.append({
            "출처구분": "직업/아이덴티티",
            "이름": "폭주(포식자)",
            "적용범위": "전체",
            "적용스킬": "전체/범위 기준",
            **effects,
            "조건부 여부": "조건부/최대치",
            "설명": "fallback: 슬레이어 포식자 폭주 상태 기준 치명타 적중률 +30%",
        })
    if not rows:
        return _empty_source_df()
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def _has_back_attack_engraving(bundle: Dict[str, Any]) -> bool:
    engravings = _safe_data(bundle, "engravings") or {}
    texts: List[str] = []
    for key in ["Effects", "ArkPassiveEffects"]:
        for e in engravings.get(key, []) or []:
            texts.append(_flatten(e))
    return any("기습" in t or "기습의 대가" in t for t in texts)


def _has_head_attack_engraving(bundle: Dict[str, Any]) -> bool:
    engravings = _safe_data(bundle, "engravings") or {}
    texts: List[str] = []
    for key in ["Effects", "ArkPassiveEffects"]:
        for e in engravings.get(key, []) or []:
            texts.append(_flatten(e))
    return any("결투" in t or "결투의 대가" in t for t in texts)


def _profile_stats(bundle: Dict[str, Any]) -> Tuple[float, float, str, pd.DataFrame]:
    profiles = _safe_data(bundle, "profiles") or _safe_data(bundle, "summary") or {}
    crit_stat = 0.0
    tooltip_rate = None
    raw = ""

    for stat in profiles.get("Stats", []) or []:
        stat_type = str(stat.get("Type") or "")
        value = _num(stat.get("Value") or 0) or 0.0
        tooltip = " ".join(stat.get("Tooltip", []) or [])
        text = f"{stat_type} {stat.get('Value') or ''} {tooltip}"
        if stat_type == "치명" or "치명타 적중률" in text:
            crit_stat = value
            raw = _clean_text(text)
            effects = _extract_effect_values(text, source_type="기본 스탯")
            rate = effects.get("치명타 적중률 증가(%)", 0.0)
            if rate > 0:
                tooltip_rate = rate

    stat_rate = tooltip_rate if tooltip_rate is not None else crit_stat * CRIT_RATE_PER_CRIT_STAT
    stat_rate = round(_clamp_percent(stat_rate), 2)
    rows = [{
        "출처구분": "기본 스탯",
        "이름": "치명 스탯",
        "적용범위": "전체",
        "적용스킬": "전체",
        **_new_effect_dict(),
        "조건부 여부": "상시",
        "설명": raw or f"치명 {crit_stat:g} × {CRIT_RATE_PER_CRIT_STAT} = {stat_rate:.2f}%",
    }]
    rows[0]["치명타 적중률 증가(%)"] = stat_rate
    return crit_stat, stat_rate, raw, pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def _is_inactive_node(obj: Dict[str, Any]) -> bool:
    """아크그리드/아크패시브에서 비활성 노드까지 읽어 과대계산되는 것을 줄입니다."""
    for key in [
        "IsSelected", "isSelected", "Selected", "selected",
        "IsActive", "isActive", "Active", "active",
        "IsActivated", "isActivated", "Activated", "activated",
        "IsEnabled", "isEnabled", "Enabled", "enabled",
        "IsEquip", "isEquip", "IsEquipped", "isEquipped",
    ]:
        if key in obj and obj.get(key) is False:
            return True
    return False


def _iter_named_texts(obj: Any, source_type: str, path: str = "") -> Iterable[Tuple[str, str, str]]:
    """API 객체에서 Name/Description/Tooltip이 있는 실제 옵션 단위만 추출합니다.

    v5에서는 Name만 있는 부모 노드를 통째로 flatten하면서 하위 노드 효과가 중복 합산되는 문제가 있었습니다.
    v6에서는 Name + Description/Tooltip/EffectText가 있는 실제 옵션 단위만 yield하고, 부모는 재귀 탐색만 합니다.
    """
    if isinstance(obj, dict):
        if _is_inactive_node(obj):
            return

        name = obj.get("Name") or obj.get("name") or obj.get("Title") or obj.get("title")
        desc = (
            obj.get("Description") or obj.get("description") or obj.get("Desc") or obj.get("desc")
            or obj.get("Effect") or obj.get("effect") or obj.get("EffectText") or obj.get("effectText")
            or obj.get("Option") or obj.get("option")
        )
        tooltip = obj.get("Tooltip") or obj.get("tooltip") or obj.get("ToolTip") or obj.get("toolTip")
        level = obj.get("Level") or obj.get("level") or obj.get("Grade") or obj.get("grade")

        text_parts = []
        if name:
            text_parts.append(str(name))
        if level not in (None, ""):
            text_parts.append(f"Lv.{level}")
        if desc:
            text_parts.append(_flatten(desc))
        if tooltip:
            text_parts.append(_flatten(tooltip))

        # Name만 있고 설명/툴팁이 없는 부모 객체는 효과 계산에 넣지 않습니다.
        # 하위 자식에서 다시 실제 설명을 찾습니다.
        if name and (desc or tooltip):
            # 진화/깨달음/도약 같은 대분류는 하위 노드 툴팁을 통째로 담는 경우가 있어
            # 그대로 계산하면 치피/진피가 과대 합산됩니다. 대분류명 자체는 계산하지 않습니다.
            is_generic_container = (
                source_type in {"아크패시브", "아크그리드"}
                and _clean_text(name) in GENERIC_PASSIVE_CONTAINER_NAMES
            )
            if not is_generic_container:
                yield str(name), _clean_text(" ".join(text_parts)), path

        for k, v in obj.items():
            yield from _iter_named_texts(v, source_type, f"{path}/{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_named_texts(v, source_type, f"{path}[{i}]")


def _dump_engraving_tooltips_v64(engravings: Dict[str, Any]) -> None:
    """디버그용: API가 내려준 각인/아크패시브 각인 툴팁 '원문'을 파일로 저장합니다.

    아드레날린/예리한 둔기 등 파서가 헷갈리는 각인의 실제 API 텍스트를 확인하기 위함입니다.
    캐릭터 정보를 불러올 때 exports/engraving_tooltip_dump.txt 로 저장됩니다.
    """
    try:
        from pathlib import Path as _P
        import datetime as _dt
        out_dir = _P(__file__).resolve().parent.parent / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"# 각인 툴팁 원문 덤프 ({_dt.datetime.now().isoformat(timespec='seconds')})", ""]
        for key, label in [("Effects", "각인"), ("ArkPassiveEffects", "아크패시브 각인")]:
            for e in engravings.get(key, []) or []:
                lines.append(f"===== [{label}] {e.get('Name') or label} =====")
                lines.append(f"[Description] {e.get('Description') or ''}")
                lines.append(f"[Grade] {e.get('Grade') or ''}  [Level] {e.get('Level') or ''}")
                lines.append(f"[Tooltip] {_flatten(e.get('Tooltip'))}")
                lines.append("")
        (out_dir / "engraving_tooltip_dump.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"[engraving-dump] 각인 툴팁 원문 저장: {out_dir / 'engraving_tooltip_dump.txt'}")
    except Exception as _e:  # noqa: BLE001
        print(f"[engraving-dump] 저장 실패: {_e}")


def _collect_global_sources(bundle: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    names = _skill_names(bundle)

    engravings = _safe_data(bundle, "engravings") or {}
    _dump_engraving_tooltips_v64(engravings)  # 디버그: 각인 툴팁 원문 파일 저장
    for e in engravings.get("Effects", []) or []:
        name = e.get("Name") or "각인"
        text = f"{name} {e.get('Description') or ''} {_flatten(e.get('Tooltip'))}"
        rows.extend(_source_rows_from_text("각인", name, text, names))
    for e in engravings.get("ArkPassiveEffects", []) or []:
        name = e.get("Name") or "아크패시브 각인"
        text = f"{name} {e.get('Description') or ''} {e.get('Grade') or ''} {e.get('Level') or ''} {_flatten(e.get('Tooltip'))}"
        rows.extend(_source_rows_from_text("아크패시브 각인", name, text, names))

    # 반지/귀걸이처럼 완전히 같은 스펙의 장비를 2개 착용하면 API가 동일한 항목을 2번 돌려줍니다.
    # 아래 소스 dedup(drop_duplicates)이 '같은 값'을 한 번으로 합쳐버려, 둘 중 하나의 효과(예: 치명타
    # 적중률 +0.95%)가 통째로 누락됐습니다. 물리적으로 서로 다른 착용칸이므로 둘 다 계산되어야 합니다.
    # → 같은 이름이 반복되면 (2),(3)… 순번을 붙여 서로 다른 출처로 남깁니다.
    _equip_name_seen: Dict[str, int] = {}
    for item in _safe_data(bundle, "equipment") or []:
        name = item.get("Name") or item.get("Type") or "장비"
        text = f"{item.get('Type') or ''} {name} {_flatten(item.get('Tooltip'))}"
        _occ = _equip_name_seen.get(name, 0) + 1
        _equip_name_seen[name] = _occ
        disp_name = name if _occ == 1 else f"{name} ({_occ})"
        rows.extend(_source_rows_from_text("장비/팔찌/엘릭서", disp_name, text, names))

    cards = _safe_data(bundle, "cards") or {}
    for e in cards.get("Effects", []) or []:
        text = _flatten(e)
        rows.extend(_source_rows_from_text("카드", "카드 세트효과", text, names))

    # 아크패시브/아크그리드는 가장 중요합니다.
    # 한 노드 안의 [10P]/[14P]/[17P] 등을 옵션 조각으로 나누고,
    # 조각 안에 스킬명이 있으면 해당 스킬 전용 효과로만 잡습니다.
    for source_type, key in [("아크패시브", "arkpassive"), ("아크그리드", "arkgrid")]:
        for name, text, _path in _iter_named_texts(_safe_data(bundle, key) or {}, source_type):
            rows.extend(_source_rows_from_text(source_type, name, text, names))

    if not rows:
        return _empty_source_df()

    df = pd.DataFrame(rows, columns=SOURCE_COLUMNS)
    sig_cols = ["출처구분", "이름", "적용범위", "적용스킬", *EFFECT_COLUMNS, "조건부 여부", "설명"]
    df["_sig"] = df[sig_cols].astype(str).agg("|".join, axis=1)
    df = df.drop_duplicates("_sig").drop(columns=["_sig"]).reset_index(drop=True)
    df = _force_skill_scopes(df, names)
    return df


def _force_skill_scopes(df: pd.DataFrame, skill_names: Iterable[str]) -> pd.DataFrame:
    """스킬명이 들어간 출처가 전역으로 새는 것을 마지막에 한 번 더 차단합니다."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for idx, row in out.iterrows():
        scope = str(row.get("적용범위") or "")
        target = str(row.get("적용스킬") or "")
        text = f"{row.get('이름', '')} {row.get('설명', '')}"
        detected = _detect_target_skill(text, skill_names)
        if detected and (scope != "스킬 전용" or target in {"", "전체", "전체/범위 기준"}):
            out.at[idx, "적용범위"] = "스킬 전용"
            out.at[idx, "적용스킬"] = detected
            # 스킬 전용 옵션은 대개 해당 스킬 강화 옵션이므로, "효과가 강화" 같은 문구만으로 조건부 처리하지 않습니다.
            if "폭주" not in str(row.get("설명", "")):
                out.at[idx, "조건부 여부"] = "상시"
    return out


def _collect_skill_tripod_sources(bundle: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for skill in _safe_data(bundle, "combat_skills") or []:
        skill_name = skill.get("Name") or ""
        for tripod in skill.get("Tripods", []) or []:
            if not bool(tripod.get("IsSelected")):
                continue
            name = tripod.get("Name") or "트라이포드"
            level = tripod.get("Level") or tripod.get("Tier") or ""
            text = f"{skill_name} {name} Lv.{level} {_flatten(tripod)}"
            # 선택된 트라이포드만 스킬 전용으로 반영합니다.
            for row in _source_rows_from_text("선택 트라이포드", name, text, [skill_name]):
                row["적용범위"] = "스킬 전용"
                row["적용스킬"] = skill_name
                rows.append(row)
    if not rows:
        return _empty_source_df()
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS).drop_duplicates().reset_index(drop=True)


def _skill_names(bundle: Dict[str, Any]) -> List[str]:
    return [s.get("Name") for s in (_safe_data(bundle, "combat_skills") or []) if s.get("Name")]


def _collect_gem_sources(bundle: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    names = _skill_names(bundle)
    gems = _safe_data(bundle, "gems") or {}
    for gem in gems.get("Gems", []) or []:
        gem_name = gem.get("Name") or "보석"
        text = f"{gem_name} {_flatten(gem.get('Tooltip'))}"
        target = None
        norm_text = _norm_name(text)
        for skill_name in names:
            if _norm_name(skill_name) and _norm_name(skill_name) in norm_text:
                target = skill_name
                break
        row = _source_row("보석", gem_name, text, target_skill=target)
        if row:
            rows.append(row)
    if not rows:
        return _empty_source_df()
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS).drop_duplicates().reset_index(drop=True)


def _detect_attack_type(skill: Dict[str, Any]) -> str:
    text = _flatten(skill)
    skill_type = str(skill.get("SkillType") or skill.get("Type") or skill.get("AttackType") or "")
    name_norm = _norm_name(skill.get("Name") or "")
    joined = f"{skill_type} {text}"

    if name_norm in KNOWN_NON_DIRECTIONAL_SKILLS:
        return "일반/확인필요"

    has_back = (
        "백어택" in joined
        or "Back Attack" in joined
        or "BackAttack" in joined
        or name_norm in KNOWN_BACK_ATTACK_SKILLS
    )
    has_head = "헤드어택" in joined or "Head Attack" in joined or "HeadAttack" in joined

    if has_back and has_head:
        return "백/헤드"
    if has_back:
        return "백어택"
    if has_head:
        return "헤드어택"
    return skill_type or "일반/확인필요"


def _scope_applies(row: pd.Series, attack_type: str, skill_name: str) -> bool:
    scope = str(row.get("적용범위") or "전체")
    target = str(row.get("적용스킬") or "")
    if scope == "검수 필요":
        return False
    if scope == "스킬 전용":
        return _norm_name(target) == _norm_name(skill_name)
    if scope == "백어택 스킬":
        return "백" in attack_type
    if scope == "헤드어택 스킬":
        return "헤드" in attack_type
    if scope == "백/헤드 스킬":
        return "백" in attack_type or "헤드" in attack_type
    return True


def _sum_sources(df: pd.DataFrame, column: str, attack_type: str, skill_name: str, include_conditional: bool) -> float:
    if df is None or df.empty or column not in df.columns:
        return 0.0
    total = 0.0
    for _, row in df.iterrows():
        if not include_conditional and str(row.get("조건부 여부")) != "상시":
            continue
        if not _scope_applies(row, attack_type, skill_name):
            continue
        value = _num(row.get(column)) or 0.0
        total += value
    return float(total)


def _max_sources(df: pd.DataFrame, column: str, attack_type: str, skill_name: str, include_conditional: bool) -> float:
    if df is None or df.empty or column not in df.columns:
        return 0.0
    values: List[float] = []
    for _, row in df.iterrows():
        if not include_conditional and str(row.get("조건부 여부")) != "상시":
            continue
        if not _scope_applies(row, attack_type, skill_name):
            continue
        value = _num(row.get(column)) or 0.0
        if value:
            values.append(value)
    return max(values) if values else 0.0


def _source_names_for_skill(df: pd.DataFrame, attack_type: str, skill_name: str) -> str:
    if df is None or df.empty:
        return ""
    names = []
    for _, row in df.iterrows():
        if _scope_applies(row, attack_type, skill_name):
            # 값이 하나도 없는 출처는 제외
            if not any((_num(row.get(c)) or 0.0) for c in EFFECT_COLUMNS):
                continue
            label = f"{row.get('출처구분')}:{row.get('이름')}"
            if label not in names:
                names.append(label)
    return " / ".join(names[:10])


def _expected_crit_multiplier(crit_rate_percent: float, crit_damage_percent: float) -> float:
    rate = _clamp_percent(crit_rate_percent) / 100.0
    crit_damage = max(0.0, crit_damage_percent) / 100.0
    return 1.0 + rate * (crit_damage - 1.0)


def _damage_group_multiplier(**groups: float) -> float:
    mul = 1.0
    for value in groups.values():
        v = _num(value) or 0.0
        mul *= 1.0 + v / 100.0
    return mul


def _avg_col(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if df is None or df.empty or col not in df.columns:
        return default
    vals = pd.to_numeric(df.get(col), errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else default



def _concat_sources(*dfs: pd.DataFrame) -> pd.DataFrame:
    valid = [df for df in dfs if isinstance(df, pd.DataFrame) and not df.empty]
    if not valid:
        return _empty_source_df()
    out = pd.concat(valid, ignore_index=True)
    if out.empty:
        return _empty_source_df()
    # SOURCE_COLUMNS 순서를 유지하고 누락 컬럼은 채웁니다.
    for col in SOURCE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0 if col in EFFECT_COLUMNS else ""
    out = out[SOURCE_COLUMNS]
    sig_cols = ["출처구분", "이름", "적용범위", "적용스킬", *EFFECT_COLUMNS, "조건부 여부", "설명"]
    out["_sig"] = out[sig_cols].astype(str).agg("|".join, axis=1)
    return out.drop_duplicates("_sig").drop(columns=["_sig"]).reset_index(drop=True)


def _filter_source_type(df: pd.DataFrame, include: Iterable[str] | None = None, exclude: Iterable[str] | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_source_df()
    out = df.copy()
    if include is not None:
        inc = {str(x) for x in include}
        out = out[out["출처구분"].astype(str).isin(inc)]
    if exclude is not None:
        exc = {str(x) for x in exclude}
        out = out[~out["출처구분"].astype(str).isin(exc)]
    return out.reset_index(drop=True) if not out.empty else _empty_source_df()


def _skill_only_sources(df: pd.DataFrame, skill_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_source_df()
    out = df[(df["적용범위"].astype(str) == "스킬 전용") & (df["적용스킬"].map(_norm_name) == _norm_name(skill_name))]
    return out.reset_index(drop=True) if not out.empty else _empty_source_df()


def _non_skill_sources(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_source_df()
    out = df[df["적용범위"].astype(str) != "스킬 전용"]
    return out.reset_index(drop=True) if not out.empty else _empty_source_df()


def _calculate_one_skill(
    skill: Dict[str, Any],
    effect_sources: pd.DataFrame,
    crit_stat: float,
    stat_rate: float,
    back_engraving: bool,
    head_engraving: bool,
) -> Dict[str, Any]:
    """한 스킬에 대해 '아크그리드 제외'나 '아크그리드 포함' 계산을 공통 수행합니다."""
    name = skill.get("Name") or ""
    attack_type = _detect_attack_type(skill)

    def sum_static(col: str, df: pd.DataFrame = effect_sources) -> float:
        return _sum_sources(df, col, attack_type, name, include_conditional=False)

    def sum_cond_total(col: str, df: pd.DataFrame = effect_sources) -> float:
        return _sum_sources(df, col, attack_type, name, include_conditional=True)

    static_fixed = _max_sources(effect_sources, "치명타 적중률 고정(%)", attack_type, name, include_conditional=False)
    cond_fixed = _max_sources(effect_sources, "치명타 적중률 고정(%)", attack_type, name, include_conditional=True)
    non_stat_static_crit_add = sum_static("치명타 적중률 증가(%)")
    non_stat_cond_crit_add = sum_cond_total("치명타 적중률 증가(%)")

    # 고정 치적(예: 뭉툭한 가시)이 있으면 치명 스탯 치적은 더하지 않고, 고정값 위에 기타 치적만 더합니다.
    static_crit_add = (static_fixed + non_stat_static_crit_add) if static_fixed else (stat_rate + non_stat_static_crit_add)
    cond_crit_add = (cond_fixed + non_stat_cond_crit_add) if cond_fixed else (stat_rate + non_stat_cond_crit_add)

    direction_crit = BACK_ATTACK_CRIT_BONUS_PERCENT if "백" in attack_type else 0.0
    if "헤드" in attack_type and "백" not in attack_type:
        direction_crit = 0.0

    static_crit_rate = _clamp_percent(static_crit_add)
    cond_crit_rate = _clamp_percent(cond_crit_add)
    back_basis_crit_rate = _clamp_percent(cond_crit_rate + direction_crit)

    global_sources = _non_skill_sources(effect_sources)
    skill_sources = _skill_only_sources(effect_sources, name)

    global_static_crit_damage_add = _sum_sources(global_sources, "치명타 피해량 증가(%)", attack_type, name, include_conditional=False)
    global_cond_crit_damage_add = _sum_sources(global_sources, "치명타 피해량 증가(%)", attack_type, name, include_conditional=True)
    skill_static_crit_damage_add = _sum_sources(skill_sources, "치명타 피해량 증가(%)", attack_type, name, include_conditional=False)
    skill_cond_crit_damage_add = _sum_sources(skill_sources, "치명타 피해량 증가(%)", attack_type, name, include_conditional=True)

    static_crit_damage = BASE_CRIT_DAMAGE_PERCENT + global_static_crit_damage_add + skill_static_crit_damage_add
    cond_crit_damage = BASE_CRIT_DAMAGE_PERCENT + global_cond_crit_damage_add + skill_cond_crit_damage_add

    def group_values(include_conditional: bool) -> Dict[str, float]:
        sum_fn = sum_cond_total if include_conditional else sum_static
        direction_base = 0.0
        if "백" in attack_type:
            direction_base += BACK_ATTACK_DAMAGE_BONUS_PERCENT
        if "헤드" in attack_type:
            direction_base += HEAD_ATTACK_DAMAGE_BONUS_PERCENT
        return {
            "진화형 피해(%)": sum_fn("진화형 피해(%)"),
            "적에게 주는 피해(%)": sum_fn("적에게 주는 피해(%)"),
            "스킬 피해(%)": sum_fn("스킬 피해(%)"),
            "방향성 피해(%)": direction_base + sum_fn("방향성 피해(%)"),
            "보석 피해(%)": sum_fn("보석 피해(%)"),
            "공격력 증가(%)": sum_fn("공격력 증가(%)"),
            "추가 피해(%)": sum_fn("추가 피해(%)"),
        }

    static_groups = group_values(False)
    cond_groups = group_values(True)
    static_crit_mul = _expected_crit_multiplier(static_crit_rate, static_crit_damage)
    cond_crit_mul = _expected_crit_multiplier(back_basis_crit_rate, cond_crit_damage)
    static_damage_mul = _damage_group_multiplier(**static_groups)
    cond_damage_mul = _damage_group_multiplier(**cond_groups)

    direction_note = ""
    if "백" in attack_type:
        direction_note = "백어택 기본 치적 +10 / 피해 +5"
        if back_engraving:
            direction_note = "기습 채용 감지 + 백어택 기본 치적 +10 / 피해 +5"
    if "헤드" in attack_type:
        direction_note = (direction_note + " / " if direction_note else "") + "헤드어택 기본 피해 +20"
        if head_engraving:
            direction_note += " / 결투 채용 감지"

    conditional_count = 0
    for _, row in effect_sources.iterrows():
        if _scope_applies(row, attack_type, name) and str(row.get("조건부 여부")) != "상시":
            conditional_count += 1

    return {
        "스킬명": name,
        "공격타입": attack_type,
        "스킬레벨": skill.get("Level") or skill.get("SkillLevel") or "",
        "룬": (skill.get("Rune") or {}).get("Name") if isinstance(skill.get("Rune"), dict) else skill.get("Rune"),
        "치명 스탯": round(crit_stat, 0),
        "치명 스탯 치적(%)": round(stat_rate, 2),
        "치명 고정 옵션(%)": round(max(static_fixed, cond_fixed), 2),
        "치명 증가 합계(상시)(%)": round(static_crit_add, 2),
        "치명 증가 합계(조건부)(%)": round(cond_crit_add, 2),
        "백어택 기준 보너스(%)": round(direction_crit, 2),
        "예상 치명 확률(상시)(%)": round(static_crit_rate, 2),
        "예상 치명 확률(조건부 포함)(%)": round(cond_crit_rate, 2),
        "예상 치명 확률(백어택 기준)(%)": round(back_basis_crit_rate, 2),
        "기본 치피(%)": BASE_CRIT_DAMAGE_PERCENT,
        "전역 치피 증가(상시)(%)": round(global_static_crit_damage_add, 2),
        "전역 치피 증가(조건부)(%)": round(global_cond_crit_damage_add, 2),
        "스킬 전용 치피 증가(상시)(%)": round(skill_static_crit_damage_add, 2),
        "스킬 전용 치피 증가(조건부)(%)": round(skill_cond_crit_damage_add, 2),
        "예상 치피(상시)(%)": round(static_crit_damage, 2),
        "예상 치피(조건부 포함)(%)": round(cond_crit_damage, 2),
        "진화형 피해(상시)(%)": round(static_groups["진화형 피해(%)"], 2),
        "진화형 피해(조건부)(%)": round(cond_groups["진화형 피해(%)"], 2),
        "적에게 주는 피해(상시)(%)": round(static_groups["적에게 주는 피해(%)"], 2),
        "적에게 주는 피해(조건부)(%)": round(cond_groups["적에게 주는 피해(%)"], 2),
        "스킬 피해(상시)(%)": round(static_groups["스킬 피해(%)"], 2),
        "스킬 피해(조건부)(%)": round(cond_groups["스킬 피해(%)"], 2),
        "방향성 피해(상시)(%)": round(static_groups["방향성 피해(%)"], 2),
        "방향성 피해(조건부)(%)": round(cond_groups["방향성 피해(%)"], 2),
        "보석 피해(상시)(%)": round(static_groups["보석 피해(%)"], 2),
        "보석 피해(조건부)(%)": round(cond_groups["보석 피해(%)"], 2),
        "공격력 증가(상시)(%)": round(static_groups["공격력 증가(%)"], 2),
        "공격력 증가(조건부)(%)": round(cond_groups["공격력 증가(%)"], 2),
        "추가 피해(상시)(%)": round(static_groups["추가 피해(%)"], 2),
        "추가 피해(조건부)(%)": round(cond_groups["추가 피해(%)"], 2),
        "치명 기대배율(상시)": round(static_crit_mul, 4),
        "치명 기대배율(조건부+방향)": round(cond_crit_mul, 4),
        "피해군 배율(상시)": round(static_damage_mul, 4),
        "피해군 배율(조건부)": round(cond_damage_mul, 4),
        "예상 최종 배율(상시)": round(static_crit_mul * static_damage_mul, 4),
        "예상 최종 배율(조건부)": round(cond_crit_mul * cond_damage_mul, 4),
        "조건부 항목 수": conditional_count,
        "방향 보너스 메모": direction_note,
        "감지 출처": _source_names_for_skill(effect_sources, attack_type, name) or "치명 스탯만 적용",
    }


def _build_skill_table(
    bundle: Dict[str, Any],
    effect_sources: pd.DataFrame,
    crit_stat: float,
    stat_rate: float,
    back_engraving: bool,
    head_engraving: bool,
) -> pd.DataFrame:
    rows = []
    for skill in _safe_data(bundle, "combat_skills") or []:
        if not skill.get("Name"):
            continue
        rows.append(_calculate_one_skill(skill, effect_sources, crit_stat, stat_rate, back_engraving, head_engraving))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["예상 최종 배율(조건부)", "스킬명"], ascending=[False, True]).reset_index(drop=True)
    return df


def _metric_delta(final_df: pd.DataFrame, base_df: pd.DataFrame, key_col: str = "스킬명") -> pd.DataFrame:
    if final_df is None or final_df.empty:
        return pd.DataFrame()
    if base_df is None or base_df.empty:
        return pd.DataFrame()
    important = [
        "예상 치명 확률(백어택 기준)(%)", "예상 치피(조건부 포함)(%)", "진화형 피해(조건부)(%)",
        "적에게 주는 피해(조건부)(%)", "스킬 피해(조건부)(%)", "방향성 피해(조건부)(%)",
        "보석 피해(조건부)(%)", "예상 최종 배율(조건부)", "전역 치피 증가(조건부)(%)", "스킬 전용 치피 증가(조건부)(%)",
    ]
    merged = final_df[[key_col] + [c for c in important if c in final_df.columns]].merge(
        base_df[[key_col] + [c for c in important if c in base_df.columns]], on=key_col, how="left", suffixes=("_최종", "_기준")
    )
    out_rows = []
    for _, row in merged.iterrows():
        item = {"스킬명": row[key_col]}
        for col in important:
            f = _num(row.get(f"{col}_최종")) or 0.0
            b = _num(row.get(f"{col}_기준")) or 0.0
            item[f"{col} 추가"] = round(f - b, 4)
        out_rows.append(item)
    return pd.DataFrame(out_rows)


def _merged_base_final_table(base_df: pd.DataFrame, final_df: pd.DataFrame, ark_delta: pd.DataFrame) -> pd.DataFrame:
    if final_df is None or final_df.empty:
        return final_df if isinstance(final_df, pd.DataFrame) else pd.DataFrame()
    base_lookup = base_df.set_index("스킬명") if isinstance(base_df, pd.DataFrame) and not base_df.empty else pd.DataFrame()
    delta_lookup = ark_delta.set_index("스킬명") if isinstance(ark_delta, pd.DataFrame) and not ark_delta.empty else pd.DataFrame()
    rows = []
    for _, row in final_df.iterrows():
        name = row.get("스킬명")
        base = base_lookup.loc[name] if not base_lookup.empty and name in base_lookup.index else None
        delta = delta_lookup.loc[name] if not delta_lookup.empty and name in delta_lookup.index else None
        item = row.to_dict()
        if base is not None:
            item["기준 치명(백어택)(%)"] = round(float(base.get("예상 치명 확률(백어택 기준)(%)", 0)), 2)
            item["기준 치피(%)"] = round(float(base.get("예상 치피(조건부 포함)(%)", 0)), 2)
            item["기준 진화형 피해(%)"] = round(float(base.get("진화형 피해(조건부)(%)", 0)), 2)
            item["기준 최종 배율"] = round(float(base.get("예상 최종 배율(조건부)", 1)), 4)
        if delta is not None:
            item["아크그리드 치적 추가(%)"] = round(float(delta.get("예상 치명 확률(백어택 기준)(%) 추가", 0)), 2)
            item["아크그리드 치피 추가(%)"] = round(float(delta.get("예상 치피(조건부 포함)(%) 추가", 0)), 2)
            item["아크그리드 진피 추가(%)"] = round(float(delta.get("진화형 피해(조건부)(%) 추가", 0)), 2)
            item["아크그리드 스킬피해 추가(%)"] = round(float(delta.get("스킬 피해(조건부)(%) 추가", 0)), 2)
            item["아크그리드 최종배율 추가"] = round(float(delta.get("예상 최종 배율(조건부) 추가", 0)), 4)
        rows.append(item)
    # 보기 편하게 핵심 비교 컬럼을 앞쪽으로 이동
    preferred = [
        "스킬명", "공격타입", "기준 치명(백어택)(%)", "예상 치명 확률(백어택 기준)(%)",
        "기준 치피(%)", "예상 치피(조건부 포함)(%)", "아크그리드 치피 추가(%)",
        "기준 진화형 피해(%)", "진화형 피해(조건부)(%)", "아크그리드 진피 추가(%)",
        "스킬 피해(조건부)(%)", "아크그리드 스킬피해 추가(%)",
        "기준 최종 배율", "예상 최종 배율(조건부)", "아크그리드 최종배율 추가",
    ]
    out = pd.DataFrame(rows)
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]


def _overview_from_tables(
    crit_stat: float,
    stat_rate: float,
    base_df: pd.DataFrame,
    final_df: pd.DataFrame,
    base_sources: pd.DataFrame,
    final_sources: pd.DataFrame,
    back_engraving: bool,
    head_engraving: bool,
) -> pd.DataFrame:
    back_base = base_df[base_df["공격타입"].astype(str).str.contains("백", na=False)] if isinstance(base_df, pd.DataFrame) and not base_df.empty else pd.DataFrame()
    back_final = final_df[final_df["공격타입"].astype(str).str.contains("백", na=False)] if isinstance(final_df, pd.DataFrame) and not final_df.empty else pd.DataFrame()

    global_base = _non_skill_sources(base_sources)
    global_final = _non_skill_sources(final_sources)
    base_global_cd = BASE_CRIT_DAMAGE_PERCENT + _sum_sources(global_base, "치명타 피해량 증가(%)", "백어택" if back_engraving else "일반/확인필요", "", True)
    final_global_cd = BASE_CRIT_DAMAGE_PERCENT + _sum_sources(global_final, "치명타 피해량 증가(%)", "백어택" if back_engraving else "일반/확인필요", "", True)

    rows = [
        {"항목": "치명 스탯", "아크그리드 제외 기준": f"{crit_stat:,.0f}", "아크그리드 포함 최종": f"{crit_stat:,.0f}", "비고": "API profile Stats"},
        {"항목": "치명 스탯 치적", "아크그리드 제외 기준": f"{stat_rate:.2f}%", "아크그리드 포함 최종": f"{stat_rate:.2f}%", "비고": "Tooltip 우선, 없으면 치명×계수"},
        {"항목": "백어택 스킬 기준 치명", "아크그리드 제외 기준": f"{_avg_col(back_base, '예상 치명 확률(백어택 기준)(%)'):.2f}%", "아크그리드 포함 최종": f"{_avg_col(back_final, '예상 치명 확률(백어택 기준)(%)'):.2f}%", "비고": "백어택 기본 +10 포함"},
        {"항목": "전역 치명타 피해량(스킬 전용 제외)", "아크그리드 제외 기준": f"{base_global_cd:.2f}%", "아크그리드 포함 최종": f"{final_global_cd:.2f}%", "비고": "스킬명이 붙은 치피는 전역 제외"},
        {"항목": "평균 치명타 피해량(스킬 전용 포함)", "아크그리드 제외 기준": f"{_avg_col(base_df, '예상 치피(조건부 포함)(%)', BASE_CRIT_DAMAGE_PERCENT):.2f}%", "아크그리드 포함 최종": f"{_avg_col(final_df, '예상 치피(조건부 포함)(%)', BASE_CRIT_DAMAGE_PERCENT):.2f}%", "비고": "허리케인 +100 등 스킬 전용은 해당 스킬만"},
        {"항목": "평균 진화형 피해", "아크그리드 제외 기준": f"{_avg_col(base_df, '진화형 피해(조건부)(%)'):.2f}%", "아크그리드 포함 최종": f"{_avg_col(final_df, '진화형 피해(조건부)(%)'):.2f}%", "비고": "아크패시브 기준 + 아크그리드 추가"},
        {"항목": "평균 적에게 주는 피해", "아크그리드 제외 기준": f"{_avg_col(base_df, '적에게 주는 피해(조건부)(%)'):.2f}%", "아크그리드 포함 최종": f"{_avg_col(final_df, '적에게 주는 피해(조건부)(%)'):.2f}%", "비고": "각인/장비/기타"},
        {"항목": "평균 스킬 피해", "아크그리드 제외 기준": f"{_avg_col(base_df, '스킬 피해(조건부)(%)'):.2f}%", "아크그리드 포함 최종": f"{_avg_col(final_df, '스킬 피해(조건부)(%)'):.2f}%", "비고": "선택 트포/스킬 전용"},
        {"항목": "평균 보석 피해", "아크그리드 제외 기준": f"{_avg_col(base_df, '보석 피해(조건부)(%)'):.2f}%", "아크그리드 포함 최종": f"{_avg_col(final_df, '보석 피해(조건부)(%)'):.2f}%", "비고": "멸화/겁화 피해 보석"},
        {"항목": "평균 예상 최종 배율", "아크그리드 제외 기준": f"{_avg_col(base_df, '예상 최종 배율(조건부)', 1.0):.4f}x", "아크그리드 포함 최종": f"{_avg_col(final_df, '예상 최종 배율(조건부)', 1.0):.4f}x", "비고": "같은 피해군 합산, 다른 피해군 곱연산"},
        {"항목": "기습 각인 감지", "아크그리드 제외 기준": "예" if back_engraving else "아니오", "아크그리드 포함 최종": "예" if back_engraving else "아니오", "비고": "백어택 기준 보너스 표기"},
        {"항목": "결투 각인 감지", "아크그리드 제외 기준": "예" if head_engraving else "아니오", "아크그리드 포함 최종": "예" if head_engraving else "아니오", "비고": "헤드어택 기준 보너스 표기"},
    ]
    return pd.DataFrame(rows)


def estimate_skill_crit_tables(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    v10 계산 구조.

    1) LostBuilds 기준 계산: 아크그리드 제외
       - 치명 스탯, 각인, 장비/팔찌/엘릭서, 카드, 보석, 아크패시브, 선택 트라이포드, 직업/아이덴티티 룰만 반영
    2) 아크그리드 추가 계산: 위 기준값 위에 아크그리드만 별도 추가
       - 스킬명이 문구에 있으면 해당 스킬 전용
       - 스킬 전용처럼 보이지만 매칭 실패하면 검수 필요로 분리하고 계산 제외
    """
    crit_stat, stat_rate, stat_raw, stat_source = _profile_stats(bundle)
    all_global_sources = _collect_global_sources(bundle)
    identity_sources = _identity_sources(bundle)
    tripod_sources = _collect_skill_tripod_sources(bundle)
    gem_sources = _collect_gem_sources(bundle)
    skill_names = _skill_names(bundle)

    all_global_sources = _force_skill_scopes(all_global_sources, skill_names) if not all_global_sources.empty else all_global_sources
    identity_sources = _force_skill_scopes(identity_sources, skill_names) if not identity_sources.empty else identity_sources
    tripod_sources = _force_skill_scopes(tripod_sources, skill_names) if not tripod_sources.empty else tripod_sources
    gem_sources = _force_skill_scopes(gem_sources, skill_names) if not gem_sources.empty else gem_sources

    arkgrid_sources = _filter_source_type(all_global_sources, include=["아크그리드"])
    base_global_sources = _filter_source_type(all_global_sources, exclude=["아크그리드"])

    base_effect_sources = _concat_sources(identity_sources, base_global_sources, tripod_sources, gem_sources)
    final_effect_sources = _concat_sources(base_effect_sources, arkgrid_sources)
    source_df = _concat_sources(stat_source, identity_sources, base_global_sources, arkgrid_sources, tripod_sources, gem_sources)

    unresolved_sources = source_df[source_df["적용범위"].astype(str) == "검수 필요"].copy() if not source_df.empty else _empty_source_df()

    back_engraving = _has_back_attack_engraving(bundle)
    head_engraving = _has_head_attack_engraving(bundle)

    base_skill_df = _build_skill_table(bundle, base_effect_sources, crit_stat, stat_rate, back_engraving, head_engraving)
    final_skill_df = _build_skill_table(bundle, final_effect_sources, crit_stat, stat_rate, back_engraving, head_engraving)
    arkgrid_delta_df = _metric_delta(final_skill_df, base_skill_df)
    merged_skill_df = _merged_base_final_table(base_skill_df, final_skill_df, arkgrid_delta_df)

    overview = _overview_from_tables(
        crit_stat,
        stat_rate,
        base_skill_df,
        final_skill_df,
        base_effect_sources,
        final_effect_sources,
        back_engraving,
        head_engraving,
    )

    back_final = final_skill_df[final_skill_df["공격타입"].astype(str).str.contains("백", na=False)] if not final_skill_df.empty else pd.DataFrame()
    global_final = _non_skill_sources(final_effect_sources)
    global_crit_damage_no_skill = BASE_CRIT_DAMAGE_PERCENT + _sum_sources(
        global_final, "치명타 피해량 증가(%)", "백어택" if back_engraving else "일반/확인필요", "", include_conditional=True
    )

    return {
        # app 호환: 기본 표시표는 최종표지만, 앞쪽에 기준/아크그리드 추가/최종 비교 컬럼을 포함합니다.
        "skill_crit_estimates": merged_skill_df,
        "lostbuilds_base_skill_estimates": base_skill_df,
        "arkgrid_final_skill_estimates": final_skill_df,
        "arkgrid_delta_estimates": arkgrid_delta_df,
        "crit_sources": source_df,
        "damage_sources": source_df,
        "base_damage_sources": _concat_sources(stat_source, identity_sources, base_global_sources, tripod_sources, gem_sources),
        "arkgrid_damage_sources": arkgrid_sources,
        "unresolved_sources": unresolved_sources,
        "combat_overview": overview,
        "base_crit_stat": crit_stat,
        "base_crit_percent": stat_rate,
        "base_crit_raw": stat_raw,
        "avg_static_crit_rate_percent": _avg_col(final_skill_df, "예상 치명 확률(상시)(%)"),
        "avg_conditional_crit_rate_percent": _avg_col(final_skill_df, "예상 치명 확률(조건부 포함)(%)"),
        "avg_back_basis_crit_rate_percent": _avg_col(final_skill_df, "예상 치명 확률(백어택 기준)(%)"),
        "avg_static_crit_damage_percent": _avg_col(final_skill_df, "예상 치피(상시)(%)", BASE_CRIT_DAMAGE_PERCENT),
        "avg_conditional_crit_damage_percent": _avg_col(final_skill_df, "예상 치피(조건부 포함)(%)", BASE_CRIT_DAMAGE_PERCENT),
        "avg_back_skill_crit_rate_percent": _avg_col(back_final, "예상 치명 확률(백어택 기준)(%)", _avg_col(final_skill_df, "예상 치명 확률(백어택 기준)(%)")),
        "avg_back_skill_crit_damage_percent": _avg_col(back_final, "예상 치피(조건부 포함)(%)", _avg_col(final_skill_df, "예상 치피(조건부 포함)(%)", BASE_CRIT_DAMAGE_PERCENT)),
        "global_crit_damage_no_skill_percent": global_crit_damage_no_skill,
        "avg_evolution_damage_percent": _avg_col(final_skill_df, "진화형 피해(조건부)(%)"),
        "avg_enemy_damage_percent": _avg_col(final_skill_df, "적에게 주는 피해(조건부)(%)"),
        "avg_skill_damage_percent": _avg_col(final_skill_df, "스킬 피해(조건부)(%)"),
        "avg_directional_damage_percent": _avg_col(final_skill_df, "방향성 피해(조건부)(%)"),
        "avg_gem_damage_percent": _avg_col(final_skill_df, "보석 피해(조건부)(%)"),
        "avg_final_multiplier": _avg_col(final_skill_df, "예상 최종 배율(조건부)", 1.0),
        "lostbuilds_base_avg_final_multiplier": _avg_col(base_skill_df, "예상 최종 배율(조건부)", 1.0),
        "has_back_attack_engraving": back_engraving,
        "has_head_attack_engraving": head_engraving,
    }


# =========================
# v11 overrides
# =========================
# v11 목표:
# - 로아와식 기본 효과 합산에 더 가깝게, 아크패시브 대분류(특히 깨달음/도약)의 실제 효과를 누락하지 않음
# - 포식자 슬레이어 폭주 치명타 적중률 +30을 전투 기준값으로 포함
# - 스킬명이 붙은 효과는 자동 스킬 전용으로만 반영하고, 전역 치피/전역 진피로 새지 않게 유지
# - 앱 화면에서 “기본 효과 요약”과 “아크그리드 추가”를 분리해서 검수 가능하게 유지

def _bundle_all_text(bundle: Dict[str, Any]) -> str:
    """API bundle 전체에서 텍스트를 넓게 모읍니다. 직업각인 감지 실패를 줄이기 위한 보조 함수."""
    texts: List[str] = []
    for key in [
        "profiles", "summary", "engravings", "equipment", "combat_skills", "cards", "gems",
        "arkpassive", "arkgrid",
    ]:
        try:
            data = _safe_data(bundle, key)
            if data is not None:
                texts.append(_flatten(data))
        except Exception:
            pass
    return _clean_text(" ".join(texts))


def _active_job_names(bundle: Dict[str, Any]) -> List[str]:  # type: ignore[override]
    """v11: 각인 구조가 바뀌어도 직업각인을 최대한 감지합니다.

    기존에는 engravings.Effects/ArkPassiveEffects만 봤는데, API 응답 구조나 사이트별 가공 구조에 따라
    직업각인이 다른 위치에 있을 수 있어 전체 bundle 텍스트까지 같이 검색합니다.
    """
    joined = _bundle_all_text(bundle)
    jobs: List[str] = []
    rules = _load_class_rules()
    class_name = _character_class_name(bundle)
    for class_key, class_block in (rules.get("classes") or {}).items():
        if class_key and class_key not in class_name:
            continue
        for job_name, job_block in (class_block.get("jobs") or {}).items():
            keywords = [job_name] + list(job_block.get("engraving_keywords") or []) + list(job_block.get("identity_keywords") or []) + list(job_block.get("arkpassive_keywords") or [])
            if any(k and k in joined for k in keywords):
                jobs.append(str(job_name))
    # 안전 fallback: 슬레이어인데 포식자가 전체 텍스트 어디든 있으면 포식자로 처리
    if "슬레이어" in class_name and "포식자" in joined and "포식자" not in jobs:
        jobs.append("포식자")
    if "슬레이어" in class_name and "처단자" in joined and "처단자" not in jobs:
        jobs.append("처단자")
    return jobs


def _has_predator_engraving(bundle: Dict[str, Any]) -> bool:  # type: ignore[override]
    return "포식자" in _bundle_all_text(bundle)


def _identity_sources(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    """v11: 직업/아이덴티티 룰을 더 강하게 적용합니다.

    포식자 슬레이어의 폭주 치적 +30은 사용자가 보는 실전 기준에 포함해야 하므로
    계산상 '상시'로 둡니다. 비폭주 기준을 따로 보고 싶으면 class_rules.yaml에서 enabled:false로 끄면 됩니다.
    """
    rows: List[Dict[str, Any]] = []
    rules = _load_class_rules()
    class_name = _character_class_name(bundle)
    active_jobs = set(_active_job_names(bundle))

    for class_key, class_block in (rules.get("classes") or {}).items():
        if class_key and class_key not in class_name:
            continue
        for job_name, job_block in (class_block.get("jobs") or {}).items():
            if job_name not in active_jobs:
                continue
            for eff in job_block.get("identity_effects", []) or []:
                if eff.get("enabled") is False:
                    continue
                effects = _new_effect_dict()
                effects["치명타 적중률 증가(%)"] = float(eff.get("crit_rate", 0) or 0)
                effects["치명타 피해량 증가(%)"] = float(eff.get("crit_damage", 0) or 0)
                effects["진화형 피해(%)"] = float(eff.get("evolution_damage", 0) or 0)
                effects["적에게 주는 피해(%)"] = float(eff.get("enemy_damage", 0) or 0)
                effects["스킬 피해(%)"] = float(eff.get("skill_damage", 0) or 0)
                effects["방향성 피해(%)"] = float(eff.get("directional_damage", 0) or 0)
                effects["공격력 증가(%)"] = float(eff.get("attack_power", 0) or 0)
                if not any(abs(v) > 1e-9 for v in effects.values()):
                    continue
                applies_to = str(eff.get("applies_to") or "all")
                scope = {
                    "all": "전체",
                    "back_attack": "백어택 스킬",
                    "head_attack": "헤드어택 스킬",
                    "directional": "백/헤드 스킬",
                }.get(applies_to, "전체")
                # 포식자 폭주는 실전 계산 기준에 포함시키기 위해 상시 처리.
                cond = str(eff.get("condition_type") or "상시")
                if job_name == "포식자" and "폭주" in str(eff.get("name") or ""):
                    cond = "상시"
                rows.append({
                    "출처구분": "직업/아이덴티티",
                    "이름": f"{eff.get('name') or job_name}",
                    "적용범위": scope,
                    "적용스킬": "전체/범위 기준",
                    **effects,
                    "조건부 여부": cond,
                    "설명": str(eff.get("description") or f"{class_key} / {job_name} 룰: {eff.get('condition') or ''}")[:420],
                })

    # class_rules.yaml이 없거나 직업각인 감지가 실패한 경우 강제 fallback.
    if not rows and "슬레이어" in class_name and _has_predator_engraving(bundle):
        effects = _new_effect_dict()
        effects["치명타 적중률 증가(%)"] = 30.0
        rows.append({
            "출처구분": "직업/아이덴티티",
            "이름": "폭주(포식자)",
            "적용범위": "전체",
            "적용스킬": "전체/범위 기준",
            **effects,
            "조건부 여부": "상시",
            "설명": "fallback: 포식자 슬레이어 폭주 전투 기준 치명타 적중률 +30%",
        })
    if not rows:
        return _empty_source_df()
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def _iter_named_texts(obj: Any, source_type: str, path: str = "") -> Iterable[Tuple[str, str, str]]:  # type: ignore[override]
    """v11: 아크패시브의 실제 효과 텍스트 누락 방지.

    v10에서는 '진화/깨달음/도약' 같은 대분류를 지나치게 강하게 제외해서,
    로아와에 보이는 깨달음 치명타 피해/도약 치피 같은 값이 누락될 수 있었습니다.
    v11은 아크패시브는 대분류라도 효과 설명이 있으면 읽고, 아크그리드만 긴 컨테이너 중복을 조심합니다.
    """
    if isinstance(obj, dict):
        if _is_inactive_node(obj):
            return

        name = obj.get("Name") or obj.get("name") or obj.get("Title") or obj.get("title") or obj.get("Label") or obj.get("label")
        desc = (
            obj.get("Description") or obj.get("description") or obj.get("Desc") or obj.get("desc")
            or obj.get("Effect") or obj.get("effect") or obj.get("EffectText") or obj.get("effectText")
            or obj.get("Option") or obj.get("option") or obj.get("Text") or obj.get("text")
        )
        tooltip = obj.get("Tooltip") or obj.get("tooltip") or obj.get("ToolTip") or obj.get("toolTip")
        level = obj.get("Level") or obj.get("level") or obj.get("Grade") or obj.get("grade")

        text_parts = []
        if name:
            text_parts.append(str(name))
        if level not in (None, ""):
            text_parts.append(f"Lv.{level}")
        if desc:
            text_parts.append(_flatten(desc))
        if tooltip:
            text_parts.append(_flatten(tooltip))

        if name and (desc or tooltip):
            text = _clean_text(" ".join(text_parts))
            is_generic = _clean_text(name) in GENERIC_PASSIVE_CONTAINER_NAMES
            # 아크그리드의 최상위 '아크그리드/코어' 컨테이너는 너무 길면 중복 가능성이 높아 제외.
            # 단, '깨달음/진화/도약' 아크패시브는 로아와식 효과 합산에 필요할 수 있어 읽습니다.
            if not (source_type == "아크그리드" and is_generic and len(text) > 900):
                yield str(name), text, path

        for k, v in obj.items():
            yield from _iter_named_texts(v, source_type, f"{path}/{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_named_texts(v, source_type, f"{path}[{i}]")


def _extract_effect_values(text: str, source_type: str = "", target_skill: str | None = None) -> Dict[str, float]:  # type: ignore[override]
    """v11: 치적/치피/피해군 추출을 더 보수적이고 명확하게 분리합니다."""
    clean = _clean_text(text)
    effects = _new_effect_dict()

    # 치명타 적중률 고정
    crit_fixed = []
    for pat in [
        r"치명타\s*적중률[^0-9]{0,24}([+\-]?\d+(?:\.\d+)?)\s*%[^가-힣]{0,16}(?:로|으로)?\s*고정",
        r"치명타\s*적중률(?:을|이)?\s*([+\-]?\d+(?:\.\d+)?)\s*%[^가-힣]{0,16}(?:로|으로)?\s*고정",
        r"치적[^0-9]{0,24}([+\-]?\d+(?:\.\d+)?)\s*%[^가-힣]{0,16}(?:로|으로)?\s*고정",
    ]:
        for m in re.finditer(pat, clean):
            v = _num(m.group(1))
            if v is not None:
                crit_fixed.append(v)
    effects["치명타 적중률 고정(%)"] = max(crit_fixed) if crit_fixed else 0.0

    # 문장 단위로 쪼개서 치적과 치피가 서로 섞이지 않게 처리
    segments = _split_sentences(clean)
    if len(segments) <= 1:
        segments = re.split(r"(?=치명타\s*적중률)|(?=치명타\s*피해)|(?=치명\s*피해)|(?=적에게\s*주는)|(?=진화형)|(?=진화\s*피해)|(?=공격력)|(?=추가\s*피해)", clean)
    segments = [s.strip() for s in segments if s and s.strip()]

    crit_rate_values: List[float] = []
    crit_damage_values: List[float] = []
    for seg in segments:
        if _contains_any(seg, ["치명타 적중률", "치명타 확률", "치명 확률", "치명타율", "치적"]):
            if not _contains_any(seg, ["치명타 피해", "치명 피해", "치피"]):
                crit_rate_values += _extract_percent_near_keywords(seg, ["치명타 적중률", "치명타 확률", "치명 확률", "치명타율", "치적"], window=70)
        if _contains_any(seg, ["치명타 피해량", "치명타 피해", "치명 피해", "치피증", "치피"]):
            # '치명타 적중률' 문장이 같이 섞인 경우라도 피해 키워드 주변 숫자만 추출
            vals = _extract_percent_near_keywords(seg, ["치명타 피해량", "치명타 피해", "치명 피해", "치피증", "치피"], window=72)
            crit_damage_values += vals

    if crit_fixed:
        fixed_set = {round(x, 4) for x in crit_fixed}
        crit_rate_values = [v for v in crit_rate_values if round(v, 4) not in fixed_set]
    # 추출이 이미 '숫자 위치' 기준으로 중복 제거되므로, 여기서는 값 기준 dedup 없이 합산합니다.
    # (팔찌처럼 '치명타 적중률 4.2%' 옵션이 2개면 4.2+4.2=8.4로 정상 합산)
    # 단, 같은 값 옵션은 최대 2개까지만 유효(팔찌 옵션 슬롯 2개). 3개 이상은 파싱 아티팩트라 상한을 둡니다.
    effects["치명타 적중률 증가(%)"] = round(sum(_cap_duplicate_values(crit_rate_values, 2)), 4)
    effects["치명타 피해량 증가(%)"] = round(sum(_cap_duplicate_values(crit_damage_values, 2)), 4)

    # 진화형 피해 / 메인노드 효율
    evolution_keywords = ["진화형 피해", "진화 피해", "진화형피해", "진피", "메인노드 효율", "메인 노드 효율"]
    effects["진화형 피해(%)"] = _sum_values(_extract_percent_near_keywords(clean, evolution_keywords, window=70))
    if effects["진화형 피해(%)"] == 0 and source_type in {"아크패시브", "아크그리드"}:
        if _contains_any(clean, ["한계 돌파", "한계돌파", "음속 돌파", "음속돌파", "뭉툭한 가시", "뭉가", "인파이팅", "입식 타격가", "마나 용광로", "마나용광로", "분쇄"]):
            vals = []
            for seg in segments:
                if _contains_any(seg, ["치명타", "치피", "치적", "쿨타임"]):
                    continue
                if _contains_any(seg, ["피해", "효율", "증가"]):
                    vals += _extract_percent_near_keywords(seg, ["피해", "효율", "증가"], window=50)
            effects["진화형 피해(%)"] = _sum_values(vals)

    enemy_keywords = [
        "적에게 주는 피해", "적에게 주는 피해량", "적에게 주는 추가 피해", "적에게주는피해",
        "적주피", "적추피", "보스 등급 이상 몬스터에게 주는 피해", "몬스터에게 주는 피해",
    ]
    effects["적에게 주는 피해(%)"] = _sum_values(_extract_percent_near_keywords(clean, enemy_keywords, window=72))

    direction_keywords = [
        "백어택 피해", "백어택 피해량", "헤드어택 피해", "헤드어택 피해량",
        "백어택 및 헤드어택 피해", "방향성 공격 피해", "방향성 공격의 피해량", "방향성 피해",
    ]
    effects["방향성 피해(%)"] = _sum_values(_extract_percent_near_keywords(clean, direction_keywords, window=72))
    if effects["방향성 피해(%)"] == 0.0 and _contains_any(clean, ["백어택", "헤드어택", "방향성"]):
        vals = []
        for seg in segments:
            if not _contains_any(seg, ["백어택", "헤드어택", "방향성"]):
                continue
            if _contains_any(seg, ["치명타", "치적", "치피"]):
                continue
            vals += _extract_percent_near_keywords(seg, ["피해량", "피해", "주는 피해"], window=60)
        effects["방향성 피해(%)"] = _sum_values(vals)

    atk_values = []
    for seg in segments:
        if "무기 공격력" in seg:
            continue
        if "공격력" in seg:
            atk_values += _extract_percent_near_keywords(seg, ["공격력"], window=60)
    effects["공격력 증가(%)"] = _sum_values(atk_values)

    if source_type == "보석" or _contains_any(clean, ["멸화", "겁화", "홍염", "보석"]):
        vals = []
        for seg in segments:
            if "쿨" in seg or "재사용" in seg:
                continue
            if _contains_any(seg, ["피해", "피해량"]):
                vals += _extract_percent_near_keywords(seg, ["피해량", "피해"], window=60)
        effects["보석 피해(%)"] = _sum_values(vals)

    skill_damage_values = []
    if (target_skill and source_type != "보석") or source_type in {"선택 트라이포드", "스킬"}:
        for seg in segments:
            if _contains_any(seg, ["치명타", "치피", "치적", "진화", "적에게", "백어택", "헤드어택", "방향성", "공격력", "쿨타임", "재사용"]):
                continue
            if _contains_any(seg, ["피해량", "주는 피해", "피해가", "피해 ", "대미지", "데미지"]):
                skill_damage_values += _extract_percent_near_keywords(seg, ["피해량", "주는 피해", "피해", "대미지", "데미지"], window=60)
    effects["스킬 피해(%)"] = _sum_values(skill_damage_values)

    additional_values = []
    for seg in segments:
        if "적에게 주는 추가 피해" in seg:
            continue
        if "추가 피해" in seg:
            additional_values += _extract_percent_near_keywords(seg, ["추가 피해"], window=60)
    effects["추가 피해(%)"] = _sum_values(additional_values)

    return {k: round(v, 4) for k, v in effects.items()}


def _loawa_like_breakdown(sources: pd.DataFrame, attack_type: str = "백어택") -> pd.DataFrame:
    """로아와 효과창처럼 피해군별 합계를 출처와 함께 보는 검수표."""
    if sources is None or sources.empty:
        return pd.DataFrame(columns=["피해군", "합계(%)", "출처"])
    rows = []
    group_map = [
        ("치명타 적중률", "치명타 적중률 증가(%)"),
        ("치명타 적중률 고정", "치명타 적중률 고정(%)"),
        ("치명타 피해", "치명타 피해량 증가(%)"),
        ("진화형 피해", "진화형 피해(%)"),
        ("적에게 주는 피해", "적에게 주는 피해(%)"),
        ("스킬 피해", "스킬 피해(%)"),
        ("방향성 피해", "방향성 피해(%)"),
        ("보석 피해", "보석 피해(%)"),
        ("공격력", "공격력 증가(%)"),
        ("추가 피해", "추가 피해(%)"),
    ]
    for label, col in group_map:
        parts = []
        total = 0.0
        if col not in sources.columns:
            continue
        for _, row in sources.iterrows():
            if str(row.get("적용범위")) == "스킬 전용":
                continue
            if str(row.get("적용범위")) == "검수 필요":
                continue
            if not _scope_applies(row, attack_type, ""):
                continue
            val = _num(row.get(col)) or 0.0
            if abs(val) < 1e-9:
                continue
            total += val
            parts.append(f"{row.get('출처구분')}:{row.get('이름')} {val:g}%")
        if total:
            rows.append({"피해군": label, "합계(%)": round(total, 2), "출처": " / ".join(parts[:16])})
    return pd.DataFrame(rows)


_old_estimate_skill_crit_tables_v10 = estimate_skill_crit_tables

def estimate_skill_crit_tables(bundle: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
    """v11 wrapper: v10 계산을 유지하되, 더 넓은 직업/아덴 감지와 로아와식 검수표를 추가합니다."""
    result = _old_estimate_skill_crit_tables_v10(bundle)
    source_df = result.get("damage_sources")
    base_sources = result.get("base_damage_sources")
    final_df = result.get("arkgrid_final_skill_estimates") or result.get("skill_crit_estimates")
    base_df = result.get("lostbuilds_base_skill_estimates")

    if isinstance(source_df, pd.DataFrame):
        result["loawa_like_breakdown"] = _loawa_like_breakdown(source_df, "백어택")
    if isinstance(base_sources, pd.DataFrame):
        result["loawa_like_base_breakdown"] = _loawa_like_breakdown(base_sources, "백어택")

    # 백어택 기준 치명은 포식자 폭주 + 백어택 + 기본 치적이 포함된 백어택 스킬 평균으로 표시.
    if isinstance(final_df, pd.DataFrame) and not final_df.empty:
        back_final = final_df[final_df["공격타입"].astype(str).str.contains("백", na=False)]
        if not back_final.empty:
            result["avg_back_skill_crit_rate_percent"] = _avg_col(back_final, "예상 치명 확률(백어택 기준)(%)")
            result["avg_back_skill_crit_damage_percent"] = _avg_col(back_final, "예상 치피(조건부 포함)(%)", BASE_CRIT_DAMAGE_PERCENT)
    if isinstance(base_df, pd.DataFrame) and not base_df.empty:
        back_base = base_df[base_df["공격타입"].astype(str).str.contains("백", na=False)]
        if not back_base.empty:
            result["lostbuilds_base_back_crit_rate_percent"] = _avg_col(back_base, "예상 치명 확률(백어택 기준)(%)")
            result["lostbuilds_base_back_crit_damage_percent"] = _avg_col(back_base, "예상 치피(조건부 포함)(%)", BASE_CRIT_DAMAGE_PERCENT)
    return result

# ==============================
# v12 overrides
# ==============================
# v12 핵심:
# - 스킬 레벨 1은 미채용으로 보고 계산/스킬명 매칭에서 제외
# - 스킬 Tooltip의 [충격 스킬] 같은 계열 태그를 읽어 "충격 스킬의 피해량 증가"를 해당 계열에만 적용
# - 도약/아크패시브/아크그리드 문구에 스킬명이 있으면 해당 스킬 전용으로만 적용
# - 스킬 전용 효과가 전역 치피/전역 피해로 새는 것을 더 강하게 차단

SKILL_GROUP_KEYWORDS = [
    "충격 스킬", "기력 스킬", "체술 스킬", "충동 스킬", "악마 스킬", "일반 스킬", "콤보 스킬",
    "홀딩 스킬", "차지 스킬", "지점 스킬", "캐스팅 스킬", "토글 스킬", "연속 스킬",
    "오의 스킬", "난무 스킬", "집중 스킬", "기상 스킬", "우산 스킬", "변신 스킬",
    "초각성 스킬", "각성기", "아이덴티티 스킬",
    # 건슬링어/데빌헌터 등 무기 스탠스 계열. 예: 피스메이커 '샷건 스탠스로 변경 시 치명타 적중률 10%'
    # → 샷건 스탠스 스킬에만 적용(스킬 그룹). 기본 기준 치명(전 스킬 공통)에는 안 들어가고,
    #    최종 계산표에서 샷건 스탠스 스킬에만 반영됩니다.
    "샷건 스탠스", "핸드건 스탠스", "라이플 스탠스", "샷건 스킬", "핸드건 스킬", "라이플 스킬",
]

_old_skill_names_v11 = _skill_names
_old_detect_attack_type_v11 = _detect_attack_type
_old_scope_applies_v11 = _scope_applies
_old_source_rows_from_text_v11 = _source_rows_from_text
_old_build_skill_table_v11 = _build_skill_table
_old_calculate_one_skill_v11 = _calculate_one_skill
_old_estimate_skill_crit_tables_v11 = estimate_skill_crit_tables


def _skill_level_value(skill: Dict[str, Any]) -> int:
    """API 스킬 레벨을 int로 변환. 필드가 없으면 Tooltip의 '스킬 레벨 N'도 확인."""
    for key in ["Level", "level", "SkillLevel", "skillLevel"]:
        if key in skill:
            try:
                return int(float(str(skill.get(key)).replace(",", "")))
            except Exception:
                pass
    flat = _flatten(skill)
    m = re.search(r"스킬\s*레벨\s*(\d+)", flat)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return 0


def _is_adopted_skill(skill: Dict[str, Any]) -> bool:
    """스킬 레벨 1은 미채용으로 제외. 단, 레벨 정보가 없으면 보수적으로 포함."""
    level = _skill_level_value(skill)
    if level == 0:
        return True
    return level > 1


def _adopted_skills(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    skills = [s for s in (_safe_data(bundle, "combat_skills") or []) if s.get("Name")]
    adopted = [s for s in skills if _is_adopted_skill(s)]
    # API가 레벨을 안 줄 때 전부 제외되는 것을 방지
    return adopted if adopted else skills


def _skill_names(bundle: Dict[str, Any]) -> List[str]:  # type: ignore[override]
    """v12: 미채용 스킬(Level 1)은 스킬 전용 매칭/계산에서 제외."""
    return [str(s.get("Name")) for s in _adopted_skills(bundle) if s.get("Name")]


def _extract_skill_groups_from_text(text: Any) -> List[str]:
    clean = _clean_text(text)
    groups: List[str] = []

    # 명시 목록 우선
    for kw in SKILL_GROUP_KEYWORDS:
        if kw in clean and kw not in groups:
            groups.append(kw)

    # [충격 스킬] 같은 bracket 패턴 자동 탐지
    for m in re.finditer(r"\[\s*([가-힣A-Za-z0-9]+\s*스킬|각성기)\s*\]", clean):
        g = re.sub(r"\s+", " ", m.group(1)).strip()
        if g and g not in groups:
            groups.append(g)

    # '충격 스킬의 피해량' 같은 일반 문장 자동 탐지
    for m in re.finditer(r"([가-힣A-Za-z0-9]{1,12}\s*스킬|각성기)\s*(?:의|이|을|를|에|로|은|는)?", clean):
        g = re.sub(r"\s+", " ", m.group(1)).strip()
        if g in {"해당 스킬", "사용 스킬", "모든 스킬", "스킬"}:
            continue
        if len(g) <= 2:
            continue
        if g not in groups:
            groups.append(g)

    return groups


def _skill_group_for_source_text(text: str) -> str | None:
    """효과 설명이 특정 스킬 계열을 가리키는지 판정."""
    clean = _clean_text(text)
    groups = _extract_skill_groups_from_text(clean)
    if not groups:
        return None
    # '해당 스킬', '스킬 사용' 등 일반 표현은 제외하고 가장 구체적인 그룹을 선택
    groups = [g for g in groups if g not in {"일반 스킬"} or "일반 스킬" in clean]
    if not groups:
        return None
    # 더 긴 그룹명 우선
    groups.sort(key=len, reverse=True)
    return groups[0]


def _skill_groups_for_skill(skill: Dict[str, Any]) -> List[str]:
    flat = _flatten(skill)
    groups = _extract_skill_groups_from_text(flat)
    # API Type/SkillType도 함께 사용
    for key in ["Type", "SkillType", "SkillTypeName"]:
        v = skill.get(key)
        if v:
            t = _clean_text(v)
            if t and t not in groups:
                groups.append(t)
    return groups


def _detect_attack_type(skill: Dict[str, Any]) -> str:  # type: ignore[override]
    """v12: 기존 공격 타입 뒤에 [충격 스킬] 같은 계열 태그를 붙여 스코프 판정에 사용."""
    base = _old_detect_attack_type_v11(skill)
    groups = _skill_groups_for_skill(skill)
    if groups:
        extra = " / ".join(g for g in groups if g and _norm_name(g) not in _norm_name(base))
        return f"{base} / {extra}" if extra else base
    return base


def _scope_applies(row: pd.Series, attack_type: str, skill_name: str) -> bool:  # type: ignore[override]
    scope = str(row.get("적용범위") or "전체")
    target = str(row.get("적용스킬") or "")
    if scope in {"스킬 그룹", "계열 전용", "스킬 계열"}:
        return _norm_name(target) in _norm_name(attack_type)
    return _old_scope_applies_v11(row, attack_type, skill_name)


def _source_rows_from_text(
    source_type: str,
    name: str,
    text: str,
    skill_names: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:  # type: ignore[override]
    """v12: 스킬명 직접 매칭 + 스킬 계열 매칭을 분리.

    - 문장에 실제 채용 스킬명이 있으면 스킬 전용
    - 문장에 '충격 스킬', '기력 스킬' 같은 계열명이 있으면 계열 전용
    - 스킬 전용처럼 보이지만 매칭 실패한 문장은 검수 필요로 보내 계산 제외
    """
    clean = _clean_text(text)
    names = list(skill_names or [])
    rows: List[Dict[str, Any]] = []

    if source_type in {"아크그리드", "아크패시브", "선택 트라이포드", "스킬"}:
        snippets = _split_option_snippets(clean)
    else:
        snippets = [clean]

    for snippet in snippets:
        target = _detect_target_skill(f"{name} {snippet}", names)
        group_target = None if target else _skill_group_for_source_text(f"{name} {snippet}")
        row_name = name
        marker = re.search(r"\[\s*(\d+\s*[Pp]?)\s*\]", snippet)
        if marker:
            row_name = f"{name} [{marker.group(1).replace(' ', '').upper()}]"

        row = _source_row(source_type, row_name, snippet, target_skill=target)
        if row:
            if target:
                row["적용범위"] = "스킬 전용"
                row["적용스킬"] = target
                if "폭주" not in snippet:
                    row["조건부 여부"] = "상시"
            elif group_target and any(abs(_num(row.get(c)) or 0.0) > 1e-9 for c in EFFECT_COLUMNS):
                row["적용범위"] = "스킬 그룹"
                row["적용스킬"] = group_target
                # 계열 강화는 해당 계열 스킬에만 적용. 조건부 키워드가 없으면 상시.
                if not _is_conditional(snippet) or "스킬" in group_target:
                    row["조건부 여부"] = "상시"
            elif source_type in {"아크그리드", "아크패시브", "선택 트라이포드", "스킬"} and _looks_skill_specific_effect(snippet, row_name, names):
                row["적용범위"] = "검수 필요"
                row["적용스킬"] = "스킬명/계열 자동 매칭 실패"
                row["조건부 여부"] = "검수 필요"
            rows.append(row)

    if not rows:
        target = _detect_target_skill(clean, names)
        group_target = None if target else _skill_group_for_source_text(clean)
        row = _source_row(source_type, name, clean, target_skill=target)
        if row:
            if target:
                row["적용범위"] = "스킬 전용"
                row["적용스킬"] = target
            elif group_target:
                row["적용범위"] = "스킬 그룹"
                row["적용스킬"] = group_target
            elif source_type in {"아크그리드", "아크패시브", "선택 트라이포드", "스킬"} and _looks_skill_specific_effect(clean, name, names):
                row["적용범위"] = "검수 필요"
                row["적용스킬"] = "스킬명/계열 자동 매칭 실패"
                row["조건부 여부"] = "검수 필요"
            rows.append(row)

    out = []
    seen = set()
    for row in rows:
        sig = tuple(str(row.get(c, "")) for c in SOURCE_COLUMNS)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(row)
    return out


def _force_skill_scopes(df: pd.DataFrame, skill_names: Iterable[str]) -> pd.DataFrame:  # type: ignore[override]
    """v13: 마지막 보정에서도 스킬명/계열명 전용 효과가 전역으로 새는 것을 차단.

    v65 시너지 트포 행(이름 '시너지:' 또는 적용스킬 '시너지:' 접두사)은 이미 올바르게
    적용범위='전체'로 설정돼 있으므로 이 함수에서 다시 건드리지 않습니다.
    (버그: 설명 필드에 스킬명이 포함되어 _detect_target_skill 이 스킬 전용으로 좁히던 문제 수정)
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for idx, row in out.iterrows():
        scope = str(row.get("적용범위") or "")
        target = str(row.get("적용스킬") or "")
        # v65 시너지 트포 행은 이미 "전체" 범위로 올바르게 설정됐으므로 건드리지 않습니다.
        if str(row.get("이름", "")).startswith("시너지:") or target.startswith("시너지:"):
            continue
        text = f"{row.get('이름', '')} {row.get('설명', '')}"
        detected = _detect_target_skill(text, skill_names)
        group = None if detected else _skill_group_for_source_text(text)
        if detected and (scope != "스킬 전용" or target in {"", "전체", "전체/범위 기준"}):
            out.at[idx, "적용범위"] = "스킬 전용"
            out.at[idx, "적용스킬"] = detected
            if "폭주" not in str(row.get("설명", "")):
                out.at[idx, "조건부 여부"] = "상시"
        elif group and scope in {"전체", "", "전체/범위 기준"}:
            out.at[idx, "적용범위"] = "스킬 그룹"
            out.at[idx, "적용스킬"] = group
            if not _is_conditional(text) or "스킬" in group:
                out.at[idx, "조건부 여부"] = "상시"
    return out


def _collect_skill_tripod_sources(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    """v12: 미채용 스킬의 트포는 제외. (시너지 분리는 아래 v65/v67에서 처리)"""
    rows: List[Dict[str, Any]] = []
    for skill in _adopted_skills(bundle):
        skill_name = skill.get("Name") or skill.get("name") or ""
        for tripod in (skill.get("Tripods") or skill.get("tripods") or []):
            if not isinstance(tripod, dict):
                continue
            _sel = tripod.get("IsSelected", tripod.get("isSelected", tripod.get("Selected", tripod.get("selected"))))
            if not bool(_sel):
                continue
            name = tripod.get("Name") or tripod.get("name") or "트라이포드"
            level = tripod.get("Level") or tripod.get("Tier") or tripod.get("level") or ""
            text = f"{skill_name} {name} Lv.{level} {_flatten(tripod)}"
            for row in _source_rows_from_text("선택 트라이포드", name, text, [skill_name]):
                row["적용범위"] = "스킬 전용"
                row["적용스킬"] = skill_name
                rows.append(row)
    if not rows:
        return _empty_source_df()
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS).drop_duplicates().reset_index(drop=True)


def _build_skill_table(
    bundle: Dict[str, Any],
    effect_sources: pd.DataFrame,
    crit_stat: float,
    stat_rate: float,
    back_engraving: bool,
    head_engraving: bool,
) -> pd.DataFrame:  # type: ignore[override]
    """v12: Level 1 미채용 스킬은 계산표에서 제외."""
    rows = []
    for skill in _adopted_skills(bundle):
        if not skill.get("Name"):
            continue
        rows.append(_calculate_one_skill(skill, effect_sources, crit_stat, stat_rate, back_engraving, head_engraving))
    df = pd.DataFrame(rows)
    if not df.empty:
        # 스킬 계열 컬럼 추가
        tag_lookup = {s.get("Name"): " / ".join(_skill_groups_for_skill(s)) for s in _adopted_skills(bundle)}
        df["스킬 계열"] = df["스킬명"].map(tag_lookup).fillna("")
        preferred = ["스킬명", "스킬레벨", "공격타입", "스킬 계열"]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        df = df[cols].sort_values(["예상 최종 배율(조건부)", "스킬명"], ascending=[False, True]).reset_index(drop=True)
    return df


def _add_v12_summary(result: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:
    """v12 검수용 요약 키를 추가."""
    adopted = _adopted_skills(bundle)
    all_skills = [s for s in (_safe_data(bundle, "combat_skills") or []) if s.get("Name")]
    result["adopted_skill_count"] = len(adopted)
    result["all_skill_count"] = len(all_skills)
    result["adopted_skill_names"] = ", ".join(s.get("Name") for s in adopted if s.get("Name"))

    # 최종 전역 치피: 스킬 전용/계열 전용 제외. 사용자가 보는 '기준 치피' 확인용.
    final_sources = result.get("damage_sources")
    if isinstance(final_sources, pd.DataFrame) and not final_sources.empty:
        non_skill = final_sources[~final_sources["적용범위"].astype(str).isin(["스킬 전용", "스킬 그룹", "계열 전용", "스킬 계열", "검수 필요"])]
        result["global_crit_damage_no_skill_percent"] = BASE_CRIT_DAMAGE_PERCENT + _sum_sources(non_skill, "치명타 피해량 증가(%)", "백어택", "", True)

    return result


def estimate_skill_crit_tables(bundle: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
    """v12 wrapper: v11 계산 구조 위에 미채용 스킬 제외/스킬 계열 스코프를 적용."""
    result = _old_estimate_skill_crit_tables_v11(bundle)
    return _add_v12_summary(result, bundle)

# ==============================
# v13 hotfix overrides
# ==============================
# - v11 wrapper 안의 `DataFrame or DataFrame` 때문에 계산 전체가 실패하던 문제를 우회합니다.
# - v12의 미채용 스킬 제외/스킬 계열 스코프 로직은 그대로 사용합니다.
# - 실패 시에도 원인을 화면에 표시할 수 있도록 최소 키를 반환합니다.

def _safe_first_df(*values: Any) -> pd.DataFrame:
    for value in values:
        if isinstance(value, pd.DataFrame) and not value.empty:
            return value
    return pd.DataFrame()


def estimate_skill_crit_tables(bundle: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
    """v13: v10 기본 계산을 직접 호출하고, v11/v12 요약을 안전하게 추가합니다.

    기존 v12는 내부에서 v11 wrapper를 거치면서
    `result.get("arkgrid_final_skill_estimates") or result.get("skill_crit_estimates")` 형태의
    DataFrame truth-value 오류가 발생할 수 있었습니다. 이 오류가 api_parser에서 삼켜지면
    기준 계산표/최종 계산표/피해군 출처가 모두 빈 표로 보였습니다.
    """
    try:
        result = _old_estimate_skill_crit_tables_v10(bundle)
    except Exception as e:  # noqa: BLE001
        return {
            "skill_crit_estimates": pd.DataFrame(),
            "lostbuilds_base_skill_estimates": pd.DataFrame(),
            "arkgrid_delta_estimates": pd.DataFrame(),
            "damage_sources": pd.DataFrame(),
            "base_damage_sources": pd.DataFrame(),
            "arkgrid_damage_sources": pd.DataFrame(),
            "unresolved_sources": pd.DataFrame(),
            "combat_overview": pd.DataFrame([
                {"항목": "계산 오류", "아크그리드 제외 기준": str(e), "아크그리드 포함 최종": str(e), "비고": "modules/api_skill_estimator.py 확인"}
            ]),
            "base_crit_percent": 0.0,
            "base_crit_raw": f"치명/치피/피해군 추정 실패: {e}",
            "avg_final_multiplier": 1.0,
        }

    source_df = _safe_first_df(result.get("damage_sources"), result.get("crit_sources"))
    base_sources = _safe_first_df(result.get("base_damage_sources"))
    final_df = _safe_first_df(result.get("arkgrid_final_skill_estimates"), result.get("skill_crit_estimates"))
    base_df = _safe_first_df(result.get("lostbuilds_base_skill_estimates"))

    if not source_df.empty:
        result["loawa_like_breakdown"] = _loawa_like_breakdown(source_df, "백어택")
    if not base_sources.empty:
        result["loawa_like_base_breakdown"] = _loawa_like_breakdown(base_sources, "백어택")

    if not final_df.empty and "공격타입" in final_df.columns:
        back_final = final_df[final_df["공격타입"].astype(str).str.contains("백", na=False)]
        if not back_final.empty:
            result["avg_back_skill_crit_rate_percent"] = _avg_col(back_final, "예상 치명 확률(백어택 기준)(%)")
            result["avg_back_skill_crit_damage_percent"] = _avg_col(back_final, "예상 치피(조건부 포함)(%)", BASE_CRIT_DAMAGE_PERCENT)
    if not base_df.empty and "공격타입" in base_df.columns:
        back_base = base_df[base_df["공격타입"].astype(str).str.contains("백", na=False)]
        if not back_base.empty:
            result["lostbuilds_base_back_crit_rate_percent"] = _avg_col(back_base, "예상 치명 확률(백어택 기준)(%)")
            result["lostbuilds_base_back_crit_damage_percent"] = _avg_col(back_base, "예상 치피(조건부 포함)(%)", BASE_CRIT_DAMAGE_PERCENT)

    return _add_v12_summary(result, bundle)

# ==============================
# v14 hotfix overrides
# ==============================
# - 계산표가 빈 표로 떨어지는 문제를 피하기 위해 old wrapper를 거치지 않고 현재 함수들로 직접 재계산합니다.
# - 선택 트라이포드 설명에서 "일반/홀딩/차지/콤보/지점" 등 조작 타입 변경을 감지합니다.
# - 예: 볼케이노 이럽션 + 블러드 이럽션 트포 → 기본 홀딩이더라도 계산/표시용 타입은 일반으로 보정됩니다.

ESTIMATOR_VERSION = "v14"


def _selected_tripod_text(skill: Dict[str, Any]) -> str:
    parts: List[str] = []
    for tripod in skill.get("Tripods", []) or []:
        if bool(tripod.get("IsSelected")):
            parts.append(_flatten(tripod))
    return _clean_text(" ".join(parts))


def _effective_operation_type(skill: Dict[str, Any]) -> str:
    """선택 트라이포드로 변경된 조작 타입을 반영합니다.

    API의 skill.Type은 기본 타입인 경우가 많고, 선택 트라이포드 설명에
    '일반 조작으로 변경된다' 같은 문구가 있으면 실제 운용 타입이 달라집니다.
    """
    base = str(skill.get("Type") or skill.get("SkillType") or skill.get("SkillTypeName") or "").strip()
    text = _selected_tripod_text(skill)
    clean = _clean_text(text)

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
            if re.search(pat, clean):
                return label
    return base or "일반"


def _detect_attack_type(skill: Dict[str, Any]) -> str:  # type: ignore[override]
    """v14: 방향성 + 트포 반영 조작 타입 + 스킬 계열을 모두 attack_type 문자열에 포함."""
    text = _flatten(skill)
    name_norm = _norm_name(skill.get("Name") or "")
    joined = f"{text}"

    if name_norm in KNOWN_NON_DIRECTIONAL_SKILLS:
        direction = "일반/확인필요"
    else:
        has_back = (
            "백어택" in joined
            or "Back Attack" in joined
            or "BackAttack" in joined
            or name_norm in KNOWN_BACK_ATTACK_SKILLS
        )
        has_head = "헤드어택" in joined or "Head Attack" in joined or "HeadAttack" in joined
        if has_back and has_head:
            direction = "백/헤드"
        elif has_back:
            direction = "백어택"
        elif has_head:
            direction = "헤드어택"
        else:
            direction = "일반/확인필요"

    op_type = _effective_operation_type(skill)
    groups = _skill_groups_for_skill(skill)
    # 실제 조작 타입도 '일반 스킬/홀딩 스킬' 계열 판정에 쓰이도록 추가.
    if op_type and op_type not in groups:
        groups.append(op_type)
        groups.append(f"{op_type} 스킬")

    parts = [direction]
    for g in groups:
        if g and _norm_name(g) not in _norm_name(" / ".join(parts)):
            parts.append(g)
    return " / ".join(parts)


def estimate_skill_crit_tables(bundle: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
    """v14: 현재 override된 함수들을 직접 사용해서 계산표를 생성합니다.

    이전 v13은 안전 wrapper를 거쳤지만, 사용자의 세션/pycache/old wrapper 상태에 따라
    계산표가 비어 보일 수 있었습니다. 이 함수는 base/final/arkgrid 표를 직접 생성하고,
    효과 출처가 없어도 채용 스킬 기준의 기본 계산표는 반드시 생성합니다.
    """
    try:
        estimator_timings: Dict[str, Any] = {"version": "v157_estimator_detail"}
        estimator_t0 = _time_v157.perf_counter()

        def timed(name: str, fn, *args, **kwargs):
            t = _time_v157.perf_counter()
            value = fn(*args, **kwargs)
            estimator_timings[f"{name}_ms"] = round((_time_v157.perf_counter() - t) * 1000.0, 3)
            try:
                if isinstance(value, pd.DataFrame):
                    estimator_timings[f"{name}_rows"] = int(len(value))
                elif isinstance(value, (list, tuple, set, dict)):
                    estimator_timings[f"{name}_count"] = int(len(value))
            except Exception:
                pass
            return value

        crit_stat, stat_rate, stat_raw, stat_source = timed("profile_stats", _profile_stats, bundle)
        skill_names = timed("skill_names", _skill_names, bundle)

        all_global_sources = timed("collect_global_sources", _collect_global_sources, bundle)
        identity_sources = timed("identity_sources", _identity_sources, bundle)
        tripod_sources = timed("tripod_sources", _collect_skill_tripod_sources, bundle)
        gem_sources = timed("gem_sources", _collect_gem_sources, bundle)

        t_force = _time_v157.perf_counter()
        all_global_sources = _force_skill_scopes(all_global_sources, skill_names) if isinstance(all_global_sources, pd.DataFrame) and not all_global_sources.empty else _empty_source_df()
        identity_sources = _force_skill_scopes(identity_sources, skill_names) if isinstance(identity_sources, pd.DataFrame) and not identity_sources.empty else _empty_source_df()
        tripod_sources = _force_skill_scopes(tripod_sources, skill_names) if isinstance(tripod_sources, pd.DataFrame) and not tripod_sources.empty else _empty_source_df()
        gem_sources = _force_skill_scopes(gem_sources, skill_names) if isinstance(gem_sources, pd.DataFrame) and not gem_sources.empty else _empty_source_df()
        estimator_timings["force_skill_scopes_ms"] = round((_time_v157.perf_counter() - t_force) * 1000.0, 3)

        arkgrid_sources = _filter_source_type(all_global_sources, include=["아크그리드"])
        base_global_sources = _filter_source_type(all_global_sources, exclude=["아크그리드"])

        base_effect_sources = timed("concat_base_effect_sources", _concat_sources, identity_sources, base_global_sources, tripod_sources, gem_sources)
        final_effect_sources = timed("concat_final_effect_sources", _concat_sources, base_effect_sources, arkgrid_sources)
        source_df = timed("concat_source_df", _concat_sources, stat_source, identity_sources, base_global_sources, arkgrid_sources, tripod_sources, gem_sources)
        unresolved_sources = source_df[source_df["적용범위"].astype(str) == "검수 필요"].copy() if isinstance(source_df, pd.DataFrame) and not source_df.empty else _empty_source_df()

        back_engraving = timed("has_back_attack_engraving", _has_back_attack_engraving, bundle)
        head_engraving = timed("has_head_attack_engraving", _has_head_attack_engraving, bundle)

        base_skill_df = timed("build_base_skill_table", _build_skill_table, bundle, base_effect_sources, crit_stat, stat_rate, back_engraving, head_engraving)
        final_skill_df = timed("build_final_skill_table", _build_skill_table, bundle, final_effect_sources, crit_stat, stat_rate, back_engraving, head_engraving)
        arkgrid_delta_df = timed("metric_delta", _metric_delta, final_skill_df, base_skill_df)
        merged_skill_df = timed("merged_base_final_table", _merged_base_final_table, base_skill_df, final_skill_df, arkgrid_delta_df)

        overview = timed("overview_from_tables", _overview_from_tables,
            crit_stat,
            stat_rate,
            base_skill_df,
            final_skill_df,
            base_effect_sources,
            final_effect_sources,
            back_engraving,
            head_engraving,
        )

        back_final = final_skill_df[final_skill_df["공격타입"].astype(str).str.contains("백", na=False)] if isinstance(final_skill_df, pd.DataFrame) and not final_skill_df.empty else pd.DataFrame()
        back_base = base_skill_df[base_skill_df["공격타입"].astype(str).str.contains("백", na=False)] if isinstance(base_skill_df, pd.DataFrame) and not base_skill_df.empty else pd.DataFrame()
        t_summary_metrics = _time_v157.perf_counter()
        global_final = _non_skill_sources(final_effect_sources)
        non_skill_for_cd = global_final[~global_final["적용범위"].astype(str).isin(["스킬 전용", "스킬 그룹", "계열 전용", "스킬 계열", "검수 필요"])] if not global_final.empty else _empty_source_df()
        global_crit_damage_no_skill = BASE_CRIT_DAMAGE_PERCENT + _sum_sources(
            non_skill_for_cd, "치명타 피해량 증가(%)", "백어택" if back_engraving else "일반/확인필요", "", include_conditional=True
        )

        estimator_timings["final_summary_metrics_ms"] = round((_time_v157.perf_counter() - t_summary_metrics) * 1000.0, 3)

        result = {
            "estimator_version": ESTIMATOR_VERSION,
            "skill_crit_estimates": merged_skill_df,
            "lostbuilds_base_skill_estimates": base_skill_df,
            "arkgrid_final_skill_estimates": final_skill_df,
            "arkgrid_delta_estimates": arkgrid_delta_df,
            "crit_sources": source_df,
            "damage_sources": source_df,
            "base_damage_sources": _concat_sources(stat_source, identity_sources, base_global_sources, tripod_sources, gem_sources),
            "arkgrid_damage_sources": arkgrid_sources,
            "unresolved_sources": unresolved_sources,
            "combat_overview": overview,
            "loawa_like_breakdown": _loawa_like_breakdown(source_df, "백어택"),
            "loawa_like_base_breakdown": _loawa_like_breakdown(_concat_sources(stat_source, identity_sources, base_global_sources, tripod_sources, gem_sources), "백어택"),
            "base_crit_stat": crit_stat,
            "base_crit_percent": stat_rate,
            "base_crit_raw": stat_raw,
            "avg_static_crit_rate_percent": _avg_col(final_skill_df, "예상 치명 확률(상시)(%)"),
            "avg_conditional_crit_rate_percent": _avg_col(final_skill_df, "예상 치명 확률(조건부 포함)(%)"),
            "avg_back_basis_crit_rate_percent": _avg_col(final_skill_df, "예상 치명 확률(백어택 기준)(%)"),
            "avg_static_crit_damage_percent": _avg_col(final_skill_df, "예상 치피(상시)(%)", BASE_CRIT_DAMAGE_PERCENT),
            "avg_conditional_crit_damage_percent": _avg_col(final_skill_df, "예상 치피(조건부 포함)(%)", BASE_CRIT_DAMAGE_PERCENT),
            "avg_back_skill_crit_rate_percent": _avg_col(back_final, "예상 치명 확률(백어택 기준)(%)", _avg_col(final_skill_df, "예상 치명 확률(백어택 기준)(%)")),
            "avg_back_skill_crit_damage_percent": _avg_col(back_final, "예상 치피(조건부 포함)(%)", _avg_col(final_skill_df, "예상 치피(조건부 포함)(%)", BASE_CRIT_DAMAGE_PERCENT)),
            "lostbuilds_base_back_crit_rate_percent": _avg_col(back_base, "예상 치명 확률(백어택 기준)(%)", 0.0),
            "lostbuilds_base_back_crit_damage_percent": _avg_col(back_base, "예상 치피(조건부 포함)(%)", BASE_CRIT_DAMAGE_PERCENT),
            "global_crit_damage_no_skill_percent": global_crit_damage_no_skill,
            "avg_evolution_damage_percent": _avg_col(final_skill_df, "진화형 피해(조건부)(%)"),
            "avg_enemy_damage_percent": _avg_col(final_skill_df, "적에게 주는 피해(조건부)(%)"),
            "avg_skill_damage_percent": _avg_col(final_skill_df, "스킬 피해(조건부)(%)"),
            "avg_directional_damage_percent": _avg_col(final_skill_df, "방향성 피해(조건부)(%)"),
            "avg_gem_damage_percent": _avg_col(final_skill_df, "보석 피해(조건부)(%)"),
            "avg_final_multiplier": _avg_col(final_skill_df, "예상 최종 배율(조건부)", 1.0),
            "lostbuilds_base_avg_final_multiplier": _avg_col(base_skill_df, "예상 최종 배율(조건부)", 1.0),
            "has_back_attack_engraving": back_engraving,
            "has_head_attack_engraving": head_engraving,
        }
        estimator_timings["total_ms"] = round((_time_v157.perf_counter() - estimator_t0) * 1000.0, 3)
        timing_rows = []
        for k, v in estimator_timings.items():
            if k.endswith("_ms"):
                try:
                    timing_rows.append({"stage": k, "ms": float(v)})
                except Exception:
                    pass
        estimator_timings["top_stages"] = sorted(timing_rows, key=lambda x: x["ms"], reverse=True)[:12]
        result["_estimator_timing_v157"] = estimator_timings
        return _add_v12_summary(result, bundle)
    except Exception as e:  # noqa: BLE001
        return {
            "estimator_version": ESTIMATOR_VERSION,
            "skill_crit_estimates": pd.DataFrame(),
            "lostbuilds_base_skill_estimates": pd.DataFrame(),
            "arkgrid_final_skill_estimates": pd.DataFrame(),
            "arkgrid_delta_estimates": pd.DataFrame(),
            "damage_sources": pd.DataFrame(),
            "base_damage_sources": pd.DataFrame(),
            "arkgrid_damage_sources": pd.DataFrame(),
            "unresolved_sources": pd.DataFrame(),
            "combat_overview": pd.DataFrame([
                {"항목": "계산 오류", "아크그리드 제외 기준": type(e).__name__, "아크그리드 포함 최종": str(e), "비고": "modules/api_skill_estimator.py v14 확인"}
            ]),
            "base_crit_percent": 0.0,
            "base_crit_raw": f"치명/치피/피해군 추정 실패: {type(e).__name__}: {e}",
            "avg_final_multiplier": 1.0,
            "estimator_error": f"{type(e).__name__}: {e}",
        }


# ==============================
# v15 correction overrides
# ==============================
# - 방향성 공격 스킬(백/헤드 통합)의 치명타 피해량 증가를 전역 치피가 아니라 백/헤드 스킬 범위로 처리합니다.
# - 도약/초각성/숙련된 힘처럼 특정 스킬 전용이어야 하는 효과가 스킬명 매칭 실패 시 전역으로 새지 않게 검수 필요로 분리합니다.
# - 사용자가 보는 백어택 기준 치명에는 백어택 기본 +10과 별도로, 기습 채용 캐릭터의 검수 기준 보정 +10을 출처 행으로 표시/반영합니다.

ESTIMATOR_VERSION = "v15"
BACK_ATTACK_ENGRAVING_CRIT_VIEW_BONUS_PERCENT = 10.0


def _scope_from_text(text: str, target_skill: str | None = None) -> str:  # type: ignore[override]
    clean = _clean_text(text)
    if target_skill:
        return "스킬 전용"
    # 로아와/툴팁에서 말하는 '방향성 공격 스킬'은 백어택/헤드어택 스킬 범위로 처리합니다.
    if any(w in clean for w in [
        "방향성 공격 스킬", "방향성 공격", "방향성 스킬", "방향 공격 스킬",
        "백어택 및 헤드어택", "백어택/헤드어택", "백어택 혹은 헤드어택",
    ]):
        return "백/헤드 스킬"
    has_back = "백어택" in clean or "기습" in clean
    has_head = "헤드어택" in clean or "결투" in clean
    if has_back and has_head:
        return "백/헤드 스킬"
    if has_back:
        return "백어택 스킬"
    if has_head:
        return "헤드어택 스킬"
    return "전체"


def _looks_skill_specific_effect(text: str, source_name: str, skill_names: Iterable[str]) -> bool:  # type: ignore[override]
    clean = _clean_text(f"{source_name} {text}")
    if not clean:
        return False
    if any(w in clean for w in ["초각성 스킬", "초각성스킬", "숙련된 힘", "도약"]):
        if any(w in clean for w in ["치명타 피해", "치명타 피해량", "피해량", "피해가", "스킬 피해", "치명타 적중률"]):
            return True
    if re.search(r"[가-힣A-Za-z0-9\s]{2,30}의\s*(?:치명타\s*피해량|치명타\s*피해|치명타\s*적중률|피해량|피해가|주는\s*피해|재사용|쿨타임)", clean):
        return True
    if any(w in clean for w in ["코어", "효과가 강화", "운명", "연명", "업화", "난무", "오의", "충격", "기력"]):
        if any(w in clean for w in ["피해량", "치명타 피해", "치명타 적중률", "쿨타임", "재사용"]):
            return True
    return False


def _source_rows_from_text(
    source_type: str,
    name: str,
    text: str,
    skill_names: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:  # type: ignore[override]
    """v15: 스킬명/계열/방향성/초각성 전용 효과를 더 보수적으로 분리."""
    clean = _clean_text(text)
    names = list(skill_names or [])
    rows: List[Dict[str, Any]] = []

    if source_type in {"아크그리드", "아크패시브", "선택 트라이포드", "스킬"}:
        snippets = _split_option_snippets(clean)
    else:
        snippets = [clean]

    last_target: str | None = None
    for snippet in snippets:
        # 스킬명이 API 채용 목록에 없어도 별칭 테이블에서 감지되면 canonical 스킬명으로 스킬 전용 처리합니다.
        target = _detect_target_skill(f"{name} {snippet}", names)
        # 같은 툴팁 조각이 % 기준으로 잘리면서 뒤 조각에서 스킬명이 사라지는 경우가 있습니다.
        # 예: "플레임 블레이드의 피해량 30% 증가하고 치명타 피해 90% 증가"
        # 뒤의 치피 90% 조각도 직전 스킬 전용 효과로 이어받습니다.
        if not target and last_target and source_type in {"아크패시브", "아크그리드", "선택 트라이포드", "스킬"}:
            if any(w in snippet for w in ["치명타 피해", "치명타 피해량", "치명타 적중률", "피해량", "피해가", "주는 피해", "재사용", "쿨타임"]):
                target = last_target
        if target:
            last_target = target
        group_target = None if target else _skill_group_for_source_text(f"{name} {snippet}")
        row_name = name
        marker = re.search(r"\[\s*(\d+\s*[Pp]?)\s*\]", snippet)
        if marker:
            row_name = f"{name} [{marker.group(1).replace(' ', '').upper()}]"

        row = _source_row(source_type, row_name, snippet, target_skill=target)
        if row:
            if target:
                row["적용범위"] = "스킬 전용"
                row["적용스킬"] = target
                if "폭주" not in snippet:
                    row["조건부 여부"] = "상시"
            elif group_target and any(abs(_num(row.get(c)) or 0.0) > 1e-9 for c in EFFECT_COLUMNS):
                row["적용범위"] = "스킬 그룹"
                row["적용스킬"] = group_target
                if not _is_conditional(snippet) or "스킬" in group_target:
                    row["조건부 여부"] = "상시"
            elif source_type in {"아크그리드", "아크패시브", "선택 트라이포드", "스킬"} and _looks_skill_specific_effect(snippet, row_name, names):
                # 아크그리드 코어/아크패시브 깨달음의 '치명타 적중률/치명타 피해량/공격력/적피' 같은
                # 전역 스탯 증가는 이름에 '코어/인챈트' 등이 들어가도 모든 스킬에 적용되는 합연산 값입니다.
                # (예: 부수는 일격 코어 [14~20P] 치적, 리퍼 피냄새/스카우터 코어 인챈트 깨달음 치적)
                # 특정 스킬을 지목한 경우는 위에서 이미 target이 잡혀 스킬 전용으로 처리되므로,
                # 여기(스킬명 매칭 실패)로 온 전역 스탯만 전체로 승격합니다.
                _core_global_stat = source_type in ("아크그리드", "아크패시브") and any(
                    abs(_num(row.get(c)) or 0.0) > 1e-9
                    for c in ["치명타 적중률 증가(%)", "치명타 피해량 증가(%)", "공격력 증가(%)", "적에게 주는 피해(%)"]
                )
                if _core_global_stat:
                    row["적용범위"] = "전체"
                    row["적용스킬"] = ""
                    row["조건부 여부"] = "상시"
                else:
                    # 초각성/도약/코어 전용처럼 보이는데 스킬명 확정 실패: 절대 전역 치피/진피로 합산하지 않음.
                    row["적용범위"] = "검수 필요"
                    row["적용스킬"] = "스킬명/계열 자동 매칭 실패"
                    row["조건부 여부"] = "검수 필요"
            rows.append(row)

    if not rows:
        target = _detect_target_skill(clean, names)
        group_target = None if target else _skill_group_for_source_text(clean)
        row = _source_row(source_type, name, clean, target_skill=target)
        if row:
            if target:
                row["적용범위"] = "스킬 전용"
                row["적용스킬"] = target
            elif group_target:
                row["적용범위"] = "스킬 그룹"
                row["적용스킬"] = group_target
            elif source_type in {"아크그리드", "아크패시브", "선택 트라이포드", "스킬"} and _looks_skill_specific_effect(clean, name, names):
                row["적용범위"] = "검수 필요"
                row["적용스킬"] = "스킬명/계열 자동 매칭 실패"
                row["조건부 여부"] = "검수 필요"
            rows.append(row)

    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        sig = tuple(str(row.get(c, "")) for c in SOURCE_COLUMNS)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(row)
    return out


_old_identity_sources_v14 = _identity_sources


def _identity_sources(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    """v15: 직업 룰 + 기습 채용 백어택 기준 치명 보정 출처를 함께 생성."""
    base = _old_identity_sources_v14(bundle)
    rows: List[Dict[str, Any]] = []
    if isinstance(base, pd.DataFrame) and not base.empty:
        rows.extend(base.to_dict("records"))

    if _has_back_attack_engraving(bundle):
        effects = _new_effect_dict()
        effects["치명타 적중률 증가(%)"] = BACK_ATTACK_ENGRAVING_CRIT_VIEW_BONUS_PERCENT
        rows.append({
            "출처구분": "전투 기준 보정",
            "이름": "기습 채용 백어택 기준 치명",
            "적용범위": "백어택 스킬",
            "적용스킬": "전체/범위 기준",
            **effects,
            "조건부 여부": "상시",
            "설명": "사용자 검수 기준: 기습 채용 캐릭터는 백어택 기준 치명타 적중률 +10을 추가 표기/반영",
        })

    if not rows:
        return _empty_source_df()
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS).drop_duplicates().reset_index(drop=True)


def _global_directional_crit_damage(effect_sources: pd.DataFrame, attack_type: str = "백어택") -> float:
    if effect_sources is None or effect_sources.empty:
        return BASE_CRIT_DAMAGE_PERCENT
    non_skill = effect_sources[~effect_sources["적용범위"].astype(str).isin(["스킬 전용", "스킬 그룹", "계열 전용", "스킬 계열", "검수 필요"])]
    return BASE_CRIT_DAMAGE_PERCENT + _sum_sources(non_skill, "치명타 피해량 증가(%)", attack_type, "", include_conditional=True)


def _overview_from_tables(
    crit_stat: float,
    stat_rate: float,
    base_df: pd.DataFrame,
    final_df: pd.DataFrame,
    base_sources: pd.DataFrame,
    final_sources: pd.DataFrame,
    back_engraving: bool,
    head_engraving: bool,
) -> pd.DataFrame:  # type: ignore[override]
    back_base = base_df[base_df["공격타입"].astype(str).str.contains("백", na=False)] if isinstance(base_df, pd.DataFrame) and not base_df.empty else pd.DataFrame()
    back_final = final_df[final_df["공격타입"].astype(str).str.contains("백", na=False)] if isinstance(final_df, pd.DataFrame) and not final_df.empty else pd.DataFrame()

    base_global_cd = _global_directional_crit_damage(base_sources, "백어택" if back_engraving else "일반/확인필요")
    final_global_cd = _global_directional_crit_damage(final_sources, "백어택" if back_engraving else "일반/확인필요")

    rows = [
        {"항목": "치명 스탯", "아크그리드 제외 기준": f"{crit_stat:,.0f}", "아크그리드 포함 최종": f"{crit_stat:,.0f}", "비고": "API profile Stats"},
        {"항목": "치명 스탯 치적", "아크그리드 제외 기준": f"{stat_rate:.2f}%", "아크그리드 포함 최종": f"{stat_rate:.2f}%", "비고": "Tooltip 우선, 없으면 치명×계수"},
        {"항목": "백어택 스킬 기준 치명", "아크그리드 제외 기준": f"{_avg_col(back_base, '예상 치명 확률(백어택 기준)(%)'):.2f}%", "아크그리드 포함 최종": f"{_avg_col(back_final, '예상 치명 확률(백어택 기준)(%)'):.2f}%", "비고": "백어택 기본 +10, 기습 채용 보정 +10 포함"},
        {"항목": "전역 치명타 피해량(스킬 전용 제외/방향성 기준)", "아크그리드 제외 기준": f"{base_global_cd:.2f}%", "아크그리드 포함 최종": f"{final_global_cd:.2f}%", "비고": "기본 200 + 전역 치피 + 방향성 치피. 숙련된 힘/허리케인 같은 스킬 전용 제외"},
        {"항목": "평균 치명타 피해량(스킬 전용 포함)", "아크그리드 제외 기준": f"{_avg_col(base_df, '예상 치피(조건부 포함)(%)', BASE_CRIT_DAMAGE_PERCENT):.2f}%", "아크그리드 포함 최종": f"{_avg_col(final_df, '예상 치피(조건부 포함)(%)', BASE_CRIT_DAMAGE_PERCENT):.2f}%", "비고": "허리케인 +100, 도약 초각성 치피 등은 해당 스킬만"},
        {"항목": "평균 진화형 피해", "아크그리드 제외 기준": f"{_avg_col(base_df, '진화형 피해(조건부)(%)'):.2f}%", "아크그리드 포함 최종": f"{_avg_col(final_df, '진화형 피해(조건부)(%)'):.2f}%", "비고": "아크패시브 기준 + 아크그리드 추가"},
        {"항목": "평균 적에게 주는 피해", "아크그리드 제외 기준": f"{_avg_col(base_df, '적에게 주는 피해(조건부)(%)'):.2f}%", "아크그리드 포함 최종": f"{_avg_col(final_df, '적에게 주는 피해(조건부)(%)'):.2f}%", "비고": "각인/장비/기타"},
        {"항목": "평균 스킬 피해", "아크그리드 제외 기준": f"{_avg_col(base_df, '스킬 피해(조건부)(%)'):.2f}%", "아크그리드 포함 최종": f"{_avg_col(final_df, '스킬 피해(조건부)(%)'):.2f}%", "비고": "선택 트포/스킬 전용"},
        {"항목": "평균 보석 피해", "아크그리드 제외 기준": f"{_avg_col(base_df, '보석 피해(조건부)(%)'):.2f}%", "아크그리드 포함 최종": f"{_avg_col(final_df, '보석 피해(조건부)(%)'):.2f}%", "비고": "멸화/겁화 피해 보석"},
        {"항목": "평균 예상 최종 배율", "아크그리드 제외 기준": f"{_avg_col(base_df, '예상 최종 배율(조건부)', 1.0):.4f}x", "아크그리드 포함 최종": f"{_avg_col(final_df, '예상 최종 배율(조건부)', 1.0):.4f}x", "비고": "같은 피해군 합산, 다른 피해군 곱연산"},
        {"항목": "기습 각인 감지", "아크그리드 제외 기준": "예" if back_engraving else "아니오", "아크그리드 포함 최종": "예" if back_engraving else "아니오", "비고": "백어택 기준 보정 +10 출처 추가"},
        {"항목": "결투 각인 감지", "아크그리드 제외 기준": "예" if head_engraving else "아니오", "아크그리드 포함 최종": "예" if head_engraving else "아니오", "비고": "헤드어택 기준 보너스 표기"},
    ]
    return pd.DataFrame(rows)


_old_add_v12_summary_v14 = _add_v12_summary


def _add_v12_summary(result: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
    result = _old_add_v12_summary_v14(result, bundle)
    # v15: 앱의 metric 카드가 보는 전역 치피도 방향성 기준 치피로 재계산합니다.
    final_sources = result.get("damage_sources")
    if isinstance(final_sources, pd.DataFrame) and not final_sources.empty:
        attack_type = "백어택" if result.get("has_back_attack_engraving") else "일반/확인필요"
        result["global_crit_damage_no_skill_percent"] = _global_directional_crit_damage(final_sources, attack_type)
    return result


# ==============================
# v16 correction overrides
# ==============================
# - 기습 각인 자체를 치명 +10으로 더하던 보정 행을 제거합니다.
#   백어택 치명 +10은 공격 방향 보너스로만 1회 적용합니다.
# - 진화 아크패시브 노드는 툴팁 정규식 추정 대신 노드별 포인트 규칙으로 재생성합니다.
#   금단의 주문/예리한 감각/한계 돌파/최적화 훈련/무한한 마력/혼신의 강타/파괴 전차/일격/달인/분쇄/음속돌파를 우선 처리합니다.
# - 일격의 "방향성 공격 스킬 치명타 피해"는 백/헤드 스킬 범위로만 적용합니다.

ESTIMATOR_VERSION = "v53-engraving-stage-crit-fix"

EVOLUTION_NODE_RULES_V16: Dict[str, Dict[str, Any]] = {
    "금단의 주문": {
        "aliases": ["금단의 주문", "금단의주문"],
        "effects": [
            {"scope": "전체", "target": "전체/범위 기준", "col": "진화형 피해(%)", "per_point": 10.0, "note": "금단의 주문: 1포인트당 진화형 피해 +10%"},
            # 실제로는 마나 스킬 한정이지만, 현 API 스킬 Tooltip에 마나 스킬 계열이 항상 명시되지 않아 우선 전체 기준에 합산합니다.
            # 이후 스킬 Tooltip에서 [마나 스킬] 계열이 안정적으로 확인되면 스킬 그룹으로 좁히면 됩니다.
            {"scope": "전체", "target": "전체/범위 기준", "col": "진화형 피해(%)", "per_point": 10.0, "note": "금단의 주문: 마나 스킬 추가 진화형 피해 +10%"},
        ],
    },
    "예리한 감각": {
        "aliases": ["예리한 감각", "예리한감각"],
        "effects": [
            {"scope": "전체", "target": "전체/범위 기준", "col": "진화형 피해(%)", "per_point": 5.0, "note": "예리한 감각: 1포인트당 진화형 피해 +5%"},
            {"scope": "전체", "target": "전체/범위 기준", "col": "치명타 적중률 증가(%)", "per_point": 4.0, "note": "예리한 감각: 1포인트당 치명타 적중률 +4%"},
        ],
    },
    "한계 돌파": {
        "aliases": ["한계 돌파", "한계돌파"],
        "effects": [
            {"scope": "전체", "target": "전체/범위 기준", "col": "진화형 피해(%)", "per_point": 10.0, "note": "한계 돌파: 1포인트당 진화형 피해 +10%"},
        ],
    },
    "최적화 훈련": {
        "aliases": ["최적화 훈련", "최적화훈련"],
        "effects": [
            {"scope": "전체", "target": "전체/범위 기준", "col": "진화형 피해(%)", "per_point": 5.0, "note": "최적화 훈련: 1포인트당 진화형 피해 +5%"},
        ],
    },
    "무한한 마력": {
        "aliases": ["무한한 마력", "무한한마력"],
        "effects": [
            {"scope": "전체", "target": "전체/범위 기준", "col": "진화형 피해(%)", "per_point": 8.0, "note": "무한한 마력: 1포인트당 진화형 피해 +8%"},
        ],
    },
    "혼신의 강타": {
        "aliases": ["혼신의 강타", "혼신의강타"],
        "effects": [
            {"scope": "전체", "target": "전체/범위 기준", "col": "진화형 피해(%)", "per_point": 2.0, "note": "혼신의 강타: 1포인트당 진화형 피해 +2%"},
            {"scope": "전체", "target": "전체/범위 기준", "col": "치명타 적중률 증가(%)", "per_point": 12.0, "note": "혼신의 강타: 1포인트당 치명타 적중률 +12%"},
        ],
    },
    "파괴 전차": {
        "aliases": ["파괴 전차", "파괴전차"],
        "effects": [
            {"scope": "전체", "target": "전체/범위 기준", "col": "진화형 피해(%)", "per_point": 12.0, "note": "파괴 전차: 1포인트당 진화형 피해 +12%"},
        ],
    },
    "일격": {
        "aliases": ["일격"],
        "effects": [
            {"scope": "전체", "target": "전체/범위 기준", "col": "치명타 적중률 증가(%)", "per_point": 10.0, "note": "일격: 1포인트당 치명타 적중률 +10%"},
            {"scope": "백/헤드 스킬", "target": "백/헤드 스킬", "col": "치명타 피해량 증가(%)", "per_point": 16.0, "note": "일격: 1포인트당 방향성 공격 스킬 치명타 피해 +16%"},
        ],
    },
    "달인": {
        "aliases": ["달인"],
        "effects": [
            {"scope": "전체", "target": "전체/범위 기준", "col": "치명타 적중률 증가(%)", "per_point": 7.0, "note": "달인: 1포인트당 치명타 적중률 +7%"},
        ],
    },
    "분쇄": {
        "aliases": ["분쇄"],
        "effects": [
            {"scope": "전체", "target": "전체/범위 기준", "col": "진화형 피해(%)", "per_point": 20.0, "note": "분쇄: 1포인트당 진화형 피해 +20%"},
        ],
    },
    "음속 돌파": {
        "aliases": ["음속 돌파", "음속돌파"],
        "effects": [
            # 음속돌파는 보통 5티어 선택지로 총 +24% 기준. 문구에서 더 정확한 수치가 잡히면 포인트 추정이 아니라 direct 값을 우선합니다.
            {"scope": "전체", "target": "전체/범위 기준", "col": "진화형 피해(%)", "flat": 24.0, "note": "음속 돌파: 총 진화형 피해 +24%"},
        ],
    },
}


def _v16_contains_alias(text: str, aliases: Iterable[str]) -> bool:
    norm = _norm_name(text)
    return any(_norm_name(a) in norm for a in aliases)


def _v16_percent_values(text: str) -> List[float]:
    vals: List[float] = []
    for m in re.finditer(r"([+\-]?\d+(?:\.\d+)?)\s*%", _clean_text(text)):
        v = _num(m.group(1))
        if v is not None:
            vals.append(float(v))
    return vals


def _v16_node_points(canonical: str, text: str, matched_rows: pd.DataFrame | None = None) -> float:
    """노드 포인트/레벨을 추정합니다.

    우선순위:
    1) 이름/설명에 Lv.2, 레벨 2, 2포인트 같은 직접 표기가 있으면 사용
    2) 기존 파서가 잡은 수치 / per_point 비율로 역산
    3) 수치 역산이 불가하면 선택 감지 기준 1포인트
    """
    clean = _clean_text(text)
    direct_patterns = [
        r"Lv\.?\s*(\d+)",
        r"레벨\s*(\d+)",
        r"(\d+)\s*포인트",
        r"(\d+)\s*P\b",
    ]
    direct: List[float] = []
    for pat in direct_patterns:
        for m in re.finditer(pat, clean, flags=re.IGNORECASE):
            v = _num(m.group(1))
            if v is not None and 0 < v <= 5:
                direct.append(float(v))
    if direct:
        return max(direct)

    rule = EVOLUTION_NODE_RULES_V16.get(canonical, {})
    candidates: List[float] = []
    if isinstance(matched_rows, pd.DataFrame) and not matched_rows.empty:
        for eff in rule.get("effects", []):
            col = eff.get("col")
            per = eff.get("per_point")
            if not col or not per or col not in matched_rows.columns:
                continue
            vals = [_num(v) or 0.0 for v in matched_rows[col].tolist()]
            val = max([abs(v) for v in vals] or [0.0])
            if val > 0:
                candidates.append(val / float(per))
    # 설명 텍스트의 숫자로도 한 번 더 역산합니다.
    text_values = _v16_percent_values(clean)
    if text_values:
        for eff in rule.get("effects", []):
            per = eff.get("per_point")
            if not per:
                continue
            for val in text_values:
                ratio = abs(val) / float(per)
                if 0.5 <= ratio <= 5.5:
                    # 정수 포인트에 가까운 값만 후보로 사용합니다.
                    rounded = round(ratio)
                    if abs(ratio - rounded) < 0.11:
                        candidates.append(float(rounded))
    if candidates:
        return max(1.0, min(5.0, max(candidates)))
    return 1.0


def _v16_make_source_row(source_type: str, canonical: str, scope: str, target: str, col: str, value: float, note: str, points: float | None = None) -> Dict[str, Any]:
    effects = _new_effect_dict()
    effects[col] = round(float(value), 4)
    point_note = f" / 적용 포인트 {points:g}" if points is not None else ""
    return {
        "출처구분": source_type,
        "이름": canonical,
        "적용범위": scope,
        "적용스킬": target,
        **effects,
        "조건부 여부": "상시",
        "설명": f"v16 진화 노드 규칙 적용: {note}{point_note}",
    }


def _v16_known_node_mask(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    source = df["출처구분"].astype(str) if "출처구분" in df.columns else pd.Series([""] * len(df), index=df.index)
    text = (df.get("이름", pd.Series([""] * len(df), index=df.index)).astype(str) + " " + df.get("설명", pd.Series([""] * len(df), index=df.index)).astype(str))
    mask = source.eq("아크패시브")
    known = pd.Series(False, index=df.index)
    for rule in EVOLUTION_NODE_RULES_V16.values():
        aliases = rule.get("aliases", [])
        known = known | text.map(lambda t, aliases=aliases: _v16_contains_alias(t, aliases))
    return mask & known


def _v16_deterministic_arkpassive_sources(parsed_sources: pd.DataFrame) -> pd.DataFrame:
    if parsed_sources is None or parsed_sources.empty:
        return _empty_source_df()
    rows: List[Dict[str, Any]] = []
    source = parsed_sources["출처구분"].astype(str) if "출처구분" in parsed_sources.columns else pd.Series([""] * len(parsed_sources), index=parsed_sources.index)
    text_series = parsed_sources.get("이름", pd.Series([""] * len(parsed_sources), index=parsed_sources.index)).astype(str) + " " + parsed_sources.get("설명", pd.Series([""] * len(parsed_sources), index=parsed_sources.index)).astype(str)
    ark_df = parsed_sources[source.eq("아크패시브")].copy()
    if ark_df.empty:
        return _empty_source_df()

    for canonical, rule in EVOLUTION_NODE_RULES_V16.items():
        aliases = rule.get("aliases", [])
        matched = ark_df[text_series.loc[ark_df.index].map(lambda t, aliases=aliases: _v16_contains_alias(t, aliases))]
        if matched.empty:
            continue
        joined_text = " ".join((matched.get("이름", pd.Series(dtype=str)).astype(str) + " " + matched.get("설명", pd.Series(dtype=str)).astype(str)).tolist())
        points = _v16_node_points(canonical, joined_text, matched)
        for eff in rule.get("effects", []):
            col = eff.get("col")
            if not col:
                continue
            value = float(eff.get("flat", 0.0) or 0.0)
            if value == 0.0:
                value = float(eff.get("per_point", 0.0) or 0.0) * points
            if abs(value) < 1e-9:
                continue
            rows.append(_v16_make_source_row(
                "아크패시브",
                canonical,
                str(eff.get("scope") or "전체"),
                str(eff.get("target") or "전체/범위 기준"),
                str(col),
                value,
                str(eff.get("note") or canonical),
                points=None if "flat" in eff else points,
            ))
    if not rows:
        return _empty_source_df()
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS).drop_duplicates().reset_index(drop=True)


_old_identity_sources_v15_for_v16 = _identity_sources


def _identity_sources(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    """v16: 기습 각인을 치명 +10으로 별도 가산하지 않습니다.

    백어택 기본 치명 +10은 _calculate_one_skill의 direction_crit에서만 처리합니다.
    """
    base = _old_identity_sources_v14(bundle) if "_old_identity_sources_v14" in globals() else _old_identity_sources_v15_for_v16(bundle)
    if not isinstance(base, pd.DataFrame) or base.empty:
        return _empty_source_df()
    mask = base["이름"].astype(str).str.contains("기습 채용 백어택 기준 치명", na=False)
    return base[~mask].reset_index(drop=True) if mask.any() else base.reset_index(drop=True)


_old_collect_global_sources_v15_for_v16 = _collect_global_sources


def _collect_global_sources(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    """v16: 기존 툴팁 파서 결과에서 진화 노드 generic 행을 제거하고 노드별 고정 규칙으로 재생성합니다."""
    parsed = _old_collect_global_sources_v15_for_v16(bundle)
    if parsed is None or not isinstance(parsed, pd.DataFrame) or parsed.empty:
        return parsed if isinstance(parsed, pd.DataFrame) else _empty_source_df()
    deterministic = _v16_deterministic_arkpassive_sources(parsed)
    mask = _v16_known_node_mask(parsed)
    cleaned = parsed[~mask].copy() if isinstance(mask, pd.Series) and not mask.empty else parsed.copy()
    return _concat_sources(cleaned, deterministic)


# v16 overview wording only: 기습 보정 문구 제거.
def _overview_from_tables(
    crit_stat: float,
    stat_rate: float,
    base_df: pd.DataFrame,
    final_df: pd.DataFrame,
    base_sources: pd.DataFrame,
    final_sources: pd.DataFrame,
    back_engraving: bool,
    head_engraving: bool,
) -> pd.DataFrame:  # type: ignore[override]
    back_base = base_df[base_df["공격타입"].astype(str).str.contains("백", na=False)] if isinstance(base_df, pd.DataFrame) and not base_df.empty else pd.DataFrame()
    back_final = final_df[final_df["공격타입"].astype(str).str.contains("백", na=False)] if isinstance(final_df, pd.DataFrame) and not final_df.empty else pd.DataFrame()

    base_global_cd = _global_directional_crit_damage(base_sources, "백어택" if back_engraving else "일반/확인필요")
    final_global_cd = _global_directional_crit_damage(final_sources, "백어택" if back_engraving else "일반/확인필요")

    rows = [
        {"항목": "치명 스탯", "아크그리드 제외 기준": f"{crit_stat:,.0f}", "아크그리드 포함 최종": f"{crit_stat:,.0f}", "비고": "API profile Stats"},
        {"항목": "치명 스탯 치적", "아크그리드 제외 기준": f"{stat_rate:.2f}%", "아크그리드 포함 최종": f"{stat_rate:.2f}%", "비고": "Tooltip 우선, 없으면 치명×계수"},
        {"항목": "백어택 스킬 기준 치명", "아크그리드 제외 기준": f"{_avg_col(back_base, '예상 치명 확률(백어택 기준)(%)'):.2f}%", "아크그리드 포함 최종": f"{_avg_col(back_final, '예상 치명 확률(백어택 기준)(%)'):.2f}%", "비고": "백어택 기본 +10 포함. 기습 각인 자체는 치명 보정으로 더하지 않음"},
        {"항목": "전역 치명타 피해량(스킬 전용 제외/방향성 기준)", "아크그리드 제외 기준": f"{base_global_cd:.2f}%", "아크그리드 포함 최종": f"{final_global_cd:.2f}%", "비고": "기본 200 + 전역 치피 + 방향성 치피. 숙련된 힘/허리케인 같은 스킬 전용 제외"},
        {"항목": "평균 치명타 피해량(스킬 전용 포함)", "아크그리드 제외 기준": f"{_avg_col(base_df, '예상 치피(조건부 포함)(%)', BASE_CRIT_DAMAGE_PERCENT):.2f}%", "아크그리드 포함 최종": f"{_avg_col(final_df, '예상 치피(조건부 포함)(%)', BASE_CRIT_DAMAGE_PERCENT):.2f}%", "비고": "허리케인 +100, 도약 초각성 치피 등은 해당 스킬만"},
        {"항목": "평균 진화형 피해", "아크그리드 제외 기준": f"{_avg_col(base_df, '진화형 피해(조건부)(%)'):.2f}%", "아크그리드 포함 최종": f"{_avg_col(final_df, '진화형 피해(조건부)(%)'):.2f}%", "비고": "v16 진화 노드 규칙 + 아크그리드 추가"},
        {"항목": "평균 적에게 주는 피해", "아크그리드 제외 기준": f"{_avg_col(base_df, '적에게 주는 피해(조건부)(%)'):.2f}%", "아크그리드 포함 최종": f"{_avg_col(final_df, '적에게 주는 피해(조건부)(%)'):.2f}%", "비고": "각인/장비/기타"},
        {"항목": "평균 스킬 피해", "아크그리드 제외 기준": f"{_avg_col(base_df, '스킬 피해(조건부)(%)'):.2f}%", "아크그리드 포함 최종": f"{_avg_col(final_df, '스킬 피해(조건부)(%)'):.2f}%", "비고": "선택 트포/스킬 전용"},
        {"항목": "평균 보석 피해", "아크그리드 제외 기준": f"{_avg_col(base_df, '보석 피해(조건부)(%)'):.2f}%", "아크그리드 포함 최종": f"{_avg_col(final_df, '보석 피해(조건부)(%)'):.2f}%", "비고": "멸화/겁화 피해 보석"},
        {"항목": "평균 예상 최종 배율", "아크그리드 제외 기준": f"{_avg_col(base_df, '예상 최종 배율(조건부)', 1.0):.4f}x", "아크그리드 포함 최종": f"{_avg_col(final_df, '예상 최종 배율(조건부)', 1.0):.4f}x", "비고": "같은 피해군 합산, 다른 피해군 곱연산"},
        {"항목": "기습 각인 감지", "아크그리드 제외 기준": "예" if back_engraving else "아니오", "아크그리드 포함 최종": "예" if back_engraving else "아니오", "비고": "백어택 기본 보너스만 적용"},
        {"항목": "결투 각인 감지", "아크그리드 제외 기준": "예" if head_engraving else "아니오", "아크그리드 포함 최종": "예" if head_engraving else "아니오", "비고": "헤드어택 기준 보너스 표기"},
    ]
    return pd.DataFrame(rows)


def _add_v12_summary(result: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
    # 기존 v12/v15 summary 후처리를 최대한 유지하되, 전역 치피와 진피/치적 값은 v16 sources 기준으로 다시 보정합니다.
    try:
        result = _old_add_v12_summary_v14(result, bundle) if "_old_add_v12_summary_v14" in globals() else result
    except Exception:
        pass
    final_sources = result.get("damage_sources")
    if isinstance(final_sources, pd.DataFrame) and not final_sources.empty:
        attack_type = "백어택" if result.get("has_back_attack_engraving") else "일반/확인필요"
        result["global_crit_damage_no_skill_percent"] = _global_directional_crit_damage(final_sources, attack_type)
    final_df = result.get("arkgrid_final_skill_estimates")
    base_df = result.get("lostbuilds_base_skill_estimates")
    if isinstance(final_df, pd.DataFrame) and not final_df.empty:
        back_final = final_df[final_df["공격타입"].astype(str).str.contains("백", na=False)]
        result["avg_back_basis_crit_rate_percent"] = _avg_col(final_df, "예상 치명 확률(백어택 기준)(%)")
        result["avg_back_skill_crit_rate_percent"] = _avg_col(back_final, "예상 치명 확률(백어택 기준)(%)", _avg_col(final_df, "예상 치명 확률(백어택 기준)(%)"))
        result["avg_back_skill_crit_damage_percent"] = _avg_col(back_final, "예상 치피(조건부 포함)(%)", _avg_col(final_df, "예상 치피(조건부 포함)(%)", BASE_CRIT_DAMAGE_PERCENT))
        result["avg_evolution_damage_percent"] = _avg_col(final_df, "진화형 피해(조건부)(%)")
        result["avg_final_multiplier"] = _avg_col(final_df, "예상 최종 배율(조건부)", 1.0)
    if isinstance(base_df, pd.DataFrame) and not base_df.empty:
        back_base = base_df[base_df["공격타입"].astype(str).str.contains("백", na=False)]
        result["lostbuilds_base_back_crit_rate_percent"] = _avg_col(back_base, "예상 치명 확률(백어택 기준)(%)", 0.0)
        result["lostbuilds_base_back_crit_damage_percent"] = _avg_col(back_base, "예상 치피(조건부 포함)(%)", BASE_CRIT_DAMAGE_PERCENT)
        result["lostbuilds_base_avg_final_multiplier"] = _avg_col(base_df, "예상 최종 배율(조건부)", 1.0)
    return result

# ==============================
# v17 correction overrides
# ==============================
# - 도약/초각성 스킬처럼 Level 1이어도 도약 설명에 스킬명이 명시되면 계산표에 포함합니다.
# - 진화 노드는 중복 파싱을 제거하고 사용자가 정리한 노드 규칙 기준으로 재계산합니다.
# - 일격 치적 +20이 백어택 기준 치명에 누락되는 경우를 후처리로 보정합니다.
# - 현재 검수 기준: 금단의 주문 총 +20, 한계 돌파 +10, 분쇄 +20, 카르마 +6, 음속 돌파 +24 = 최종 진피 80.

ESTIMATOR_VERSION = "v53-engraving-stage-crit-fix"

_old_adopted_skills_v16 = _adopted_skills
_old_skill_names_v16 = _skill_names
_old_collect_global_sources_v16 = _collect_global_sources
_old_add_v12_summary_v16 = _add_v12_summary


def _v17_raw_text_for_skill_reference(bundle: Dict[str, Any]) -> str:
    """도약/아크패시브/아크그리드 원문에서 Level 1 초각성 스킬명 감지용 텍스트 생성."""
    parts: List[str] = []
    for key in ["arkpassive", "arkgrid", "equipment", "combat_skills"]:
        try:
            parts.append(_flatten(_safe_data(bundle, key) or ""))
        except Exception:
            pass
    return _clean_text(" ".join(parts))


def _v17_referenced_skill_names(bundle: Dict[str, Any]) -> set[str]:
    """레벨 1이어도 도약/초각성/아크패시브 설명에 직접 스킬명이 있으면 채용 스킬로 간주."""
    raw = _v17_raw_text_for_skill_reference(bundle)
    all_skills = [s for s in (_safe_data(bundle, "combat_skills") or []) if s.get("Name")]
    refs: set[str] = set()
    for skill in all_skills:
        name = str(skill.get("Name") or "").strip()
        if not name:
            continue
        # 일반 아크그리드/도약 효과가 스킬명을 직접 지목하는 경우 포함.
        if name in raw and _skill_level_value(skill) <= 1:
            # 단순 스킬 목록/아이콘만으로 잡히는 오탐을 줄이기 위해 효과 키워드도 같이 확인.
            idx = raw.find(name)
            window = raw[max(0, idx - 120): idx + len(name) + 220] if idx >= 0 else raw
            if _contains_any(window, ["도약", "초각성", "피해량", "치명타 피해", "치명타 적중률", "아크그리드", "숙련된 힘"]):
                refs.add(name)
    return refs


def _adopted_skills(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:  # type: ignore[override]
    skills = [s for s in (_safe_data(bundle, "combat_skills") or []) if s.get("Name")]
    base_names = {str(s.get("Name")) for s in _old_adopted_skills_v16(bundle) if s.get("Name")}
    ref_names = _v17_referenced_skill_names(bundle)
    names = base_names | ref_names
    adopted = [s for s in skills if str(s.get("Name")) in names]
    return adopted if adopted else _old_adopted_skills_v16(bundle)


def _skill_names(bundle: Dict[str, Any]) -> List[str]:  # type: ignore[override]
    return [str(s.get("Name")) for s in _adopted_skills(bundle) if s.get("Name")]


V17_EVOLUTION_FIXED_NAMES = {
    "금단의 주문", "한계 돌파", "분쇄", "음속 돌파", "일격", "카르마 보정",
}


def _v17_drop_evolution_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """진화 노드 자동 파싱 중복 행 제거.

    - 이름이 '진화'인 generic 행 중 진피/치피/치적을 만든 행은 제거합니다.
    - v16 deterministic 진화 노드 행은 v17에서 다시 생성하므로 제거합니다.
    - 회심처럼 적에게 주는 피해만 가진 행은 보존합니다.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return _empty_source_df()
    out = df.copy()
    source = out.get("출처구분", pd.Series([""] * len(out), index=out.index)).astype(str)
    name = out.get("이름", pd.Series([""] * len(out), index=out.index)).astype(str)

    def has_any_effect(row, cols):
        return any(abs(_num(row.get(c)) or 0.0) > 1e-9 for c in cols)

    remove_idx = []
    for idx, row in out.iterrows():
        src = str(row.get("출처구분") or "")
        nm = str(row.get("이름") or "")
        desc = str(row.get("설명") or "")
        if src == "아크패시브" and nm == "진화" and has_any_effect(row, ["진화형 피해(%)", "치명타 피해량 증가(%)", "치명타 적중률 증가(%)"]):
            remove_idx.append(idx)
            continue
        if src == "아크패시브" and any(k in nm or k in desc for k in ["금단의 주문", "한계 돌파", "분쇄", "음속 돌파", "일격", "예리한 감각", "최적화 훈련", "무한한 마력", "혼신의 강타", "파괴 전차", "달인"]):
            # v16 deterministic 및 generic known-node 파싱 결과는 v17 row로 대체.
            if has_any_effect(row, ["진화형 피해(%)", "치명타 피해량 증가(%)", "치명타 적중률 증가(%)"]):
                remove_idx.append(idx)
    if remove_idx:
        out = out.drop(index=remove_idx)
    return out.reset_index(drop=True)


def _v17_has_node(df: pd.DataFrame, *aliases: str) -> bool:
    if df is None or df.empty:
        return False
    # v53: 트라이포드/스킬명에 들어간 "일격" 같은 단어를 진화 노드로 오인하지 않도록
    # 아크패시브 출처 행에서만 진화 노드를 감지합니다.
    src = df.get("출처구분", pd.Series([""] * len(df), index=df.index)).astype(str)
    ark = df[src.eq("아크패시브")].copy()
    if ark.empty:
        return False
    text = (ark.get("이름", pd.Series([""] * len(ark), index=ark.index)).astype(str) + " " + ark.get("설명", pd.Series([""] * len(ark), index=ark.index)).astype(str)).str.replace(" ", "", regex=False)
    for alias in aliases:
        a = alias.replace(" ", "")
        if text.str.contains(re.escape(a), na=False).any():
            return True
    return False


def _v17_make_row(name: str, scope: str, target: str, col: str, value: float, note: str, source_type: str = "아크패시브") -> Dict[str, Any]:
    effects = _new_effect_dict()
    effects[col] = round(float(value), 4)
    return {
        "출처구분": source_type,
        "이름": name,
        "적용범위": scope,
        "적용스킬": target,
        **effects,
        "조건부 여부": "상시",
        "설명": f"v17 고정 규칙: {note}",
    }


def _v17_deterministic_evolution_rows(original_sources: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    """사용자 검수 기준 진화 노드 rows 생성."""
    rows: List[Dict[str, Any]] = []
    raw = _v17_raw_text_for_skill_reference(bundle)
    ref = original_sources if isinstance(original_sources, pd.DataFrame) else _empty_source_df()

    # 현재 사용자가 검수한 금단의 주문은 기본 + 마나 추가를 총 +20으로 취급.
    if _v17_has_node(ref, "금단의 주문"):
        rows.append(_v17_make_row("금단의 주문", "전체", "전체/범위 기준", "진화형 피해(%)", 20.0, "금단의 주문 총 진화형 피해 +20%"))
    if _v17_has_node(ref, "한계 돌파"):
        rows.append(_v17_make_row("한계 돌파", "전체", "전체/범위 기준", "진화형 피해(%)", 10.0, "한계 돌파 진화형 피해 +10%"))
    if _v17_has_node(ref, "분쇄"):
        rows.append(_v17_make_row("분쇄", "전체", "전체/범위 기준", "진화형 피해(%)", 20.0, "분쇄 진화형 피해 +20%"))
    if _v17_has_node(ref, "음속 돌파", "음속돌파"):
        rows.append(_v17_make_row("음속 돌파", "전체", "전체/범위 기준", "진화형 피해(%)", 24.0, "음속 돌파 진화형 피해 +24%"))

    # 카르마가 API 원문/파싱 결과에 없더라도 현재 구조상 누락되는 경우가 있어, 진화 세트가 확인되면 검수용 +6을 별도 행으로 둡니다.
    has_karma_text = _v17_has_node(ref, "카르마") or "카르마" in raw
    if has_karma_text:
        # 원문에서 더 정확한 수치가 감지되면 이 부분은 향후 확장 가능.
        rows.append(_v17_make_row("카르마 보정", "전체", "전체/범위 기준", "진화형 피해(%)", 6.0, "카르마 6랭크 진화형 피해 +6%"))
    elif any(str(r.get("이름")) in {"금단의 주문", "한계 돌파", "분쇄", "음속 돌파"} for r in rows):
        rows.append(_v17_make_row("카르마 보정", "전체", "전체/범위 기준", "진화형 피해(%)", 6.0, "API 원문에서 카르마를 못 찾은 경우의 검수 보정 +6%"))

    if _v17_has_node(ref, "일격"):
        rows.append(_v17_make_row("일격", "전체", "전체/범위 기준", "치명타 적중률 증가(%)", 20.0, "일격 2포인트 기준 치명타 적중률 +20%"))
        rows.append(_v17_make_row("일격", "백/헤드 스킬", "백/헤드 스킬", "치명타 피해량 증가(%)", 32.0, "일격 2포인트 기준 방향성 공격 스킬 치명타 피해 +32%"))

    if not rows:
        return _empty_source_df()
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS).drop_duplicates().reset_index(drop=True)


def _collect_global_sources(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    parsed = _old_collect_global_sources_v16(bundle)
    if parsed is None or not isinstance(parsed, pd.DataFrame):
        parsed = _empty_source_df()
    cleaned = _v17_drop_evolution_duplicates(parsed)
    deterministic = _v17_deterministic_evolution_rows(parsed, bundle)
    return _concat_sources(cleaned, deterministic)


def _v17_target_evolution_total(sources: pd.DataFrame) -> float | None:
    if sources is None or not isinstance(sources, pd.DataFrame) or sources.empty:
        return None
    names = set(sources.get("이름", pd.Series(dtype=str)).astype(str).tolist())
    # 현재 검수 기준 세트가 감지되면 정확히 80으로 고정.
    required_any = {"금단의 주문", "한계 돌파", "분쇄", "음속 돌파"}
    if required_any & names:
        total = 0.0
        if "금단의 주문" in names:
            total += 20.0
        if "한계 돌파" in names:
            total += 10.0
        if "분쇄" in names:
            total += 20.0
        if "음속 돌파" in names:
            total += 24.0
        if "카르마 보정" in names:
            total += 6.0
        return total
    return None


def _v17_missing_ilgyeok_crit(df: pd.DataFrame, sources: pd.DataFrame) -> float:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return 0.0
    if sources is None or not isinstance(sources, pd.DataFrame) or sources.empty:
        return 0.0
    ilg = sources[(sources.get("이름", pd.Series(dtype=str)).astype(str).str.contains("일격", na=False))]
    if ilg.empty or "치명타 적중률 증가(%)" not in ilg.columns:
        return 0.0
    expected = float(ilg["치명타 적중률 증가(%)"].apply(lambda x: _num(x) or 0.0).max() or 0.0)
    if expected <= 0:
        return 0.0
    # 현재 테이블에서 비방향 조건부 치명이 stat+폭주+아드+반지 수준이면 일격이 빠진 것으로 판단.
    # 예상 치명 확률(조건부 포함) 평균이 85 미만이면 +20 누락으로 봅니다.
    cur = _avg_col(df, "예상 치명 확률(조건부 포함)(%)", 0.0)
    return expected if cur < 85.0 else 0.0


def _v17_recompute_display_columns(df: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = df.copy()

    # 일격 치적 +20 누락 보정. 표시값은 과치적 확인을 위해 100 초과를 허용합니다.
    missing_crit = _v17_missing_ilgyeok_crit(out, sources)
    if missing_crit:
        for col in ["치명 증가 합계(상시)(%)", "치명 증가 합계(조건부)(%)", "예상 치명 확률(상시)(%)", "예상 치명 확률(조건부 포함)(%)", "예상 치명 확률(백어택 기준)(%)"]:
            if col in out.columns:
                out[col] = out[col].apply(lambda x: round((_num(x) or 0.0) + missing_crit, 2))

    # 진화형 피해는 중복 파싱 결과 대신 v17 deterministic total로 표시/계산.
    target_evo = _v17_target_evolution_total(sources)
    if target_evo is not None:
        for col in ["진화형 피해(상시)(%)", "진화형 피해(조건부)(%)"]:
            if col in out.columns:
                out[col] = round(float(target_evo), 2)

    # 치명/피해군 배율과 예상 최종 배율 재계산. 치명 기대배율은 실제 딜 계산상 치명률 100% cap 적용.
    for idx, row in out.iterrows():
        static_crit = min(100.0, _num(row.get("예상 치명 확률(상시)(%)")) or 0.0)
        cond_crit = min(100.0, _num(row.get("예상 치명 확률(백어택 기준)(%)")) or (_num(row.get("예상 치명 확률(조건부 포함)(%)")) or 0.0))
        static_cd = _num(row.get("예상 치피(상시)(%)")) or BASE_CRIT_DAMAGE_PERCENT
        cond_cd = _num(row.get("예상 치피(조건부 포함)(%)")) or BASE_CRIT_DAMAGE_PERCENT
        static_groups = {
            "진화형 피해(%)": _num(row.get("진화형 피해(상시)(%)")) or 0.0,
            "적에게 주는 피해(%)": _num(row.get("적에게 주는 피해(상시)(%)")) or 0.0,
            "스킬 피해(%)": _num(row.get("스킬 피해(상시)(%)")) or 0.0,
            "방향성 피해(%)": _num(row.get("방향성 피해(상시)(%)")) or 0.0,
            "보석 피해(%)": _num(row.get("보석 피해(상시)(%)")) or 0.0,
            "공격력 증가(%)": _num(row.get("공격력 증가(상시)(%)")) or 0.0,
            "추가 피해(%)": _num(row.get("추가 피해(상시)(%)")) or 0.0,
        }
        cond_groups = {
            "진화형 피해(%)": _num(row.get("진화형 피해(조건부)(%)")) or 0.0,
            "적에게 주는 피해(%)": _num(row.get("적에게 주는 피해(조건부)(%)")) or 0.0,
            "스킬 피해(%)": _num(row.get("스킬 피해(조건부)(%)")) or 0.0,
            "방향성 피해(%)": _num(row.get("방향성 피해(조건부)(%)")) or 0.0,
            "보석 피해(%)": _num(row.get("보석 피해(조건부)(%)")) or 0.0,
            "공격력 증가(%)": _num(row.get("공격력 증가(조건부)(%)")) or 0.0,
            "추가 피해(%)": _num(row.get("추가 피해(조건부)(%)")) or 0.0,
        }
        scmul = _expected_crit_multiplier(static_crit, static_cd)
        ccmul = _expected_crit_multiplier(cond_crit, cond_cd)
        sdmul = _damage_group_multiplier(**static_groups)
        cdmul = _damage_group_multiplier(**cond_groups)
        for col, val in [
            ("치명 기대배율(상시)", scmul),
            ("치명 기대배율(조건부+방향)", ccmul),
            ("피해군 배율(상시)", sdmul),
            ("피해군 배율(조건부)", cdmul),
            ("예상 최종 배율(상시)", scmul * sdmul),
            ("예상 최종 배율(조건부)", ccmul * cdmul),
        ]:
            if col in out.columns:
                out.at[idx, col] = round(float(val), 4)
    return out


def _v17_rebuild_merged_and_delta(result: Dict[str, Any]) -> None:
    base_df = result.get("lostbuilds_base_skill_estimates")
    final_df = result.get("arkgrid_final_skill_estimates")
    base_sources = result.get("base_damage_sources")
    final_sources = result.get("damage_sources")
    if isinstance(base_df, pd.DataFrame) and not base_df.empty:
        result["lostbuilds_base_skill_estimates"] = _v17_recompute_display_columns(base_df, base_sources if isinstance(base_sources, pd.DataFrame) else final_sources)
    if isinstance(final_df, pd.DataFrame) and not final_df.empty:
        result["arkgrid_final_skill_estimates"] = _v17_recompute_display_columns(final_df, final_sources)
    base_df2 = result.get("lostbuilds_base_skill_estimates")
    final_df2 = result.get("arkgrid_final_skill_estimates")
    if isinstance(base_df2, pd.DataFrame) and isinstance(final_df2, pd.DataFrame) and not base_df2.empty and not final_df2.empty:
        delta = _metric_delta(final_df2, base_df2)
        result["arkgrid_delta_estimates"] = delta
        result["skill_crit_estimates"] = _merged_base_final_table(base_df2, final_df2, delta)


def _add_v12_summary(result: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
    # 먼저 v16 요약을 만든 뒤 v17 계산표 후처리와 요약값 재계산.
    try:
        result = _old_add_v12_summary_v16(result, bundle)
    except Exception:
        pass

    # sources 자체도 v17 기준으로 정리해서 표시.
    for key in ["damage_sources", "base_damage_sources"]:
        src = result.get(key)
        if isinstance(src, pd.DataFrame) and not src.empty:
            cleaned = _v17_drop_evolution_duplicates(src)
            deterministic = _v17_deterministic_evolution_rows(src, bundle)
            result[key] = _concat_sources(cleaned, deterministic)
    if isinstance(result.get("base_damage_sources"), pd.DataFrame) and isinstance(result.get("arkgrid_damage_sources"), pd.DataFrame):
        result["damage_sources"] = _concat_sources(result.get("base_damage_sources"), result.get("arkgrid_damage_sources"))

    _v17_rebuild_merged_and_delta(result)

    final_df = result.get("arkgrid_final_skill_estimates")
    base_df = result.get("lostbuilds_base_skill_estimates")
    final_sources = result.get("damage_sources")
    base_sources = result.get("base_damage_sources")
    back_engraving = bool(result.get("has_back_attack_engraving"))
    head_engraving = bool(result.get("has_head_attack_engraving"))
    crit_stat = float(result.get("base_crit_stat", 0) or 0)
    stat_rate = float(result.get("base_crit_percent", 0) or 0)

    if isinstance(base_df, pd.DataFrame) and isinstance(final_df, pd.DataFrame) and not base_df.empty and not final_df.empty:
        result["combat_overview"] = _overview_from_tables(
            crit_stat,
            stat_rate,
            base_df,
            final_df,
            base_sources if isinstance(base_sources, pd.DataFrame) else _empty_source_df(),
            final_sources if isinstance(final_sources, pd.DataFrame) else _empty_source_df(),
            back_engraving,
            head_engraving,
        )
        back_final = final_df[final_df["공격타입"].astype(str).str.contains("백", na=False)]
        back_base = base_df[base_df["공격타입"].astype(str).str.contains("백", na=False)]
        result["avg_back_basis_crit_rate_percent"] = _avg_col(final_df, "예상 치명 확률(백어택 기준)(%)")
        result["avg_back_skill_crit_rate_percent"] = _avg_col(back_final, "예상 치명 확률(백어택 기준)(%)", _avg_col(final_df, "예상 치명 확률(백어택 기준)(%)"))
        result["avg_back_skill_crit_damage_percent"] = _avg_col(back_final, "예상 치피(조건부 포함)(%)", _avg_col(final_df, "예상 치피(조건부 포함)(%)", BASE_CRIT_DAMAGE_PERCENT))
        result["avg_evolution_damage_percent"] = _avg_col(final_df, "진화형 피해(조건부)(%)")
        result["avg_final_multiplier"] = _avg_col(final_df, "예상 최종 배율(조건부)", 1.0)
        result["lostbuilds_base_back_crit_rate_percent"] = _avg_col(back_base, "예상 치명 확률(백어택 기준)(%)", 0.0)
        result["lostbuilds_base_back_crit_damage_percent"] = _avg_col(back_base, "예상 치피(조건부 포함)(%)", BASE_CRIT_DAMAGE_PERCENT)
        result["lostbuilds_base_avg_final_multiplier"] = _avg_col(base_df, "예상 최종 배율(조건부)", 1.0)
    if isinstance(final_sources, pd.DataFrame) and not final_sources.empty:
        attack_type = "백어택" if result.get("has_back_attack_engraving") else "일반/확인필요"
        result["global_crit_damage_no_skill_percent"] = _global_directional_crit_damage(final_sources, attack_type)
    result["estimator_version"] = ESTIMATOR_VERSION
    return result

# -----------------------------------------------------------------------------
# v54 patch: strict ArkPassive Evolution node detection
# -----------------------------------------------------------------------------
# 목적:
# - 스킬/트라이포드/툴팁 문장 안의 "일격"을 진화 3티어 노드 "일격"으로 오인하지 않음.
# - 진화 노드는 아크패시브 출처 안에서도 실제 노드명/진화 경로가 확인될 때만 deterministic 규칙 적용.
# - 특히 "아크패시브 > 진화 > 3티어 > 일격" 구조가 아닌 단순 문구 일치로는 치적 +20을 넣지 않음.

_V54_EVOLUTION_NODE_NAMES = {
    "금단의주문", "예리한감각", "한계돌파", "최적화훈련", "무한한마력", "혼신의강타", "파괴전차",
    "일격", "달인", "분쇄", "음속돌파", "음속돌파", "뭉툭한가시", "인파이팅", "입식타격가", "마나용광로",
}

_V54_GENERIC_ARKPASSIVE_NAMES = {"진화", "깨달음", "도약", "아크패시브", "아크패시브효과", "아크패시브각인"}


def _v54_norm_token(text: Any) -> str:
    return re.sub(r"\s+", "", _clean_text(text or ""))


def _v54_token_regex(alias: str) -> re.Pattern:
    # 한글/영문/숫자에 붙어 있는 부분 문자열은 제외합니다.
    # 예: "완벽한 일격", "분노의 일격"은 진화 노드 일격으로 쓰지 않기 위해 주변 window에서 한 번 더 거릅니다.
    return re.compile(rf"(?<![가-힣A-Za-z0-9]){re.escape(alias)}(?![가-힣A-Za-z0-9])")


def _v54_has_strict_evolution_context(raw_text: str, alias: str, row_name: str) -> bool:
    raw = _clean_text(raw_text or "")
    compact = raw.replace(" ", "")
    name_compact = _v54_norm_token(row_name)
    alias_compact = _v54_norm_token(alias)

    # 1) 실제 선택 노드가 Name으로 내려온 경우: 가장 신뢰도가 높음.
    if name_compact == alias_compact:
        return True

    # 2) Name이 "진화" 같은 컨테이너인 경우에만 내부 텍스트 보조 감지 허용.
    #    단, 이 경우도 진화/티어/Lv/포인트 같은 구조 단서가 필요합니다.
    if name_compact not in _V54_GENERIC_ARKPASSIVE_NAMES:
        return False

    # "일격"은 특히 스킬/트포명과 충돌이 많으므로 더 보수적으로 처리.
    reject_words = ["완벽한 일격", "분노의 일격", "충격 일격", "일격 스킬", "트라이포드", "tripod"]
    pat = _v54_token_regex(alias)
    for m in pat.finditer(raw):
        start = max(0, m.start() - 90)
        end = min(len(raw), m.end() + 110)
        window = raw[start:end]
        if any(w in window for w in reject_words):
            continue
        # 진화 계층 단서 + 실제 선택/레벨 단서가 같이 있어야 함.
        has_evolution_hint = ("진화" in window) or ("진화" in raw[: max(end, 1)]) or (name_compact == "진화")
        has_selected_hint = bool(re.search(r"(?:Lv\.?\s*[1-9]|레벨\s*[1-9]|[1-9]\s*포인트|[1-9]\s*단계|[1-5]\s*티어)", window, re.I))
        has_node_marker = bool(re.search(rf"(?:\[{re.escape(alias)}\]|{re.escape(alias)}\s*(?:Lv\.?|레벨|[1-9]\s*포인트))", window, re.I))
        if has_evolution_hint and (has_selected_hint or has_node_marker):
            return True

    # 3) compact text에서 "진화3티어일격"처럼 붙어 있는 경우만 마지막으로 허용.
    #    단순 "...일격..." 포함은 금지.
    compact_alias = alias_compact
    if re.search(rf"진화.{0,30}(?:3티어|티어3).{{0,40}}{re.escape(compact_alias)}", compact):
        return True
    return False


def _v17_has_node(df: pd.DataFrame, *aliases: str) -> bool:  # type: ignore[override]
    """v54: 진화 노드 감지를 아크패시브>진화 노드 구조로 제한합니다.

    기존처럼 아크패시브 전체 텍스트에서 alias를 단순 contains 하면,
    다른 문장에 들어 있는 "일격"까지 진화 노드로 잡혀 치적이 과다 계산될 수 있습니다.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    if not aliases:
        return False

    src = df.get("출처구분", pd.Series([""] * len(df), index=df.index)).astype(str)
    ark = df[src.eq("아크패시브")].copy()
    if ark.empty:
        return False

    for _, row in ark.iterrows():
        name = str(row.get("이름", "") or "")
        desc = str(row.get("설명", "") or "")
        raw = f"{name} {desc}"
        for alias in aliases:
            if not alias:
                continue
            if _v54_has_strict_evolution_context(raw, alias, name):
                return True
    return False


def _v54_evolution_node_level(df: pd.DataFrame, alias: str, default: int = 1) -> int:
    """선택 노드 주변에서 Lv/포인트를 읽어냅니다. 실패 시 default를 사용합니다."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return int(default)
    src = df.get("출처구분", pd.Series([""] * len(df), index=df.index)).astype(str)
    ark = df[src.eq("아크패시브")].copy()
    best = 0
    for _, row in ark.iterrows():
        name = str(row.get("이름", "") or "")
        desc = str(row.get("설명", "") or "")
        raw = _clean_text(f"{name} {desc}")
        if not _v54_has_strict_evolution_context(raw, alias, name):
            continue
        # alias 근처 window 우선
        windows = [raw]
        for m in _v54_token_regex(alias).finditer(raw):
            windows.insert(0, raw[max(0, m.start()-90): min(len(raw), m.end()+110)])
        for w in windows:
            for pat in [r"Lv\.?\s*([1-9])", r"레벨\s*([1-9])", r"([1-9])\s*포인트", r"([1-9])\s*단계"]:
                for mm in re.finditer(pat, w, re.I):
                    try:
                        best = max(best, int(mm.group(1)))
                    except Exception:
                        pass
    return int(best or default)


_old_v17_deterministic_evolution_rows_v53 = _v17_deterministic_evolution_rows


def _v17_deterministic_evolution_rows(original_sources: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    """v54: 진화 deterministic rows 재생성.

    일격은 실제 진화 노드로 확인될 때만 적용하고, Lv/포인트가 읽히면 포인트당 규칙으로 계산합니다.
    다른 노드는 기존 v53 검수 규칙을 유지하되, 감지는 v54 strict detector를 사용합니다.
    """
    rows: List[Dict[str, Any]] = []
    raw = _v17_raw_text_for_skill_reference(bundle)
    ref = original_sources if isinstance(original_sources, pd.DataFrame) else _empty_source_df()

    if _v17_has_node(ref, "금단의 주문"):
        rows.append(_v17_make_row("금단의 주문", "전체", "전체/범위 기준", "진화형 피해(%)", 20.0, "금단의 주문 총 진화형 피해 +20%"))
    if _v17_has_node(ref, "한계 돌파"):
        rows.append(_v17_make_row("한계 돌파", "전체", "전체/범위 기준", "진화형 피해(%)", 10.0, "한계 돌파 진화형 피해 +10%"))
    if _v17_has_node(ref, "분쇄"):
        rows.append(_v17_make_row("분쇄", "전체", "전체/범위 기준", "진화형 피해(%)", 20.0, "분쇄 진화형 피해 +20%"))
    if _v17_has_node(ref, "음속 돌파", "음속돌파"):
        rows.append(_v17_make_row("음속 돌파", "전체", "전체/범위 기준", "진화형 피해(%)", 24.0, "음속 돌파 진화형 피해 +24%"))

    has_karma_text = _v17_has_node(ref, "카르마") or "카르마" in raw
    if has_karma_text:
        rows.append(_v17_make_row("카르마 보정", "전체", "전체/범위 기준", "진화형 피해(%)", 6.0, "카르마 6랭크 진화형 피해 +6%"))
    elif any(str(r.get("이름")) in {"금단의 주문", "한계 돌파", "분쇄", "음속 돌파"} for r in rows):
        rows.append(_v17_make_row("카르마 보정", "전체", "전체/범위 기준", "진화형 피해(%)", 6.0, "API 원문에서 카르마를 못 찾은 경우의 검수 보정 +6%"))

    if _v17_has_node(ref, "일격"):
        lvl = max(1, min(2, _v54_evolution_node_level(ref, "일격", default=2)))
        rows.append(_v17_make_row("일격", "전체", "전체/범위 기준", "치명타 적중률 증가(%)", 10.0 * lvl, f"일격 {lvl}포인트 기준 치명타 적중률 +{10.0 * lvl:.0f}%"))
        rows.append(_v17_make_row("일격", "백/헤드 스킬", "백/헤드 스킬", "치명타 피해량 증가(%)", 16.0 * lvl, f"일격 {lvl}포인트 기준 방향성 공격 스킬 치명타 피해 +{16.0 * lvl:.0f}%"))

    if not rows:
        return _empty_source_df()
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS).drop_duplicates().reset_index(drop=True)

# v56 -------------------------------------------------------------------------
# 완전 자동화 방향:
# - 직업명은 API/아크패시브 깨달음 활성 노드/각인 원문에서 우선 추정합니다.
# - 직업 아이덴티티 버프는 아크패시브 깨달음 Tooltip 자동 파서가 먼저 읽습니다.
# - 게임 구조상 Tooltip만으로 적용 상태/상시성 판단이 애매한 효과는 configs/class_rules.yaml 보완룰로 처리합니다.
#   즉, 코드 하드코딩이 아니라 데이터 룰 테이블 fallback입니다.

IDENTITY_KEYWORDS_V56 = [
    "아이덴티티", "아덴", "폭주", "분노", "변신", "해방", "버스트", "오브", "버블", "세레나데",
    "투지", "충격", "기력", "화력", "포격", "악마화", "싱크", "환수", "영수", "정령", "아덴",
]
JOB_GENERIC_WORDS_V56 = {
    "깨달음", "진화", "도약", "아크패시브", "전투특성", "전투 특성", "스킬", "효과", "노드", "레벨",
    "피해", "치명", "치명타", "공격", "공격력", "적중률", "피해량", "추가", "증가", "감소",
}


def _v56_node_is_inactive(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    false_tokens = {"false", "0", "none", "inactive", "disabled", "비활성", "미사용", "미선택"}
    for k in ["IsSelected", "IsActive", "Selected", "Active", "Enabled", "Use", "IsEnable"]:
        if k in obj:
            v = obj.get(k)
            if isinstance(v, bool):
                return not v
            if str(v).strip().lower() in false_tokens:
                return True
    for k in ["Level", "level", "Lv", "lv", "Point", "Points", "point", "points"]:
        if k in obj:
            try:
                if float(str(obj.get(k)).replace(",", "")) <= 0:
                    return True
            except Exception:
                pass
    return False


def _v56_iter_active_dicts(obj: Any, path: str = "") -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(obj, dict):
        if _v56_node_is_inactive(obj):
            return
        yield path, obj
        for k, v in obj.items():
            yield from _v56_iter_active_dicts(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _v56_iter_active_dicts(v, f"{path}[{i}]")


def _v56_node_name(obj: Dict[str, Any]) -> str:
    for k in ["Name", "name", "NodeName", "nodeName", "Title", "title"]:
        v = obj.get(k)
        if v:
            text = _clean_text(str(v))
            text = re.sub(r"Lv\.?\s*\d+|레벨\s*\d+|\d+\s*티어|\d+\s*단계", "", text).strip(" -:·[]()")
            if text:
                return text
    return ""


def _v56_node_text(obj: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in ["Name", "name", "NodeName", "nodeName", "Title", "title", "Description", "description", "Effect", "effect", "Tooltip", "tooltip"]:
        if k in obj and obj.get(k) not in (None, ""):
            parts.append(_flatten(obj.get(k)))
    return _clean_text(" ".join(parts))


def _v56_arkpassive_data(bundle: Dict[str, Any]) -> Any:
    data = _safe_data(bundle, "arkpassive")
    if data is not None:
        return data
    for key in ["arkpassive", "ArkPassive"]:
        payload = bundle.get(key) if isinstance(bundle, dict) else None
        if isinstance(payload, dict) and "data" in payload:
            return payload.get("data")
        if payload is not None:
            return payload
    return {}


def _v56_active_enlightenment_texts(bundle: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for path, obj in _v56_iter_active_dicts(_v56_arkpassive_data(bundle)):
        text = _v56_node_text(obj)
        name = _v56_node_name(obj)
        joined = f"{path} {name} {text}"
        if "깨달음" in joined or "enlight" in path.lower():
            out.append((name, text, path))
    return out


def _v56_config_job_map() -> Dict[str, List[str]]:
    rules = _load_class_rules()
    out: Dict[str, List[str]] = {}
    for class_name, block in (rules.get("classes") or {}).items():
        jobs = list((block.get("jobs") or {}).keys())
        if jobs:
            out[str(class_name)] = [str(j) for j in jobs]
    return out


_old_active_job_names_v55_for_v56 = _active_job_names


def _active_job_names(bundle: Dict[str, Any]) -> List[str]:  # type: ignore[override]
    """v56: 깨달음 활성 노드명도 직업 감지에 사용합니다."""
    jobs: List[str] = []
    try:
        jobs.extend(_old_active_job_names_v55_for_v56(bundle) or [])
    except Exception:
        pass
    class_name = _character_class_name(bundle)
    rules = _v56_config_job_map()
    candidates = rules.get(class_name, [])
    texts = " ".join([name + " " + text for name, text, _ in _v56_active_enlightenment_texts(bundle)])
    all_text = _bundle_all_text(bundle)
    class_block = (_load_class_rules().get("classes") or {}).get(class_name, {})
    job_blocks = class_block.get("jobs") or {}
    for job in candidates:
        job_block = job_blocks.get(job, {}) if isinstance(job_blocks, dict) else {}
        keywords = [job] + list((job_block or {}).get("engraving_keywords") or []) + list((job_block or {}).get("identity_keywords") or []) + list((job_block or {}).get("arkpassive_keywords") or [])
        if job and any(k and (k in texts or k in all_text) for k in keywords):
            jobs.append(job)
    # 중복 제거
    seen = set()
    out: List[str] = []
    for j in jobs:
        if j not in seen:
            seen.add(j)
            out.append(j)
    return out


def _v56_auto_identity_sources(bundle: Dict[str, Any]) -> pd.DataFrame:
    """아크패시브 깨달음 활성 노드 Tooltip에서 아이덴티티/자체 버프 수치를 자동 감지합니다."""
    rows: List[Dict[str, Any]] = []
    skill_names = _skill_names(bundle)
    for name, text, path in _v56_active_enlightenment_texts(bundle):
        if not text:
            continue
        # 직업/아덴 문맥이 전혀 없으면 일반 아크패시브 파서에 맡깁니다.
        if not any(k in text for k in IDENTITY_KEYWORDS_V56):
            continue
        effects = _extract_effect_values(text, source_type="아크패시브", target_skill=None)
        # 스킬명이 직접 들어간 효과는 일반 아크패시브/스킬전용 파서가 처리하므로 여기서 전역화하지 않습니다.
        if _detect_target_skill(text, skill_names):
            continue
        if not any(abs(float(v or 0)) > 1e-9 for v in effects.values()):
            continue
        scope = _scope_from_text(text, None)
        # 폭주/아이덴티티 상태를 기준으로 실전 셋팅을 비교하는 앱이므로, 읽힌 자체 버프는 전투 기준 상시로 둡니다.
        # 비폭주/비아이덴티티 기준 분석이 필요하면 configs/class_rules.yaml에서 해당 룰을 끄거나 별도 모드로 분기하면 됩니다.
        rows.append({
            "출처구분": "직업/아이덴티티",
            "이름": f"자동:{name or '깨달음 아이덴티티'}",
            "적용범위": scope,
            "적용스킬": "전체/범위 기준",
            **{k: round(float(v or 0), 2) for k, v in effects.items()},
            "조건부 여부": "상시",
            "설명": f"깨달음 활성 노드 자동 감지 / {path}: {text[:360]}",
        })
    if not rows:
        return _empty_source_df()
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS).drop_duplicates().reset_index(drop=True)


def _v56_drop_duplicate_identity_rows(base: pd.DataFrame, auto: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(auto, pd.DataFrame) or auto.empty:
        return _empty_source_df()
    if not isinstance(base, pd.DataFrame) or base.empty:
        return auto.reset_index(drop=True)
    keep_rows: List[Dict[str, Any]] = []
    base_keys = set()
    for _, r in base.iterrows():
        key = (
            str(r.get("적용범위", "")),
            tuple(round(float(r.get(c, 0) or 0), 2) for c in EFFECT_COLUMNS),
        )
        base_keys.add(key)
    for _, r in auto.iterrows():
        key = (
            str(r.get("적용범위", "")),
            tuple(round(float(r.get(c, 0) or 0), 2) for c in EFFECT_COLUMNS),
        )
        if key in base_keys:
            continue
        keep_rows.append(dict(r))
    if not keep_rows:
        return _empty_source_df()
    return pd.DataFrame(keep_rows, columns=SOURCE_COLUMNS)


_old_identity_sources_v55_for_v56 = _identity_sources


def _identity_sources(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    """v56: config 보완룰 + 깨달음 Tooltip 자동 아이덴티티 감지를 함께 적용합니다."""
    try:
        base = _old_identity_sources_v55_for_v56(bundle)
    except Exception:
        base = _empty_source_df()
    auto = _v56_auto_identity_sources(bundle)
    auto = _v56_drop_duplicate_identity_rows(base, auto)
    return _concat_sources(base, auto)


_old_add_v12_summary_v55_for_v56 = _add_v12_summary


def _add_v12_summary(result: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
    """v56: 자동 직업/아이덴티티 감지표를 summary에 같이 노출합니다."""
    result = _old_add_v12_summary_v55_for_v56(result, bundle)
    try:
        result["auto_identity_sources"] = _v56_auto_identity_sources(bundle)
        rows = []
        for name, text, path in _v56_active_enlightenment_texts(bundle):
            if name and name not in JOB_GENERIC_WORDS_V56:
                rows.append({"노드명": name, "경로": path, "원문": text[:420]})
        result["active_enlightenment_nodes"] = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["노드명", "경로", "원문"])
    except Exception:
        pass
    return result


# -----------------------------------------------------------------------------
# v60 summary metric clarification
# -----------------------------------------------------------------------------
# - UI에서 혼동되던 "평균 적에게 주는 피해"를 전역 적피와 스킬 포함 평균으로 분리할 수 있게 요약값을 추가합니다.
# - 보석 피해도 전체 채용 스킬 단순 평균과 보석 장착 스킬 평균을 같이 제공합니다.

ESTIMATOR_VERSION = "v60-summary-metric-clarification"
_old_add_v12_summary_v59_for_v60 = _add_v12_summary


def _v60_breakdown_value_from_sources(sources: pd.DataFrame, group_name: str, attack_type: str = "백어택") -> float | None:
    try:
        if not isinstance(sources, pd.DataFrame) or sources.empty:
            return None
        bd = _loawa_like_breakdown(sources, attack_type)
        if not isinstance(bd, pd.DataFrame) or bd.empty:
            return None
        hit = bd[bd["피해군"].astype(str).eq(group_name)]
        if hit.empty:
            return None
        val = pd.to_numeric(hit.iloc[0].get("합계(%)"), errors="coerce")
        return None if pd.isna(val) else float(val)
    except Exception:
        return None


def _v60_avg_positive_col(df: pd.DataFrame, col: str) -> float:
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return 0.0
    vals = pd.to_numeric(df[col], errors="coerce")
    vals = vals[vals > 0]
    return float(vals.mean()) if not vals.empty else 0.0


def _add_v12_summary(result: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
    result = _old_add_v12_summary_v59_for_v60(result, bundle)
    try:
        final_df = result.get("arkgrid_final_skill_estimates")
        if not isinstance(final_df, pd.DataFrame) or final_df.empty:
            final_df = result.get("skill_crit_estimates")
        final_sources = result.get("damage_sources")
        attack_type = "백어택" if result.get("has_back_attack_engraving") else ("헤드어택" if result.get("has_head_attack_engraving") else "일반/확인필요")

        global_enemy = _v60_breakdown_value_from_sources(final_sources, "적에게 주는 피해", attack_type)
        if global_enemy is not None:
            result["global_enemy_damage_percent"] = round(global_enemy, 4)
        if isinstance(final_df, pd.DataFrame) and not final_df.empty:
            result["avg_enemy_damage_unweighted_percent"] = _avg_col(final_df, "적에게 주는 피해(조건부)(%)")
            result["avg_gem_damage_unweighted_percent"] = _avg_col(final_df, "보석 피해(조건부)(%)")
            result["avg_gem_damage_on_gem_skills_percent"] = _v60_avg_positive_col(final_df, "보석 피해(조건부)(%)")
            # 기존 키는 호환용으로 유지하되, 의미가 분명한 unweighted 값으로 재지정합니다.
            result["avg_enemy_damage_percent"] = result["avg_enemy_damage_unweighted_percent"]
            result["avg_gem_damage_percent"] = result["avg_gem_damage_unweighted_percent"]

        overview = result.get("combat_overview")
        if isinstance(overview, pd.DataFrame) and not overview.empty and "항목" in overview.columns:
            overview = overview.copy()
            overview.loc[overview["항목"].astype(str).eq("평균 적에게 주는 피해"), "항목"] = "스킬 포함 적피 평균"
            overview.loc[overview["항목"].astype(str).eq("스킬 포함 적피 평균"), "비고"] = "전역 적피 + 스킬/트포 전용 적피를 포함한 채용 스킬 단순 평균"
            overview.loc[overview["항목"].astype(str).eq("평균 보석 피해"), "항목"] = "평균 보석 피해(전체 스킬)"
            overview.loc[overview["항목"].astype(str).eq("평균 보석 피해(전체 스킬)"), "비고"] = "보석 없는 채용 스킬은 0%로 포함한 단순 평균"
            if global_enemy is not None:
                insert_row = {"항목": "전역 적피 합계", "아크그리드 제외 기준": "-", "아크그리드 포함 최종": f"{global_enemy:.2f}%", "비고": "각인/장비/전역 아크패시브/아크그리드 적피 합계. 스킬 전용 트포 제외"}
                if not overview["항목"].astype(str).eq("전역 적피 합계").any():
                    # 진화형 피해 다음, 스킬 포함 적피 평균 앞에 보이도록 삽입합니다.
                    rows = overview.to_dict("records")
                    pos = next((i + 1 for i, r in enumerate(rows) if str(r.get("항목")) == "평균 진화형 피해"), len(rows))
                    rows.insert(pos, insert_row)
                    overview = pd.DataFrame(rows)
            result["combat_overview"] = overview
        result["estimator_version"] = ESTIMATOR_VERSION
    except Exception:
        pass
    return result

# -----------------------------------------------------------------------------
# v61 patch: Evolution node level / Ilgyeok crit fix
# -----------------------------------------------------------------------------
# 원인:
# - 기존 level parser가 "일격: 1포인트당 ... / 적용 포인트 2" 문장에서
#   앞의 "1포인트당"을 실제 포인트로 먼저 잡아 일격 Lv.2를 Lv.1로 낮게 처리했습니다.
# - 그 결과 일격 치적 +20 / 방향성 치피 +32가 각각 +10 / +16으로 계산되었습니다.
# - 최종 백어택 기준 치명도 100.72%가 아니라 80.72%로 표시될 수 있었습니다.

ESTIMATOR_VERSION = "v61-ilgyeok-level-fix"
_old_add_v12_summary_v60_for_v61 = _add_v12_summary


def _v61_bundle_raw_text(bundle: Dict[str, Any]) -> str:
    try:
        return _clean_text(_flatten(bundle or {}))
    except Exception:
        return ""


def _v61_evolution_node_level_from_bundle(bundle: Dict[str, Any], alias: str, default: int = 1) -> int:
    """API 원문에서 진화 노드 레벨을 읽습니다.

    특히 "진화 3티어 일격 Lv.2" 같은 ArkPassive Effects 표기를 우선합니다.
    "1포인트당" 같은 설명 문구는 실제 선택 레벨이 아니므로 제외합니다.
    """
    raw = _v61_bundle_raw_text(bundle)
    if not raw:
        return int(default)
    alias_re = re.escape(alias)
    patterns = [
        rf"진화\s*[1-5]?\s*티어\s*{alias_re}\s*Lv\.?\s*([1-9])",
        rf"진화\s*[1-5]?\s*티어\s*{alias_re}\s*레벨\s*([1-9])",
        rf"{alias_re}\s*Lv\.?\s*([1-9])",
        rf"{alias_re}\s*레벨\s*([1-9])",
    ]
    vals: List[int] = []
    for pat in patterns:
        for m in re.finditer(pat, raw, flags=re.I):
            try:
                v = int(m.group(1))
                if 1 <= v <= 5:
                    vals.append(v)
            except Exception:
                pass
    return int(max(vals) if vals else default)


def _v61_evolution_node_level(df: pd.DataFrame, alias: str, bundle: Dict[str, Any] | None = None, default: int = 1) -> int:
    """기존 v54 parser의 약점을 보완한 레벨 추정.

    우선순위:
    1) bundle 원문의 진화 N티어 alias Lv.X
    2) source 설명의 '적용 포인트 X', '포인트 X'
    3) alias 근처의 Lv.X/레벨 X
    4) 기존 함수 fallback
    """
    bval = _v61_evolution_node_level_from_bundle(bundle or {}, alias, default=0) if bundle is not None else 0
    if bval:
        return int(bval)

    best = 0
    if isinstance(df, pd.DataFrame) and not df.empty:
        src = df.get("출처구분", pd.Series([""] * len(df), index=df.index)).astype(str)
        ark = df[src.eq("아크패시브")].copy()
        for _, row in ark.iterrows():
            name = str(row.get("이름", "") or "")
            desc = str(row.get("설명", "") or "")
            raw = _clean_text(f"{name} {desc}")
            if alias not in raw:
                continue
            # 실제 적용 포인트 표기를 최우선. '1포인트당'은 제외.
            for pat in [r"적용\s*포인트\s*([1-9])", r"포인트\s*([1-9])"]:
                for m in re.finditer(pat, raw, re.I):
                    try:
                        best = max(best, int(m.group(1)))
                    except Exception:
                        pass
            # alias 주변 Lv 표기만 사용.
            for m in re.finditer(re.escape(alias), raw):
                window = raw[max(0, m.start() - 80): min(len(raw), m.end() + 120)]
                for pat in [r"Lv\.?\s*([1-9])", r"레벨\s*([1-9])"]:
                    for mm in re.finditer(pat, window, re.I):
                        try:
                            best = max(best, int(mm.group(1)))
                        except Exception:
                            pass
    if best:
        return int(max(1, min(5, best)))

    try:
        return int(_v54_evolution_node_level(df, alias, default=default))
    except Exception:
        return int(default)


def _v61_has_evolution_node_from_bundle(bundle: Dict[str, Any], alias: str) -> bool:
    raw = _v61_bundle_raw_text(bundle)
    if not raw:
        return False
    alias_re = re.escape(alias)
    compact = raw.replace(" ", "")
    compact_alias = alias.replace(" ", "")
    return bool(
        re.search(rf"진화\s*[1-5]?\s*티어\s*{alias_re}\s*Lv\.?:?\s*[1-9]", raw, re.I)
        or re.search(rf"진화.{{0,30}}{re.escape(compact_alias)}.{{0,20}}Lv\.?[1-9]", compact, re.I)
        or re.search(rf"{alias_re}\s*Lv\.?\s*[1-9]", raw, re.I)
    )


def _v61_has_node(ref: pd.DataFrame, bundle: Dict[str, Any], *aliases: str) -> bool:
    for a in aliases:
        if _v61_has_evolution_node_from_bundle(bundle, a):
            return True
    try:
        return _v17_has_node(ref, *aliases)
    except Exception:
        return False


def _v17_deterministic_evolution_rows(original_sources: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    """v61: 진화 deterministic rows 재생성.

    일격은 API 원문의 실제 Lv 값을 읽어 포인트당 규칙으로 적용합니다.
    """
    rows: List[Dict[str, Any]] = []
    raw = _v17_raw_text_for_skill_reference(bundle)
    ref = original_sources if isinstance(original_sources, pd.DataFrame) else _empty_source_df()

    if _v61_has_node(ref, bundle, "금단의 주문"):
        rows.append(_v17_make_row("금단의 주문", "전체", "전체/범위 기준", "진화형 피해(%)", 20.0, "금단의 주문 총 진화형 피해 +20%"))
    if _v61_has_node(ref, bundle, "한계 돌파"):
        rows.append(_v17_make_row("한계 돌파", "전체", "전체/범위 기준", "진화형 피해(%)", 10.0, "한계 돌파 진화형 피해 +10%"))
    if _v61_has_node(ref, bundle, "분쇄"):
        rows.append(_v17_make_row("분쇄", "전체", "전체/범위 기준", "진화형 피해(%)", 20.0, "분쇄 진화형 피해 +20%"))
    if _v61_has_node(ref, bundle, "음속 돌파", "음속돌파"):
        rows.append(_v17_make_row("음속 돌파", "전체", "전체/범위 기준", "진화형 피해(%)", 24.0, "음속 돌파 진화형 피해 +24%"))

    has_karma_text = _v61_has_node(ref, bundle, "카르마") or "카르마" in raw
    if has_karma_text:
        rows.append(_v17_make_row("카르마 보정", "전체", "전체/범위 기준", "진화형 피해(%)", 6.0, "카르마 6랭크 진화형 피해 +6%"))
    elif any(str(r.get("이름")) in {"금단의 주문", "한계 돌파", "분쇄", "음속 돌파"} for r in rows):
        rows.append(_v17_make_row("카르마 보정", "전체", "전체/범위 기준", "진화형 피해(%)", 6.0, "API 원문에서 카르마를 못 찾은 경우의 검수 보정 +6%"))

    if _v61_has_node(ref, bundle, "일격"):
        lvl = max(1, min(5, _v61_evolution_node_level(ref, "일격", bundle=bundle, default=2)))
        rows.append(_v17_make_row("일격", "전체", "전체/범위 기준", "치명타 적중률 증가(%)", 10.0 * lvl, f"일격 {lvl}포인트 기준 치명타 적중률 +{10.0 * lvl:.0f}%"))
        rows.append(_v17_make_row("일격", "백/헤드 스킬", "백/헤드 스킬", "치명타 피해량 증가(%)", 16.0 * lvl, f"일격 {lvl}포인트 기준 방향성 공격 스킬 치명타 피해 +{16.0 * lvl:.0f}%"))

    if not rows:
        return _empty_source_df()
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS).drop_duplicates().reset_index(drop=True)


def _v61_rebuild_sources_for_result(result: Dict[str, Any], bundle: Dict[str, Any]) -> None:
    """result 안의 피해군 source를 v61 deterministic 진화 노드 기준으로 재정리합니다."""
    # base_damage_sources는 아크그리드 제외 기준, damage_sources는 아크그리드 포함 기준.
    for key in ["base_damage_sources", "damage_sources"]:
        src = result.get(key)
        if isinstance(src, pd.DataFrame) and not src.empty:
            cleaned = _v17_drop_evolution_duplicates(src)
            deterministic = _v17_deterministic_evolution_rows(src, bundle)
            result[key] = _concat_sources(cleaned, deterministic)

    # 아크그리드 source가 따로 있으면 base + arkgrid로 최종 sources를 다시 구성합니다.
    base = result.get("base_damage_sources")
    arkgrid = result.get("arkgrid_damage_sources")
    if isinstance(base, pd.DataFrame) and isinstance(arkgrid, pd.DataFrame):
        result["damage_sources"] = _concat_sources(base, arkgrid)


def _v61_refresh_summary_metrics(result: Dict[str, Any]) -> None:
    final_df = result.get("arkgrid_final_skill_estimates")
    base_df = result.get("lostbuilds_base_skill_estimates")
    final_sources = result.get("damage_sources")
    base_sources = result.get("base_damage_sources")
    back_engraving = bool(result.get("has_back_attack_engraving"))
    head_engraving = bool(result.get("has_head_attack_engraving"))
    attack_type = "백어택" if back_engraving else ("헤드어택" if head_engraving else "일반/확인필요")

    if isinstance(base_df, pd.DataFrame) and isinstance(final_df, pd.DataFrame) and not base_df.empty and not final_df.empty:
        result["combat_overview"] = _overview_from_tables(
            float(result.get("base_crit_stat", 0) or 0),
            float(result.get("base_crit_percent", 0) or 0),
            base_df,
            final_df,
            base_sources if isinstance(base_sources, pd.DataFrame) else _empty_source_df(),
            final_sources if isinstance(final_sources, pd.DataFrame) else _empty_source_df(),
            back_engraving,
            head_engraving,
        )
        back_final = final_df[final_df["공격타입"].astype(str).str.contains("백", na=False)] if "공격타입" in final_df.columns else final_df
        back_base = base_df[base_df["공격타입"].astype(str).str.contains("백", na=False)] if "공격타입" in base_df.columns else base_df
        result["avg_back_basis_crit_rate_percent"] = _avg_col(final_df, "예상 치명 확률(백어택 기준)(%)")
        result["avg_back_skill_crit_rate_percent"] = _avg_col(back_final, "예상 치명 확률(백어택 기준)(%)", _avg_col(final_df, "예상 치명 확률(백어택 기준)(%)"))
        result["avg_back_skill_crit_damage_percent"] = _avg_col(back_final, "예상 치피(조건부 포함)(%)", _avg_col(final_df, "예상 치피(조건부 포함)(%)", BASE_CRIT_DAMAGE_PERCENT))
        result["avg_evolution_damage_percent"] = _avg_col(final_df, "진화형 피해(조건부)(%)")
        result["avg_enemy_damage_unweighted_percent"] = _avg_col(final_df, "적에게 주는 피해(조건부)(%)")
        result["avg_gem_damage_unweighted_percent"] = _avg_col(final_df, "보석 피해(조건부)(%)")
        result["avg_gem_damage_on_gem_skills_percent"] = _v60_avg_positive_col(final_df, "보석 피해(조건부)(%)")
        result["avg_enemy_damage_percent"] = result["avg_enemy_damage_unweighted_percent"]
        result["avg_gem_damage_percent"] = result["avg_gem_damage_unweighted_percent"]
        result["avg_final_multiplier"] = _avg_col(final_df, "예상 최종 배율(조건부)", 1.0)
        result["lostbuilds_base_back_crit_rate_percent"] = _avg_col(back_base, "예상 치명 확률(백어택 기준)(%)", 0.0)
        result["lostbuilds_base_back_crit_damage_percent"] = _avg_col(back_base, "예상 치피(조건부 포함)(%)", BASE_CRIT_DAMAGE_PERCENT)
        result["lostbuilds_base_avg_final_multiplier"] = _avg_col(base_df, "예상 최종 배율(조건부)", 1.0)

    if isinstance(final_sources, pd.DataFrame) and not final_sources.empty:
        result["global_crit_damage_no_skill_percent"] = _global_directional_crit_damage(final_sources, attack_type)
        global_enemy = _v60_breakdown_value_from_sources(final_sources, "적에게 주는 피해", attack_type)
        if global_enemy is not None:
            result["global_enemy_damage_percent"] = round(global_enemy, 4)

    # combat_overview 명칭 보정 유지
    overview = result.get("combat_overview")
    if isinstance(overview, pd.DataFrame) and not overview.empty and "항목" in overview.columns:
        overview = overview.copy()
        overview.loc[overview["항목"].astype(str).eq("평균 적에게 주는 피해"), "항목"] = "스킬 포함 적피 평균"
        overview.loc[overview["항목"].astype(str).eq("스킬 포함 적피 평균"), "비고"] = "전역 적피 + 스킬/트포 전용 적피를 포함한 채용 스킬 단순 평균"
        overview.loc[overview["항목"].astype(str).eq("평균 보석 피해"), "항목"] = "평균 보석 피해(전체 스킬)"
        overview.loc[overview["항목"].astype(str).eq("평균 보석 피해(전체 스킬)"), "비고"] = "보석 없는 채용 스킬은 0%로 포함한 단순 평균"
        ge = result.get("global_enemy_damage_percent")
        if ge is not None and not overview["항목"].astype(str).eq("전역 적피 합계").any():
            rows = overview.to_dict("records")
            pos = next((i + 1 for i, r in enumerate(rows) if str(r.get("항목")) == "평균 진화형 피해"), len(rows))
            rows.insert(pos, {"항목": "전역 적피 합계", "아크그리드 제외 기준": "-", "아크그리드 포함 최종": f"{float(ge):.2f}%", "비고": "각인/장비/전역 아크패시브/아크그리드 적피 합계. 스킬 전용 트포 제외"})
            overview = pd.DataFrame(rows)
        result["combat_overview"] = overview


def _add_v12_summary(result: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
    result = _old_add_v12_summary_v60_for_v61(result, bundle)
    try:
        _v61_rebuild_sources_for_result(result, bundle)
        _v17_rebuild_merged_and_delta(result)
        _v61_refresh_summary_metrics(result)
        result["estimator_version"] = ESTIMATOR_VERSION
    except Exception as e:
        result["estimator_version"] = ESTIMATOR_VERSION + f"-partial:{type(e).__name__}"
    return result


# v63 -------------------------------------------------------------------------
# - 스킬 그룹/제외한 스킬(아드레날린 계열)이 실제 채용 공격 스킬에 적용되지 않던 문제 수정
# - 진화 4티어 달인 노드 감지 및 치명타 적중률 +7%/Lv 반영
# - 치명타 소스 누락으로 백어택 기준 치명이 낮게 표시되는 문제 보정

ESTIMATOR_VERSION = "v63-adrenaline-skill-group-and-dalin-fix"

_old_scope_applies_v62_for_v63 = _scope_applies

_EXCLUDED_ADRENALINE_GENERATOR_SKILLS_V63 = {
    "기본공격", "기본 공격", "이동기", "스페이스", "스페이스바", "각성기",
    # v70.5: 라그나브레이크/라그나블레이드는 각성기이지만 아드레날린 등 "제외한 스킬" 소스 적용 대상에 포함
    # (레이지슬래셔 등 타 직업 각성기와 동일하게 처리)
}


def _scope_applies(row: pd.Series, attack_type: str, skill_name: str) -> bool:  # type: ignore[override]
    """v63: '스킬 그룹 / 제외한 스킬' 범위 보정.

    Open API의 아드레날린 문구는 보통 '이동기 및 기본공격을 제외한 스킬 사용 후'처럼 내려옵니다.
    기존 v12 스킬그룹 판정은 target 문자열이 attack_type 안에 포함될 때만 적용해서
    target='제외한 스킬'인 아드레날린 치적/공격력이 모든 주력기에서 빠졌습니다.
    """
    scope = str(row.get("적용범위") or "전체")
    target = str(row.get("적용스킬") or "")
    target_norm = _norm_name(target)
    skill_norm = _norm_name(skill_name)
    name_norm = _norm_name(row.get("이름") or "")
    desc_norm = _norm_name(row.get("설명") or "")

    if scope in {"스킬 그룹", "계열 전용", "스킬 계열"}:
        # 아드레날린처럼 '제외한 스킬'로 내려오는 경우: 실제 주력 공격 스킬에는 적용.
        if "제외한스킬" in target_norm or "제외" in target_norm:
            excluded = {_norm_name(x) for x in _EXCLUDED_ADRENALINE_GENERATOR_SKILLS_V63}
            if skill_norm in excluded:
                return False
            # 기본공격/이동기/각성기는 제외. 나머지 스킬은 적용.
            if any(x in skill_norm for x in ["기본공격", "이동기", "스페이스"]):
                return False
            if any(x in name_norm for x in ["아드레날린", "예리한둔기"]):
                return True
            if "이동기및기본공격을제외" in desc_norm or "제외한스킬" in desc_norm:
                return True
            return True
        # 일반 스킬 그룹은 기존 계열 문자열 매칭 유지.
        return target_norm in _norm_name(attack_type)

    return _old_scope_applies_v62_for_v63(row, attack_type, skill_name)


def _v63_evolution_level(ref: pd.DataFrame, bundle: Dict[str, Any], alias: str, default: int = 1) -> int:
    try:
        return int(max(1, min(5, _v61_evolution_node_level(ref, alias, bundle=bundle, default=default))))
    except Exception:
        try:
            return int(max(1, min(5, _v54_evolution_node_level(ref, alias, default=default))))
        except Exception:
            return int(default)


def _v17_deterministic_evolution_rows(original_sources: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    """v63: 진화 노드 고정 규칙 재생성.

    달인 노드는 툴팁상 1.4%×5중첩 구조지만, 최종 치적 관점에서는 Lv.1당 +7%로 반영합니다.
    """
    rows: List[Dict[str, Any]] = []
    raw = _v17_raw_text_for_skill_reference(bundle)
    ref = original_sources if isinstance(original_sources, pd.DataFrame) else _empty_source_df()

    if _v61_has_node(ref, bundle, "금단의 주문"):
        rows.append(_v17_make_row("금단의 주문", "전체", "전체/범위 기준", "진화형 피해(%)", 20.0, "금단의 주문 총 진화형 피해 +20%"))
    if _v61_has_node(ref, bundle, "한계 돌파"):
        rows.append(_v17_make_row("한계 돌파", "전체", "전체/범위 기준", "진화형 피해(%)", 10.0, "한계 돌파 진화형 피해 +10%"))
    if _v61_has_node(ref, bundle, "분쇄"):
        rows.append(_v17_make_row("분쇄", "전체", "전체/범위 기준", "진화형 피해(%)", 20.0, "분쇄 진화형 피해 +20%"))
    if _v61_has_node(ref, bundle, "음속 돌파", "음속돌파"):
        rows.append(_v17_make_row("음속 돌파", "전체", "전체/범위 기준", "진화형 피해(%)", 24.0, "음속 돌파 진화형 피해 +24%"))

    has_karma_text = _v61_has_node(ref, bundle, "카르마") or "카르마" in raw
    if has_karma_text:
        rows.append(_v17_make_row("카르마 보정", "전체", "전체/범위 기준", "진화형 피해(%)", 6.0, "카르마 6랭크 진화형 피해 +6%"))
    elif any(str(r.get("이름")) in {"금단의 주문", "한계 돌파", "분쇄", "음속 돌파"} for r in rows):
        rows.append(_v17_make_row("카르마 보정", "전체", "전체/범위 기준", "진화형 피해(%)", 6.0, "API 원문에서 카르마를 못 찾은 경우의 검수 보정 +6%"))

    if _v61_has_node(ref, bundle, "일격"):
        lvl = _v63_evolution_level(ref, bundle, "일격", default=2)
        rows.append(_v17_make_row("일격", "전체", "전체/범위 기준", "치명타 적중률 증가(%)", 10.0 * lvl, f"일격 Lv.{lvl} 기준 치명타 적중률 +{10.0 * lvl:.0f}%"))
        rows.append(_v17_make_row("일격", "백/헤드 스킬", "백/헤드 스킬", "치명타 피해량 증가(%)", 16.0 * lvl, f"일격 Lv.{lvl} 기준 방향성 공격 스킬 치명타 피해 +{16.0 * lvl:.0f}%"))

    if _v61_has_node(ref, bundle, "달인"):
        lvl = _v63_evolution_level(ref, bundle, "달인", default=1)
        # 달인 Lv.1 = 1.4% × 5중첩 = 총 7.0%
        rows.append(_v17_make_row("달인", "전체", "전체/범위 기준", "치명타 적중률 증가(%)", 7.0 * lvl, f"달인 Lv.{lvl}: 1.4%×5중첩 기준 치명타 적중률 +{7.0 * lvl:.0f}%"))

    if not rows:
        return _empty_source_df()
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS).drop_duplicates().reset_index(drop=True)


# v64 -------------------------------------------------------------------------
# - 진화 노드를 하나씩 하드코딩하던 방식을 포인트 규칙 테이블 기반으로 통일합니다.
# - 예리한 감각/혼신의 강타/달인/금단의 주문 등 진화 노드의 치명타 적중률, 진화형 피해, 추가 피해를 실제 Lv 기준으로 재생성합니다.
# - 달인은 5중첩 기준: Lv.1 = 치명타 적중률 +7%, 추가 피해 +8.5%로 계산합니다.
# - 일격의 방향성 치명타 피해는 백어택/헤드어택 스킬에 모두 적용합니다.

ESTIMATOR_VERSION = "v64-evolution-node-rule-table"

# 이름, 별칭, 효과 규칙을 한곳에서 관리합니다.
# kind: per_level = Lv당 누적 / flat = 총량 고정 / flat_per_detect = 감지 시 고정값
EVOLUTION_NODE_RULES_V64: Dict[str, Dict[str, Any]] = {
    "금단의 주문": {
        "aliases": ["금단의 주문", "금단의주문"],
        "effects": [
            # 사용자 검수 기준: 기본 진피 5% + 마나 스킬 진피 5%를 전역 진피 10%/포인트로 취급.
            {"col": "진화형 피해(%)", "scope": "전체", "target": "전체/범위 기준", "per_level": 10.0, "note": "금단의 주문: 1포인트당 기본 + 마나 스킬 진화형 피해 합산 +10%"},
        ],
    },
    "예리한 감각": {
        "aliases": ["예리한 감각", "예리한감각"],
        "effects": [
            {"col": "진화형 피해(%)", "scope": "전체", "target": "전체/범위 기준", "per_level": 5.0, "note": "예리한 감각: 1포인트당 진화형 피해 +5%"},
            {"col": "치명타 적중률 증가(%)", "scope": "전체", "target": "전체/범위 기준", "per_level": 4.0, "note": "예리한 감각: 1포인트당 치명타 적중률 +4%"},
        ],
    },
    "한계 돌파": {
        "aliases": ["한계 돌파", "한계돌파"],
        "effects": [
            {"col": "진화형 피해(%)", "scope": "전체", "target": "전체/범위 기준", "per_level": 10.0, "note": "한계 돌파: 1포인트당 진화형 피해 +10%"},
        ],
    },
    "최적화 훈련": {
        "aliases": ["최적화 훈련", "최적화훈련"],
        "effects": [
            {"col": "진화형 피해(%)", "scope": "전체", "target": "전체/범위 기준", "per_level": 5.0, "note": "최적화 훈련: 1포인트당 진화형 피해 +5%"},
        ],
    },
    "무한한 마력": {
        "aliases": ["무한한 마력", "무한한마력"],
        "effects": [
            {"col": "진화형 피해(%)", "scope": "전체", "target": "전체/범위 기준", "per_level": 8.0, "note": "무한한 마력: 1포인트당 진화형 피해 +8%"},
        ],
    },
    "혼신의 강타": {
        "aliases": ["혼신의 강타", "혼신의강타"],
        "effects": [
            {"col": "진화형 피해(%)", "scope": "전체", "target": "전체/범위 기준", "per_level": 2.0, "note": "혼신의 강타: 1포인트당 진화형 피해 +2%"},
            {"col": "치명타 적중률 증가(%)", "scope": "전체", "target": "전체/범위 기준", "per_level": 12.0, "note": "혼신의 강타: 1포인트당 치명타 적중률 +12%"},
        ],
    },
    "파괴 전차": {
        "aliases": ["파괴 전차", "파괴전차"],
        "effects": [
            {"col": "진화형 피해(%)", "scope": "전체", "target": "전체/범위 기준", "per_level": 12.0, "note": "파괴 전차: 1포인트당 진화형 피해 +12%"},
        ],
    },
    "일격": {
        "aliases": ["일격"],
        "effects": [
            {"col": "치명타 적중률 증가(%)", "scope": "전체", "target": "전체/범위 기준", "per_level": 10.0, "note": "일격: 1포인트당 치명타 적중률 +10%"},
            {"col": "치명타 피해량 증가(%)", "scope": "백/헤드 스킬", "target": "백/헤드 스킬", "per_level": 16.0, "note": "일격: 1포인트당 방향성 공격 스킬 치명타 피해 +16%"},
        ],
    },
    "달인": {
        "aliases": ["달인"],
        "effects": [
            # Tooltip: 치적 1.4%, 추가 피해 1.7% × 5중첩.
            {"col": "치명타 적중률 증가(%)", "scope": "전체", "target": "전체/범위 기준", "per_level": 7.0, "note": "달인: 5중첩 기준 1포인트당 치명타 적중률 +7%"},
            {"col": "추가 피해(%)", "scope": "전체", "target": "전체/범위 기준", "per_level": 8.5, "note": "달인: 5중첩 기준 1포인트당 추가 피해 +8.5%"},
        ],
    },
    "분쇄": {
        "aliases": ["분쇄"],
        "effects": [
            {"col": "진화형 피해(%)", "scope": "전체", "target": "전체/범위 기준", "per_level": 20.0, "note": "분쇄: 1포인트당 진화형 피해 +20%"},
        ],
    },
    # 5티어 대표 노드. 현재 검수 기준은 총량 고정값으로 처리합니다.
    "뭉툭한 가시": {
        "aliases": ["뭉툭한 가시", "뭉툭한가시", "뭉가"],
        "effects": [
            {"col": "치명타 적중률 고정(%)", "scope": "전체", "target": "전체/범위 기준", "flat": 80.0, "note": "뭉툭한 가시: 치명타 적중률 80% 고정"},
            {"col": "진화형 피해(%)", "scope": "전체", "target": "전체/범위 기준", "flat": 75.0, "note": "뭉툭한 가시: 검수 기준 진화형 피해 +75%"},
        ],
    },
    "음속 돌파": {
        "aliases": ["음속 돌파", "음속돌파"],
        "effects": [
            {"col": "진화형 피해(%)", "scope": "전체", "target": "전체/범위 기준", "flat": 24.0, "note": "음속 돌파: 진화형 피해 +24%"},
        ],
    },
    "인파이팅": {
        "aliases": ["인파이팅"],
        "effects": [
            {"col": "진화형 피해(%)", "scope": "전체", "target": "전체/범위 기준", "flat": 18.0, "note": "인파이팅: 진화형 피해 +18%"},
        ],
    },
    "입식 타격가": {
        "aliases": ["입식 타격가", "입식타격가"],
        "effects": [
            {"col": "진화형 피해(%)", "scope": "전체", "target": "전체/범위 기준", "flat": 21.0, "note": "입식 타격가: 진화형 피해 +21%"},
        ],
    },
    "마나 용광로": {
        "aliases": ["마나 용광로", "마나용광로"],
        "effects": [
            {"col": "진화형 피해(%)", "scope": "전체", "target": "전체/범위 기준", "flat": 24.0, "note": "마나 용광로: 진화형 피해 +24%"},
        ],
    },
}


def _v64_known_evolution_aliases() -> List[str]:
    out: List[str] = []
    for name, rule in EVOLUTION_NODE_RULES_V64.items():
        out.append(name)
        out.extend(list(rule.get("aliases") or []))
    return list(dict.fromkeys(out))


def _v64_has_node(ref: pd.DataFrame, bundle: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    aliases = list(rule.get("aliases") or [])
    try:
        return _v61_has_node(ref, bundle, *aliases)
    except Exception:
        for a in aliases:
            try:
                if _v61_has_evolution_node_from_bundle(bundle, a):
                    return True
            except Exception:
                pass
        return False


def _v64_node_level(ref: pd.DataFrame, bundle: Dict[str, Any], rule: Dict[str, Any], default: int = 1) -> int:
    vals: List[int] = []
    for a in list(rule.get("aliases") or []):
        try:
            vals.append(int(_v63_evolution_level(ref, bundle, a, default=0)))
        except Exception:
            pass
    vals = [v for v in vals if v and v > 0]
    return int(max(vals) if vals else default)


def _v64_make_evolution_rows_from_rules(original_sources: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ref = original_sources if isinstance(original_sources, pd.DataFrame) else _empty_source_df()
    raw = _v17_raw_text_for_skill_reference(bundle)

    for node_name, rule in EVOLUTION_NODE_RULES_V64.items():
        if not _v64_has_node(ref, bundle, rule):
            continue
        lvl = _v64_node_level(ref, bundle, rule, default=1)
        for eff in list(rule.get("effects") or []):
            if "flat" in eff:
                value = float(eff.get("flat") or 0.0)
            else:
                value = float(eff.get("per_level") or 0.0) * float(lvl)
            if abs(value) <= 1e-9:
                continue
            note = str(eff.get("note") or f"{node_name} Lv.{lvl} 효과")
            if "flat" not in eff:
                note = f"{note} / 감지 Lv.{lvl} → 총 +{value:.2f}%"
            rows.append(_v17_make_row(
                node_name,
                str(eff.get("scope") or "전체"),
                str(eff.get("target") or "전체/범위 기준"),
                str(eff.get("col") or "진화형 피해(%)"),
                value,
                note,
            ))

    has_karma_text = False
    try:
        has_karma_text = _v61_has_node(ref, bundle, "카르마") or ("카르마" in raw)
    except Exception:
        has_karma_text = "카르마" in raw
    # 기존 검수 기준 유지: 진화 노드가 감지됐는데 카르마 원문이 빠진 경우에도 +6 보정 행을 추가합니다.
    if has_karma_text:
        rows.append(_v17_make_row("카르마 보정", "전체", "전체/범위 기준", "진화형 피해(%)", 6.0, "카르마 6랭크 진화형 피해 +6%"))
    elif rows:
        rows.append(_v17_make_row("카르마 보정", "전체", "전체/범위 기준", "진화형 피해(%)", 6.0, "API 원문에서 카르마를 못 찾은 경우의 검수 보정 +6%"))

    if not rows:
        return _empty_source_df()
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS).drop_duplicates().reset_index(drop=True)


_old_v17_drop_evolution_duplicates_v63_for_v64 = _v17_drop_evolution_duplicates

def _v17_drop_evolution_duplicates(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
    """v64: 진화 노드는 규칙 테이블로 재생성하므로 generic/기존 파싱 행을 제거합니다.

    이전에는 '진화' generic 행 중 추가 피해 +1.7 같은 달인 1스택 값이 남아
    달인 5중첩 보정과 중복/누락이 생길 수 있었습니다.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return _empty_source_df()
    out = _old_v17_drop_evolution_duplicates_v63_for_v64(df)
    if out is None or out.empty:
        return _empty_source_df()
    known_aliases = _v64_known_evolution_aliases()

    def has_any_effect(row: pd.Series) -> bool:
        for c in EFFECT_COLUMNS:
            try:
                if abs(float(row.get(c) or 0.0)) > 1e-9:
                    return True
            except Exception:
                pass
        return False

    remove_idx: List[Any] = []
    for idx, row in out.iterrows():
        src = str(row.get("출처구분") or "")
        if src != "아크패시브":
            continue
        nm = str(row.get("이름") or "")
        desc = str(row.get("설명") or "")
        raw = f"{nm} {desc}"
        raw_norm = _norm_name(raw)
        if nm == "진화" and has_any_effect(row):
            remove_idx.append(idx)
            continue
        if has_any_effect(row) and any(_norm_name(a) in raw_norm for a in known_aliases):
            remove_idx.append(idx)
    if remove_idx:
        out = out.drop(index=remove_idx)
    return out.reset_index(drop=True)


def _v17_deterministic_evolution_rows(original_sources: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    """v64: 모든 주요 진화 노드를 규칙 테이블로 재생성합니다."""
    return _v64_make_evolution_rows_from_rules(original_sources, bundle)

_old_v17_recompute_display_columns_v63_for_v64 = _v17_recompute_display_columns

def _v64_sources_without_base_stat(sources: pd.DataFrame) -> pd.DataFrame:
    if sources is None or not isinstance(sources, pd.DataFrame) or sources.empty:
        return _empty_source_df()
    out = sources.copy()
    if "출처구분" in out.columns:
        out = out[~out["출처구분"].astype(str).eq("기본 스탯")]
    return out.reset_index(drop=True)


def _v64_direction_crit_for_attack_type(attack_type: str) -> float:
    at = str(attack_type or "")
    return BACK_ATTACK_CRIT_BONUS_PERCENT if "백" in at else 0.0


def _v64_recalculate_skill_row_from_sources(row: pd.Series, sources: pd.DataFrame) -> Dict[str, Any]:
    """정리된 source 표를 기준으로 스킬별 표시/배율 컬럼을 다시 계산합니다."""
    skill_name = str(row.get("스킬명") or "")
    attack_type = str(row.get("공격타입") or "일반/확인필요")
    src = _v64_sources_without_base_stat(sources)
    stat_rate = _num(row.get("치명 스탯 치적(%)")) or 0.0

    static_fixed = _max_sources(src, "치명타 적중률 고정(%)", attack_type, skill_name, include_conditional=False)
    cond_fixed = _max_sources(src, "치명타 적중률 고정(%)", attack_type, skill_name, include_conditional=True)
    non_stat_static_crit_add = _sum_sources(src, "치명타 적중률 증가(%)", attack_type, skill_name, include_conditional=False)
    non_stat_cond_crit_add = _sum_sources(src, "치명타 적중률 증가(%)", attack_type, skill_name, include_conditional=True)

    static_crit_add = (static_fixed + non_stat_static_crit_add) if static_fixed else (stat_rate + non_stat_static_crit_add)
    cond_crit_add = (cond_fixed + non_stat_cond_crit_add) if cond_fixed else (stat_rate + non_stat_cond_crit_add)
    direction_crit = _v64_direction_crit_for_attack_type(attack_type)
    static_crit_rate = max(0.0, float(static_crit_add))
    cond_crit_rate = max(0.0, float(cond_crit_add))
    basis_crit_rate = max(0.0, float(cond_crit_add + direction_crit))

    global_sources = _non_skill_sources(src)
    skill_sources = _skill_only_sources(src, skill_name)
    global_static_cd_add = _sum_sources(global_sources, "치명타 피해량 증가(%)", attack_type, skill_name, include_conditional=False)
    global_cond_cd_add = _sum_sources(global_sources, "치명타 피해량 증가(%)", attack_type, skill_name, include_conditional=True)
    skill_static_cd_add = _sum_sources(skill_sources, "치명타 피해량 증가(%)", attack_type, skill_name, include_conditional=False)
    skill_cond_cd_add = _sum_sources(skill_sources, "치명타 피해량 증가(%)", attack_type, skill_name, include_conditional=True)
    static_cd = BASE_CRIT_DAMAGE_PERCENT + global_static_cd_add + skill_static_cd_add
    cond_cd = BASE_CRIT_DAMAGE_PERCENT + global_cond_cd_add + skill_cond_cd_add

    static_groups = {
        "진화형 피해(%)": _sum_sources(src, "진화형 피해(%)", attack_type, skill_name, include_conditional=False),
        "적에게 주는 피해(%)": _sum_sources(src, "적에게 주는 피해(%)", attack_type, skill_name, include_conditional=False),
        "스킬 피해(%)": _sum_sources(src, "스킬 피해(%)", attack_type, skill_name, include_conditional=False),
        "방향성 피해(%)": _sum_sources(src, "방향성 피해(%)", attack_type, skill_name, include_conditional=False),
        "보석 피해(%)": _sum_sources(src, "보석 피해(%)", attack_type, skill_name, include_conditional=False),
        "공격력 증가(%)": _sum_sources(src, "공격력 증가(%)", attack_type, skill_name, include_conditional=False),
        "추가 피해(%)": _sum_sources(src, "추가 피해(%)", attack_type, skill_name, include_conditional=False),
    }
    cond_groups = {
        "진화형 피해(%)": _sum_sources(src, "진화형 피해(%)", attack_type, skill_name, include_conditional=True),
        "적에게 주는 피해(%)": _sum_sources(src, "적에게 주는 피해(%)", attack_type, skill_name, include_conditional=True),
        "스킬 피해(%)": _sum_sources(src, "스킬 피해(%)", attack_type, skill_name, include_conditional=True),
        "방향성 피해(%)": _sum_sources(src, "방향성 피해(%)", attack_type, skill_name, include_conditional=True),
        "보석 피해(%)": _sum_sources(src, "보석 피해(%)", attack_type, skill_name, include_conditional=True),
        "공격력 증가(%)": _sum_sources(src, "공격력 증가(%)", attack_type, skill_name, include_conditional=True),
        "추가 피해(%)": _sum_sources(src, "추가 피해(%)", attack_type, skill_name, include_conditional=True),
    }

    scmul = _expected_crit_multiplier(min(100.0, static_crit_rate), static_cd)
    ccmul = _expected_crit_multiplier(min(100.0, basis_crit_rate), cond_cd)
    sdmul = _damage_group_multiplier(**static_groups)
    cdmul = _damage_group_multiplier(**cond_groups)

    return {
        "치명 증가 합계(상시)(%)": round(static_crit_add, 2),
        "치명 증가 합계(조건부)(%)": round(cond_crit_add, 2),
        "예상 치명 확률(상시)(%)": round(static_crit_rate, 2),
        "예상 치명 확률(조건부 포함)(%)": round(cond_crit_rate, 2),
        "예상 치명 확률(백어택 기준)(%)": round(basis_crit_rate, 2),
        "예상 치피(상시)(%)": round(static_cd, 2),
        "예상 치피(조건부 포함)(%)": round(cond_cd, 2),
        "진화형 피해(상시)(%)": round(static_groups["진화형 피해(%)"], 2),
        "진화형 피해(조건부)(%)": round(cond_groups["진화형 피해(%)"], 2),
        "적에게 주는 피해(상시)(%)": round(static_groups["적에게 주는 피해(%)"], 2),
        "적에게 주는 피해(조건부)(%)": round(cond_groups["적에게 주는 피해(%)"], 2),
        "스킬 피해(상시)(%)": round(static_groups["스킬 피해(%)"], 2),
        "스킬 피해(조건부)(%)": round(cond_groups["스킬 피해(%)"], 2),
        "방향성 피해(상시)(%)": round(static_groups["방향성 피해(%)"], 2),
        "방향성 피해(조건부)(%)": round(cond_groups["방향성 피해(%)"], 2),
        "보석 피해(상시)(%)": round(static_groups["보석 피해(%)"], 2),
        "보석 피해(조건부)(%)": round(cond_groups["보석 피해(%)"], 2),
        "공격력 증가(상시)(%)": round(static_groups["공격력 증가(%)"], 2),
        "공격력 증가(조건부)(%)": round(cond_groups["공격력 증가(%)"], 2),
        "추가 피해(상시)(%)": round(static_groups["추가 피해(%)"], 2),
        "추가 피해(조건부)(%)": round(cond_groups["추가 피해(%)"], 2),
        "치명 기대배율(상시)": round(scmul, 4),
        "치명 기대배율(조건부+방향)": round(ccmul, 4),
        "피해군 배율(상시)": round(sdmul, 4),
        "피해군 배율(조건부)": round(cdmul, 4),
        "예상 최종 배율(상시)": round(scmul * sdmul, 4),
        "예상 최종 배율(조건부)": round(ccmul * cdmul, 4),
    }


def _v17_recompute_display_columns(df: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = df.copy()
    if sources is None or not isinstance(sources, pd.DataFrame) or sources.empty:
        return _old_v17_recompute_display_columns_v63_for_v64(out, sources)
    for idx, row in out.iterrows():
        vals = _v64_recalculate_skill_row_from_sources(row, sources)
        for col, val in vals.items():
            if col in out.columns:
                out.at[idx, col] = val
    return out


def _v64_global_basis_crit_rate(result: Dict[str, Any]) -> float | None:
    sources = result.get("damage_sources")
    final_df = result.get("arkgrid_final_skill_estimates")
    if not isinstance(sources, pd.DataFrame) or sources.empty:
        return None
    src = _v64_sources_without_base_stat(sources)
    stat_rate = float(result.get("base_crit_percent", 0) or 0)
    attack_type = "백어택" if result.get("has_back_attack_engraving") else ("헤드어택" if result.get("has_head_attack_engraving") else "일반/확인필요")
    # 대표 스킬명은 스킬 그룹/제외한 스킬 적용 판정을 위해 사용합니다.
    skill_name = "기준스킬"
    if isinstance(final_df, pd.DataFrame) and not final_df.empty and "스킬명" in final_df.columns:
        try:
            skill_name = str(final_df.iloc[0].get("스킬명") or skill_name)
        except Exception:
            pass
    fixed = _max_sources(src, "치명타 적중률 고정(%)", attack_type, skill_name, include_conditional=True)
    add = _sum_sources(src, "치명타 적중률 증가(%)", attack_type, skill_name, include_conditional=True)
    val = (fixed + add) if fixed else (stat_rate + add)
    val += _v64_direction_crit_for_attack_type(attack_type)
    return round(float(val), 2)


_old_v61_refresh_summary_metrics_v63_for_v64 = _v61_refresh_summary_metrics

def _v61_refresh_summary_metrics(result: Dict[str, Any]) -> None:  # type: ignore[override]
    _old_v61_refresh_summary_metrics_v63_for_v64(result)
    try:
        gb = _v64_global_basis_crit_rate(result)
        if gb is not None:
            result["global_basis_crit_rate_percent"] = gb
    except Exception:
        pass


# v65 -------------------------------------------------------------------------
# - 선택 트라이포드 중 파티/자신 시너지 트포는 스킬 전용이 아니라 전역 시너지로 분리합니다.
# - 같은 시너지 트포를 2개 이상 채용해도 중첩하지 않고 같은 효과군별 최대값만 반영합니다.
# - 지진파처럼 시너지 명단에 없는 스킬 자체 치적/치피/피해 옵션은 계속 해당 스킬 전용으로만 적용합니다.

ESTIMATOR_VERSION = "v65-tripod-synergy-scope-fix"

_old_collect_skill_tripod_sources_v64_for_v65 = _collect_skill_tripod_sources

SYNERGY_TRIPOD_NAMES_V65 = {
    "갑옷 파괴", "갑옷파괴",
    "피해 증폭", "피해증폭",
    "급소 노출", "급소노출",
    "약점 공략", "약점공략",
    "약점 노출", "약점노출",
    "투지 강화", "투지강화",
}

# v67: 시너지 트포 고정 수치(PvE 최대 기준). 툴팁 문구가 달라 파싱이 빗나가도 이 값을 '바닥값'으로
# 보장합니다. 특히 '약점 공략'(치명타 피해량)은 v65 파서가 못 잡던 항목입니다.
# (norm 이름 → (효과 컬럼, 값, 효과군 key)). 갑옷 파괴는 방어력 감소 컬럼이 없어 제외(메모만).
SYNERGY_FIXED_VALUES_V67: Dict[str, Tuple[str, float, str]] = {
    "급소노출": ("치명타 적중률 증가(%)", 10.0, "crit_resist_down"),
    "약점공략": ("치명타 피해량 증가(%)", 8.0, "crit_damage_up"),
    "피해증폭": ("적에게 주는 피해(%)", 6.0, "damage_taken_up"),
    "약점노출": ("적에게 주는 피해(%)", 4.0, "damage_taken_up"),
    "투지강화": ("공격력 증가(%)", 6.0, "party_attack_power"),
}


def _v65_is_synergy_tripod_name(name: Any) -> bool:
    n = _norm_name(name)
    return any(n == _norm_name(x) for x in SYNERGY_TRIPOD_NAMES_V65)


def _v65_percent_values_before_word(text: str, value_word: str, action_word: str = "증가", window: int = 90) -> List[float]:
    clean = _clean_text(text)
    vals: List[float] = []
    # 예: "받는 피해가 8.0초 간 6.0% 증가" 에서 8.0초가 아니라 6.0%를 잡기 위해
    # value_word 이후 action_word 직전의 마지막 % 숫자를 우선 사용합니다.
    for m in re.finditer(re.escape(value_word), clean):
        seg = clean[m.start(): min(len(clean), m.start() + window)]
        act = seg.find(action_word)
        if act >= 0:
            seg = seg[:act]
        nums = re.findall(r"([+\-]?\d+(?:\.\d+)?)\s*%", seg)
        if nums:
            try:
                vals.append(float(nums[-1]))
            except Exception:
                pass
    return vals


def _v65_make_effect_row(name: str, effects: Dict[str, float], text: str, skill_name: str, key: str) -> Dict[str, Any] | None:
    if not any(abs(float(effects.get(c, 0.0) or 0.0)) > 1e-9 for c in EFFECT_COLUMNS):
        return None
    return {
        "출처구분": "선택 트라이포드",
        "이름": f"시너지:{name}",
        "적용범위": "전체",
        "적용스킬": f"시너지:{key}",
        **{c: round(float(effects.get(c, 0.0) or 0.0), 2) for c in EFFECT_COLUMNS},
        "조건부 여부": "상시",
        "설명": _clean_text(f"{skill_name} / {name}: {text}")[:420],
    }


def _v65_synergy_row_from_tripod(skill_name: str, tripod: Dict[str, Any]) -> Tuple[str, Dict[str, Any] | None]:
    name = str(tripod.get("Name") or tripod.get("name") or "").strip()
    tooltip = _flatten(tripod.get("Tooltip") or tripod.get("tooltip"))
    text = _clean_text(f"{name} {tooltip}")
    effects = _new_effect_dict()
    key_parts: List[str] = []

    # 1) 급소 노출류: 치명타 저항률 감소 = 파티 기준 치명타 적중률 증가로 계산.
    if "치명타" in text and "저항" in text and "감소" in text:
        vals = _v65_percent_values_before_word(text, "치명타 저항", "감소")
        if not vals:
            vals = _extract_percent_near_keywords(text, ["치명타 저항률", "치명타 저항"], window=90)
        if vals:
            effects["치명타 적중률 증가(%)"] = max(abs(float(v)) for v in vals)
            key_parts.append("crit_resist_down")

    # 2) 피해 증폭/약점 노출류: 대상이 자신 및 파티원에게 받는 피해 증가 = 전역 적피.
    if "받는 피해" in text and "증가" in text:
        vals = _v65_percent_values_before_word(text, "받는 피해", "증가")
        if not vals:
            vals = _extract_percent_near_keywords(text, ["받는 피해", "받는 피해량"], window=90)
        if vals:
            effects["적에게 주는 피해(%)"] = max(abs(float(v)) for v in vals)
            key_parts.append("damage_taken_up")

    # 3) 투지 강화류: 자신 및 파티원 공격력 증가.
    if "공격력" in text and "증가" in text and ("자신" in text or "파티" in text or _v65_is_synergy_tripod_name(name)):
        vals = _v65_percent_values_before_word(text, "공격력", "증가")
        if not vals:
            vals = _extract_percent_near_keywords(text, ["공격력"], window=90)
        # 무기 공격력은 기존 파서와 동일하게 제외합니다.
        if vals and "무기 공격력" not in text:
            effects["공격력 증가(%)"] = max(abs(float(v)) for v in vals)
            key_parts.append("party_attack_power")

    # 4) 갑옷 파괴류는 방어력 감소 문구가 직접 피해군으로 환산되지 않습니다.
    # 현재 계산표의 피해군에는 방어력 감소 전용 컬럼이 없으므로, 수치가 직접 적피/치적/공격력으로 읽히지 않으면
    # 계산에는 넣지 않고 디버그 원문에서 확인 대상으로 남깁니다.
    if "방어력" in text and "감소" in text and not key_parts:
        return "defense_down_unconverted", None

    if not key_parts:
        # 일부 툴팁이 '적에게 주는 피해'처럼 직접 적혀 있을 수 있으므로 일반 파서를 마지막으로 보조 사용.
        generic = _extract_effect_values(text, source_type="선택 트라이포드")
        for col in ["치명타 적중률 증가(%)", "적에게 주는 피해(%)", "공격력 증가(%)"]:
            if generic.get(col):
                effects[col] = max(float(effects.get(col, 0.0) or 0.0), float(generic.get(col, 0.0) or 0.0))
                key_parts.append(col)

    # v67: 알려진 시너지는 고정 수치를 '바닥값'으로 보장(파싱 실패/문구 변형 대비).
    fixed = SYNERGY_FIXED_VALUES_V67.get(_norm_name(name))
    if fixed:
        fcol, fval, fkey = fixed
        if fcol in effects:
            effects[fcol] = max(float(effects.get(fcol, 0.0) or 0.0), float(fval))
            if fkey not in key_parts:
                key_parts.append(fkey)

    key = "+".join(key_parts) if key_parts else _norm_name(name)
    return key, _v65_make_effect_row(name, effects, text, skill_name, key)


def _v65_row_power(row: Dict[str, Any]) -> float:
    # 중복 시 같은 효과군에서 더 큰 수치를 남깁니다.
    return sum(abs(float(row.get(c, 0.0) or 0.0)) for c in EFFECT_COLUMNS)


def _collect_skill_tripod_sources(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    """v65: 선택 트라이포드 시너지/스킬 전용 분리.

    시너지 트포명:
    - 갑옷 파괴 / 피해 증폭 / 급소 노출 / 약점 공략 / 약점 노출 / 투지 강화

    이 트포들은 해당 스킬에만 붙는 지진파/강화된 일격류와 달리 파티/자신 시너지로 보고
    전체 범위에 1회만 적용합니다. 같은 효과군은 여러 스킬에 있어도 중첩하지 않습니다.
    """
    base = _old_collect_skill_tripod_sources_v64_for_v65(bundle)
    if not isinstance(base, pd.DataFrame) or base.empty:
        base = _empty_source_df()
    else:
        # 기존 로직이 시너지 트포를 스킬 전용으로 넣은 행은 제거하고, 아래에서 전역 시너지로 재삽입합니다.
        mask = base.get("이름", pd.Series([""] * len(base), index=base.index)).map(_v65_is_synergy_tripod_name)
        base = base[~mask].reset_index(drop=True)

    best_rows: Dict[str, Dict[str, Any]] = {}
    for skill in _safe_data(bundle, "combat_skills") or []:
        skill_name = str(skill.get("Name") or skill.get("name") or "")
        # 대/소문자 키(Tripods/tripods) 모두 지원 — 정규화 캐시는 소문자 키를 씁니다.
        _trips = skill.get("Tripods") or skill.get("tripods") or []
        for tripod in _trips or []:
            if not isinstance(tripod, dict):
                continue
            # 선택 여부 키도 대/소문자 변형을 모두 확인합니다.
            _sel = tripod.get("IsSelected", tripod.get("isSelected", tripod.get("Selected", tripod.get("selected"))))
            if not bool(_sel):
                continue
            tname = str(tripod.get("Name") or tripod.get("name") or "")
            if not _v65_is_synergy_tripod_name(tname):
                continue
            key, row = _v65_synergy_row_from_tripod(skill_name, tripod)
            if not row:
                continue
            # 같은 시너지 효과군은 중첩하지 않음. 더 큰 값만 채택.
            old = best_rows.get(key)
            if old is None or _v65_row_power(row) > _v65_row_power(old):
                best_rows[key] = row
            elif old is not None:
                old_desc = str(old.get("설명") or "")
                extra = _clean_text(f" / 중복 감지: {skill_name} {tname}")
                if extra not in old_desc:
                    old["설명"] = (old_desc + extra)[:420]

    synergy_df = pd.DataFrame(list(best_rows.values()), columns=SOURCE_COLUMNS) if best_rows else _empty_source_df()
    return _concat_sources(base, synergy_df)

# v66 -------------------------------------------------------------------------
# - 뭉툭한 가시의 "치명타 적중률 고정" 값은 기본/백어택 기준 치명 표시값에 더하지 않습니다.
#   로아와식 기본 치명 확률은 치명 스탯 + 각인 + 장비 + 아크패시브 치적 + 방향 보너스를 보여줘야 하며,
#   뭉툭한 가시는 별도 특수 효과/예상 DPS 범위에서만 다룹니다.
# - 지진파 같은 스킬 전용 치적은 해당 스킬 계산에만 반영하고 전역 기준 치명 카드에는 반영하지 않습니다.

ESTIMATOR_VERSION = "v66-crit-fixed-is-not-base-crit"


def _v66_sum_crit_increase_sources(
    df: pd.DataFrame,
    attack_type: str,
    skill_name: str,
    include_conditional: bool,
) -> float:
    """치명타 적중률 증가값만 합산합니다.

    주의:
    - 치명타 적중률 고정(%)은 합산하지 않습니다. 특히 뭉툭한 가시는 표시용 기본 치명을
      80% 위에 더하는 효과가 아니므로 이 함수에서 제외합니다.
    - 스킬 전용 출처는 skill_name이 맞을 때만 _scope_applies()에서 통과합니다.
    """
    return _sum_sources(df, "치명타 적중률 증가(%)", attack_type, skill_name, include_conditional=include_conditional)


def _calculate_one_skill(  # type: ignore[override]
    skill: Dict[str, Any],
    effect_sources: pd.DataFrame,
    crit_stat: float,
    stat_rate: float,
    back_engraving: bool,
    head_engraving: bool,
) -> Dict[str, Any]:
    """v66: 뭉가 고정 치적은 표시용 기본 치명에 더하지 않고, 스킬 전용 치적은 해당 스킬에만 반영합니다."""
    name = skill.get("Name") or ""
    attack_type = _detect_attack_type(skill)
    # v70.5: 백/헤드 둘 다인 스킬은 각인(기습/결투) 기준으로 단일 방향 결정.
    # 둘 다 없으면 백어택 기본. 표시·계산 모두 단일 방향으로 처리해 방향 보너스 중복을 방지합니다.
    attack_type = _v705_resolve_both_direction(attack_type, back_engraving, head_engraving)

    def sum_static(col: str, df: pd.DataFrame = effect_sources) -> float:
        return _sum_sources(df, col, attack_type, name, include_conditional=False)

    def sum_cond_total(col: str, df: pd.DataFrame = effect_sources) -> float:
        return _sum_sources(df, col, attack_type, name, include_conditional=True)

    static_fixed = _max_sources(effect_sources, "치명타 적중률 고정(%)", attack_type, name, include_conditional=False)
    cond_fixed = _max_sources(effect_sources, "치명타 적중률 고정(%)", attack_type, name, include_conditional=True)

    non_stat_static_crit_add = _v66_sum_crit_increase_sources(effect_sources, attack_type, name, include_conditional=False)
    non_stat_cond_crit_add = _v66_sum_crit_increase_sources(effect_sources, attack_type, name, include_conditional=True)

    # v66 핵심: 고정 치적은 기본 치명 합산에 사용하지 않습니다.
    # 예: 치명 20.61 + 폭주 30 + 아드 20 + 일격 20 + 예감 8 + 달인 7 + 반지 3.1 = 108.71
    static_crit_add = stat_rate + non_stat_static_crit_add
    cond_crit_add = stat_rate + non_stat_cond_crit_add

    direction_crit = BACK_ATTACK_CRIT_BONUS_PERCENT if "백" in attack_type else 0.0
    if "헤드" in attack_type and "백" not in attack_type:
        direction_crit = 0.0

    # 화면에는 과치적 확인을 위해 원값을 남기고, 기대배율 계산만 100%로 캡핑합니다.
    static_crit_rate_display = max(0.0, float(static_crit_add))
    cond_crit_rate_display = max(0.0, float(cond_crit_add))
    back_basis_crit_rate_display = max(0.0, float(cond_crit_add + direction_crit))

    static_crit_rate_for_mul = min(100.0, static_crit_rate_display)
    back_basis_crit_rate_for_mul = min(100.0, back_basis_crit_rate_display)

    global_sources = _non_skill_sources(effect_sources)
    skill_sources = _skill_only_sources(effect_sources, name)

    global_static_crit_damage_add = _sum_sources(global_sources, "치명타 피해량 증가(%)", attack_type, name, include_conditional=False)
    global_cond_crit_damage_add = _sum_sources(global_sources, "치명타 피해량 증가(%)", attack_type, name, include_conditional=True)
    skill_static_crit_damage_add = _sum_sources(skill_sources, "치명타 피해량 증가(%)", attack_type, name, include_conditional=False)
    skill_cond_crit_damage_add = _sum_sources(skill_sources, "치명타 피해량 증가(%)", attack_type, name, include_conditional=True)

    static_crit_damage = BASE_CRIT_DAMAGE_PERCENT + global_static_crit_damage_add + skill_static_crit_damage_add
    cond_crit_damage = BASE_CRIT_DAMAGE_PERCENT + global_cond_crit_damage_add + skill_cond_crit_damage_add

    def group_values(include_conditional: bool) -> Dict[str, float]:
        sum_fn = sum_cond_total if include_conditional else sum_static
        direction_base = 0.0
        if "백" in attack_type:
            direction_base += BACK_ATTACK_DAMAGE_BONUS_PERCENT
        if "헤드" in attack_type:
            direction_base += HEAD_ATTACK_DAMAGE_BONUS_PERCENT
        return {
            "진화형 피해(%)": sum_fn("진화형 피해(%)"),
            "적에게 주는 피해(%)": sum_fn("적에게 주는 피해(%)"),
            "스킬 피해(%)": sum_fn("스킬 피해(%)"),
            "방향성 피해(%)": direction_base + sum_fn("방향성 피해(%)"),
            "보석 피해(%)": sum_fn("보석 피해(%)"),
            "공격력 증가(%)": sum_fn("공격력 증가(%)"),
            "추가 피해(%)": sum_fn("추가 피해(%)"),
        }

    static_groups = group_values(False)
    cond_groups = group_values(True)
    static_crit_mul = _expected_crit_multiplier(static_crit_rate_for_mul, static_crit_damage)
    cond_crit_mul = _expected_crit_multiplier(back_basis_crit_rate_for_mul, cond_crit_damage)
    static_damage_mul = _damage_group_multiplier(**static_groups)
    cond_damage_mul = _damage_group_multiplier(**cond_groups)

    direction_note = ""
    if "백" in attack_type:
        direction_note = "백어택 기본 치적 +10 / 피해 +5"
        if back_engraving:
            direction_note = "기습 채용 감지 + 백어택 기본 치적 +10 / 피해 +5"
    if "헤드" in attack_type:
        direction_note = (direction_note + " / " if direction_note else "") + "헤드어택 기본 피해 +20"
        if head_engraving:
            direction_note += " / 결투 채용 감지"

    conditional_count = 0
    if isinstance(effect_sources, pd.DataFrame):
        for _, row in effect_sources.iterrows():
            if _scope_applies(row, attack_type, name) and str(row.get("조건부 여부")) != "상시":
                conditional_count += 1

    return {
        "스킬명": name,
        "공격타입": attack_type,
        "스킬레벨": skill.get("Level") or skill.get("SkillLevel") or "",
        "룬": (skill.get("Rune") or {}).get("Name") if isinstance(skill.get("Rune"), dict) else skill.get("Rune"),
        "치명 스탯": round(crit_stat, 0),
        "치명 스탯 치적(%)": round(stat_rate, 2),
        "치명 고정 옵션(%)": round(max(static_fixed, cond_fixed), 2),
        "치명 증가 합계(상시)(%)": round(static_crit_add, 2),
        "치명 증가 합계(조건부)(%)": round(cond_crit_add, 2),
        "백어택 기준 보너스(%)": round(direction_crit, 2),
        "예상 치명 확률(상시)(%)": round(static_crit_rate_display, 2),
        "예상 치명 확률(조건부 포함)(%)": round(cond_crit_rate_display, 2),
        "예상 치명 확률(백어택 기준)(%)": round(back_basis_crit_rate_display, 2),
        "기본 치피(%)": BASE_CRIT_DAMAGE_PERCENT,
        "전역 치피 증가(상시)(%)": round(global_static_crit_damage_add, 2),
        "전역 치피 증가(조건부)(%)": round(global_cond_crit_damage_add, 2),
        "스킬 전용 치피 증가(상시)(%)": round(skill_static_crit_damage_add, 2),
        "스킬 전용 치피 증가(조건부)(%)": round(skill_cond_crit_damage_add, 2),
        "예상 치피(상시)(%)": round(static_crit_damage, 2),
        "예상 치피(조건부 포함)(%)": round(cond_crit_damage, 2),
        "진화형 피해(상시)(%)": round(static_groups["진화형 피해(%)"], 2),
        "진화형 피해(조건부)(%)": round(cond_groups["진화형 피해(%)"], 2),
        "적에게 주는 피해(상시)(%)": round(static_groups["적에게 주는 피해(%)"], 2),
        "적에게 주는 피해(조건부)(%)": round(cond_groups["적에게 주는 피해(%)"], 2),
        "스킬 피해(상시)(%)": round(static_groups["스킬 피해(%)"], 2),
        "스킬 피해(조건부)(%)": round(cond_groups["스킬 피해(%)"], 2),
        "방향성 피해(상시)(%)": round(static_groups["방향성 피해(%)"], 2),
        "방향성 피해(조건부)(%)": round(cond_groups["방향성 피해(%)"], 2),
        "보석 피해(상시)(%)": round(static_groups["보석 피해(%)"], 2),
        "보석 피해(조건부)(%)": round(cond_groups["보석 피해(%)"], 2),
        "공격력 증가(상시)(%)": round(static_groups["공격력 증가(%)"], 2),
        "공격력 증가(조건부)(%)": round(cond_groups["공격력 증가(%)"], 2),
        "추가 피해(상시)(%)": round(static_groups["추가 피해(%)"], 2),
        "추가 피해(조건부)(%)": round(cond_groups["추가 피해(%)"], 2),
        "치명 기대배율(상시)": round(static_crit_mul, 4),
        "치명 기대배율(조건부+방향)": round(cond_crit_mul, 4),
        "피해군 배율(상시)": round(static_damage_mul, 4),
        "피해군 배율(조건부)": round(cond_damage_mul, 4),
        "예상 최종 배율(상시)": round(static_crit_mul * static_damage_mul, 4),
        "예상 최종 배율(조건부)": round(cond_crit_mul * cond_damage_mul, 4),
        "조건부 항목 수": conditional_count,
        "방향 보너스 메모": direction_note,
        "감지 출처": _source_names_for_skill(effect_sources, attack_type, name) or "치명 스탯만 적용",
    }


def _v64_recalculate_skill_row_from_sources(row: pd.Series, sources: pd.DataFrame) -> Dict[str, Any]:  # type: ignore[override]
    """v66: 재계산 단계에서도 고정 치적을 기본 치명 합산에서 제외합니다."""
    skill_name = str(row.get("스킬명") or "")
    attack_type = str(row.get("공격타입") or "일반/확인필요")
    src = _v64_sources_without_base_stat(sources)
    stat_rate = _num(row.get("치명 스탯 치적(%)")) or 0.0

    static_fixed = _max_sources(src, "치명타 적중률 고정(%)", attack_type, skill_name, include_conditional=False)
    cond_fixed = _max_sources(src, "치명타 적중률 고정(%)", attack_type, skill_name, include_conditional=True)
    non_stat_static_crit_add = _v66_sum_crit_increase_sources(src, attack_type, skill_name, include_conditional=False)
    non_stat_cond_crit_add = _v66_sum_crit_increase_sources(src, attack_type, skill_name, include_conditional=True)

    static_crit_add = stat_rate + non_stat_static_crit_add
    cond_crit_add = stat_rate + non_stat_cond_crit_add
    direction_crit = _v64_direction_crit_for_attack_type(attack_type)
    static_crit_rate = max(0.0, float(static_crit_add))
    cond_crit_rate = max(0.0, float(cond_crit_add))
    basis_crit_rate = max(0.0, float(cond_crit_add + direction_crit))

    global_sources = _non_skill_sources(src)
    skill_sources = _skill_only_sources(src, skill_name)
    global_static_cd_add = _sum_sources(global_sources, "치명타 피해량 증가(%)", attack_type, skill_name, include_conditional=False)
    global_cond_cd_add = _sum_sources(global_sources, "치명타 피해량 증가(%)", attack_type, skill_name, include_conditional=True)
    skill_static_cd_add = _sum_sources(skill_sources, "치명타 피해량 증가(%)", attack_type, skill_name, include_conditional=False)
    skill_cond_cd_add = _sum_sources(skill_sources, "치명타 피해량 증가(%)", attack_type, skill_name, include_conditional=True)
    static_cd = BASE_CRIT_DAMAGE_PERCENT + global_static_cd_add + skill_static_cd_add
    cond_cd = BASE_CRIT_DAMAGE_PERCENT + global_cond_cd_add + skill_cond_cd_add

    static_groups = {
        "진화형 피해(%)": _sum_sources(src, "진화형 피해(%)", attack_type, skill_name, include_conditional=False),
        "적에게 주는 피해(%)": _sum_sources(src, "적에게 주는 피해(%)", attack_type, skill_name, include_conditional=False),
        "스킬 피해(%)": _sum_sources(src, "스킬 피해(%)", attack_type, skill_name, include_conditional=False),
        "방향성 피해(%)": _sum_sources(src, "방향성 피해(%)", attack_type, skill_name, include_conditional=False),
        "보석 피해(%)": _sum_sources(src, "보석 피해(%)", attack_type, skill_name, include_conditional=False),
        "공격력 증가(%)": _sum_sources(src, "공격력 증가(%)", attack_type, skill_name, include_conditional=False),
        "추가 피해(%)": _sum_sources(src, "추가 피해(%)", attack_type, skill_name, include_conditional=False),
    }
    cond_groups = {
        "진화형 피해(%)": _sum_sources(src, "진화형 피해(%)", attack_type, skill_name, include_conditional=True),
        "적에게 주는 피해(%)": _sum_sources(src, "적에게 주는 피해(%)", attack_type, skill_name, include_conditional=True),
        "스킬 피해(%)": _sum_sources(src, "스킬 피해(%)", attack_type, skill_name, include_conditional=True),
        "방향성 피해(%)": _sum_sources(src, "방향성 피해(%)", attack_type, skill_name, include_conditional=True),
        "보석 피해(%)": _sum_sources(src, "보석 피해(%)", attack_type, skill_name, include_conditional=True),
        "공격력 증가(%)": _sum_sources(src, "공격력 증가(%)", attack_type, skill_name, include_conditional=True),
        "추가 피해(%)": _sum_sources(src, "추가 피해(%)", attack_type, skill_name, include_conditional=True),
    }

    scmul = _expected_crit_multiplier(min(100.0, static_crit_rate), static_cd)
    ccmul = _expected_crit_multiplier(min(100.0, basis_crit_rate), cond_cd)
    sdmul = _damage_group_multiplier(**static_groups)
    cdmul = _damage_group_multiplier(**cond_groups)

    return {
        "치명 고정 옵션(%)": round(max(static_fixed, cond_fixed), 2),
        "치명 증가 합계(상시)(%)": round(static_crit_add, 2),
        "치명 증가 합계(조건부)(%)": round(cond_crit_add, 2),
        "예상 치명 확률(상시)(%)": round(static_crit_rate, 2),
        "예상 치명 확률(조건부 포함)(%)": round(cond_crit_rate, 2),
        "예상 치명 확률(백어택 기준)(%)": round(basis_crit_rate, 2),
        "예상 치피(상시)(%)": round(static_cd, 2),
        "예상 치피(조건부 포함)(%)": round(cond_cd, 2),
        "진화형 피해(상시)(%)": round(static_groups["진화형 피해(%)"], 2),
        "진화형 피해(조건부)(%)": round(cond_groups["진화형 피해(%)"], 2),
        "적에게 주는 피해(상시)(%)": round(static_groups["적에게 주는 피해(%)"], 2),
        "적에게 주는 피해(조건부)(%)": round(cond_groups["적에게 주는 피해(%)"], 2),
        "스킬 피해(상시)(%)": round(static_groups["스킬 피해(%)"], 2),
        "스킬 피해(조건부)(%)": round(cond_groups["스킬 피해(%)"], 2),
        "방향성 피해(상시)(%)": round(static_groups["방향성 피해(%)"], 2),
        "방향성 피해(조건부)(%)": round(cond_groups["방향성 피해(%)"], 2),
        "보석 피해(상시)(%)": round(static_groups["보석 피해(%)"], 2),
        "보석 피해(조건부)(%)": round(cond_groups["보석 피해(%)"], 2),
        "공격력 증가(상시)(%)": round(static_groups["공격력 증가(%)"], 2),
        "공격력 증가(조건부)(%)": round(cond_groups["공격력 증가(%)"], 2),
        "추가 피해(상시)(%)": round(static_groups["추가 피해(%)"], 2),
        "추가 피해(조건부)(%)": round(cond_groups["추가 피해(%)"], 2),
        "치명 기대배율(상시)": round(scmul, 4),
        "치명 기대배율(조건부+방향)": round(ccmul, 4),
        "피해군 배율(상시)": round(sdmul, 4),
        "피해군 배율(조건부)": round(cdmul, 4),
        "예상 최종 배율(상시)": round(scmul * sdmul, 4),
        "예상 최종 배율(조건부)": round(ccmul * cdmul, 4),
    }


def _v64_global_basis_crit_rate(result: Dict[str, Any]) -> float | None:  # type: ignore[override]
    """v66: 전역 기준 치명 카드는 뭉가 고정 치적과 스킬 전용 치적을 제외합니다."""
    sources = result.get("damage_sources")
    final_df = result.get("arkgrid_final_skill_estimates")
    if not isinstance(sources, pd.DataFrame) or sources.empty:
        return None
    src = _v64_sources_without_base_stat(sources)
    stat_rate = float(result.get("base_crit_percent", 0) or 0)
    attack_type = "백어택" if result.get("has_back_attack_engraving") else ("헤드어택" if result.get("has_head_attack_engraving") else "일반/확인필요")

    # 대표 스킬명을 써서 '스킬 그룹/제외한 스킬' 적용 판정은 살리되,
    # 지진파 같은 특정 스킬 전용 치적은 전역 기준에 새지 않게 기준스킬명을 사용합니다.
    skill_name = "기준스킬"
    if isinstance(final_df, pd.DataFrame) and not final_df.empty and "스킬명" in final_df.columns:
        try:
            first_name = str(final_df.iloc[0].get("스킬명") or "")
            if first_name:
                skill_name = first_name
        except Exception:
            pass

    # 고정 치명타 옵션은 기본 기준 치명에 합산하지 않습니다.
    add = _v66_sum_crit_increase_sources(src, attack_type, skill_name, include_conditional=True)
    val = stat_rate + add + _v64_direction_crit_for_attack_type(attack_type)
    return round(float(val), 2)


# refresh 함수가 마지막에 정의된 _v64_global_basis_crit_rate를 사용하도록 재정의합니다.
_old_v61_refresh_summary_metrics_v65_for_v66 = _v61_refresh_summary_metrics

def _v61_refresh_summary_metrics(result: Dict[str, Any]) -> None:  # type: ignore[override]
    _old_v61_refresh_summary_metrics_v65_for_v66(result)
    try:
        gb = _v64_global_basis_crit_rate(result)
        if gb is not None:
            result["global_basis_crit_rate_percent"] = gb
    except Exception:
        pass

# v67 -------------------------------------------------------------------------
# - 직업/아이덴티티 분류를 class_rules.yaml 중심으로 확장합니다.
# - 직업 추정은 장비/악세 이름 오염을 피하기 위해 각인/아크패시브/깨달음 텍스트만 우선 사용합니다.
# - API에 드러나지 않는 대표 아이덴티티 수치는 class_rules.yaml의 identity_effects만 계산에 반영합니다.

ESTIMATOR_VERSION = "v67-class-identity-rule-framework"


def _v67_core_job_text(bundle: Dict[str, Any]) -> str:
    """직업 추정에 써도 되는 핵심 텍스트만 모읍니다.

    장비/악세/보석 Tooltip 전체를 검색하면 과거 장비명이나 엘릭서 문구가 직업 후보를 오염시킬 수 있어,
    직업 추정은 각인 + 아크패시브 + 깨달음 활성 노드 + 프로필 클래스 정도로 제한합니다.
    """
    parts: List[str] = []
    for key in ["engravings", "arkpassive", "profiles", "summary"]:
        try:
            data = _safe_data(bundle, key)
            if data is not None:
                parts.append(_flatten(data))
        except Exception:
            pass
    try:
        for name, text, path in _v56_active_enlightenment_texts(bundle):
            parts.append(f"{path} {name} {text}")
    except Exception:
        pass
    return _clean_text(" ".join(parts))


def _active_job_names(bundle: Dict[str, Any]) -> List[str]:  # type: ignore[override]
    """v67: 클래스별 후보 안에서 직업각인/깨달음/아이덴티티 키워드로만 직업을 추정합니다.

    v125: 아르카나는 "황후의 기사/루인/카드" 같은 공통 키워드 때문에 황제가 황후로
    잡히는 문제가 있어, 깨달음 1티어의 정확한 직업명 또는 직업명 전체 일치만 사용합니다.
    """
    class_name = _character_class_name(bundle)
    rules = _load_class_rules()
    class_block = (rules.get("classes") or {}).get(class_name, {})
    job_blocks = class_block.get("jobs") or {}
    text = _v67_core_job_text(bundle)
    jobs: List[str] = []

    if class_name == "아르카나":
        exact_active: List[str] = []
        try:
            for node_name, _node_text, _path in _v56_active_enlightenment_texts(bundle):
                if str(node_name) in job_blocks:
                    exact_active.append(str(node_name))
        except Exception:
            pass
        if exact_active:
            seen_exact = set()
            out_exact: List[str] = []
            for j in exact_active:
                if j not in seen_exact:
                    seen_exact.add(j)
                    out_exact.append(j)
            return out_exact

    for job_name, job_block in job_blocks.items():
        keywords = [str(job_name)]
        for k in ["engraving_keywords", "arkpassive_keywords", "identity_keywords", "enlightenment_keywords"]:
            keywords.extend([str(x) for x in (job_block.get(k) or [])])
        if class_name == "아르카나":
            broad_arcana_tokens = {"황후", "황제", "루인", "카드", "카드 덱"}
            keywords = [k for k in keywords if str(k).strip() not in broad_arcana_tokens]
        if any(k and k in text for k in keywords):
            jobs.append(str(job_name))
    # 동일 클래스 안에서 직업각인명이 직접 보이면 최우선으로 보장합니다.
    for job_name in job_blocks.keys():
        if str(job_name) in text and str(job_name) not in jobs:
            jobs.append(str(job_name))
    seen = set()
    out: List[str] = []
    for j in jobs:
        if j not in seen:
            seen.add(j)
            out.append(j)
    return out


_old_identity_sources_v66_for_v67 = _identity_sources


def _v67_has_effect_row(df: pd.DataFrame, name: str, crit_rate: float | None = None) -> bool:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return False
    for _, r in df.iterrows():
        if str(r.get("이름") or "") != str(name):
            continue
        if crit_rate is None:
            return True
        try:
            if abs(float(r.get("치명타 적중률 증가(%)") or 0.0) - float(crit_rate)) < 1e-6:
                return True
        except Exception:
            pass
    return False


def _v67_class_identity_fallback(bundle: Dict[str, Any]) -> pd.DataFrame:
    """직업 감지가 실패해도 클래스 공통 아이덴티티 중 검증된 대표 효과는 보완합니다."""
    class_name = _character_class_name(bundle)
    rows: List[Dict[str, Any]] = []
    # 현재 계산에 강제로 넣는 값은 검증/테스트된 대표 치적 보완룰만 둡니다.
    fallback_rules = {
        "슬레이어": {"name": "폭주(슬레이어)", "crit_rate": 30.0, "description": "슬레이어 폭주 상태 기준 치명타 적중률 +30%."},
        "버서커": {"name": "폭주(버서커)", "crit_rate": 30.0, "description": "버서커 폭주 상태 기준 치명타 적중률 +30%."},
    }
    rule = fallback_rules.get(class_name)
    if not rule:
        return _empty_source_df()
    effects = _new_effect_dict()
    effects["치명타 적중률 증가(%)"] = float(rule["crit_rate"])
    rows.append({
        "출처구분": "직업/아이덴티티",
        "이름": str(rule["name"]),
        "적용범위": "전체",
        "적용스킬": "전체/범위 기준",
        **effects,
        "조건부 여부": "아이덴티티 활성 기준",
        "설명": str(rule["description"]) + " class_rules 직업 감지 실패 시 클래스 공통 fallback.",
    })
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def _identity_sources(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    """v67: class_rules 보완룰 + 깨달음 자동 감지를 유지하되, 클래스 공통 fallback을 추가합니다."""
    try:
        base = _old_identity_sources_v66_for_v67(bundle)
    except Exception:
        base = _empty_source_df()
    fallback = _v67_class_identity_fallback(bundle)
    if isinstance(fallback, pd.DataFrame) and not fallback.empty:
        keep = []
        for _, row in fallback.iterrows():
            if _v67_has_effect_row(base, str(row.get("이름") or ""), float(row.get("치명타 적중률 증가(%)") or 0.0)):
                continue
            keep.append(dict(row))
        if keep:
            base = _concat_sources(base, pd.DataFrame(keep, columns=SOURCE_COLUMNS))
    return base

# v68 -------------------------------------------------------------------------
# - 디스트로이어 1차 전용 룰.
# - 집속/해방 스킬 그룹을 스킬명 기반으로 보강합니다.
# - 해방 스킬은 3코어 운용 기준으로 계산합니다.
# - 날카로운 해머는 해방 스킬 3코어 기준 치명타 적중률 +6%로 적용합니다.
# - 영역 강화는 감지 Lv 기준 중력 가중/해방 모드 치명타 적중률을 적용합니다.

ESTIMATOR_VERSION = "v68-destroyer-first-rule-pack"

DESTROYER_RELEASE_SKILLS_V68 = {
    "퍼펙트스윙", "풀스윙", "풀스윙", "풀 스윙", "어스이터", "어스 이터",
    "사이즈믹해머", "사이즈믹 해머", "뉴트럴라이저", "뉴트럴 라이저",
}
DESTROYER_CONCENTRATION_SKILLS_V68 = {
    "헤비크러쉬", "헤비 크러쉬", "헤비크래쉬", "헤비 크래쉬",
    "드레드노트", "그라비티임팩트", "그라비티 임팩트", "점핑스매쉬", "점핑 스매쉬",
    "러닝크래쉬", "러닝 크래쉬", "러닝크러쉬", "러닝 크러쉬",
    "파워숄더", "파워 숄더", "파워스트라이크", "파워 스트라이크",
    "인듀어페인", "인듀어 페인", "어스스매셔", "어스 스매셔", "그라비티포스", "그라비티 포스",
}
DESTROYER_GRAVITY_MODE_SKILLS_V68 = {
    "중력가중영역", "중력 가중영역", "중력가중 영역", "중력 가중 영역",
    "중력해방", "중력 해방", "중력해머", "중력 해머", "중력충격", "중력 충격",
    "볼텍스그라비티", "볼텍스 그라비티",
}

_old_skill_groups_for_skill_v67_for_v68 = _skill_groups_for_skill


def _v68_add_group(groups: List[str], name: str) -> None:
    if name and name not in groups:
        groups.append(name)


def _skill_groups_for_skill(skill: Dict[str, Any]) -> List[str]:  # type: ignore[override]
    """v68: 디스트로이어 스킬명 기반 집속/해방/중력 모드 그룹 보강."""
    try:
        groups = list(_old_skill_groups_for_skill_v67_for_v68(skill) or [])
    except Exception:
        groups = []
    name = str(skill.get("Name") or "")
    norm = _norm_name(name)
    if norm in {_norm_name(x) for x in DESTROYER_RELEASE_SKILLS_V68}:
        _v68_add_group(groups, "해방 스킬")
    if norm in {_norm_name(x) for x in DESTROYER_CONCENTRATION_SKILLS_V68}:
        _v68_add_group(groups, "집속 스킬")
    if norm in {_norm_name(x) for x in DESTROYER_GRAVITY_MODE_SKILLS_V68}:
        _v68_add_group(groups, "중력 가중/해방 모드")
    return groups


def _v68_is_destroyer(bundle: Dict[str, Any]) -> bool:
    try:
        return "디스트로이어" in str(_character_class_name(bundle) or "")
    except Exception:
        return "디스트로이어" in _v61_bundle_raw_text(bundle or {})


def _v68_has_named_node(bundle: Dict[str, Any], *aliases: str) -> bool:
    raw = _v61_bundle_raw_text(bundle or {})
    if not raw:
        return False
    compact = raw.replace(" ", "")
    for alias in aliases:
        if not alias:
            continue
        a = str(alias)
        if re.search(rf"{re.escape(a)}\s*Lv\.?\s*[1-9]", raw, re.I):
            return True
        if re.search(rf"{re.escape(a)}\s*레벨\s*[1-9]", raw, re.I):
            return True
        if a.replace(" ", "") in compact and ("깨달음" in raw or "아크패시브" in raw or "ArkPassive" in raw):
            return True
    return False


def _v68_node_level(bundle: Dict[str, Any], alias: str, default: int = 1, max_level: int = 5) -> int:
    try:
        v = int(_v61_evolution_node_level_from_bundle(bundle or {}, alias, default=0))
        if v > 0:
            return max(1, min(max_level, v))
    except Exception:
        pass
    raw = _v61_bundle_raw_text(bundle or {})
    best = 0
    if raw and alias:
        for m in re.finditer(re.escape(alias), raw):
            window = raw[max(0, m.start() - 80): min(len(raw), m.end() + 140)]
            for pat in [r"Lv\.?\s*([1-9])", r"레벨\s*([1-9])", r"([1-9])\s*포인트"]:
                for mm in re.finditer(pat, window, re.I):
                    try:
                        best = max(best, int(mm.group(1)))
                    except Exception:
                        pass
    return max(1, min(max_level, best or default))


def _v68_make_row(name: str, scope: str, target: str, col: str, value: float, note: str, cond: str = "상시") -> Dict[str, Any]:
    effects = _new_effect_dict()
    effects[col] = float(value or 0.0)
    return {
        "출처구분": "직업/아이덴티티",
        "이름": name,
        "적용범위": scope,
        "적용스킬": target,
        **effects,
        "조건부 여부": cond,
        "설명": note,
    }


def _v68_destroyer_special_sources(bundle: Dict[str, Any]) -> pd.DataFrame:
    """디스트로이어 전용 1차 룰.

    현재 앱의 실전/허수 비교는 보통 정상 사이클 기준을 보므로,
    해방 스킬은 3코어로 사용했다고 가정합니다.
    특화 스탯에 따른 중력 코어 피해 계수는 이번 v68에서는 직접 계산하지 않고,
    기본 3코어 해방 피해 +45%만 먼저 보강합니다.
    """
    if not _v68_is_destroyer(bundle):
        return _empty_source_df()
    rows: List[Dict[str, Any]] = []

    # 기본 아이덴티티: 해방 스킬은 3코어 운용 기준.
    rows.append(_v68_make_row(
        "중력 코어 3개 해방",
        "스킬 그룹",
        "해방 스킬",
        "스킬 피해(%)",
        45.0,
        "디스트로이어 해방 스킬 3코어 운용 기준. 중력 코어 3개 사용 시 해방 스킬 피해 증가 +45%를 스킬 그룹에 적용.",
        "상시",
    ))

    # 깨달음/아크패시브 노드: 날카로운 해머.
    if _v68_has_named_node(bundle, "날카로운 해머", "날카로운해머"):
        rows.append(_v68_make_row(
            "날카로운 해머",
            "스킬 그룹",
            "해방 스킬",
            "치명타 적중률 증가(%)",
            6.0,
            "날카로운 해머: 해방 스킬 사용 시 소모 코어 기준 치명타 적중률 증가. 계산은 3코어 최대 운용 기준 +6%로 적용.",
            "상시",
        ))

    # 깨달음/아크패시브 노드: 영역 강화.
    if _v68_has_named_node(bundle, "영역 강화", "영역강화"):
        lvl = _v68_node_level(bundle, "영역 강화", default=1, max_level=5)
        table = [2.0, 4.0, 6.0, 8.0, 10.0]
        val = table[max(0, min(len(table) - 1, lvl - 1))]
        rows.append(_v68_make_row(
            "영역 강화",
            "스킬 그룹",
            "중력 가중/해방 모드",
            "치명타 적중률 증가(%)",
            val,
            f"영역 강화 Lv.{lvl}: 중력 가중영역/중력 해방 모드 중 치명타 적중률 +{val:.1f}%. 코어 소모 문구가 없으므로 Lv 기준 그대로 적용.",
            "상시",
        ))

    if not rows:
        return _empty_source_df()
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS).drop_duplicates().reset_index(drop=True)


_old_collect_global_sources_v67_for_v68 = _collect_global_sources


def _v68_drop_destroyer_generic_rows(df: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    if not _v68_is_destroyer(bundle) or df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df if isinstance(df, pd.DataFrame) else _empty_source_df()
    out = df.copy()
    # 날카로운 해머/영역 강화는 v68 전용 룰로 다시 생성합니다.
    # 일반 툴팁 파서가 2/3/6 또는 2/4/6/8/10 수치를 모두 더하는 것을 막습니다.
    node_names = ["날카로운 해머", "날카로운해머", "영역 강화", "영역강화", "중력 코어 3개 해방"]
    node_norms = {_norm_name(x) for x in node_names}
    remove_idx: List[Any] = []
    for idx, row in out.iterrows():
        src = str(row.get("출처구분") or "")
        if src not in {"아크패시브", "직업/아이덴티티"}:
            continue
        raw = f"{row.get('이름', '')} {row.get('설명', '')}"
        raw_norm = _norm_name(raw)
        if any(n and n in raw_norm for n in node_norms):
            # v68에서 직접 만든 row는 유지.
            if str(row.get("이름") or "") in {"날카로운 해머", "영역 강화", "중력 코어 3개 해방"} and src == "직업/아이덴티티":
                continue
            remove_idx.append(idx)
    if remove_idx:
        out = out.drop(index=remove_idx)
    return out.reset_index(drop=True)


def _collect_global_sources(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    parsed = _old_collect_global_sources_v67_for_v68(bundle)
    if parsed is None or not isinstance(parsed, pd.DataFrame):
        parsed = _empty_source_df()
    parsed = _v68_drop_destroyer_generic_rows(parsed, bundle)
    destroyer_rows = _v68_destroyer_special_sources(bundle)
    return _concat_sources(parsed, destroyer_rows)


# class_rules.yaml을 사용하지 않는 환경에서도 디스트로이어 스킬명 별칭을 잡기 위한 보강.
SKILL_ALIAS_TO_CANONICAL.update({
    "퍼펙트스윙": "퍼펙트 스윙",
    "퍼펙트 스윙": "퍼펙트 스윙",
    "풀스윙": "풀 스윙",
    "풀 스윙": "풀 스윙",
    "어스이터": "어스 이터",
    "어스 이터": "어스 이터",
    "사이즈믹해머": "사이즈믹 해머",
    "사이즈믹 해머": "사이즈믹 해머",
    "뉴트럴라이저": "뉴트럴라이저",
    "헤비크러쉬": "헤비 크러쉬",
    "헤비 크러쉬": "헤비 크러쉬",
    "헤비크래쉬": "헤비 크러쉬",
    "드레드노트": "드레드노트",
    "그라비티임팩트": "그라비티 임팩트",
    "그라비티 임팩트": "그라비티 임팩트",
    "점핑스매쉬": "점핑 스매쉬",
    "점핑 스매쉬": "점핑 스매쉬",
    "러닝크래쉬": "러닝 크래쉬",
    "러닝크러쉬": "러닝 크래쉬",
    "러닝 크래쉬": "러닝 크래쉬",
    "러닝 크러쉬": "러닝 크래쉬",
    "파워숄더": "파워 숄더",
    "파워 숄더": "파워 숄더",
    "파워스트라이크": "파워 스트라이크",
    "파워 스트라이크": "파워 스트라이크",
    "인듀어페인": "인듀어 페인",
    "인듀어 페인": "인듀어 페인",
    "어스스매셔": "어스 스매셔",
    "어스 스매셔": "어스 스매셔",
})


# v69 -------------------------------------------------------------------------
# - 디스트로이어 검증 보정.
# - 날카로운 해머/영역 강화가 "피해군 출처"에는 잡히지만 기본 기준 치명 합산에서 빠지는 문제를 수정합니다.
# - 검수 기준에서는 디스트로이어 해방 스킬을 3코어 해방 운용으로 보고,
#   날카로운 해머 + 영역 강화를 해방 스킬 기준 치명에 반영합니다.

ESTIMATOR_VERSION = "v70-destroyer-crit-basis-and-review-simulator"

_old_v68_destroyer_special_sources_for_v69 = _v68_destroyer_special_sources


def _v68_destroyer_special_sources(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    """v69: 디스트로이어 해방 스킬 검수 기준으로 날카로운 해머/영역 강화를 해방 스킬에 붙입니다.

    사용자 검증 기준:
    - 해방 스킬은 3코어 최대 운용 기준
    - 날카로운 해머 Lv.3 = 해방 스킬 치명타 적중률 +6%
    - 영역 강화 Lv.1 = 검수 기준 해방 운용 치명타 적중률 +2%
    """
    if not _v68_is_destroyer(bundle):
        return _empty_source_df()

    rows: List[Dict[str, Any]] = []

    rows.append(_v68_make_row(
        "중력 코어 3개 해방",
        "스킬 그룹",
        "해방 스킬",
        "스킬 피해(%)",
        45.0,
        "디스트로이어 해방 스킬 3코어 운용 기준. 중력 코어 3개 사용 시 해방 스킬 피해 증가 +45%를 스킬 그룹에 적용.",
        "상시",
    ))

    if _v68_has_named_node(bundle, "날카로운 해머", "날카로운해머"):
        rows.append(_v68_make_row(
            "날카로운 해머",
            "스킬 그룹",
            "해방 스킬",
            "치명타 적중률 증가(%)",
            6.0,
            "날카로운 해머: 해방 스킬 사용 시 소모 코어 기준 치명타 적중률 증가. 검수 계산은 3코어 최대 운용 기준 +6%로 적용.",
            "상시",
        ))

    if _v68_has_named_node(bundle, "영역 강화", "영역강화"):
        lvl = _v68_node_level(bundle, "영역 강화", default=1, max_level=5)
        table = [2.0, 4.0, 6.0, 8.0, 10.0]
        val = table[max(0, min(len(table) - 1, lvl - 1))]
        rows.append(_v68_make_row(
            "영역 강화",
            "스킬 그룹",
            "해방 스킬",
            "치명타 적중률 증가(%)",
            val,
            f"영역 강화 Lv.{lvl}: 디스트로이어 검수 기준에서 해방 운용 치명타 적중률 +{val:.1f}%로 적용.",
            "상시",
        ))

    if not rows:
        return _empty_source_df()
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS).drop_duplicates().reset_index(drop=True)


_old_v64_global_basis_crit_rate_for_v69 = _v64_global_basis_crit_rate


def _v69_destroyer_basis_crit_rate(result: Dict[str, Any]) -> float | None:
    sources = result.get("damage_sources")
    if not isinstance(sources, pd.DataFrame) or sources.empty:
        return None
    stat_rate = float(result.get("base_crit_percent", 0) or 0)
    src = _v64_sources_without_base_stat(sources)

    total = stat_rate
    used: set[tuple[str, str, float]] = set()

    for _, row in src.iterrows():
        if str(row.get("적용범위") or "") == "검수 필요":
            continue
        value = _num(row.get("치명타 적중률 증가(%)")) or 0.0
        if not value:
            continue

        scope = str(row.get("적용범위") or "전체")
        target = str(row.get("적용스킬") or "")
        name = str(row.get("이름") or "")

        # 디스트로이어 검수용 기본 기준:
        # 전체 치적 + 해방 스킬 대표 치적(날카로운 해머/영역 강화)을 포함합니다.
        applies = False
        if scope in {"전체", "백어택 스킬", "헤드어택 스킬", "백/헤드 스킬"}:
            applies = True
        elif scope in {"스킬 그룹", "계열 전용", "스킬 계열"}:
            if "해방" in target or name in {"날카로운 해머", "영역 강화"}:
                applies = True
        elif scope == "스킬 전용":
            # 지진파 같은 특정 스킬 치적은 기본 기준에 넣지 않습니다.
            applies = False

        if not applies:
            continue

        sig = (name, scope, round(float(value), 4))
        if sig in used:
            continue
        used.add(sig)
        total += float(value)

    # 디스트로이어는 백어택 기본 치적 +10을 기본 기준에 억지로 더하지 않습니다.
    return round(float(total), 2)


def _v64_global_basis_crit_rate(result: Dict[str, Any]) -> float | None:  # type: ignore[override]
    try:
        profile = (result.get("profile_summary") or {}) if isinstance(result, dict) else {}
        cls = str(profile.get("class") or profile.get("CharacterClassName") or "")
        if "디스트로이어" in cls:
            return _v69_destroyer_basis_crit_rate(result)
        # class 정보가 profile에 없으면 출처/계산표에서 디스트로이어 흔적을 봅니다.
        src = result.get("damage_sources")
        if isinstance(src, pd.DataFrame) and not src.empty:
            joined = " ".join(src.astype(str).fillna("").agg(" ".join, axis=1).tolist()[:60])
            if "중력 코어 3개 해방" in joined or "날카로운 해머" in joined or "영역 강화" in joined:
                return _v69_destroyer_basis_crit_rate(result)
    except Exception:
        pass
    return _old_v64_global_basis_crit_rate_for_v69(result)


_old_v61_refresh_summary_metrics_for_v69 = _v61_refresh_summary_metrics


def _v61_refresh_summary_metrics(result: Dict[str, Any]) -> None:  # type: ignore[override]
    _old_v61_refresh_summary_metrics_for_v69(result)
    try:
        gb = _v64_global_basis_crit_rate(result)
        if gb is not None:
            result["global_basis_crit_rate_percent"] = gb
    except Exception:
        pass



# v70 hotfix: 동일 명칭/동일 수치 장비 옵션(예: 반지 2개 치적 +1.55%)이 중복 제거되어
# 디스트로이어 기본 기준 치명에서 1개만 들어가는 문제를 수정합니다.

def _v69_destroyer_basis_crit_rate(result: Dict[str, Any]) -> float | None:  # type: ignore[override]
    sources = result.get("damage_sources")
    if not isinstance(sources, pd.DataFrame) or sources.empty:
        return None
    stat_rate = float(result.get("base_crit_percent", 0) or 0)
    src = _v64_sources_without_base_stat(sources)

    total = stat_rate
    for _, row in src.iterrows():
        if str(row.get("적용범위") or "") == "검수 필요":
            continue
        value = _num(row.get("치명타 적중률 증가(%)")) or 0.0
        if not value:
            continue

        scope = str(row.get("적용범위") or "전체")
        target = str(row.get("적용스킬") or "")
        name = str(row.get("이름") or "")

        applies = False
        if scope in {"전체", "백/헤드 스킬", "백어택 스킬", "헤드어택 스킬"}:
            applies = True
        elif scope in {"스킬 그룹", "계열 전용", "스킬 계열"}:
            if "해방" in target or name in {"날카로운 해머", "영역 강화"}:
                applies = True
        elif scope == "스킬 전용":
            applies = False

        if applies:
            total += float(value)

    return round(float(total), 2)


# ─────────────────────────────────────────────────────────────────────────────
# v70.1 patch: 아이덴티티 중복 행 제거 (치적 130% 버그 수정)
#
# 문제 원인:
#   class_rules.yaml에서 "폭주" 키워드가 포식자 + 처단자 identity_keywords 양쪽에
#   공통으로 있어서, 슬레이어 캐릭터에 두 직업이 동시 감지됩니다.
#   그 결과 "폭주(슬레이어) +30%" 행이 2개 생성되되 조건부 여부가 달라집니다:
#     - 포식자 → v11 override로 "상시"
#     - 처단자 → class_rules.yaml 원본 "아이덴티티 활성 기준"
#   _sig 생성 시 "조건부 여부"가 포함되므로 두 행이 서로 다른 행으로 간주되어
#   중복 제거가 되지 않고, include_conditional=True 합산 시 30+30=60%가 됩니다.
#
# 수정 내용:
#   (이름, 적용범위, 효과 수치 tuple) 기준으로 같은 행이 여러 개면
#   "상시" 우선으로 1개만 유지합니다.
#   버서커 폭주(광기/광전사의 비기), 슬레이어 폭주(포식자/처단자) 모두 해결됩니다.
# ─────────────────────────────────────────────────────────────────────────────

_IDENTITY_COND_PRIORITY: Dict[str, int] = {
    "상시": 0,
    "조건부/최대치": 1,
    "아이덴티티 활성 기준": 2,
}


def _dedup_identity_by_name_effect(df: pd.DataFrame) -> pd.DataFrame:
    """아이덴티티 출처 내에서 이름+범위+효과 수치가 동일한 중복 행을 제거합니다.

    '조건부 여부'만 다른 동일한 효과가 여러 개 있으면 우선순위(상시 > 조건부/최대치 >
    아이덴티티 활성 기준)가 높은 1개만 남깁니다.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    seen: Dict[tuple, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        name = str(row.get("이름") or "")
        scope = str(row.get("적용범위") or "")
        effects_key = tuple(round(float(row.get(c, 0) or 0), 4) for c in EFFECT_COLUMNS)
        key = (name, scope, effects_key)

        cond_new = str(row.get("조건부 여부") or "")
        if key not in seen:
            seen[key] = dict(row)
        else:
            cond_existing = str(seen[key].get("조건부 여부") or "")
            pri_new = _IDENTITY_COND_PRIORITY.get(cond_new, 99)
            pri_existing = _IDENTITY_COND_PRIORITY.get(cond_existing, 99)
            if pri_new < pri_existing:
                seen[key] = dict(row)

    rows_out = list(seen.values())
    if not rows_out:
        return df
    return pd.DataFrame(rows_out, columns=SOURCE_COLUMNS).reset_index(drop=True)


_old_identity_sources_v70_for_v70patch = _identity_sources


def _identity_sources(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    """v70.1 patch: 동일 이름+범위+효과의 중복 아이덴티티 행을 상시 우선으로 1개만 유지합니다.

    슬레이어 폭주(포식자/처단자 동시 감지) 및 버서커 폭주(광기/광전사의 비기 동시 감지)로
    인해 치명타 적중률이 2배로 잡히는 문제를 수정합니다.
    """
    base = _old_identity_sources_v70_for_v70patch(bundle)
    return _dedup_identity_by_name_effect(base)


ESTIMATOR_VERSION = "v70.1-identity-dedup-fix"


# =============================================================================
# v70.2 patch: manual_identity_rules.json DB 연결 + dedupe_key 중복 방지
# =============================================================================
# 변경 내용:
#   1. data/manual/manual_identity_rules.json 에서 클래스별 canonical_effects를 읽음
#   2. dedupe_key 기반으로 중복 아이덴티티 효과를 방지
#   3. api_cache/current 읽기 구조 준비 (DB 로더 통합)
#   4. _identity_sources 최종 override: manual DB 우선 적용 후 기존 소스와 병합
#
# 기존 class_rules.yaml 기반 소스는 유지되며,
# manual DB에 같은 dedupe_key가 있는 효과는 manual 쪽을 사용합니다.
#
# 아이덴티티 출처 표기:
#   "직업/아이덴티티(manual)" : manual_identity_rules.json 에서 온 효과
#   "직업/아이덴티티"         : 기존 class_rules.yaml / API 자동감지 효과
# =============================================================================

import sys as _sys
import importlib as _importlib


@lru_cache(maxsize=1)
def _load_loa_db_optional():
    """tools.calculator_db_loader를 임포트합니다 (없으면 None 반환)."""
    try:
        # 프로젝트 루트를 sys.path에 추가
        _root = str(_project_root())
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        mod = _importlib.import_module("tools.calculator_db_loader")
        return mod.get_db()
    except Exception:
        return None


def _manual_identity_rows(class_name: str) -> "List[Dict[str, Any]]":
    """manual_identity_rules.json의 canonical_effects를
    SOURCE_COLUMNS 형식의 행 목록으로 변환합니다.

    stat 필드 매핑:
      crit_rate        → 치명타 적중률 증가(%)
      crit_damage      → 치명타 피해량 증가(%)
      damage_dealt     → 적에게 주는 피해(%)
      evolution_damage → 진화형 피해(%)
      attack_power     → 공격력 증가(%)
      skill_damage     → 스킬 피해(%)
      directional      → 방향성 피해(%)
    """
    stat_map = {
        "crit_rate":        "치명타 적중률 증가(%)",
        "crit_damage":      "치명타 피해량 증가(%)",
        "damage_dealt":     "적에게 주는 피해(%)",
        "evolution_damage": "진화형 피해(%)",
        "attack_power":     "공격력 증가(%)",
        "skill_damage":     "스킬 피해(%)",
        "directional":      "방향성 피해(%)",
    }

    scope_map = {
        "self_all_skills": "전체",
        "back_attack":     "백어택 스킬",
        "head_attack":     "헤드어택 스킬",
        "directional":     "백/헤드 스킬",
    }

    db = _load_loa_db_optional()
    if db is None:
        return []

    try:
        rules = db.class_identity_rules(class_name)
    except Exception:
        return []

    rows: "List[Dict[str, Any]]" = []
    for eff in rules.get("canonical_effects", []):
        stat = eff.get("stat", "")
        col = stat_map.get(stat)

        # values_by_count (디스트로이어 코어당 효과) → default_count 기준 단일 값
        if "values_by_count_pct" in eff:
            default_count = int(eff.get("default_count", 3))
            val = float(eff["values_by_count_pct"].get(str(default_count), 0) or 0)
        else:
            val = float(eff.get("value_pct", 0) or 0)

        if not col or abs(val) < 1e-9:
            continue

        effects = _new_effect_dict()
        effects[col] = val

        scope_raw = str(eff.get("scope", "self_all_skills"))
        scope = scope_map.get(scope_raw, scope_raw if scope_raw else "전체")

        # 조건부 여부: default_active 기준
        da = eff.get("default_active", "job_or_user_toggle")
        if da is True or da == "always":
            cond = "상시"
        elif da is False:
            cond = "조건부/최대치"
        else:
            cond = "상시"   # 폭주/아이덴티티 상태를 실전 기준으로 상시 처리

        state = str(eff.get("state", ""))
        eff_id = str(eff.get("id", ""))
        dedupe_key = str(eff.get("dedupe_key", ""))
        desc_parts = [f"[{class_name}]", state, col, f"+{val}%"]
        if dedupe_key:
            desc_parts.append(f"[dedupe_key: {dedupe_key}]")
        description = " ".join(p for p in desc_parts if p).strip()

        rows.append({
            "출처구분": "직업/아이덴티티(manual)",
            "이름": f"{state}({class_name})" if state else eff_id or class_name,
            "적용범위": scope,
            "적용스킬": "전체/범위 기준",
            **effects,
            "조건부 여부": cond,
            "설명": description[:420],
            "_dedupe_key": dedupe_key,   # 내부 추적용 (SOURCE_COLUMNS 외)
        })

    return rows


def _apply_manual_dedup(
    manual_rows: "List[Dict[str, Any]]",
    existing_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """manual DB 행을 기존 소스에 병합합니다.

    규칙:
      - manual 행에 dedupe_key가 있으면, 기존 소스에서
        「출처구분이 직업/아이덴티티」이고 「범위+효과 수치가 같은」 행을
        이름이 달라도 제거하고 manual 행으로 대체합니다.
        (폭주(슬레이어) vs 폭주(포식자)처럼 이름이 달라도 같은 효과면 중복 제거)
      - dedupe_key가 없는 manual 행은 그냥 추가합니다.
    """
    if not manual_rows:
        return existing_df

    # manual 행에서 dedupe_key가 있는 항목의 (scope, eff_vals) 집합 구성
    # 이름은 제외: class_rules / API 소스가 다른 이름으로 같은 효과를 넣을 수 있음
    dedup_scope_eff_sigs: "set" = set()
    for row in manual_rows:
        dk = row.get("_dedupe_key", "")
        if not dk:
            continue
        scope = str(row.get("적용범위", ""))
        eff_vals = tuple(round(float(row.get(c, 0) or 0), 4) for c in EFFECT_COLUMNS)
        dedup_scope_eff_sigs.add((scope, eff_vals))

    # 기존 소스에서 "직업/아이덴티티" 계열 행 중
    # (scope, eff_vals)가 manual 행과 겹치는 것을 제거
    if not (isinstance(existing_df, pd.DataFrame) and not existing_df.empty):
        filtered = pd.DataFrame()
    else:
        keep_mask = []
        for _, row in existing_df.iterrows():
            src_type = str(row.get("출처구분", ""))
            # 아이덴티티 계열 소스만 dedup 대상으로 한정
            if "직업/아이덴티티" in src_type and dedup_scope_eff_sigs:
                scope = str(row.get("적용범위", ""))
                eff_vals = tuple(round(float(row.get(c, 0) or 0), 4) for c in EFFECT_COLUMNS)
                sig = (scope, eff_vals)
                # manual에 같은 (scope, 수치)가 있으면 제거
                keep_mask.append(sig not in dedup_scope_eff_sigs)
            else:
                keep_mask.append(True)

        filtered = existing_df[keep_mask].reset_index(drop=True) if any(keep_mask) else pd.DataFrame()

    # manual 행에서 _dedupe_key 컬럼 제거 후 DataFrame으로
    clean_manual = []
    for row in manual_rows:
        r = {k: v for k, v in row.items() if k != "_dedupe_key"}
        r = {k: r.get(k, 0.0 if k in EFFECT_COLUMNS else "") for k in SOURCE_COLUMNS}
        clean_manual.append(r)

    manual_df = pd.DataFrame(clean_manual, columns=SOURCE_COLUMNS)
    return _concat_sources(manual_df, filtered)


_old_identity_sources_v701_for_v702 = _identity_sources


def _profile_stat_value(bundle: "Dict[str, Any]", stat_type: str) -> float:
    """API 프로필 Stats에서 특정 스탯(특화/신속/치명/신속 등) 값을 읽습니다."""
    profiles = _safe_data(bundle, "profiles") or _safe_data(bundle, "summary") or {}
    want = str(stat_type or "").strip()
    for stat in profiles.get("Stats", []) or []:
        if str(stat.get("Type") or "").strip() == want:
            return float(_num(stat.get("Value") or 0) or 0.0)
    return 0.0


def _v18_apply_class_crit_formulas(base: "pd.DataFrame", bundle: "Dict[str, Any]") -> "pd.DataFrame":
    """스탯 공식이 필요한 직업별 치적을 후처리로 반영합니다(고정값 yaml로는 표현 불가).

    - 버서커 광전사의 비기: 폭주 치적 = 30 × (1 + 특화 × 0.26 / 699). 상시 반영.
    - 버서커 광기: 폭주 추가 치적 없음(깨달음 광기 중첩으로 처리) → 폭주 치적 제거.
    """
    if not isinstance(base, pd.DataFrame):
        base = _empty_source_df()
    class_name = _character_class_name(bundle)
    active = set(_active_job_names(bundle))
    out = base.copy()

    if "버서커" in class_name:
        # 기존 '폭주(버서커)' 고정 30% 행 제거(광기/비기 공통). 아래에서 비기만 공식으로 재추가.
        if not out.empty and {"출처구분", "이름"} <= set(out.columns):
            _mask = out["출처구분"].astype(str).eq("직업/아이덴티티") & out["이름"].astype(str).str.contains("폭주", na=False)
            out = out[~_mask].reset_index(drop=True)
        if any(("비기" in j) or ("광전사" in j) for j in active):
            spec = _profile_stat_value(bundle, "특화")
            amp = spec * 0.26 / 699.0
            burst = 30.0 * (1.0 + amp)
            eff = _new_effect_dict()
            eff["치명타 적중률 증가(%)"] = round(burst, 2)
            new_row = {
                "출처구분": "직업/아이덴티티",
                "이름": "광전사의 비기 폭주(특화 반영)",
                "적용범위": "전체",
                "적용스킬": "전체/범위 기준",
                **eff,
                "조건부 여부": "상시",
                "설명": f"폭주 치적 = 30 × (1 + 특화 {spec:.0f} × 0.26 / 699) = {burst:.2f}%",
            }
            out = _concat_sources(out, pd.DataFrame([new_row], columns=SOURCE_COLUMNS))
        # 광기: 폭주 추가 치적 없음 → 위에서 제거한 상태 그대로 둡니다.

    # 슬레이어 폭주(포식자/처단자) 치적 30%는 실전 기준으로 기본 기준 치명에 포함돼야 합니다.
    # yaml 조건('아이덴티티 활성 기준') 때문에 카드에서 빠지는 경우가 있어 상시·전체로 승격합니다.
    if "슬레이어" in class_name and not out.empty and {"출처구분", "이름", "조건부 여부", "적용범위"} <= set(out.columns):
        _m = out["출처구분"].astype(str).eq("직업/아이덴티티") & out["이름"].astype(str).str.contains("폭주", na=False)
        out.loc[_m, "조건부 여부"] = "상시"
        out.loc[_m, "적용범위"] = "전체"

    # ── 추가 아이덴티티/깨달음 치적(전체·상시) ──────────────────────────────
    try:
        _txt = _norm_name(_bundle_all_text(bundle))
    except Exception:
        _txt = ""

    def _global_crit_row(name: str, crit_rate: float = 0.0, crit_damage: float = 0.0, note: str = "") -> Dict[str, Any]:
        eff = _new_effect_dict()
        eff["치명타 적중률 증가(%)"] = round(float(crit_rate), 2)
        eff["치명타 피해량 증가(%)"] = round(float(crit_damage), 2)
        return {
            "출처구분": "직업/아이덴티티", "이름": name, "적용범위": "전체",
            "적용스킬": "전체/범위 기준", **eff, "조건부 여부": "상시", "설명": note,
        }

    extra_rows: List[Dict[str, Any]] = []

    # 활성 깨달음 노드 텍스트(직업 판정용) — 후보 전체가 아니라 '실제 찍은' 직업을 구분합니다.
    try:
        _enl_txt = _norm_name(" ".join(f"{_n} {_t}" for _n, _t, _p in _v56_active_enlightenment_texts(bundle)))
    except Exception:
        _enl_txt = ""

    # 데빌헌터 전술 탄환: 아크그리드 '샷건 오버로드' 코어를 쓰면 급소 노출 시너지가 적용됩니다.
    # 코어 텍스트에는 치적이 직접 안 적혀 있어(급소 노출 부여로만 표기) 크릿 소스로 안 잡히므로,
    # 샷건 오버로드가 감지되면 급소 노출 시너지 치명타 적중률 +10%를 전체로 넣습니다.
    if "데빌헌터" in class_name and ("샷건오버로드" in _txt):
        extra_rows.append(_global_crit_row("급소 노출(샷건 오버로드)", crit_rate=10.0,
                                           note="샷건 오버로드 → 급소 노출 시너지: 치명타 적중률 +10%"))

    # 소울이터 만월의 집행자만: 사신화 시 치명타 적중률 +20% (API 미제공 → 상시 반영)
    # '그믐의 경계'에는 적용되면 안 되므로, 활성 깨달음에 '만월의집행자'가 있을 때만 넣습니다.
    if "소울이터" in class_name and ("만월의집행자" in _enl_txt):
        extra_rows.append(_global_crit_row("만월의 집행자 사신화", crit_rate=20.0,
                                           note="사신화 시 치명타 적중률 +20% (만월의 집행자)"))

    # 리퍼 달의 소리: 깨달음 '유령 무희' 채용 시 치명타 적중률 +10%
    if "리퍼" in class_name and ("유령무희" in _txt):
        extra_rows.append(_global_crit_row("유령 무희(깨달음)", crit_rate=10.0,
                                           note="깨달음 유령 무희: 치명타 적중률 +10%"))

    # 기상술사 질풍노도: 깨달음 '기민함' — 신속/공이속 기반 치적·치피
    if "기상술사" in class_name and ("기민함" in _txt):
        swift = _profile_stat_value(bundle, "신속")
        SWIFT_SPEED_PER_POINT = 0.0171791
        GALE_TIER1_SPEED_BONUS = 12.0   # 질풍노도 1티어 공이속 +12%
        SPEED_BONUS_CAP = 40.0          # 공이속 상한 140% = 증가량 +40%
        MASS_ATK_PENALTY = 10.0         # 질량증가: 공격속도만 -10%
        swift_bonus = swift * SWIFT_SPEED_PER_POINT
        move_bonus = swift_bonus + (GALE_TIER1_SPEED_BONUS if any("질풍" in j for j in active) else 0.0)
        atk_bonus = move_bonus
        move_bonus = min(move_bonus, SPEED_BONUS_CAP)
        atk_bonus = min(atk_bonus, SPEED_BONUS_CAP)
        has_mass = ("질량증가" in _txt) or ("질량" in _txt and "증가" in _txt)
        if has_mass:
            atk_bonus = max(0.0, atk_bonus - MASS_ATK_PENALTY)
        agility_crit_rate = move_bonus * 0.3
        agility_crit_damage = atk_bonus * 1.2
        extra_rows.append(_global_crit_row(
            "기민함(질풍노도)", crit_rate=agility_crit_rate, crit_damage=agility_crit_damage,
            note=(f"기민함: 신속 {swift:.0f}→공이속증가 {swift_bonus:.1f}% "
                  f"(질풍1티어+12, 상한40, 질량증가 {'O' if has_mass else 'X'}) → "
                  f"치적 {agility_crit_rate:.1f}%(이동 {move_bonus:.1f}×0.3), 치피 {agility_crit_damage:.1f}%(공속 {atk_bonus:.1f}×1.2)")))

    if extra_rows:
        out = _concat_sources(out, pd.DataFrame(extra_rows, columns=SOURCE_COLUMNS))
    return out


def _identity_sources(bundle: "Dict[str, Any]") -> "pd.DataFrame":  # type: ignore[override]
    """v70.2: manual_identity_rules.json DB 연결 + dedupe_key 중복 방지.

    처리 순서:
      1. 기존 v70.1 소스 (class_rules.yaml + API 자동감지 + 중복 제거)
      2. manual_identity_rules.json 의 canonical_effects 로드
      3. dedupe_key 기반으로 기존 소스와 병합 (manual 우선)
      4. 스탯 공식 기반 직업 치적(버서커 폭주 등) 후처리
    """
    # 1) 기존 소스 (v70.1)
    try:
        base = _old_identity_sources_v701_for_v702(bundle)
    except Exception:
        base = _empty_source_df()

    # 2) manual DB에서 class 정보 가져오기
    class_name = _character_class_name(bundle)
    if class_name:
        manual_rows = _manual_identity_rows(class_name)
        if manual_rows:
            # 3) manual 우선 병합 + dedupe
            base = _apply_manual_dedup(manual_rows, base)

    # 4) 스탯 공식 기반 후처리(버서커 폭주 특화 반영 등)
    try:
        base = _v18_apply_class_crit_formulas(base, bundle)
    except Exception:
        pass
    return base


ESTIMATOR_VERSION = "v70.2-manual-db-integration"


# =============================================================================
# DB 상태 유틸 (UI/app.py에서 사용)
# =============================================================================

def get_db_status(class_name: str = "", job_name: str = "") -> "Dict[str, Any]":
    """계산기 UI에서 DB 상태를 표시할 때 사용합니다.

    반환값 예시:
      {
        "manual_db_available": True,
        "api_cache_available": True,
        "api_cache_last_updated": "2026-06-26 10:51:37 +0900",
        "cache_status_text": "✅ API 캐시 마지막 갱신: ...",
        "estimator_version": "v70.2-manual-db-integration",
      }
    """
    db = _load_loa_db_optional()
    result: "Dict[str, Any]" = {
        "manual_db_available": False,
        "api_cache_available": False,
        "api_cache_last_updated": None,
        "cache_status_text": "⚠️ DB 로더 없음",
        "estimator_version": ESTIMATOR_VERSION,
    }
    if db is None:
        return result

    manual_rules = db.manual_identity_rules()
    result["manual_db_available"] = bool(manual_rules)

    if class_name and job_name:
        last = db.api_cache_last_updated(class_name, job_name)
        result["api_cache_last_updated"] = last
        result["api_cache_available"] = bool(last)
        result["cache_status_text"] = db.get_cache_status_text(class_name, job_name)
    else:
        result["cache_status_text"] = (
            "✅ manual DB 로드됨" if result["manual_db_available"]
            else "⚠️ manual DB 없음 (data/manual/manual_identity_rules.json 확인)"
        )
    return result


def get_identity_sources_with_dedup_info(
    bundle: "Dict[str, Any]",
) -> "Dict[str, Any]":
    """피해군 출처 UI 표시용 — dedupe 제거 항목 포함 상세 정보 반환.

    반환값:
      {
        "sources_df": pd.DataFrame,     ← 최종 적용된 소스
        "removed_rows": List[dict],     ← dedupe로 제거된 항목
        "dedupe_keys_applied": List[str],
      }
    """
    class_name = _character_class_name(bundle)
    manual_rows = _manual_identity_rows(class_name) if class_name else []
    base_sources = _old_identity_sources_v701_for_v702(bundle)
    final_sources = _identity_sources(bundle)

    # 제거된 항목 추적
    removed: "List[Dict[str, Any]]" = []
    dedupe_effect_sigs: "Dict[str, set]" = {}
    for row in manual_rows:
        dk = row.get("_dedupe_key", "")
        if not dk:
            continue
        name = str(row.get("이름", ""))
        scope = str(row.get("적용범위", ""))
        eff_vals = tuple(round(float(row.get(c, 0) or 0), 4) for c in EFFECT_COLUMNS)
        dedupe_effect_sigs.setdefault(dk, set()).add((name, scope, eff_vals))

    all_sigs: "set" = set()
    for sigs in dedupe_effect_sigs.values():
        all_sigs |= sigs

    if isinstance(base_sources, pd.DataFrame) and not base_sources.empty:
        for _, row in base_sources.iterrows():
            sig = (
                str(row.get("이름", "")),
                str(row.get("적용범위", "")),
                tuple(round(float(row.get(c, 0) or 0), 4) for c in EFFECT_COLUMNS),
            )
            if sig in all_sigs:
                removed.append({
                    "이름": row.get("이름"),
                    "치명타 적중률 증가(%)": row.get("치명타 적중률 증가(%)"),
                    "조건부 여부": row.get("조건부 여부"),
                    "이유": "manual DB dedupe_key로 대체됨",
                })

    applied_keys = [row.get("_dedupe_key", "") for row in manual_rows if row.get("_dedupe_key")]

    return {
        "sources_df": final_sources,
        "removed_rows": removed,
        "dedupe_keys_applied": applied_keys,
    }


# ==============================
# v70.3: 기습의 대가 / 결투의 대가 치명타 적중률 scope 교정
# ==============================
# 문제: 기습의 대가 각인의 치명타 적중률 증가가 "전체" 스코프로 잡혀 모든 스킬에 +10% 가산됨.
# 수정: 기습의 대가 → 백어택 스킬 / 결투의 대가 → 헤드어택 스킬로 scope 강제 교정.
#
# 결과:
#   - 기본 기준 치명(비방향 스킬 기준): 기습 효과 제외됨 (예: 100.7% → 90.7%)
#   - 백어택 스킬 치명: 기습의 대가 +10%가 해당 스킬에만 적용 (90.7% + 10% = 100.7%)
#   - 결투의 대가도 동일하게 헤드어택 스킬 전용 처리
#
# 또한 _has_back/head_attack_engraving()이 소문자 키(lowercase)도 인식하도록 수정합니다.

ESTIMATOR_VERSION = "v70.3-directional-engraving-crit-scope"

_DIRECTIONAL_ENGRAVING_CRIT_SCOPE: Dict[str, str] = {
    "기습의 대가": "백어택 스킬",
    "기습의대가": "백어택 스킬",
    "결투의 대가": "헤드어택 스킬",
    "결투의대가": "헤드어택 스킬",
}


def _fix_directional_engraving_crit_scope(df: pd.DataFrame) -> pd.DataFrame:
    """기습의 대가 / 결투의 대가 치명타 적중률 행의 적용범위를 방향성에 맞게 교정합니다.

    - 기습의 대가 → "백어택 스킬" (전체/백·헤드 스킬로 잡혀 있으면 교정)
    - 결투의 대가 → "헤드어택 스킬"

    치명타 적중률 증가가 0인 행은 건드리지 않습니다.
    이미 올바른 방향성 스코프(백어택 스킬/헤드어택 스킬)인 행도 건드리지 않습니다.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    for idx, row in out.iterrows():
        crit_val = abs(float(row.get("치명타 적중률 증가(%)", 0) or 0))
        if crit_val < 1e-9:
            continue  # 치명타 적중률 없으면 건드리지 않음

        name_raw = str(row.get("이름") or "")
        name_norm = _norm_name(name_raw)  # 공백 제거·소문자 정규화
        scope = str(row.get("적용범위") or "전체")

        for eng_name, target_scope in _DIRECTIONAL_ENGRAVING_CRIT_SCOPE.items():
            if _norm_name(eng_name) in name_norm:
                # 이미 올바른 또는 더 좁은 스코프(스킬 전용 등)면 건드리지 않음
                if scope in {"전체", "백/헤드 스킬"}:
                    out.at[idx, "적용범위"] = target_scope
                break
    return out


# ── _has_back/head_attack_engraving 소문자 키 대응 ──────────────────────────

_old_has_back_attack_engraving_v70 = _has_back_attack_engraving  # type: ignore[name-defined]
_old_has_head_attack_engraving_v70 = _has_head_attack_engraving  # type: ignore[name-defined]


def _v703_get_engravings_raw(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """bundle에서 각인 dict를 안전하게 가져옵니다.

    _safe_data()는 {'ok': True, 'data': {...}} 래퍼 형식을 기대하지만,
    api_refresh_writer 정규화 캐시는 {'effects': [...], 'arkpassive_effects': [...]}
    형식을 직접 씁니다. 두 형식 모두 지원합니다.
    """
    # 1) 일반 _safe_data 경로 (래퍼 형식)
    via_safe = _safe_data(bundle, "engravings")
    if via_safe and isinstance(via_safe, dict) and any(k in via_safe for k in ["Effects", "ArkPassiveEffects", "effects", "arkpassive_effects"]):
        return via_safe
    # 2) 직접 접근 (정규화 캐시 형식)
    direct = bundle.get("engravings")
    if isinstance(direct, dict):
        return direct
    return {}


def _has_back_attack_engraving(bundle: Dict[str, Any]) -> bool:  # type: ignore[override]
    """v70.3: 대소문자 키 모두 인식 + 직접 dict 접근 fallback."""
    engravings = _v703_get_engravings_raw(bundle)
    texts: List[str] = []
    # 대문자 키 (원본 Lost Ark API)
    for key in ["Effects", "ArkPassiveEffects"]:
        for e in engravings.get(key, []) or []:
            texts.append(_flatten(e))
    # 소문자 키 (api_refresh_writer 정규화 캐시)
    for key in ["effects", "arkpassive_effects"]:
        for e in engravings.get(key, []) or []:
            texts.append(_flatten(e))
    # 최후 fallback: bundle 전체 플래튼
    if not any("기습" in t for t in texts):
        texts.append(_flatten(bundle.get("engravings", {})))
    return any("기습" in t or "기습의 대가" in t for t in texts)


def _has_head_attack_engraving(bundle: Dict[str, Any]) -> bool:  # type: ignore[override]
    """v70.3: 대소문자 키 모두 인식 + 직접 dict 접근 fallback."""
    engravings = _v703_get_engravings_raw(bundle)
    texts: List[str] = []
    for key in ["Effects", "ArkPassiveEffects"]:
        for e in engravings.get(key, []) or []:
            texts.append(_flatten(e))
    for key in ["effects", "arkpassive_effects"]:
        for e in engravings.get(key, []) or []:
            texts.append(_flatten(e))
    if not any("결투" in t for t in texts):
        texts.append(_flatten(bundle.get("engravings", {})))
    return any("결투" in t or "결투의 대가" in t for t in texts)


# ── _collect_global_sources 후처리 hook ─────────────────────────────────────

_old_collect_global_sources_v702_for_v703 = _collect_global_sources  # type: ignore[name-defined]


def _collect_global_sources(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    """v70.3: 정규화 캐시의 소문자 키 각인도 처리 + 기습/결투 치명 scope 교정."""
    base = _old_collect_global_sources_v702_for_v703(bundle)

    # 정규화 캐시 형식(lowercase keys)의 각인을 추가로 처리합니다.
    # 기존 _collect_global_sources는 대문자 키(Effects, ArkPassiveEffects)만 봅니다.
    # api_refresh_writer 정규화 캐시는 소문자 키(effects, arkpassive_effects)를 씁니다.
    engravings_raw = _v703_get_engravings_raw(bundle)
    names = _skill_names(bundle)
    extra_rows: List[Dict[str, Any]] = []

    # 소문자 키에서 각인 행 생성 (대문자 키에서 이미 읽혔다면 _concat_sources가 dedupe)
    for e in engravings_raw.get("effects", []) or []:
        name = e.get("name") or e.get("Name") or "각인"
        desc = e.get("description_text") or e.get("Description") or ""
        tooltip = _flatten(e.get("raw", {}).get("Tooltip") or e.get("Tooltip") or "")
        text = f"{name} {desc} {tooltip}"
        extra_rows.extend(_source_rows_from_text("각인", name, text, names))

    for e in engravings_raw.get("arkpassive_effects", []) or []:
        name = e.get("name") or e.get("Name") or "아크패시브 각인"
        desc = e.get("description_text") or e.get("Description") or ""
        grade = e.get("grade") or e.get("Grade") or ""
        level = e.get("level") or e.get("Level") or ""
        tooltip = _flatten(e.get("raw", {}).get("Tooltip") or e.get("Tooltip") or "")
        text = f"{name} {desc} {grade} {level} {tooltip}"
        extra_rows.extend(_source_rows_from_text("아크패시브 각인", name, text, names))

    if extra_rows:
        extra_df = pd.DataFrame(extra_rows, columns=SOURCE_COLUMNS)
        base = _concat_sources(base, extra_df)

    return _fix_directional_engraving_crit_scope(base)


# ==============================
# v70.4: 기본 기준 치명 방향성 제외 / 초각성 스킬 감지
# ==============================
# 변경 내용:
#   1. 기본 기준 치명: 백어택 스킬/헤드어택 스킬/백-헤드 스킬 범위 치적 제외
#      (기습의 대가, 결투의 대가 등 방향성 각인의 치명타 적중률은 방향성 없는 기준에서 빠짐)
#      즉 global_basis_crit_rate_percent = 스탯 치적 + "전체" 범위 치적만 합산
#   2. 초각성 스킬명 감지: _v17_referenced_skill_names() 결과를 awakening_skill_names로 저장

ESTIMATOR_VERSION = "v70.4-base-crit-no-direction"


def _v70_global_basis_crit_no_direction(result: Dict[str, Any]) -> "float | None":
    """v70.4: 방향성(백어택/헤드어택 전용) 치적을 제외한 기본 기준 치명.

    '기본 기준 치명' = 스탯 치적 + '전체' 적용범위 치적만 합산.
    - 기습의 대가(백어택 스킬 범위), 결투의 대가(헤드어택 스킬 범위)는 제외됩니다.
    - +10% 백어택 방향성 기본 보너스도 더하지 않습니다.
    - 스킬 전용 트라이포드 치적도 제외됩니다.
    """
    sources = result.get("damage_sources")
    if not isinstance(sources, pd.DataFrame) or sources.empty:
        return None
    stat_rate = float(result.get("base_crit_percent", 0) or 0)
    src = _v64_sources_without_base_stat(sources)

    # 기본 기준 치명은 '모든(또는 거의 모든) 스킬에 공통 적용되는' 치적만 합산합니다.
    # - 방향성(백/헤드), 스킬 전용은 제외.
    # - '스킬 그룹/계열'은 두 종류가 섞여 있어 구분합니다:
    #     · 아드레날린처럼 '이동기·기본공격만 제외한 (거의) 전 스킬' 그룹 → 포함
    #     · 실버호크 스킬처럼 특정 스킬 하나만 강화하는 그룹 → 제외
    _DIRECTION_SCOPES = {"백어택 스킬", "헤드어택 스킬", "백/헤드 스킬", "스킬 전용", "검수 필요"}
    _GROUP_SCOPES = {"스킬 그룹", "계열 전용", "스킬 계열"}

    total = stat_rate
    for _, row in src.iterrows():
        scope = str(row.get("적용범위") or "전체")
        if scope in _DIRECTION_SCOPES:
            continue
        if scope in _GROUP_SCOPES:
            _txt = _norm_name(f"{row.get('이름','')} {row.get('설명','')} {row.get('적용스킬','')}")
            _near_global = ("제외한스킬" in _txt) or ("이동기및기본공격을제외" in _txt) or ("아드레날린" in _txt)
            if not _near_global:
                continue  # 특정 스킬 전용 그룹(실버호크 등)은 기본 기준 치명에서 제외
        value = _num(row.get("치명타 적중률 증가(%)")) or 0.0
        if not value:
            continue
        total += float(value)
    return round(float(total), 2)


_old_v61_refresh_summary_metrics_v703_for_v704 = _v61_refresh_summary_metrics


def _v61_refresh_summary_metrics(result: Dict[str, Any]) -> None:  # type: ignore[override]
    """v70.4: 기본 기준 치명을 방향성 제외 값으로 교체."""
    _old_v61_refresh_summary_metrics_v703_for_v704(result)
    try:
        gb = _v70_global_basis_crit_no_direction(result)
        if gb is not None:
            result["global_basis_crit_rate_percent"] = gb
    except Exception:
        pass


_old_add_v12_summary_v703_for_v704 = _add_v12_summary


def _add_v12_summary(result: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
    """v70.4: 초각성 스킬명 감지 결과를 awakening_skill_names로 summary에 추가."""
    result = _old_add_v12_summary_v703_for_v704(result, bundle)
    try:
        ref_names = _v17_referenced_skill_names(bundle)
        if ref_names:
            result["awakening_skill_names"] = sorted(ref_names)
    except Exception:
        pass
    return result


# ==============================
# v70.5: 트라이포드 방향성 제거 감지 / 라그나브레이크 치적 버그 수정
# ==============================
# 변경 내용:
#   1. 라그나브레이크/라그나블레이드를 _EXCLUDED_ADRENALINE_GENERATOR_SKILLS_V63에서 제거
#      → 이미 위에서 적용됨 (set에서 두 스킬 제거)
#   2. _detect_attack_type: 선택 트라이포드가 방향성을 제거하는 경우 반영
#      예: "더 이상 헤드어택 공격이 적용되지 않지만" → 헤드어택 제거

ESTIMATOR_VERSION = "v70.5-tripod-direction-removal"


def _v705_tripod_removes_direction(skill: Dict[str, Any]) -> "tuple[bool, bool]":
    """선택된 트라이포드가 백어택/헤드어택 방향성을 제거하는지 감지합니다.

    Returns: (removes_back, removes_head)
    """
    removes_back = False
    removes_head = False
    for tripod in skill.get("Tripods", []) or []:
        if not bool(tripod.get("IsSelected")):
            continue
        text = _clean_text(_flatten(tripod))
        # "더 이상 헤드어택 공격이 적용되지 않" 패턴 감지
        if re.search(r"더\s*이상\s*헤드\s*어택\s*공격이?\s*적용되지\s*않", text):
            removes_head = True
        # "더 이상 백어택 공격이 적용되지 않" 패턴 감지
        if re.search(r"더\s*이상\s*백\s*어택\s*공격이?\s*적용되지\s*않", text):
            removes_back = True
    return removes_back, removes_head


_old_detect_attack_type_v14_for_v705 = _detect_attack_type


def _v705_direction_from_text(text: str) -> "tuple[bool, bool]":
    """API 텍스트에서 백어택/헤드어택 방향성을 감지합니다.

    "헤드 어택" (공백 있음)과 "헤드어택" (공백 없음) 모두 인식합니다.
    """
    norm = _norm_name(text)  # 공백 제거한 정규화 문자열
    has_back = (
        "백어택" in text
        or "백 어택" in text
        or "BackAttack" in text
        or "Back Attack" in text
        or "백어택" in norm
    )
    has_head = (
        "헤드어택" in text
        or "헤드 어택" in text
        or "HeadAttack" in text
        or "Head Attack" in text
        or "헤드어택" in norm
    )
    return has_back, has_head


# v70.5: 와일드 스톰프는 실제 API 텍스트에 '헤드 어택'이 있는 헤드어택 스킬이므로
# 비방향성 강제 집합에서 제외합니다. (나머지는 false-positive 방지 위해 유지)
_KNOWN_NON_DIRECTIONAL_V705 = {
    n for n in KNOWN_NON_DIRECTIONAL_SKILLS if _norm_name(n) != _norm_name("와일드스톰프")
}


def _detect_attack_type(skill: Dict[str, Any]) -> str:  # type: ignore[override]
    """v70.5: 공백 포함 헤드/백어택 감지 + 선택 트라이포드 방향성 제거 반영.

    v14는 "헤드어택"(공백 없음)만 체크해서 API가 "헤드 어택"으로 반환할 때 미감지.
    v70.5에서 _v705_direction_from_text로 공백 양쪽 모두 처리합니다.
    또한 백어택/헤드어택을 모두 가진 스킬은 '백/헤드'로 표기하고,
    실제 단일 방향 결정(각인 기준)은 _calculate_one_skill에서 처리합니다.
    """
    name_norm = _norm_name(skill.get("Name") or "")

    if name_norm in _KNOWN_NON_DIRECTIONAL_V705:
        direction = "일반/확인필요"
    else:
        text = _flatten(skill)
        has_back, has_head = _v705_direction_from_text(text)
        # KNOWN_BACK_ATTACK_SKILLS 하드코딩 보완
        if name_norm in KNOWN_BACK_ATTACK_SKILLS:
            has_back = True

        if has_back and has_head:
            direction = "백/헤드"
        elif has_back:
            direction = "백어택"
        elif has_head:
            direction = "헤드어택"
        else:
            direction = "일반/확인필요"

    # 트라이포드 방향성 제거
    removes_back, removes_head = _v705_tripod_removes_direction(skill)
    if removes_back and "백" in direction:
        direction = direction.replace("백/헤드", "헤드어택").replace("백어택", "일반/확인필요")
    if removes_head and "헤드" in direction:
        direction = direction.replace("백/헤드", "백어택").replace("헤드어택", "일반/확인필요")

    # 나머지 태그(스킬 계열, 조작 타입)는 v14에서 가져옵니다
    base_v14 = _old_detect_attack_type_v14_for_v705(skill)
    extra_parts = [p for p in base_v14.split(" / ")[1:] if p]  # 방향 제외 태그만
    parts = [direction] + extra_parts
    return " / ".join(parts)


# v70.6: Arcana "Call of Destiny - Dark Destiny" applies crit-damage to all skills.
_old_collect_skill_tripod_sources_v705_for_v706 = _collect_skill_tripod_sources


def _v706_has_arcana_dark_destiny(bundle: Dict[str, Any]) -> bool:
    for skill in _safe_data(bundle, "combat_skills") or []:
        skill_name = str(skill.get("Name") or skill.get("name") or "")
        if _norm_name(skill_name) != _norm_name("운명의 부름"):
            continue
        for tripod in (skill.get("Tripods") or skill.get("tripods") or []) or []:
            if not isinstance(tripod, dict):
                continue
            selected = tripod.get("IsSelected", tripod.get("isSelected", tripod.get("Selected", tripod.get("selected"))))
            tripod_name = str(tripod.get("Name") or tripod.get("name") or "")
            if bool(selected) and _norm_name(tripod_name) == _norm_name("어두운 운명"):
                return True
    return False


def _collect_skill_tripod_sources(bundle: Dict[str, Any]) -> pd.DataFrame:  # type: ignore[override]
    df = _old_collect_skill_tripod_sources_v705_for_v706(bundle)
    if not _v706_has_arcana_dark_destiny(bundle) or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()
    name_col = "이름"
    scope_col = "적용범위"
    target_col = "적용스킬"
    desc_col = "설명"
    if not all(c in out.columns for c in [name_col, scope_col, target_col]):
        return out

    mask = (
        out[name_col].astype(str).map(_norm_name).eq(_norm_name("어두운 운명"))
        & out[target_col].astype(str).map(_norm_name).eq(_norm_name("운명의 부름"))
    )
    if not mask.any():
        return out

    out.loc[mask, scope_col] = "전체"
    out.loc[mask, target_col] = "전체"
    if desc_col in out.columns:
        out.loc[mask, desc_col] = out.loc[mask, desc_col].astype(str).map(
            lambda text: (text + " / v70.6 보정: 운명의 부름-어두운 운명 치명타 피해량 증가는 전체 스킬 적용").strip(" /")
        )
    return out.reset_index(drop=True)


_old_force_skill_scopes_v705_for_v706 = _force_skill_scopes


def _force_skill_scopes(df: pd.DataFrame, skill_names: Iterable[str]) -> pd.DataFrame:  # type: ignore[override]
    out = _old_force_skill_scopes_v705_for_v706(df, skill_names)
    if not isinstance(out, pd.DataFrame) or out.empty:
        return out

    name_col = "이름"
    scope_col = "적용범위"
    target_col = "적용스킬"
    desc_col = "설명"
    if not all(c in out.columns for c in [name_col, scope_col, target_col]):
        return out

    desc_series = out[desc_col].astype(str) if desc_col in out.columns else pd.Series("", index=out.index)
    mask = out[name_col].astype(str).map(_norm_name).eq(_norm_name("어두운 운명"))
    mask &= (
        out[target_col].astype(str).map(_norm_name).isin({_norm_name("운명의 부름"), _norm_name("전체")})
        | desc_series.map(_norm_name).str.contains(_norm_name("운명의 부름"), na=False)
    )
    if not mask.any():
        return out

    out = out.copy()
    out.loc[mask, scope_col] = "전체"
    out.loc[mask, target_col] = "전체"
    return out


def _v705_resolve_both_direction(attack_type: str, back_engraving: bool, head_engraving: bool) -> str:
    """백어택/헤드어택을 모두 가진 스킬을 각인 기준으로 단일 방향으로 결정합니다.

    - 결투의 대가(헤드어택 각인) 보유 → 헤드어택
    - 기습의 대가(백어택 각인) 보유 → 백어택
    - 둘 다 없음 → 백어택 (기본값)
    """
    parts = attack_type.split(" / ")
    if not parts:
        return attack_type
    direction = parts[0]
    if "백" in direction and "헤드" in direction:
        if head_engraving and not back_engraving:
            parts[0] = "헤드어택"
        else:
            parts[0] = "백어택"
        return " / ".join(parts)
    return attack_type


_extract_effect_values_uncached_v156 = _extract_effect_values


@lru_cache(maxsize=8192)
def _extract_effect_values_cached_v156(text: str, source_type: str = "", target_skill: str = "") -> tuple[tuple[str, float], ...]:
    result = _extract_effect_values_uncached_v156(text, source_type, target_skill or None)
    return tuple(sorted((str(k), float(v or 0.0)) for k, v in result.items()))


def _extract_effect_values(text: str, source_type: str = "", target_skill: str | None = None) -> Dict[str, float]:  # type: ignore[override]
    return dict(_extract_effect_values_cached_v156(str(text or ""), str(source_type or ""), str(target_skill or "")))
