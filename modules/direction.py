"""방향(백어택/헤드어택) 평가 전용 모듈.

이 모듈은 로스트아크 전투분석기 기반 '실전 수행률' 계산기에서
방향 관련 계산을 앱 로직과 분리해 순수 함수로 제공합니다.

설계 원칙
---------
1. 스킬 '구조 방향'(BACK_ONLY/HEAD_ONLY/DUAL/NONE)과
   '착용 방향 각인'(기습의 대가/결투의 대가/없음)을 분리합니다.
2. 백/헤드 '기본 보너스'(백 +5% 및 치적 +10%, 헤드 +20%)와
   '방향 각인 보너스'(기습/결투)를 분리해서 각각 효율을 계산합니다.
3. 전투분석기의 백/헤드 피해 비중은 이미 보너스가 적용된 결과값이므로
   raw(보너스 제거) 방향 비중으로 역산한 뒤 평가합니다.
4. 데이터가 없는(None) 방향은 0으로 처리하지 않습니다.
   대신 방향 데이터 커버리지/신뢰도를 낮춥니다.

이 모듈은 모든 값을 '분수(0.0~1.0)'와 '배율(1.05 등)' 기준으로 다룹니다.
퍼센트(예: 98.0)를 쓰는 앱 경계에서는 호출 전에 /100 변환을 해야 합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 열거형 / 예외
# ---------------------------------------------------------------------------


class AttackDirectionType(str, Enum):
    """스킬이 구조적으로 낼 수 있는 방향 공격 타입."""

    BACK_ONLY = "BACK_ONLY"      # 백어택만 가능
    HEAD_ONLY = "HEAD_ONLY"      # 헤드어택만 가능
    DUAL = "DUAL"                # 백/헤드 둘 다 가능
    NONE = "NONE"                # 백/헤드 둘 다 없음(무방향)


class DirectionalEngravingState(str, Enum):
    """캐릭터가 착용한 방향 각인 상태."""

    AMBUSH_MASTER = "AMBUSH_MASTER"      # 기습의 대가(백어택 조건부 보너스)
    MASTER_BRAWLER = "MASTER_BRAWLER"    # 결투의 대가(헤드어택 조건부 보너스)
    NONE = "NONE"                        # 둘 다 없음


class DirectionValidationError(ValueError):
    """기습의 대가와 결투의 대가가 동시에 감지되는 등 잘못된 입력."""


# ---------------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectionSettings:
    """방향 계산에 쓰는 설정값. 각인 배율은 하드코딩하지 않고 여기서 관리합니다.

    ambush/master 배율은 API 툴팁에서 감지한 조건부 방향 피해(%)가 있으면
    앱 쪽에서 `from_engraving_bonus`로 덮어써서 넘겨줍니다.
    감지값이 없을 때만 아래 기본값을 폴백으로 사용합니다.
    """

    # 방향 각인 조건부 배율(예: 1.10 = 성공 시 +10%). 기본값은 폴백용입니다.
    ambush_master_multiplier: float = 1.10      # 기습의 대가
    master_brawler_multiplier: float = 1.10     # 결투의 대가

    # 백어택 기본 보너스
    back_damage_bonus: float = 0.05             # 피해량 +5%
    back_crit_rate_bonus: float = 0.10          # 치명타 적중률 +10%(치적 100% 상한)

    # 헤드어택 기본 보너스
    head_damage_bonus: float = 0.20             # 피해량 +20%
    head_stagger_bonus: float = 0.10            # 무력화 +10%(DPS 미포함, 별도 지표)

    def with_engraving_bonuses(
        self,
        ambush_bonus_percent: Optional[float] = None,
        brawler_bonus_percent: Optional[float] = None,
    ) -> "DirectionSettings":
        """감지된 조건부 방향 피해(%)로 각인 배율을 대체한 새 설정을 만듭니다.

        ambush_bonus_percent=15.0 이면 ambush_master_multiplier=1.15가 됩니다.
        None이거나 0 이하이면 기존 값을 유지합니다.
        """
        amb = self.ambush_master_multiplier
        brw = self.master_brawler_multiplier
        if ambush_bonus_percent is not None and ambush_bonus_percent > 0:
            amb = 1.0 + float(ambush_bonus_percent) / 100.0
        if brawler_bonus_percent is not None and brawler_bonus_percent > 0:
            brw = 1.0 + float(brawler_bonus_percent) / 100.0
        return DirectionSettings(
            ambush_master_multiplier=amb,
            master_brawler_multiplier=brw,
            back_damage_bonus=self.back_damage_bonus,
            back_crit_rate_bonus=self.back_crit_rate_bonus,
            head_damage_bonus=self.head_damage_bonus,
            head_stagger_bonus=self.head_stagger_bonus,
        )


DEFAULT_SETTINGS = DirectionSettings()


# ---------------------------------------------------------------------------
# 1) 스킬 구조 방향 분류
# ---------------------------------------------------------------------------


def classify_skill_attack_direction(
    api_attack_text: Optional[str],
    observed_has_back: bool = False,
    observed_has_head: bool = False,
) -> AttackDirectionType:
    """스킬의 구조 방향 타입을 분류합니다.

    v137: 스킬 구조 방향은 '오직 API 스킬 툴팁/트라이포드 공격타입'으로만 판정합니다.
    전투분석기에 찍힌 백/헤드 비중은 '그 전투에서 어느 방향으로 맞았는지'의 관측값일
    뿐, 스킬의 구조 타입이 아닙니다. 관측값으로 구조 타입을 지어내면(예: 무방향 스킬의
    작은 백 비중을 백어택으로 승격) 왜곡이 생깁니다. 그래서 관측값은 구조 판정에
    쓰지 않습니다(파라미터는 하위호환용으로 남겨두지만 사용하지 않습니다).

    판정
    ----
    - API 텍스트에 '백'과 '헤드'가 모두 있으면 DUAL
    - '백'만 있으면 BACK_ONLY
    - '헤드'만 있으면 HEAD_ONLY
    - 둘 다 없으면(트라이포드로 방향성이 제거된 경우 포함) NONE

    트라이포드로 방향성이 사라지는 스킬(예: 파천섬광 + 도전자)은 API 공격타입에서
    이미 방향 토큰이 빠져 있어야 하며, 그 경우 자동으로 NONE으로 분류됩니다.
    """
    text = str(api_attack_text or "")
    has_back_api = "백" in text
    has_head_api = "헤드" in text

    if has_back_api and has_head_api:
        return AttackDirectionType.DUAL
    if has_back_api:
        return AttackDirectionType.BACK_ONLY
    if has_head_api:
        return AttackDirectionType.HEAD_ONLY
    return AttackDirectionType.NONE


def resolve_engraving_state(
    ambush_detected: bool,
    brawler_detected: bool,
) -> DirectionalEngravingState:
    """캐릭터 전역 방향 각인 상태를 결정합니다.

    기습의 대가와 결투의 대가는 동시에 착용할 수 없으므로,
    둘 다 감지되면 DirectionValidationError를 발생시킵니다.
    """
    if ambush_detected and brawler_detected:
        raise DirectionValidationError(
            "기습의 대가와 결투의 대가가 동시에 감지되었습니다. 방향 각인은 하나만 착용할 수 있습니다."
        )
    if ambush_detected:
        return DirectionalEngravingState.AMBUSH_MASTER
    if brawler_detected:
        return DirectionalEngravingState.MASTER_BRAWLER
    return DirectionalEngravingState.NONE


def effective_engraving_state(
    skill_type: AttackDirectionType,
    global_state: DirectionalEngravingState,
) -> DirectionalEngravingState:
    """전역 각인 상태 중, 이 스킬에 '실제로 의미 있는' 각인만 남깁니다.

    - BACK_ONLY 스킬에는 기습의 대가만 의미가 있습니다(헤드 각인은 무의미).
    - HEAD_ONLY 스킬에는 결투의 대가만 의미가 있습니다.
    - DUAL 스킬에는 착용한 각인이 그대로 적용됩니다.
    - NONE 스킬에는 어떤 방향 각인도 의미가 없습니다.
    """
    if skill_type == AttackDirectionType.NONE:
        return DirectionalEngravingState.NONE
    if skill_type == AttackDirectionType.BACK_ONLY:
        return global_state if global_state == DirectionalEngravingState.AMBUSH_MASTER else DirectionalEngravingState.NONE
    if skill_type == AttackDirectionType.HEAD_ONLY:
        return global_state if global_state == DirectionalEngravingState.MASTER_BRAWLER else DirectionalEngravingState.NONE
    return global_state  # DUAL


# ---------------------------------------------------------------------------
# 2) 목표 방향
# ---------------------------------------------------------------------------


def resolve_directional_target(
    skill_type: AttackDirectionType,
    engraving_state: DirectionalEngravingState,
) -> Optional[str]:
    """스킬 구조 + 착용 각인으로 '평가 목표 방향'을 결정합니다.

    반환: "BACK" | "HEAD" | None
    DUAL + 방향 각인 없음은 목표 방향 None(각인 목표 없음)이지만,
    기본 방향 보너스 자체는 여전히 계산해야 합니다(무방향 스킬 아님).
    """
    if skill_type == AttackDirectionType.BACK_ONLY:
        return "BACK"
    if skill_type == AttackDirectionType.HEAD_ONLY:
        return "HEAD"
    if skill_type == AttackDirectionType.DUAL:
        if engraving_state == DirectionalEngravingState.AMBUSH_MASTER:
            return "BACK"
        if engraving_state == DirectionalEngravingState.MASTER_BRAWLER:
            return "HEAD"
        return None
    return None  # NONE


# ---------------------------------------------------------------------------
# 3) 기본 보너스 배율
# ---------------------------------------------------------------------------


def calc_crit_expected_multiplier(crit_rate: float, crit_damage_multiplier: float) -> float:
    """치명 기대 배율. crit_rate는 분수(0~1), crit_damage_multiplier는 배율(예: 2.0)."""
    crit_rate = max(0.0, min(1.0, crit_rate))
    return 1.0 + crit_rate * (crit_damage_multiplier - 1.0)


def calc_back_base_multiplier(
    base_crit_rate: float,
    crit_damage_multiplier: float,
    settings: DirectionSettings = DEFAULT_SETTINGS,
) -> float:
    """백어택 기본 피해 기대 배율.

    피해량 +5%에 더해, 치명타 적중률 +10%의 기대 피해 상승분을 곱합니다.
    치적 10%는 100% 상한을 적용하므로, 현재 치적이 높을수록 유효분이 줄어듭니다.
    base_crit_rate는 분수(0~1)입니다.
    """
    base_dmg = 1.0 + settings.back_damage_bonus
    without = calc_crit_expected_multiplier(base_crit_rate, crit_damage_multiplier)
    with_rate = min(base_crit_rate + settings.back_crit_rate_bonus, 1.0)
    with_bonus = calc_crit_expected_multiplier(with_rate, crit_damage_multiplier)
    if without <= 0:
        return base_dmg
    return base_dmg * with_bonus / without


def calc_head_base_multiplier(settings: DirectionSettings = DEFAULT_SETTINGS) -> float:
    """헤드어택 기본 피해 기대 배율. 무력화 10%는 DPS에 넣지 않습니다."""
    return 1.0 + settings.head_damage_bonus


def calc_stagger_bonus_expected(
    raw_head_share: Optional[float],
    settings: DirectionSettings = DEFAULT_SETTINGS,
) -> Optional[float]:
    """헤드어택 무력화 기대치(별도 지표, DPS 미포함).

    헤드로 들어간 raw 비중 × 무력화 10%로, DPS와 분리해 표시합니다.
    """
    if raw_head_share is None:
        return None
    return max(0.0, raw_head_share) * settings.head_stagger_bonus


def direction_total_multipliers(
    skill_type: AttackDirectionType,
    engraving_state: DirectionalEngravingState,
    back_base_multiplier: float,
    head_base_multiplier: float,
    settings: DirectionSettings = DEFAULT_SETTINGS,
) -> Tuple[float, float, float]:
    """방향별 총 배율 (back_total, head_total, neutral)을 반환합니다.

    각인 보너스는 해당 방향에만 곱해지고, 무방향(neutral)은 항상 1.0입니다.
    NONE 스킬은 모든 방향 배율이 1.0입니다.
    """
    if skill_type == AttackDirectionType.NONE:
        return 1.0, 1.0, 1.0
    back_total = back_base_multiplier
    head_total = head_base_multiplier
    if engraving_state == DirectionalEngravingState.AMBUSH_MASTER:
        back_total = back_base_multiplier * settings.ambush_master_multiplier
    elif engraving_state == DirectionalEngravingState.MASTER_BRAWLER:
        head_total = head_base_multiplier * settings.master_brawler_multiplier
    return back_total, head_total, 1.0


# ---------------------------------------------------------------------------
# 6) 관측 피해 비중 → raw 방향 비중 역산
# ---------------------------------------------------------------------------


def normalize_observed_direction_shares(
    observed_back_share: Optional[float],
    observed_head_share: Optional[float],
    back_total_multiplier: float,
    head_total_multiplier: float,
    skill_type: AttackDirectionType,
    observed_neutral_share: Optional[float] = None,
) -> Optional[dict]:
    """관측 피해 비중(보너스 적용 후)을 raw 방향 비중으로 역산합니다.

    입력은 분수(0~1)입니다. None은 '데이터 없음'이며 0이 아닙니다.
    - 구조상 존재하지 않는 방향(BACK_ONLY의 헤드 등)의 None은 0으로 간주합니다.
    - 구조상 존재하는데 데이터가 None이면 역산이 불가능하므로 None을 반환합니다
      (호출부에서 신뢰도를 낮춰야 합니다).

    반환: {"raw_back", "raw_head", "raw_neutral", "has_data"} 또는 None.
    """
    b = observed_back_share
    h = observed_head_share

    # 구조상 없는 방향의 결측은 0으로 채웁니다.
    if skill_type == AttackDirectionType.BACK_ONLY:
        if h is None:
            h = 0.0
    elif skill_type == AttackDirectionType.HEAD_ONLY:
        if b is None:
            b = 0.0
    elif skill_type == AttackDirectionType.NONE:
        b = 0.0
        h = 0.0

    # 구조상 존재하는데 데이터가 없으면 역산 불가.
    if b is None or h is None:
        return None

    b = max(0.0, float(b))
    h = max(0.0, float(h))
    if observed_neutral_share is not None:
        n = max(0.0, float(observed_neutral_share))
    else:
        n = max(0.0, 1.0 - b - h)

    denom = 0.0
    if back_total_multiplier > 0:
        denom += b / back_total_multiplier
    if head_total_multiplier > 0:
        denom += h / head_total_multiplier
    denom += n  # neutral_multiplier = 1.0

    if denom <= 0:
        # 관측 피해 비중이 전부 0 → 방향 정보 없음(무방향으로 들어간 것으로 간주)
        return {"raw_back": 0.0, "raw_head": 0.0, "raw_neutral": 1.0, "has_data": False}

    raw_back = (b / back_total_multiplier) / denom if back_total_multiplier > 0 else 0.0
    raw_head = (h / head_total_multiplier) / denom if head_total_multiplier > 0 else 0.0
    raw_neutral = n / denom
    return {
        "raw_back": raw_back,
        "raw_head": raw_head,
        "raw_neutral": raw_neutral,
        "has_data": (b > 0 or h > 0),
    }


# ---------------------------------------------------------------------------
# 7) 기본 방향 보너스 활용률
# ---------------------------------------------------------------------------


def calc_basic_direction_bonus_utilization(
    raw_back_share: float,
    raw_head_share: float,
    raw_neutral_share: float,
    skill_type: AttackDirectionType,
    back_base_multiplier: float,
    head_base_multiplier: float,
) -> Tuple[Optional[float], float]:
    """기습/결투 각인을 제외한, 백/헤드 '자체' 보너스 활용률을 계산합니다.

    반환: (활용률 또는 None, actual_basic_direction_multiplier)
    """
    actual = (
        raw_back_share * back_base_multiplier
        + raw_head_share * head_base_multiplier
        + raw_neutral_share * 1.0
    )
    if skill_type == AttackDirectionType.BACK_ONLY:
        max_basic = back_base_multiplier
    elif skill_type == AttackDirectionType.HEAD_ONLY:
        max_basic = head_base_multiplier
    elif skill_type == AttackDirectionType.DUAL:
        max_basic = max(back_base_multiplier, head_base_multiplier)
    else:
        max_basic = 1.0

    if max_basic > 1.0:
        return (actual - 1.0) / (max_basic - 1.0), actual
    return None, actual


# ---------------------------------------------------------------------------
# 8) 방향 각인 효율
# ---------------------------------------------------------------------------


def calc_directional_engraving_efficiency(
    raw_back_share: float,
    raw_head_share: float,
    raw_neutral_share: float,
    engraving_state: DirectionalEngravingState,
    back_base_multiplier: float,
    head_base_multiplier: float,
    settings: DirectionSettings = DEFAULT_SETTINGS,
) -> Optional[float]:
    """기습/결투 각인을 실제 딜로 얼마나 전환했는지(0~1).

    이론상 최대(각인 배율 - 1)를 100%로 봤을 때, 실제로 얻은 조건부 딜증의 비율입니다.
    각인이 없으면 None을 반환합니다.
    """
    if engraving_state == DirectionalEngravingState.AMBUSH_MASTER:
        with_eng = (
            raw_back_share * back_base_multiplier * settings.ambush_master_multiplier
            + raw_head_share * head_base_multiplier
            + raw_neutral_share * 1.0
        )
        without_eng = (
            raw_back_share * back_base_multiplier
            + raw_head_share * head_base_multiplier
            + raw_neutral_share * 1.0
        )
        theoretical = settings.ambush_master_multiplier - 1.0
    elif engraving_state == DirectionalEngravingState.MASTER_BRAWLER:
        with_eng = (
            raw_back_share * back_base_multiplier
            + raw_head_share * head_base_multiplier * settings.master_brawler_multiplier
            + raw_neutral_share * 1.0
        )
        without_eng = (
            raw_back_share * back_base_multiplier
            + raw_head_share * head_base_multiplier
            + raw_neutral_share * 1.0
        )
        theoretical = settings.master_brawler_multiplier - 1.0
    else:
        return None

    if without_eng <= 0 or theoretical <= 0:
        return None
    actual_gain = with_eng / without_eng - 1.0
    return actual_gain / theoretical


# ---------------------------------------------------------------------------
# 9) 방향 손실 피해
# ---------------------------------------------------------------------------


def calc_direction_loss_damage(
    current_crit_adjusted_damage: float,
    actual_direction_multiplier: Optional[float],
    optimal_direction_multiplier: Optional[float],
    skill_type: AttackDirectionType,
) -> Optional[float]:
    """방향만 최적으로 들어갔다면 얼마나 더 나왔을지(추가 가능 피해).

    - NONE 스킬: 0
    - actual 배율이 0 이하이거나 정보 없음: None
    """
    if skill_type == AttackDirectionType.NONE:
        return 0.0
    if actual_direction_multiplier is None or optimal_direction_multiplier is None:
        return None
    if actual_direction_multiplier <= 0:
        return None
    return current_crit_adjusted_damage * (optimal_direction_multiplier / actual_direction_multiplier - 1.0)


# ---------------------------------------------------------------------------
# 11) 데이터 신뢰도
# ---------------------------------------------------------------------------


def calc_direction_data_confidence(coverage: Optional[float]) -> str:
    """방향 데이터 커버리지(0~1)를 신뢰도 등급 문자열로 변환합니다."""
    if coverage is None:
        return "없음"
    if coverage >= 0.90:
        return "높음"
    if coverage >= 0.60:
        return "보통"
    return "낮음"


# ---------------------------------------------------------------------------
# 오케스트레이터: 스킬 한 개의 방향 평가를 한 번에
# ---------------------------------------------------------------------------


@dataclass
class DirectionEvaluation:
    skill_type: AttackDirectionType
    engraving_state: DirectionalEngravingState          # 이 스킬에 유효한 각인
    target_direction: Optional[str]                     # "BACK" | "HEAD" | None
    back_base_multiplier: float
    head_base_multiplier: float
    back_total_multiplier: float
    head_total_multiplier: float
    raw_back_share: Optional[float] = None
    raw_head_share: Optional[float] = None
    raw_neutral_share: Optional[float] = None
    actual_direction_multiplier: Optional[float] = None
    optimal_direction_multiplier: Optional[float] = None
    actual_basic_direction_multiplier: Optional[float] = None
    basic_direction_bonus_utilization: Optional[float] = None
    directional_engraving_efficiency: Optional[float] = None
    direction_loss_damage: Optional[float] = None
    directional_engraving_loss_damage: float = 0.0
    stagger_bonus_expected: Optional[float] = None
    has_direction_data: bool = False


def evaluate_skill_direction(
    api_attack_text: Optional[str],
    observed_back_share: Optional[float],
    observed_head_share: Optional[float],
    base_crit_rate: float,
    crit_damage_multiplier: float,
    current_crit_adjusted_damage: float,
    global_engraving_state: DirectionalEngravingState,
    settings: DirectionSettings = DEFAULT_SETTINGS,
    observed_neutral_share: Optional[float] = None,
) -> DirectionEvaluation:
    """스킬 한 개의 방향 평가를 한 번에 수행합니다.

    shares/crit는 분수(0~1). current_crit_adjusted_damage는 치명운 보정된 실전 피해.
    """
    observed_has_back = observed_back_share is not None
    observed_has_head = observed_head_share is not None
    skill_type = classify_skill_attack_direction(api_attack_text, observed_has_back, observed_has_head)
    eng_state = effective_engraving_state(skill_type, global_engraving_state)
    target = resolve_directional_target(skill_type, eng_state)

    back_base = calc_back_base_multiplier(base_crit_rate, crit_damage_multiplier, settings)
    head_base = calc_head_base_multiplier(settings)
    back_total, head_total, _ = direction_total_multipliers(skill_type, eng_state, back_base, head_base, settings)

    ev = DirectionEvaluation(
        skill_type=skill_type,
        engraving_state=eng_state,
        target_direction=target,
        back_base_multiplier=back_base,
        head_base_multiplier=head_base,
        back_total_multiplier=back_total,
        head_total_multiplier=head_total,
    )

    if skill_type == AttackDirectionType.NONE:
        ev.actual_direction_multiplier = 1.0
        ev.optimal_direction_multiplier = 1.0
        ev.direction_loss_damage = 0.0
        ev.directional_engraving_loss_damage = 0.0
        ev.has_direction_data = False
        return ev

    raw = normalize_observed_direction_shares(
        observed_back_share, observed_head_share,
        back_total, head_total, skill_type, observed_neutral_share,
    )
    if raw is None:
        # 구조상 존재하는 방향인데 데이터가 없음 → 방향 평가 불가(신뢰도 낮춤)
        ev.has_direction_data = False
        return ev

    rb, rh, rn = raw["raw_back"], raw["raw_head"], raw["raw_neutral"]
    ev.raw_back_share = rb
    ev.raw_head_share = rh
    ev.raw_neutral_share = rn
    ev.has_direction_data = raw["has_data"]

    # 실제 방향 배율(관측 raw 기준)
    ev.actual_direction_multiplier = rb * back_total + rh * head_total + rn * 1.0

    # 최적 방향 배율(목표 방향 100% 성공 가정)
    if target == "BACK":
        ev.optimal_direction_multiplier = back_total
    elif target == "HEAD":
        ev.optimal_direction_multiplier = head_total
    else:
        # DUAL + 각인 없음: 각인 목표 없음. 기본 보너스 기준 최대치를 참고용으로만 사용.
        ev.optimal_direction_multiplier = max(back_total, head_total)

    # 기본 방향 보너스 활용률
    util, actual_basic = calc_basic_direction_bonus_utilization(rb, rh, rn, skill_type, back_base, head_base)
    ev.basic_direction_bonus_utilization = util
    ev.actual_basic_direction_multiplier = actual_basic

    # 방향 각인 효율
    ev.directional_engraving_efficiency = calc_directional_engraving_efficiency(
        rb, rh, rn, eng_state, back_base, head_base, settings,
    )

    # 무력화 기대치(헤드)
    ev.stagger_bonus_expected = calc_stagger_bonus_expected(rh, settings)

    # 방향 손실 피해
    if target is None:
        # DUAL + 각인 없음: 각인 기준 방향 손실은 0. 기본 보너스 기준 손실은 참고용.
        ev.directional_engraving_loss_damage = 0.0
        ev.direction_loss_damage = calc_direction_loss_damage(
            current_crit_adjusted_damage, actual_basic, max(back_base, head_base), skill_type,
        )
    else:
        loss = calc_direction_loss_damage(
            current_crit_adjusted_damage, ev.actual_direction_multiplier, ev.optimal_direction_multiplier, skill_type,
        )
        ev.direction_loss_damage = loss
        ev.directional_engraving_loss_damage = loss if (loss is not None and eng_state != DirectionalEngravingState.NONE) else 0.0

    return ev
