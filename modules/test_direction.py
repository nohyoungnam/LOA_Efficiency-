"""modules/direction.py 단위 테스트.

실행: python3 -m pytest modules/test_direction.py -q
또는:  python3 modules/test_direction.py   (pytest 없이도 동작)
"""

from __future__ import annotations

import math

from direction import (  # 같은 폴더 실행 기준
    AttackDirectionType as T,
    DirectionalEngravingState as E,
    DirectionSettings,
    DirectionValidationError,
    classify_skill_attack_direction,
    resolve_engraving_state,
    resolve_directional_target,
    calc_crit_expected_multiplier,
    calc_back_base_multiplier,
    calc_head_base_multiplier,
    normalize_observed_direction_shares,
    calc_basic_direction_bonus_utilization,
    calc_directional_engraving_efficiency,
    calc_direction_loss_damage,
    calc_direction_data_confidence,
    evaluate_skill_direction,
)


# 기습/결투 각각 +10% 조건부 가정으로 테스트합니다.
SET = DirectionSettings(ambush_master_multiplier=1.10, master_brawler_multiplier=1.10)


def approx(a, b, tol=1e-6):
    return a is not None and b is not None and abs(a - b) <= tol


# ---------------------------------------------------------------------------
# 1. 구조 분류
# ---------------------------------------------------------------------------


def test_classify_from_api():
    assert classify_skill_attack_direction("백어택") == T.BACK_ONLY
    assert classify_skill_attack_direction("헤드어택") == T.HEAD_ONLY
    assert classify_skill_attack_direction("백/헤드") == T.DUAL
    assert classify_skill_attack_direction("없음") == T.NONE
    assert classify_skill_attack_direction("") == T.NONE


def test_classify_is_api_authoritative_ignores_observed():
    # v137: 구조 방향은 오직 API 공격타입으로만 판정. 관측값은 무시합니다.
    # API가 백만 명시하면, 헤드가 관측돼도 BACK_ONLY(관측으로 DUAL 승격 안 함).
    assert classify_skill_attack_direction("백어택", observed_has_head=True) == T.BACK_ONLY
    # API가 헤드만 명시하면, 백이 관측돼도 HEAD_ONLY (와일드 스톰프 케이스).
    assert classify_skill_attack_direction("헤드어택", observed_has_back=True) == T.HEAD_ONLY
    # API에 방향 정보가 없으면, 백/헤드가 관측돼도 무방향(NONE) (라그나 브레이크 케이스).
    assert classify_skill_attack_direction("", True, True) == T.NONE
    assert classify_skill_attack_direction(None, True, False) == T.NONE
    assert classify_skill_attack_direction(None, False, False) == T.NONE


# ---------------------------------------------------------------------------
# 2. 각인 상태 / 목표 방향
# ---------------------------------------------------------------------------


def test_engraving_state_conflict_raises():
    try:
        resolve_engraving_state(True, True)
    except DirectionValidationError:
        pass
    else:
        raise AssertionError("동시 착용은 validation error여야 합니다")
    assert resolve_engraving_state(True, False) == E.AMBUSH_MASTER
    assert resolve_engraving_state(False, True) == E.MASTER_BRAWLER
    assert resolve_engraving_state(False, False) == E.NONE


def test_resolve_target():
    assert resolve_directional_target(T.BACK_ONLY, E.NONE) == "BACK"
    assert resolve_directional_target(T.HEAD_ONLY, E.NONE) == "HEAD"
    assert resolve_directional_target(T.DUAL, E.AMBUSH_MASTER) == "BACK"
    assert resolve_directional_target(T.DUAL, E.MASTER_BRAWLER) == "HEAD"
    assert resolve_directional_target(T.DUAL, E.NONE) is None
    assert resolve_directional_target(T.NONE, E.NONE) is None


# ---------------------------------------------------------------------------
# 3. 기본 배율
# ---------------------------------------------------------------------------


def test_crit_and_base_multipliers():
    # 치적 50%, 치피 200%: 기대배율 = 1 + 0.5*(2-1) = 1.5
    assert approx(calc_crit_expected_multiplier(0.5, 2.0), 1.5)
    # 헤드 기본 = 1.20
    assert approx(calc_head_base_multiplier(SET), 1.20)
    # 백 기본(치적 50%): 1.05 * (1+0.6*1) / (1+0.5*1) = 1.05 * 1.6/1.5
    expected = 1.05 * 1.6 / 1.5
    assert approx(calc_back_base_multiplier(0.5, 2.0, SET), expected)


def test_back_crit_caps_at_100():
    # 치적 98% → 백 치적 10% 중 유효분은 2%뿐
    m = calc_back_base_multiplier(0.98, 2.0, SET)
    # with_rate = min(0.98+0.10, 1.0) = 1.0
    expected = 1.05 * (1 + 1.0 * 1.0) / (1 + 0.98 * 1.0)
    assert approx(m, expected)


# ---------------------------------------------------------------------------
# 6. raw 비중 역산
# ---------------------------------------------------------------------------


def test_normalize_backout_removes_head_bonus():
    # 헤드는 20% 보너스가 있으므로, 결과 피해 비중 그대로면 raw 헤드는 더 낮아야 한다.
    raw = normalize_observed_direction_shares(
        observed_back_share=0.0, observed_head_share=0.6,
        back_total_multiplier=1.05, head_total_multiplier=1.20,
        skill_type=T.HEAD_ONLY,
    )
    # neutral = 0.4. denom = 0.6/1.2 + 0.4 = 0.9 ; raw_head = 0.5/0.9
    assert approx(raw["raw_head"], (0.6 / 1.2) / 0.9)
    assert raw["raw_head"] < 0.6  # 과대평가 제거됨


def test_normalize_none_missing_structural_direction():
    # BACK_ONLY인데 헤드 데이터가 None → 헤드 0으로 간주, 역산 성공
    raw = normalize_observed_direction_shares(
        0.7, None, 1.05, 1.20, T.BACK_ONLY,
    )
    assert raw is not None
    # DUAL인데 한 방향이 None → 역산 불가(None)
    raw2 = normalize_observed_direction_shares(
        None, 0.5, 1.05, 1.20, T.DUAL,
    )
    assert raw2 is None


# ---------------------------------------------------------------------------
# 데이터 신뢰도
# ---------------------------------------------------------------------------


def test_confidence():
    assert calc_direction_data_confidence(0.95) == "높음"
    assert calc_direction_data_confidence(0.7) == "보통"
    assert calc_direction_data_confidence(0.3) == "낮음"
    assert calc_direction_data_confidence(None) == "없음"


# ---------------------------------------------------------------------------
# 스킬 타입별 오케스트레이터 테스트
# ---------------------------------------------------------------------------


def test_back_only_skill():
    ev = evaluate_skill_direction(
        api_attack_text="백어택",
        observed_back_share=0.8, observed_head_share=None,
        base_crit_rate=0.5, crit_damage_multiplier=2.0,
        current_crit_adjusted_damage=1000.0,
        global_engraving_state=E.AMBUSH_MASTER,
        settings=SET,
    )
    assert ev.skill_type == T.BACK_ONLY
    assert ev.engraving_state == E.AMBUSH_MASTER
    assert ev.target_direction == "BACK"
    # 기습 착용 → 백 실패는 각인 손실로 계산되므로 loss > 0 (80%만 성공)
    assert ev.direction_loss_damage is not None and ev.direction_loss_damage > 0
    assert ev.basic_direction_bonus_utilization is not None
    assert ev.directional_engraving_efficiency is not None


def test_back_only_without_ambush_no_engraving_loss():
    ev = evaluate_skill_direction(
        "백어택", 0.8, None, 0.5, 2.0, 1000.0,
        global_engraving_state=E.NONE, settings=SET,
    )
    assert ev.skill_type == T.BACK_ONLY
    assert ev.engraving_state == E.NONE
    # 각인이 없으니 각인 효율 None, 각인 손실 0. 하지만 기본 보너스 활용률은 존재.
    assert ev.directional_engraving_efficiency is None
    assert ev.directional_engraving_loss_damage == 0.0
    assert ev.basic_direction_bonus_utilization is not None


def test_head_only_skill():
    ev = evaluate_skill_direction(
        "헤드어택", None, 0.9, 0.5, 2.0, 1000.0,
        global_engraving_state=E.MASTER_BRAWLER, settings=SET,
    )
    assert ev.skill_type == T.HEAD_ONLY
    assert ev.target_direction == "HEAD"
    assert ev.stagger_bonus_expected is not None and ev.stagger_bonus_expected > 0
    assert ev.directional_engraving_efficiency is not None


def test_none_skill_excluded():
    ev = evaluate_skill_direction(
        "없음", None, None, 0.5, 2.0, 1000.0,
        global_engraving_state=E.AMBUSH_MASTER, settings=SET,
    )
    assert ev.skill_type == T.NONE
    assert ev.direction_loss_damage == 0.0
    assert ev.basic_direction_bonus_utilization is None
    assert ev.directional_engraving_efficiency is None
    assert approx(ev.actual_direction_multiplier, 1.0)


def test_dual_with_ambush():
    # DUAL + 기습: 목표 BACK. 헤드로 들어간 피해도 헤드 20% 기본 보너스는 받음.
    ev = evaluate_skill_direction(
        "백/헤드", observed_back_share=0.5, observed_head_share=0.4,
        base_crit_rate=0.5, crit_damage_multiplier=2.0,
        current_crit_adjusted_damage=1000.0,
        global_engraving_state=E.AMBUSH_MASTER, settings=SET,
    )
    assert ev.skill_type == T.DUAL
    assert ev.target_direction == "BACK"
    assert ev.directional_engraving_efficiency is not None
    # 헤드 피해가 완전 실패로 처리되지 않았는지: raw_head > 0
    assert ev.raw_head_share is not None and ev.raw_head_share > 0
    # 최적 = back_total (기습 포함)
    assert approx(ev.optimal_direction_multiplier, ev.back_total_multiplier)


def test_dual_with_brawler():
    ev = evaluate_skill_direction(
        "백/헤드", 0.3, 0.6, 0.5, 2.0, 1000.0,
        global_engraving_state=E.MASTER_BRAWLER, settings=SET,
    )
    assert ev.skill_type == T.DUAL
    assert ev.target_direction == "HEAD"
    # 백으로 들어간 피해도 백 5% 기본 보너스는 받음 → raw_back > 0
    assert ev.raw_back_share is not None and ev.raw_back_share > 0
    assert approx(ev.optimal_direction_multiplier, ev.head_total_multiplier)
    assert ev.directional_engraving_efficiency is not None


def test_dual_without_engraving():
    # DUAL + 각인 없음: 목표 방향 None, 각인 효율 None, 각인 손실 0.
    # 하지만 무방향 스킬로 처리하면 안 됨: 기본 보너스 활용률은 계산.
    ev = evaluate_skill_direction(
        "백/헤드", 0.5, 0.4, 0.5, 2.0, 1000.0,
        global_engraving_state=E.NONE, settings=SET,
    )
    assert ev.skill_type == T.DUAL
    assert ev.target_direction is None
    assert ev.directional_engraving_efficiency is None
    assert ev.directional_engraving_loss_damage == 0.0
    assert ev.basic_direction_bonus_utilization is not None
    # 배율 자체는 계산됨(무방향 아님)
    assert ev.actual_direction_multiplier is not None and ev.actual_direction_multiplier > 1.0


def test_dual_missing_data_low_confidence():
    # DUAL인데 한 방향 데이터가 없음 → raw 계산 불가, has_direction_data False
    ev = evaluate_skill_direction(
        "백/헤드", None, 0.5, 0.5, 2.0, 1000.0,
        global_engraving_state=E.AMBUSH_MASTER, settings=SET,
    )
    assert ev.skill_type == T.DUAL
    assert ev.has_direction_data is False
    assert ev.raw_back_share is None


def test_engraving_efficiency_monotonic():
    # 목표 방향 성공률이 높을수록 각인 효율이 높아야 함(단조성).
    low = evaluate_skill_direction(
        "백/헤드", 0.2, 0.7, 0.5, 2.0, 1000.0, E.AMBUSH_MASTER, SET,
    ).directional_engraving_efficiency
    high = evaluate_skill_direction(
        "백/헤드", 0.9, 0.05, 0.5, 2.0, 1000.0, E.AMBUSH_MASTER, SET,
    ).directional_engraving_efficiency
    assert low is not None and high is not None and high > low


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  ok  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
