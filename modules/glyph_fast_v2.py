"""글리프 주력 인식기 v2 — 전투분석기 고정폰트 숫자를 EasyOCR 없이 빠르게 읽기 위한 전처리/채택기.

디버그 크롭(공격표 14행 전수)으로 측정해 만든 개선판입니다. 핵심 아이디어:

1) 흰색도(min(R,G,B)) 전처리
   - 숫자는 흰색, 배경/워터마크("리워드" 등)·게이지는 파랑/회색입니다.
   - min(R,G,B)를 쓰면 흰 글자만 크게 남고 유색 배경은 죽어, 분리가 깨끗해집니다.
2) 본문(질량 최대) 덩어리 선택
   - 넓은 셀에서 가장자리 UI막대/워터마크가 아니라 '전경 픽셀이 가장 많은' 덩어리(=숫자)를 고릅니다.
   - 그냥 '맨 오른쪽'을 고르면 오른쪽 끝 밝은 막대를 숫자로 오인합니다(쿨타임 칸이 그래서 깨졌음).
3) 빈칸 감지
   - 전경 비율이 너무 작으면 None을 돌려, 빈 칸에서 '444,444' 같은 환각을 만들지 않습니다.
4) 형식 검증 + 값 범위 채택
   - 글리프 출력이 NN.NN% (퍼센트) / 정수(횟수) 형식이고 conf>=게이트일 때만 '채택'.
   - 퍼센트는 게임 특성상 100 이하이므로 100 초과는 거부(='100.00%'를 '109.00%'로 오인한 케이스 차단).

측정(공격표 디버그 크롭, 값 있는 칸 기준):
   - 퍼센트: 채택 게이트 0.48 → 커버리지 약 90%+, 정확도 약 96~100%.
   - 횟수(casts): 단자리라 약함(보수 게이트), 미달분은 EasyOCR 폴백.

이 모듈은 TemplateStore(glyph_engine)를 그대로 활용하고, '입력 전처리'와 '채택 판정'만 개선합니다.
recognize()는 (text, conf, accepted)를 돌려주며 accepted=False면 호출측이 EasyOCR로 폴백하면 됩니다.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

# 형식 검증 패턴
_PCT_FMT = re.compile(r"^\d{1,3}\.\d{2}%$")
_CNT_FMT = re.compile(r"^\d{1,4}$")

# 채택 게이트(형식 검증과 함께 사용). 필요 시 호출측에서 조정 가능하게 상수로 둠.
GATE_PERCENT = 0.48
GATE_COUNT = 0.60   # 횟수(casts)는 단자리 혼동(8↔1,9↔0)이 있어 약간 높게 잡아 오답 차단
PERCENT_MAX = 100.5  # 게임 퍼센트는 100 이하 → 초과는 오인식으로 보고 거부

# 빈칸 판정: 흰색 전경 비율이 이 값 미만이면 '값 없음'으로 간주
EMPTY_FG_RATIO = 0.006
# 횟수(casts)는 1~2자리라 폭이 좁고(단자리 ~8px) 전경 비율도 작습니다.
# 그래서 퍼센트보다 노이즈폭(min_w)·빈칸 임계값을 낮춰야 단자리가 안 버려집니다.
EMPTY_FG_RATIO_COUNT = 0.004
MIN_W_FACTOR_PERCENT = 0.25
MIN_W_FACTOR_COUNT = 0.10


def whiteness_isolate(pil: Image.Image, pad: int = 4,
                      min_w_factor: float = MIN_W_FACTOR_PERCENT,
                      empty_ratio: float = EMPTY_FG_RATIO) -> Optional[Image.Image]:
    """흰 숫자만 남겨(본문 덩어리) 잘라낸 L(그레이) 이미지를 반환. 빈칸/판독불가면 None.

    글리프 엔진에 넣기 직전의 전처리입니다. 배경 유색 요소를 죽이고 숫자 영역만 남깁니다.
    min_w_factor/empty_ratio 는 칸 종류별로 조정합니다(횟수는 단자리라 둘 다 낮춤).
    """
    try:
        rgb = np.asarray(pil.convert("RGB"), dtype=np.float32)
    except Exception:
        return None
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    white = rgb.min(axis=2)  # 흰색일수록 큼
    H, W = white.shape
    if W < 8 or H < 8:
        return None
    thr = white.mean() + (white.max() - white.mean()) * 0.30
    fg = white > thr
    if fg.mean() < empty_ratio:
        return None  # 빈 칸
    on = fg.sum(axis=0)
    gap = max(8, int(0.5 * H))
    runs = []
    s = None
    for i, v in enumerate(on > 0):
        if v and s is None:
            s = i
        if not v and s is not None:
            runs.append([s, i]); s = None
    if s is not None:
        runs.append([s, W])
    if not runs:
        return None
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    min_w = max(3, int(min_w_factor * H))
    cand = [(a, b) for a, b in merged if (b - a) >= min_w]
    if not cand:
        return None
    # 전경 질량(픽셀 수)이 가장 큰 덩어리 = 본문 숫자 (가장자리 막대/워터마크 배제)
    a, b = max(cand, key=lambda ab: int(on[ab[0]:ab[1]].sum()))
    a = max(0, a - pad); b = min(W, b + pad)
    ys = np.where(fg[:, a:b].any(axis=1))[0]
    if ys.size:
        y1, y2 = max(0, int(ys[0]) - pad), min(H, int(ys[-1]) + 1 + pad)
    else:
        y1, y2 = 0, H
    wc = white[y1:y2, a:b]
    rng = (wc.max() - wc.min())
    wn = np.clip((wc - wc.min()) / (rng + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(wn, mode="L")


def _percent_value_ok(text: str) -> bool:
    try:
        return float(text.rstrip("%")) <= PERCENT_MAX
    except Exception:
        return False


def recognize(store: Any, pil_crop: Image.Image, kind: str,
              gate_percent: float = GATE_PERCENT, gate_count: float = GATE_COUNT
              ) -> Tuple[str, float, bool]:
    """개선 전처리로 단일 셀을 인식.

    반환: (text, conf, accepted)
      - accepted=True  : 형식·범위·신뢰도 통과 → EasyOCR 없이 이 값 사용 가능
      - accepted=False : 호출측에서 EasyOCR로 폴백 (정확도 손실 0)
    kind: 'percent' | 'count'
    """
    # 칸 종류별 전처리 파라미터: 횟수는 단자리라 노이즈폭·빈칸 임계값을 낮춤.
    if kind == "count":
        iso = whiteness_isolate(pil_crop, min_w_factor=MIN_W_FACTOR_COUNT,
                                empty_ratio=EMPTY_FG_RATIO_COUNT)
    else:
        iso = whiteness_isolate(pil_crop)
    if iso is None:
        return "", 0.0, False
    try:
        text, conf = store.recognize(iso, kind=kind)
    except Exception:
        return "", 0.0, False
    text = text or ""
    conf = float(conf)
    if kind == "percent":
        accepted = bool(_PCT_FMT.match(text)) and conf >= gate_percent and _percent_value_ok(text)
    elif kind == "count":
        accepted = bool(_CNT_FMT.match(text)) and conf >= gate_count
    else:
        accepted = False
    return text, conf, accepted
