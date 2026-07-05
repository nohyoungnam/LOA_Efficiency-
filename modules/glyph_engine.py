"""실제 게임 화면에서 만든 글자 템플릿으로 고정폰트 숫자를 인식하는 엔진.

- 추출: 값을 아는 숫자 크롭을 글자 수(N)만큼 강제 분할(deepest-valley)하여 글자 템플릿 수집.
- 인식: OpenCV matchTemplate 다중 검출 + NMS로 분할 없이 글자열을 디코딩.
- 텍스트 높이를 H로 정규화하므로 화면/창 크기가 달라도 동작합니다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

NORM_H = 40          # 정규화 텍스트 높이(px)
UPSCALE = 4          # 작은 글자 대비 초기 확대
DET_THRESHOLD = 0.58 # matchTemplate 검출 임계값
ALLOWED = set("0123456789.,%-억만조초:")
MAX_SAMPLES_PER_CHAR = 200  # 글자당 보관 표본 상한 (평균 템플릿이라 이 정도면 충분)
REP_K = 6                   # (구버전 호환용)
MIN_GLYPH_SCORE = 0.40      # DP에서 글자로 인정하는 최소 매칭 점수
SKIP_PENALTY = 0.08         # 배경 1픽셀 건너뛸 때 페널티


def _thousands(intstr: str) -> str:
    s = (intstr or "").lstrip("0") or "0"
    try:
        return format(int(s), ",")
    except Exception:
        return s


def _band_gray_fg(crop: Image.Image) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """크롭 → (텍스트밴드 그레이 float, 전경 bool). 높이 NORM_H로 정규화."""
    if crop.width < 3 or crop.height < 3:
        return None
    big = crop.convert("L").resize((max(1, crop.width * UPSCALE), max(1, crop.height * UPSCALE)), Image.Resampling.LANCZOS)
    g = np.asarray(big, dtype=np.float32)
    thr = g.mean() + (g.max() - g.mean()) * 0.25
    fg = g > thr
    if fg.mean() > 0.5:
        fg = ~fg
    ys = np.where(fg.any(axis=1))[0]
    xs = np.where(fg.any(axis=0))[0]
    if ys.size == 0 or xs.size == 0:
        return None
    band = g[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
    fgb = fg[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
    scale = NORM_H / band.shape[0]
    bw = max(1, int(round(band.shape[1] * scale)))
    bandn = np.asarray(Image.fromarray(band.astype(np.uint8)).resize((bw, NORM_H), Image.Resampling.LANCZOS), dtype=np.float32)
    fgn = np.asarray(Image.fromarray((fgb.astype(np.uint8) * 255)).resize((bw, NORM_H), Image.Resampling.NEAREST)) > 127
    return bandn, fgn


def _force_split(fgb: np.ndarray, n: int) -> List[Tuple[int, int]]:
    proj = fgb.sum(axis=0).astype(float)
    w = fgb.shape[1]
    if n <= 1:
        return [(0, w)]
    bounds = [0]
    for k in range(1, n):
        c = int(round(k * w / n))
        lo = max(1, c - w // (2 * n))
        hi = min(w - 1, c + w // (2 * n))
        pos = c if hi <= lo else lo + int(np.argmin(proj[lo:hi]))
        bounds.append(pos)
    bounds = sorted(set(bounds)) + [w]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


class TemplateStore:
    def __init__(self) -> None:
        # char -> list of (H, w) float32 band glyphs
        self.samples: Dict[str, List[np.ndarray]] = {}
        self._mean_cache: Dict[str, np.ndarray] = {}

    def add_from_crop(self, crop: Image.Image, text: str) -> int:
        chars = [c for c in str(text).strip() if not c.isspace()]
        if not chars or any(c not in ALLOWED for c in chars):
            return 0
        bg = _band_gray_fg(crop)
        if bg is None:
            return 0
        band, fgb = bg
        segs = _force_split(fgb, len(chars))
        if len(segs) != len(chars):
            return 0
        added = 0
        for (x1, x2), ch in zip(segs, chars):
            if x2 - x1 < 3:
                continue
            lst = self.samples.setdefault(ch, [])
            lst.append(band[:, x1:x2].copy())
            if len(lst) > MAX_SAMPLES_PER_CHAR:
                del lst[0]
            added += 1
        self._mean_cache.clear()
        return added

    def char_count(self) -> int:
        return len(self.samples)

    def total_samples(self) -> int:
        return sum(len(v) for v in self.samples.values())

    def ready(self) -> bool:
        return sum(1 for d in "0123456789" if self.samples.get(d)) >= 7

    def mean_template(self, ch: str) -> Optional[np.ndarray]:
        if ch in self._mean_cache:
            return self._mean_cache[ch]
        lst = self.samples.get(ch)
        if not lst:
            return None
        widths = sorted(int(g.shape[1]) for g in lst)
        mw = max(3, widths[len(widths) // 2])
        acc = np.zeros((NORM_H, mw), dtype=np.float32)
        for g in lst:
            r = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((mw, NORM_H), Image.Resampling.LANCZOS), dtype=np.float32)
            acc += r
        acc /= len(lst)
        self._mean_cache[ch] = acc
        return acc

    def _reps(self, ch: str, k: int = REP_K) -> List[np.ndarray]:
        """글자별 '선명한' 대표 샘플 최대 k개(고르게 추출). 번진 평균 대신 사용."""
        lst = self.samples.get(ch)
        if not lst:
            return []
        if len(lst) <= k:
            return list(lst)
        idx = np.linspace(0, len(lst) - 1, k).round().astype(int)
        return [lst[i] for i in sorted(set(idx.tolist()))]

    def _decode_dp(self, crop: Image.Image, min_glyph: float = MIN_GLYPH_SCORE,
                   skip_pen: float = SKIP_PENALTY) -> Optional[Tuple[List[str], float]]:
        """DP로 밴드를 왼→오른쪽으로 빠짐없이 덮는 최적 글자열을 찾습니다.

        탐욕적 NMS의 중복/앞자리 누락 문제가 없고, '.'와 ','는 한 클래스(sep)로
        배치한 뒤 포맷 규칙으로 결정합니다. 반환: (글자 시퀀스, 신뢰도).
        """
        if not _HAS_CV2:
            return None
        bg = _band_gray_fg(crop)
        if bg is None:
            return None
        band, fg = bg
        H, W = band.shape
        bandf = ((band - band.mean()) / (band.std() + 1e-6)).astype(np.float32)
        corr: Dict[str, Tuple[np.ndarray, int]] = {}
        for ch in self.samples:
            t = self.mean_template(ch)
            if t is None or t.shape[1] > W or t.shape[0] > H:
                continue
            tf = ((t - t.mean()) / (t.std() + 1e-6)).astype(np.float32)
            corr[ch] = (cv2.matchTemplate(bandf, tf, cv2.TM_CCOEFF_NORMED)[0], int(t.shape[1]))
        if not corr:
            return None
        # '.'와 ','는 sep 한 클래스로 묶음(둘 중 높은 corr). 실제 글자는 포맷에서 결정.
        sep = [corr[c] for c in (".", ",") if c in corr]
        units: List[Tuple[str, np.ndarray, int]] = [(c, r, w) for c, (r, w) in corr.items() if c not in (".", ",")]
        if sep:
            L = min(len(a[0]) for a in sep)
            units.append(("sep", np.maximum.reduce([a[0][:L] for a in sep]), int(round(np.mean([a[1] for a in sep])))))
        fcols = np.where(fg.any(axis=0))[0]
        if fcols.size == 0:
            return None
        x0, x1 = int(fcols[0]), int(fcols[-1]) + 1
        NEG = -1e9
        dp = [NEG] * (W + 1); dp[x0] = 0.0
        back: List[Optional[Tuple[int, Optional[str], float]]] = [None] * (W + 1)
        for x in range(x0, W):
            if dp[x] <= NEG / 2:
                continue
            if dp[x] - skip_pen > dp[x + 1]:
                dp[x + 1] = dp[x] - skip_pen; back[x + 1] = (x, None, 0.0)
            for c, res, w in units:
                if x < len(res):
                    sc = float(res[x])
                    if sc < min_glyph:
                        continue
                    xe = x + w
                    if xe <= W and dp[x] + sc > dp[xe]:
                        dp[xe] = dp[x] + sc; back[xe] = (x, c, sc)
        end = max(range(x1, W + 1), key=lambda i: dp[i])
        seq: List[str] = []; scores: List[float] = []; covered = 0; x = end
        while x is not None and back[x] is not None:
            px, c, sc = back[x]
            if c is not None:
                seq.append(c); scores.append(sc); covered += (x - px)
            x = px
        seq.reverse()
        if not scores:
            return None
        coverage = covered / max(1, (x1 - x0))
        conf = float(min(scores)) * min(1.0, coverage)
        return seq, conf

    def recognize(self, crop: Image.Image, kind: Optional[str] = None) -> Tuple[str, float]:
        """글자 템플릿으로 값을 인식합니다. (텍스트, 신뢰도 0~1).

        DP로 글자 시퀀스를 뽑고, 분리기호('.'/',')는 숫자 포맷 규칙으로 복원합니다.
        kind: 'percent'|'korean'|'count'|'int'|'time'|None(자동추정).
        신뢰도가 낮으면 호출측이 EasyOCR로 폴백하는 용도입니다.
        """
        dec = self._decode_dp(crop)
        if dec is None:
            return "", 0.0
        seq, conf = dec
        digits = "".join(ch for ch in seq if ch.isdigit())
        has_pct = "%" in seq
        suf = "억" if "억" in seq else ("만" if "만" in seq else None)
        has_colon = ":" in seq
        if not digits:
            return "", 0.0
        # 자동 추정
        if kind is None:
            if has_pct:
                kind = "percent"
            elif suf:
                kind = "korean"
            elif has_colon:
                kind = "time"
            else:
                kind = "count"
        if kind == "percent":
            if len(digits) < 3:
                return digits + "%", conf * 0.5
            return f"{int(digits[:-2])}.{digits[-2:]}%", conf
        if kind == "korean":
            if suf is None or len(digits) < 3:
                return digits + (suf or ""), conf * 0.5
            return f"{_thousands(digits[:-2])}.{digits[-2:]}{suf}", conf
        if kind == "time":
            if len(digits) == 4:
                return f"{digits[:2]}:{digits[2:]}", conf
            return digits, conf * 0.5
        if kind == "count":
            return _thousands(digits), conf
        return digits, conf  # int / casts

    # ---- 영속화 ----
    def save(self, path: str | Path) -> bool:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            out: Dict[str, Any] = {"norm_h": NORM_H, "chars": {}}
            for ch, lst in self.samples.items():
                out["chars"][ch] = [
                    {"w": int(g.shape[1]), "data": np.clip(g, 0, 255).astype(np.uint8).flatten().tolist()}
                    for g in lst
                ]
            p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            return True
        except Exception:
            return False

    def load(self, path: str | Path) -> "TemplateStore":
        try:
            p = Path(path)
            if not p.exists():
                return self
            data = json.loads(p.read_text(encoding="utf-8"))
            for ch, lst in data.get("chars", {}).items():
                arrs = []
                for item in lst:
                    w = int(item["w"])
                    a = np.asarray(item["data"], dtype=np.float32).reshape(NORM_H, w)
                    arrs.append(a)
                if arrs:
                    # 기존 샘플에 '누적'(이어서 학습). 상한 초과분은 오래된 것부터 제거.
                    cur = self.samples.setdefault(ch, [])
                    cur.extend(arrs)
                    if len(cur) > MAX_SAMPLES_PER_CHAR:
                        del cur[: len(cur) - MAX_SAMPLES_PER_CHAR]
            self._mean_cache.clear()
        except Exception:
            pass
        return self
