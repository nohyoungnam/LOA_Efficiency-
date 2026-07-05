"""RapidOCR 어댑터 — EasyOCR readtext() API 호환 래퍼.

EasyOCR(torch 기반) → rapidocr-onnxruntime(ONNX 기반) 교체 어댑터입니다.
torch 없이 동작하므로 Hugging Face Spaces 무료 CPU 배포 메모리를 크게 줄입니다.

## 한국어(Korean) 지원
피해량 단위(억/만)·스킬명 등 한글 인식을 위해 Korean ONNX 모델을 사용합니다.
- 자동 우선순위: 환경변수 RAPIDOCR_KO_MODEL → data/ko_rec.onnx → 자동 다운로드
- 자동 다운로드 소스: HuggingFace RapidAI (~10MB), 최초 1회만
- 모델이 없으면 중국어+영어 모델로 폴백(숫자·%는 OK, 억/만 등 한글 누락 가능)

## readtext() 반환값 (EasyOCR detail=1 형식과 동일)
    [(bbox_4points, text, conf), ...]
    bbox_4points: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]

## 한국어 정규화
CJK 한자와 한글 단위 문자 간 자동 변환:
    億→억, 兆→조, 萬→만, 万→만, 秒→초, 分→분
"""
from __future__ import annotations

import logging
import os
import re
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CPU 사용량 제한 (웹/저사양 배포 대비)
# ---------------------------------------------------------------------------
# onnxruntime는 기본적으로 코어 수만큼 스레드를 띄우고 spin-wait를 해서, 추론이 아닐 때도
# CPU를 크게 점유합니다. import 전에 아래 환경변수를 설정해 스레드 수를 제한하고 유휴 시
# busy-wait를 끕니다. RAPIDOCR_THREADS 로 직접 지정할 수 있고, 미지정 시 최대 4개로 제한합니다.
def _init_cpu_limits() -> None:
    """v124: 웹 배포용 OCR/ONNX/OpenCV 스레드 제한을 가장 이른 시점에 적용합니다."""
    mode = str(os.environ.get("LOA_OCR_CPU_MODE", "web_low")).strip().lower()
    if mode in {"local_fast", "fast", "desktop_fast"}:
        default_threads = "4"
        force = str(os.environ.get("LOA_FORCE_CPU_LIMIT", "0")).strip().lower() in {"1", "true", "yes", "on"}
    elif mode in {"balanced", "local_balanced"}:
        default_threads = "2"
        force = True
    else:
        default_threads = "1"
        force = True
    threads = str(os.environ.get("LOA_OCR_THREADS", default_threads)).strip() or default_threads
    try:
        n = max(1, min(8, int(float(threads))))
    except Exception:
        n = int(default_threads)
    threads = str(n)
    os.environ["LOA_EFFECTIVE_OCR_THREADS"] = threads
    for var in ("RAPIDOCR_THREADS", "OMP_NUM_THREADS", "ORT_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        if force or not os.environ.get(var):
            os.environ[var] = threads
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("KMP_SETTINGS", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        import cv2  # type: ignore
        cv2.setNumThreads(n)
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
    except Exception:
        pass


def _patch_onnxruntime_threads() -> None:
    """RapidOCR 내부 onnxruntime InferenceSession 생성 시 thread options를 강제로 주입합니다."""
    try:
        import onnxruntime as ort  # type: ignore
        if getattr(ort, "_loa_v124_thread_patch", False):
            return
        try:
            threads = max(1, min(8, int(os.environ.get("LOA_EFFECTIVE_OCR_THREADS") or os.environ.get("RAPIDOCR_THREADS") or "1")))
        except Exception:
            threads = 1
        original = ort.InferenceSession
        def patched_inference_session(*args: Any, **kwargs: Any) -> Any:
            sess_options = kwargs.get("sess_options")
            if sess_options is None and len(args) >= 2 and isinstance(args[1], ort.SessionOptions):
                sess_options = args[1]
            if sess_options is None:
                sess_options = ort.SessionOptions()
                kwargs["sess_options"] = sess_options
            try:
                sess_options.intra_op_num_threads = threads
                sess_options.inter_op_num_threads = 1
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                try:
                    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                    sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
                except Exception:
                    pass
            except Exception:
                pass
            return original(*args, **kwargs)
        ort.InferenceSession = patched_inference_session  # type: ignore[assignment]
        ort._loa_v124_thread_patch = True  # type: ignore[attr-defined]
    except Exception:
        pass


_init_cpu_limits()
_patch_onnxruntime_threads()
_RAPIDOCR_THREADS = int(os.environ.get("RAPIDOCR_THREADS") or os.environ.get("LOA_EFFECTIVE_OCR_THREADS") or 1)

# ---------------------------------------------------------------------------
# Korean 정규화 테이블 (CJK/중국어 출력 → 한글 단위 문자)
# ---------------------------------------------------------------------------
_CJK_TO_KO: dict[str, str] = {
    "億": "억",  # 1억 = 10^8
    "兆": "조",  # 1조 = 10^12
    "萬": "만",  # 1만 = 10^4
    "万": "만",  # 1만 (간체)
    "秒": "초",  # 초
    "分": "분",  # 분
    "時": "시",  # 시
    "개": "개",
}


def _normalize_ko(text: str) -> str:
    """CJK 문자를 한글 단위 문자로 정규화합니다."""
    return "".join(_CJK_TO_KO.get(c, c) for c in text)


# ---------------------------------------------------------------------------
# Korean ONNX 모델 자동 다운로드
# ---------------------------------------------------------------------------
# rapidocr-onnxruntime 기본 Chinese+English 모델은 한글 Hangul 을 인식하지 못합니다.
# 아래 Korean 인식 모델(~10MB)을 1회 다운로드해 data/ 에 보관합니다.
_KO_REC_MODEL_FILENAME = "ko_rec.onnx"
# monkt/paddleocr-onnx: Korean PP-OCRv3 rec 모델 (13.4MB, 실제 동작 확인)
_KO_REC_MODEL_URL = (
    "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/korean/rec.onnx"
)
# 다운로드 실패 시 fallback URL (구 URL → 404가 되었으므로 대체)
_KO_REC_MODEL_URL_FALLBACK = (
    "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/korean/rec.onnx?download=true"
)


def _find_data_dir() -> Path:
    """data/ 폴더를 프로젝트 루트 또는 modules/ 기준으로 찾습니다."""
    try:
        here = Path(__file__).resolve().parent
        candidates = [
            here.parent / "data",
            here / "data",
            Path.cwd() / "data",
        ]
        for c in candidates:
            if c.is_dir():
                return c
        # 없으면 첫 번째 경로에 생성
        candidates[0].mkdir(parents=True, exist_ok=True)
        return candidates[0]
    except Exception:
        return Path("data")


def _ko_rec_model_path() -> Optional[Path]:
    """Korean rec 모델 경로를 반환합니다. 우선순위: 환경변수 → data/ → None."""
    # 1. 환경변수
    env = os.environ.get("RAPIDOCR_KO_MODEL")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    # 2. data/ 폴더
    data_dir = _find_data_dir()
    p = data_dir / _KO_REC_MODEL_FILENAME
    if p.is_file():
        return p
    return None


def download_ko_model(force: bool = False) -> Optional[Path]:
    """Korean rec ONNX 모델을 data/ 에 다운로드합니다. 이미 있으면 스킵.

    반환: 저장된 경로 (실패 시 None)
    """
    dest = _find_data_dir() / _KO_REC_MODEL_FILENAME
    if dest.is_file() and not force:
        return dest

    logger.info(
        "[RapidOCR] Korean 인식 모델 다운로드 중 (%s) ...", _KO_REC_MODEL_URL
    )
    print(f"[RapidOCR] Korean 모델 다운로드: {_KO_REC_MODEL_URL}")
    for url in [_KO_REC_MODEL_URL, _KO_REC_MODEL_URL_FALLBACK]:
        try:
            urllib.request.urlretrieve(url, str(dest))
            size_mb = dest.stat().st_size / 1e6
            logger.info("[RapidOCR] 다운로드 완료: %s (%.1f MB)", dest, size_mb)
            print(f"[RapidOCR] 다운로드 완료: {dest} ({size_mb:.1f} MB)")
            return dest
        except Exception as e:
            logger.warning("[RapidOCR] 다운로드 실패 (%s): %s", url, e)
    logger.error("[RapidOCR] Korean 모델 다운로드 실패. 중국어 모델로 폴백합니다.")
    return None


# ---------------------------------------------------------------------------
# RapidOCR 엔진 초기화 (1회 캐시)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _build_engine(use_korean_model: bool) -> Any:
    """RapidOCR 엔진을 생성합니다 (1회만).

    rapidocr-onnxruntime 1.2.x(Python 3.13 호환) / 1.3.x 양쪽에서 동작합니다.
    Korean 모델은 rec_model_path kwarg 로 전달합니다(1.2.x에서도 **kwargs 수용).
    """
    # v124 ensure ORT thread patch before RapidOCR import
    _init_cpu_limits()
    _patch_onnxruntime_threads()
    from rapidocr_onnxruntime import RapidOCR  # type: ignore

    if use_korean_model:
        ko_path = _ko_rec_model_path()
        if ko_path is None:
            print("[RapidOCR] Korean 모델 경로를 찾지 못해 다운로드를 시도합니다.")
            ko_path = download_ko_model()
        print(f"[RapidOCR] Korean 모델 경로: {ko_path} (exists={ko_path.is_file() if ko_path else False})")
        if ko_path is not None and ko_path.is_file():
            # rapidocr-onnxruntime 버전별로 rec 모델 경로 kwarg 형식이 달라
            # 세 가지 포맷을 차례로 시도합니다.
            #  1) rec_model_path(str)  : 1.3.x style
            #  2) Rec.model_path(str)  : 1.2.x params(dict) style
            #  3) rec_model_path(Path) : 일부 빌드가 Path 객체를 요구
            attempts = [
                ("rec_model_path(str)", {"rec_model_path": str(ko_path)}),
                ("Rec.model_path(str)", {"Rec": {"model_path": str(ko_path)}}),
                ("rec_model_path(Path)", {"rec_model_path": ko_path}),
            ]
            for kwarg_name, kwargs in attempts:
                try:
                    engine = RapidOCR(**kwargs)
                    logger.info("[RapidOCR] Korean 모델 로드 성공 (%s): %s", kwarg_name, ko_path)
                    print(f"[RapidOCR] ✓ Korean 모델 로드 성공 ({kwarg_name}): {ko_path}")
                    return engine
                except Exception as e:
                    logger.debug("[RapidOCR] Korean 모델 kwarg 실패 (%s): %s", kwarg_name, e)
                    print(f"[RapidOCR] ✗ Korean 모델 kwarg 실패 ({kwarg_name}): {e}")
            logger.warning("[RapidOCR] Korean 모델 로드 실패 — 기본(Chinese) 모델 사용")
            print("[RapidOCR] ⚠ Korean 모델 로드 실패 — 기본(Chinese) 모델 사용. 한글 스킬명 인식 불가!")
        else:
            print("[RapidOCR] ⚠ Korean 모델 파일이 없습니다 — 기본(Chinese) 모델 사용. 한글 스킬명 인식 불가!")

    return RapidOCR()


# ---------------------------------------------------------------------------
# RapidOCR 어댑터 클래스
# ---------------------------------------------------------------------------

class RapidOCRAdapter:
    """EasyOCR reader.readtext() 호환 래퍼.

    사용법:
        adapter = RapidOCRAdapter(use_korean_model=True)
        results = adapter.readtext(arr, detail=1)
    """

    def __init__(self, use_korean_model: bool = True):
        self._use_ko = use_korean_model
        self._engine = None  # lazy init

    def _get_engine(self) -> Any:
        if self._engine is None:
            self._engine = _build_engine(self._use_ko)
        return self._engine

    def readtext(
        self,
        arr: "np.ndarray",
        detail: int = 1,
        paragraph: bool = False,
        allowlist: Optional[str] = None,
    ) -> List[Any]:
        """EasyOCR 호환 readtext.

        detail=1: [(bbox, text, conf), ...]  ← EasyOCR 기본 형식
        detail=0: [text, ...]
        allowlist: 허용 문자 집합 (허용 문자 외 문자 제거). None이면 전체 허용.
        """
        engine = self._get_engine()
        try:
            # 각도 분류기(use_cls)는 전투분석기 텍스트가 회전될 일이 없어 끕니다(모델 1패스 절약 = CPU 절감).
            try:
                result, _ = engine(arr, use_cls=False)
            except TypeError:
                result, _ = engine(arr)  # 구버전: use_cls kwarg 미지원
        except Exception as e:
            logger.warning("[RapidOCR] 인식 오류: %s", e)
            return []

        if not result:
            return []

        out: List[Any] = []
        for item in result:
            try:
                box = item[0]   # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                text = _normalize_ko(str(item[1]))
                conf = float(item[2]) if len(item) > 2 else 0.5
            except Exception:
                continue

            # allowlist 필터링: 허용 문자만 남김
            if allowlist is not None:
                text = "".join(c for c in text if c in allowlist)

            if detail == 0:
                out.append(text)
            else:
                out.append((box, text, conf))

        return out


# ---------------------------------------------------------------------------
# 팩토리 함수 (fixed_grid_ocr.py 에서 get_easyocr_reader 대신 호출)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def get_rapidocr_reader(gpu: bool = False, use_korean_model: bool = True) -> RapidOCRAdapter:
    """RapidOCR 어댑터를 생성합니다 (lru_cache 로 1회만).

    gpu: 무시됩니다 (rapidocr-onnxruntime 은 CPU ONNX 전용).
    use_korean_model: True 이면 Korean rec 모델 사용 시도 (한글 인식 향상).
    """
    return RapidOCRAdapter(use_korean_model=use_korean_model)


def rapidocr_available() -> bool:
    """rapidocr-onnxruntime 설치 여부를 확인합니다."""
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI: 한국어 모델 수동 다운로드
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if "--download-ko" in sys.argv:
        result = download_ko_model(force="--force" in sys.argv)
        if result:
            print(f"[OK] {result}")
        else:
            print("[FAIL] 다운로드 실패")
            sys.exit(1)
    elif "--check" in sys.argv:
        print("rapidocr_available:", rapidocr_available())
        ko = _ko_rec_model_path()
        print("ko_rec_model:", ko)
    else:
        print("사용법: python -m modules.rapidocr_adapter --download-ko [--force]")
        print("       python -m modules.rapidocr_adapter --check")
