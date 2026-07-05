@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM v121 clean static CSS UI: 필요한 표만 CSS 정적 렌더 + 브라우저 자동 열기
set LOA_UI_FAST_MODE=static_first
set LOA_RENDER_STATIC_TABLE=1
set LOA_ATTACK_FAST_GRID=1
set LOA_ATTACK_ROW_PASS_SCALE=3
set LOA_ATTACK_DAMAGE_RECHECK=smart
set LOA_ATTACK_CRITICAL_CELL_SCALE=3
set LOA_ATTACK_PERCENT_FALLBACK=1
set LOA_ATTACK_CAST_COOLDOWN_FALLBACK=1
set LOA_ATTACK_RECHECK=0
set LOA_FORCE_DAMAGE_CELL_OCR=0
set LOA_FORCE_CASTS_CELL_OCR=0
set LOA_ATTACK_CASTS_RECHECK=0
set LOA_ICON_HEAVY_TOPK=5
set LOA_ICON_VARIANT_RETRY=0
set LOA_ICON_LOWCONF_EXTRA_THRESHOLD=68
set LOA_ICON_TIGHT_GUIDE=0
set LOA_ICON_SQUARE_CROP=1
set LOA_ICON_SQUARE_SIDE_RATIO=1.00
set LOA_ICON_SQUARE_FALLBACK_MARGIN=4.0
set LOA_SUMMARY_RAW_SCALE=3
set LOA_GLYPH_ONLY_NUMBERS=0
set LOA_API_CACHE_TTL_SEC=0
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
set STREAMLIT_SERVER_RUN_ON_SAVE=false
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_SERVER_HEADLESS=false

REM OCR/ONNX CPU 스레드 과점유 방지
set LOA_OCR_CPU_MODE=web_low
set LOA_OCR_THREADS=1
set LOA_FORCE_CPU_LIMIT=1
set RAPIDOCR_THREADS=1
set OMP_NUM_THREADS=1
set ORT_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set NUMEXPR_MAX_THREADS=1
set OMP_WAIT_POLICY=PASSIVE
set KMP_BLOCKTIME=0

set LOA_URL=http://localhost:8501

echo [LOA] v121 clean static CSS UI 실행 중...
echo [LOA] 현재 실행 폴더: %CD%
echo [LOA] 브라우저가 자동으로 안 열리면 주소창에 %LOA_URL% 를 입력하세요.

python -m pip install -r requirements.txt

REM Streamlit이 자동으로 브라우저를 못 여는 환경이 있어서 3초 뒤 강제로 한 번 더 엽니다.
start "" cmd /c "timeout /t 3 /nobreak >nul & start "" "%LOA_URL%""

python -m streamlit run app.py --server.port 8501 --server.headless false --server.fileWatcherType none --server.runOnSave false --browser.gatherUsageStats false

pause
