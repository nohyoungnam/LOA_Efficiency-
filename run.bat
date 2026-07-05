@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM v127: browser-only OCR mode. ASCII-only to avoid codepage corruption.
set LOA_DEFAULT_OCR_MODE=client_browser_only
set LOA_API_CACHE_TTL_SEC=0
set LOA_RENDER_STATIC_TABLE=1
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
set STREAMLIT_SERVER_RUN_ON_SAVE=false
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_SERVER_HEADLESS=false
set LOA_PORT=8501
set LOA_URL=http://localhost:%LOA_PORT%

echo [LOA] Installing requirements...
python -m pip install -r requirements.txt

REM Free port 8501 right before launch: kill any process LISTENING on it, twice, then wait.
echo [LOA] Freeing port %LOA_PORT% if a previous instance is still running...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr LISTENING ^| findstr ":%LOA_PORT%"') do taskkill /f /pid %%p >nul 2>&1
timeout /t 1 /nobreak >nul
for /f "tokens=5" %%p in ('netstat -ano ^| findstr LISTENING ^| findstr ":%LOA_PORT%"') do taskkill /f /pid %%p >nul 2>&1
timeout /t 2 /nobreak >nul

REM Streamlit opens the browser by itself (server.headless=false) exactly once when ready.
echo [LOA] Starting Streamlit on port %LOA_PORT% ... (browser opens automatically)
python -m streamlit run app.py --server.port %LOA_PORT% --server.headless false --server.fileWatcherType none --server.runOnSave false --browser.gatherUsageStats false

echo.
echo [LOA] If you saw 'Port 8501 is not available', CLOSE all old black cmd windows and run this again.
pause
