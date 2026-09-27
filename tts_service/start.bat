@echo off
setlocal
cd /d "%~dp0\.."
if not exist tts_service\.venv\Scripts\python.exe (
  echo Run tts_service\setup.bat first.
  exit /b 1
)
set HF_HOME=
set HF_HUB_CACHE=
set TRANSFORMERS_CACHE=
set HF_HUB_DOWNLOAD_TIMEOUT=120
tts_service\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir tts_service --host 127.0.0.1 --port 8002
endlocal
