@echo off
setlocal
cd /d "%~dp0\.."
if not exist embedding_service\.venv\Scripts\python.exe (
  echo Run embedding_service\setup.bat first.
  exit /b 1
)
set HF_HOME=
set HF_HUB_CACHE=
set TRANSFORMERS_CACHE=
call embedding_service\.venv\Scripts\python.exe -m uvicorn reranker_service.app.main:app --host 127.0.0.1 --port 8003
endlocal

