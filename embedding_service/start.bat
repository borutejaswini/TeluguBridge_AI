@echo off
setlocal
cd /d "%~dp0"
cd /d "%~dp0\.."
if not exist embedding_service\.venv\Scripts\python.exe (
  echo Run embedding_service\setup.bat first.
  exit /b 1
)
embedding_service\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir embedding_service --host 127.0.0.1 --port 8005
endlocal
