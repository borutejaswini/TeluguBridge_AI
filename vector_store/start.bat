@echo off
setlocal
cd /d "%~dp0\.."
if not exist vector_store\.venv\Scripts\python.exe (
  echo Run vector_store\setup.bat first.
  exit /b 1
)
vector_store\.venv\Scripts\python.exe vector_store\server.py
endlocal
