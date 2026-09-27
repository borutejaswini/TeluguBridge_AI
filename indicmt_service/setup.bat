@echo off
setlocal
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
if errorlevel 1 exit /b %errorlevel%
if not exist .env copy .env.example .env
echo IndicMT CPU environment is ready.
