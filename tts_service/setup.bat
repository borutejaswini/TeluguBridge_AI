@echo off
setlocal
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0+cpu
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
if errorlevel 1 exit /b %errorlevel%
echo TTS CPU environment ready.
