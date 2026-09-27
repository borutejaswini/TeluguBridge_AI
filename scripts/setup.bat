@echo off
setlocal
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 exit /b %errorlevel%
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
if errorlevel 1 exit /b %errorlevel%
if not exist .env copy .env.example .env
echo Base environment ready. Whisper packages are installed separately with requirements-whisper.txt.
