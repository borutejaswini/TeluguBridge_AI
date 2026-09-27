@echo off
setlocal
cd /d "%~dp0"
py -3.12 -m venv .venv
call .venv\Scripts\python.exe -m pip install --upgrade pip
call .venv\Scripts\python.exe -m pip install -r requirements-dev.txt
endlocal

