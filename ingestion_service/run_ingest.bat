@echo off
setlocal
cd /d "%~dp0"
call .venv\Scripts\python.exe -m app.cli --knowledge-root ..\knowledge_base --output-dir ..\data\processed_documents
endlocal

