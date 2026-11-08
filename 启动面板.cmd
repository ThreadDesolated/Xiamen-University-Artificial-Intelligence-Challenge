@echo off
cd /d "%~dp0"
python -m autolab serve --port 8000
pause
