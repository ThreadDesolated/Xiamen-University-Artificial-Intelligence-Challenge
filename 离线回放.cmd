@echo off
cd /d "%~dp0"
python -m autolab replay --run-id acceptance_1 --delay 1
pause
