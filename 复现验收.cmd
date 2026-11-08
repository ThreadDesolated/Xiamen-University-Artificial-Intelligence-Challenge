@echo off
cd /d "%~dp0"
python -m autolab run --task t1_demo --plan reports/acceptance_1/plan.json
pause
