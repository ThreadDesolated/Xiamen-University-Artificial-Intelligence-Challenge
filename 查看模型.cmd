@echo off
chcp 936 >nul
cd /d "%~dp0"

rem 查看某次运行的模型全貌
rem 用法：查看模型.cmd <run_id> [--features] [--top 5] [--json]
if not "%~1"=="" goto run

echo 最近运行（左侧为 run_id）:
python -m autolab list --limit 5
echo.
set /p RUN_ID=请输入要查看的 run_id（直接回车退出）: 
if not defined RUN_ID goto end
python -m autolab model-info --run-id "%RUN_ID%"
echo.
pause
goto end

:run
set "RUN_ID=%~1"
python -m autolab model-info --run-id "%RUN_ID%" %~2 %~3 %~4
echo.
pause

:end
