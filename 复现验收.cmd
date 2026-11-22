@echo off
chcp 936 >nul
cd /d "%~dp0"

rem 复现验收：优先按交付包里的 acceptance_1 方案原样复现；
rem 仓库工作树没有该验收证据时，自动改用最近一次实验的方案复现。
echo 复现验收：优先使用 reports\acceptance_1\plan.json（交付包自带）
echo.
if exist "reports\acceptance_1\plan.json" (
  python -m autolab run --task t1_demo --plan reports/acceptance_1/plan.json
) else (
  echo [提示] 未找到 reports\acceptance_1\plan.json（该文件随交付包提供）
  echo        改为按最近一次实验的方案复现；若还没有任何实验，可先运行 python -m autolab demo
  echo.
  python -m autolab run --plan latest
)
echo.
pause
