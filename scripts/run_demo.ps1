# AutoLab 一键演示（Windows PowerShell）
# 用法：在仓库根目录执行  pwsh -File scripts\run_demo.ps1
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
Write-Host "== AutoLab 离线演示：失败 -> 归因 -> 重规划 -> 达标 ==" -ForegroundColor Cyan
python -m autolab demo --no-recipe
