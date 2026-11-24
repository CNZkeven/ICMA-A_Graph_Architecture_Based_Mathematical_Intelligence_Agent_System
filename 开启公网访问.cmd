@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_tunnel.ps1" %*
set "ICMA_EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %ICMA_EXIT_CODE%
