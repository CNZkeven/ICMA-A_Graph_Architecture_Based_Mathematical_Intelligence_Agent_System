@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\deploy_demo.ps1" %*
set "ICMA_EXIT_CODE=%ERRORLEVEL%"
if not "%ICMA_EXIT_CODE%"=="0" (
    echo.
    echo ICMA Demo deployment failed. Review the error above.
    pause
)
exit /b %ICMA_EXIT_CODE%
