@echo off
rem TrustFuse: run (Windows). Runs scripts\windows\run.ps1 without changing the execution policy.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\run.ps1" %*
exit /b %ERRORLEVEL%
