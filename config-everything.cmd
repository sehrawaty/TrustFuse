@echo off
rem TrustFuse: config-everything (Windows). Runs scripts\windows\config-everything.ps1 without changing the execution policy.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\config-everything.ps1" %*
exit /b %ERRORLEVEL%
