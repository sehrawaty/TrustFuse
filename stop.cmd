@echo off
rem TrustFuse: stop (Windows). Runs scripts\windows\stop.ps1 without changing the execution policy.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\stop.ps1" %*
exit /b %ERRORLEVEL%
