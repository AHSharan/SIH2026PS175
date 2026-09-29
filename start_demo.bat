@echo off
rem DepthWizard demo: double-click this file.
rem Starts the local server with live mode and opens the viewer in the browser.
rem Works without internet once the DINOv3 weights have been downloaded once.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python is not installed or not on PATH. & pause & exit /b 1)
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8777/?assets=assets_live"
python serve.py --live
pause
