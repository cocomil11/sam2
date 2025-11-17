@echo off
REM Windows batch script to run the webcam streaming server
REM This allows Windows to access the script in WSL filesystem

REM Try to find Python in common locations
set PYTHON_CMD=python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found in PATH. Please ensure Python is installed on Windows.
    pause
    exit /b 1
)

REM Get the WSL path - adjust "Ubuntu" to your WSL distro name if different
set WSL_DISTRO=Ubuntu
set SCRIPT_PATH=\\wsl$\%WSL_DISTRO%\home\kenta\sam2\tools\webcam_stream_server.py

REM Check if file exists
if not exist "%SCRIPT_PATH%" (
    echo Script not found at: %SCRIPT_PATH%
    echo Please adjust WSL_DISTRO in this batch file if your distro name is different.
    pause
    exit /b 1
)

echo Starting webcam streaming server...
echo Make sure your webcam is connected and not in use by another application.
echo.
%PYTHON_CMD% "%SCRIPT_PATH%" --port 8080 --camera 0 %*

pause

