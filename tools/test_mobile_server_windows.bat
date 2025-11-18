@echo off
REM Test SAM2 Mobile Server from Windows
REM This script gets the WSL IP address and runs the test client

echo ========================================
echo SAM2 Mobile Server Test Client (Windows)
echo ========================================
echo.

REM Check if image path is provided
if "%~1"=="" (
    echo Usage: test_mobile_server_windows.bat ^<image_path^> [bbox1_x0] [bbox1_y0] [bbox1_x1] [bbox1_y1] [bbox2_x0] [bbox2_y0] [bbox2_x1] [bbox2_y1] ...
    echo.
    echo Example:
    echo   test_mobile_server_windows.bat C:\path\to\image.jpg 100 100 200 200
    echo   test_mobile_server_windows.bat C:\path\to\image.jpg 100 100 200 200 300 300 400 400
    echo.
    exit /b 1
)

set IMAGE_PATH=%~1
shift

REM Get WSL IP address (the actual WSL VM IP, not Windows host IP)
echo Getting WSL IP address...
REM Method 1: Use hostname -I (gives WSL VM's IP address)
for /f "tokens=1" %%i in ('wsl hostname -I') do set WSL_IP=%%i

REM If that didn't work, try alternative method
if "%WSL_IP%"=="" (
    for /f "tokens=*" %%i in ('wsl bash -c "ip addr show eth0 ^| grep '\''inet '\'' ^| awk '\''{print $2}'\'' ^| cut -d/ -f1"') do set WSL_IP=%%i
)

if "%WSL_IP%"=="" (
    echo.
    echo ERROR: Could not automatically detect WSL IP address.
    echo.
    echo To get the WSL IP manually, run in WSL:
    echo   hostname -I
    echo.
    echo Or from Windows:
    echo   wsl hostname -I
    echo.
    set /p WSL_IP="Please enter the WSL IP address manually: "
    if "%WSL_IP%"=="" (
        echo ERROR: WSL IP address is required.
        exit /b 1
    )
)

echo WSL IP address: %WSL_IP%
echo.

REM Build bbox arguments
set BB_ARGS=
:parse_bbox
if "%~1"=="" goto :done_parse
set BB_ARGS=%BB_ARGS% --bbox %~1 %~2 %~3 %~4
shift
shift
shift
shift
goto :parse_bbox
:done_parse

if "%BB_ARGS%"=="" (
    echo ERROR: At least one bounding box is required.
    echo Usage: test_mobile_server_windows.bat ^<image_path^> x0 y0 x1 y1 [x0 y0 x1 y1 ...]
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not found in PATH.
    echo Please make sure Python is installed and added to PATH.
    exit /b 1
)

REM Get the script directory (assuming we're in the sam2 root)
set SCRIPT_DIR=%~dp0
set TEST_SCRIPT=%SCRIPT_DIR%test_mobile_server.py

if not exist "%TEST_SCRIPT%" (
    echo ERROR: Test script not found at %TEST_SCRIPT%
    exit /b 1
)

if not exist "%IMAGE_PATH%" (
    echo ERROR: Image file not found: %IMAGE_PATH%
    exit /b 1
)

echo Running test client...
echo Server URL: http://%WSL_IP%:8080
echo Image: %IMAGE_PATH%
echo.

python "%TEST_SCRIPT%" --server-url http://%WSL_IP%:8080 --image "%IMAGE_PATH%" %BB_ARGS%

if errorlevel 1 (
    echo.
    echo ERROR: Test failed. Make sure:
    echo   1. The server is running in WSL: python tools/sam_mobile_server.py --port 8080
    echo   2. The WSL IP address is correct: %WSL_IP%
    echo   3. The image path is correct: %IMAGE_PATH%
    exit /b 1
)

echo.
echo Test completed successfully!

