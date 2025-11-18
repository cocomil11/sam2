@echo off
REM Test SAM2 Mobile Server from Windows
REM This script gets the WSL IP address and runs the test client

echo ========================================
echo SAM2 Mobile Server Test Client (Windows)
echo ========================================
echo.

REM Check if arguments are provided
if "%~1"=="" (
    echo Usage: test_mobile_server_windows.bat [--camera 0] [image_path] [bbox1_x0] [bbox1_y0] [bbox1_x1] [bbox1_y1] [--stream] [--fps 2.0] [--duration 10] [--max-frames 20]
    echo.
    echo Example (single frame from image):
    echo   test_mobile_server_windows.bat C:\path\to\image.jpg 100 100 200 200
    echo.
    echo Example (streaming from camera at 2fps):
    echo   test_mobile_server_windows.bat --camera 0 100 100 200 200 --stream --fps 2.0 --duration 10
    echo.
    echo Example (streaming from image at 2fps, max 20 frames):
    echo   test_mobile_server_windows.bat C:\path\to\image.jpg 100 100 200 200 --stream --fps 2.0 --max-frames 20
    echo.
    exit /b 1
)

set IMAGE_PATH=
set CAMERA_ID=-1
set FIRST_ARG=%~1

REM Check if first argument is --camera
if "%FIRST_ARG%"=="--camera" (
    set CAMERA_ID=%~2
    shift
    shift
) else (
    REM Assume first argument is image path
    set IMAGE_PATH=%~1
    shift
)

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

REM Build bbox arguments and check for streaming flags
set BB_ARGS=
set STREAM_ARGS=
set STREAM_MODE=0
set FPS=2.0
set DURATION=
set MAX_FRAMES=
:parse_args
if "%~1"=="" goto :done_parse
if "%~1"=="--stream" (
    set STREAM_MODE=1
    set STREAM_ARGS=%STREAM_ARGS% --stream
    shift
    goto :parse_args
)
if "%~1"=="--fps" (
    set FPS=%~2
    set STREAM_ARGS=%STREAM_ARGS% --fps %~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--duration" (
    set DURATION=%~2
    set STREAM_ARGS=%STREAM_ARGS% --duration %~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--max-frames" (
    set MAX_FRAMES=%~2
    set STREAM_ARGS=%STREAM_ARGS% --max-frames %~2
    shift
    shift
    goto :parse_args
)
REM Assume it's a bbox coordinate
set BB_ARGS=%BB_ARGS% --bbox %~1 %~2 %~3 %~4
shift
shift
shift
shift
goto :parse_args
:done_parse

if "%BB_ARGS%"=="" (
    echo ERROR: At least one bounding box is required.
    echo Usage: test_mobile_server_windows.bat ^<image_path^> x0 y0 x1 y1 [x0 y0 x1 y1 ...] [--stream] [--fps 2.0] [--duration 10] [--max-frames 20]
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

REM Check if image exists (if using image file)
if "%CAMERA_ID%"=="-1" (
    if not exist "%IMAGE_PATH%" (
        echo ERROR: Image file not found: %IMAGE_PATH%
        exit /b 1
    )
)

echo Running test client...
echo Server URL: http://%WSL_IP%:8080
if "%CAMERA_ID%"=="-1" (
    echo Image: %IMAGE_PATH%
) else (
    echo Camera: %CAMERA_ID%
)
if "%STREAM_MODE%"=="1" (
    echo Mode: Streaming at %FPS% FPS
    if not "%DURATION%"=="" echo Duration: %DURATION% seconds
    if not "%MAX_FRAMES%"=="" echo Max frames: %MAX_FRAMES%
) else (
    echo Mode: Single frame test
)
echo.

REM Build command arguments
set CMD_ARGS=--server-url http://%WSL_IP%:8080
if "%CAMERA_ID%"=="-1" (
    set CMD_ARGS=%CMD_ARGS% --image "%IMAGE_PATH%"
) else (
    set CMD_ARGS=%CMD_ARGS% --camera %CAMERA_ID%
)

python "%TEST_SCRIPT%" %CMD_ARGS% %BB_ARGS% %STREAM_ARGS%

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

