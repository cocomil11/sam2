@echo off
REM Get WSL IP address for connecting from Windows
REM This shows the actual WSL VM IP address (not the Windows host IP)

echo Getting WSL IP address...
echo.

REM Method 1: Use hostname -I (gives WSL VM's IP address)
for /f "tokens=1" %%i in ('wsl hostname -I') do set WSL_IP=%%i

if not "%WSL_IP%"=="" (
    echo WSL IP address: %WSL_IP%
    echo.
    echo Use this IP to connect from Windows:
    echo   http://%WSL_IP%:8080
    echo.
    exit /b 0
)

REM Fallback method
for /f "tokens=*" %%i in ('wsl bash -c "ip addr show eth0 ^| grep '\''inet '\'' ^| awk '\''{print $2}'\'' ^| cut -d/ -f1"') do set WSL_IP=%%i

if not "%WSL_IP%"=="" (
    echo WSL IP address: %WSL_IP%
    echo.
    echo Use this IP to connect from Windows:
    echo   http://%WSL_IP%:8080
    echo.
    exit /b 0
)

echo ERROR: Could not detect WSL IP address.
echo.
echo Try running this command manually in WSL:
echo   hostname -I
echo.
exit /b 1

