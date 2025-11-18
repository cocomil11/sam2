# Test SAM2 Mobile Server from Windows (PowerShell)
# This script gets the WSL IP address and runs the test client

param(
    [Parameter(Mandatory=$true)]
    [string]$ImagePath,
    
    [Parameter(Mandatory=$true)]
    [float[]]$Bbox,
    
    [string]$ServerPort = "8080"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SAM2 Mobile Server Test Client (Windows)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Validate bbox arguments (must be multiple of 4)
if ($Bbox.Count -eq 0 -or ($Bbox.Count % 4) -ne 0) {
    Write-Host "ERROR: Bounding boxes must be provided in groups of 4 (x0, y0, x1, y1)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage: .\test_mobile_server_windows.ps1 -ImagePath 'C:\path\to\image.jpg' -Bbox 100,100,200,200" -ForegroundColor Yellow
    Write-Host "       .\test_mobile_server_windows.ps1 -ImagePath 'C:\path\to\image.jpg' -Bbox 100,100,200,200,300,300,400,400" -ForegroundColor Yellow
    exit 1
}

# Get WSL IP address (the actual WSL VM IP, not Windows host IP)
Write-Host "Getting WSL IP address..." -ForegroundColor Yellow
try {
    # Method 1: Use hostname -I (gives WSL VM's IP address)
    $wslIpOutput = wsl hostname -I
    $WslIp = ($wslIpOutput -split '\s+')[0].Trim()
    
    if ([string]::IsNullOrWhiteSpace($WslIp)) {
        # Method 2: Use ip addr show (fallback)
        $wslIpOutput = wsl bash -c "ip addr show eth0 | grep 'inet ' | awk '{print `$2}' | cut -d/ -f1"
        $WslIp = $wslIpOutput.Trim()
    }
    
    if ([string]::IsNullOrWhiteSpace($WslIp)) {
        # Method 3: Try ip route (but this gives Windows host IP, so we'll skip it)
        throw "Could not detect WSL IP"
    }
    
    # Validate IP format
    $ipRegex = [regex]'^(\d{1,3}\.){3}\d{1,3}$'
    if (-not ($ipRegex.Match($WslIp).Success)) {
        throw "Invalid IP format detected"
    }
} catch {
    Write-Host ""
    Write-Host "ERROR: Could not automatically detect WSL IP address." -ForegroundColor Red
    Write-Host ""
    Write-Host "To get the WSL IP manually, run in WSL:" -ForegroundColor Yellow
    Write-Host "  hostname -I" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Or from Windows:" -ForegroundColor Yellow
    Write-Host "  wsl hostname -I" -ForegroundColor Cyan
    Write-Host ""
    $WslIp = Read-Host "Please enter the WSL IP address manually"
    
    if ([string]::IsNullOrWhiteSpace($WslIp)) {
        Write-Host "ERROR: WSL IP address is required." -ForegroundColor Red
        exit 1
    }
}

Write-Host "WSL IP address: $WslIp" -ForegroundColor Green
Write-Host ""

# Check if image exists
if (-not (Test-Path $ImagePath)) {
    Write-Host "ERROR: Image file not found: $ImagePath" -ForegroundColor Red
    exit 1
}

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TestScript = Join-Path $ScriptDir "test_mobile_server.py"

if (-not (Test-Path $TestScript)) {
    Write-Host "ERROR: Test script not found at $TestScript" -ForegroundColor Red
    exit 1
}

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not found in PATH." -ForegroundColor Red
    Write-Host "Please make sure Python is installed and added to PATH." -ForegroundColor Yellow
    exit 1
}

# Build bbox arguments
$bboxArgs = @()
for ($i = 0; $i -lt $Bbox.Count; $i += 4) {
    $bboxArgs += "--bbox"
    $bboxArgs += $Bbox[$i]
    $bboxArgs += $Bbox[$i + 1]
    $bboxArgs += $Bbox[$i + 2]
    $bboxArgs += $Bbox[$i + 3]
}

Write-Host "Running test client..." -ForegroundColor Yellow
Write-Host "Server URL: http://${WslIp}:${ServerPort}" -ForegroundColor Cyan
Write-Host "Image: $ImagePath" -ForegroundColor Cyan
Write-Host ""

# Run the test script
$serverUrl = "http://${WslIp}:${ServerPort}"
$allArgs = @(
    $TestScript,
    "--server-url", $serverUrl,
    "--image", $ImagePath
) + $bboxArgs

try {
    python $allArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Test failed. Make sure:" -ForegroundColor Red
        Write-Host "  1. The server is running in WSL: python tools/sam_mobile_server.py --port $ServerPort" -ForegroundColor Yellow
        Write-Host "  2. The WSL IP address is correct: $WslIp" -ForegroundColor Yellow
        Write-Host "  3. The image path is correct: $ImagePath" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host ""
    Write-Host "Test completed successfully!" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "ERROR: Failed to run test script: $_" -ForegroundColor Red
    exit 1
}

