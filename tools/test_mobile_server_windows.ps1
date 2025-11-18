# Test SAM2 Mobile Server from Windows (PowerShell)
# This script gets the WSL IP address and runs the test client

param(
    [string]$ImagePath,
    
    [Parameter(Mandatory=$true)]
    [float[]]$Bbox,
    
    [string]$ServerPort = "8080",
    
    [int]$Camera = -1,
    
    [switch]$Stream,
    
    [float]$Fps = 2.0,
    
    [float]$Duration = 0,
    
    [int]$MaxFrames = 0
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SAM2 Mobile Server Test Client (Windows)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Validate input source
if ($Camera -lt 0 -and [string]::IsNullOrWhiteSpace($ImagePath)) {
    Write-Host "ERROR: Either -ImagePath or -Camera must be specified" -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage: .\test_mobile_server_windows.ps1 -ImagePath 'C:\path\to\image.jpg' -Bbox 100,100,200,200" -ForegroundColor Yellow
    Write-Host "       .\test_mobile_server_windows.ps1 -Camera 0 -Bbox 100,100,200,200 -Stream -Fps 2.0" -ForegroundColor Yellow
    exit 1
}

# Validate bbox arguments (must be multiple of 4)
if ($Bbox.Count -eq 0 -or ($Bbox.Count % 4) -ne 0) {
    Write-Host "ERROR: Bounding boxes must be provided in groups of 4 (x0, y0, x1, y1)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage: .\test_mobile_server_windows.ps1 -ImagePath 'C:\path\to\image.jpg' -Bbox 100,100,200,200" -ForegroundColor Yellow
    Write-Host "       .\test_mobile_server_windows.ps1 -Camera 0 -Bbox 100,100,200,200 -Stream -Fps 2.0" -ForegroundColor Yellow
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

# Check if image exists (if using image file)
if ($Camera -lt 0 -and -not (Test-Path $ImagePath)) {
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

# Build streaming arguments if enabled
$streamArgs = @()
if ($Stream) {
    $streamArgs += "--stream"
    $streamArgs += "--fps"
    $streamArgs += $Fps
    if ($Duration -gt 0) {
        $streamArgs += "--duration"
        $streamArgs += $Duration
    }
    if ($MaxFrames -gt 0) {
        $streamArgs += "--max-frames"
        $streamArgs += $MaxFrames
    }
}

Write-Host "Running test client..." -ForegroundColor Yellow
Write-Host "Server URL: http://${WslIp}:${ServerPort}" -ForegroundColor Cyan
if ($Camera -ge 0) {
    Write-Host "Source: Camera $Camera" -ForegroundColor Cyan
} else {
    Write-Host "Image: $ImagePath" -ForegroundColor Cyan
}
if ($Stream) {
    Write-Host "Mode: Streaming at $Fps FPS" -ForegroundColor Cyan
    if ($Duration -gt 0) {
        Write-Host "Duration: $Duration seconds" -ForegroundColor Cyan
    }
    if ($MaxFrames -gt 0) {
        Write-Host "Max frames: $MaxFrames" -ForegroundColor Cyan
    }
} else {
    Write-Host "Mode: Single frame test" -ForegroundColor Cyan
}
Write-Host ""

# Run the test script
$serverUrl = "http://${WslIp}:${ServerPort}"
$allArgs = @(
    $TestScript,
    "--server-url", $serverUrl
)

# Add image or camera argument
if ($Camera -ge 0) {
    $allArgs += "--camera"
    $allArgs += $Camera
} else {
    $allArgs += "--image"
    $allArgs += $ImagePath
}

$allArgs = $allArgs + $bboxArgs + $streamArgs

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

