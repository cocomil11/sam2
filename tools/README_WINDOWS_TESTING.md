# Testing SAM2 Mobile Server from Windows

This guide explains how to test the SAM2 Mobile Server from Windows.

## Prerequisites

1. **Server must be running in WSL:**
   ```bash
   # Inside WSL
   python tools/sam_mobile_server.py \
       --config sam2/configs/sam2.1/sam2.1_hiera_s.yaml \
       --checkpoint checkpoints/sam2.1_hiera_small.pt \
       --port 8080
   ```

2. **Python installed on Windows** (for running the test client)

## Quick Start

### Option 1: Using Batch Script (Easiest)

**In Windows CMD:**
```cmd
cd C:\path\to\sam2\tools
test_mobile_server_windows.bat C:\Users\YourName\Pictures\test.jpg 100 100 200 200
```

**In PowerShell (batch script also works):**
```powershell
cd C:\path\to\sam2\tools
.\test_mobile_server_windows.bat C:\Users\YourName\Pictures\test.jpg 100 100 200 200
```

**With multiple bounding boxes:**
```cmd
test_mobile_server_windows.bat C:\path\to\image.jpg 100 100 200 200 300 300 400 400
```

### Option 2: Using PowerShell Script (Recommended for PowerShell)

**In PowerShell:**
```powershell
cd C:\path\to\sam2\tools
.\test_mobile_server_windows.ps1 -ImagePath "C:\Users\YourName\Pictures\test.jpg" -Bbox 100,100,200,200
```

**With multiple bounding boxes:**
```powershell
.\test_mobile_server_windows.ps1 -ImagePath "C:\path\to\image.jpg" -Bbox 100,100,200,200,300,300,400,400
```

**Note:** In PowerShell, you need to use the `.\` prefix to run scripts. If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Option 3: Manual Python Command

1. **Get WSL IP address:**
   ```cmd
   wsl bash -c "ip route show default | grep -oP 'via \K\S+'"
   ```
   Or use the helper script:
   ```cmd
   wsl bash tools/get_wsl_ip.sh
   ```

2. **Run the test client:**
   ```cmd
   python tools\test_mobile_server.py --server-url http://<WSL_IP>:8080 --image C:\path\to\image.jpg --bbox 100 100 200 200
   ```

   Replace `<WSL_IP>` with the IP address from step 1.

## Script Details

### Batch Script (`test_mobile_server_windows.bat`)

**Usage:**
```cmd
test_mobile_server_windows.bat <image_path> [bbox1_x0] [bbox1_y0] [bbox1_x1] [bbox1_y1] [bbox2_x0] [bbox2_y0] [bbox2_x1] [bbox2_y1] ...
```

**Features:**
- Automatically detects WSL IP address
- Prompts for WSL IP if auto-detection fails
- Validates inputs
- Shows clear error messages

**Example:**
```cmd
test_mobile_server_windows.bat C:\Users\YourName\Pictures\test.jpg 100 100 200 200
```

### PowerShell Script (`test_mobile_server_windows.ps1`)

**Usage:**
```powershell
.\test_mobile_server_windows.ps1 -ImagePath "<path>" -Bbox <x0>,<y0>,<x1>,<y1> [<x0>,<y0>,<x1>,<y1> ...]
```

**Parameters:**
- `-ImagePath`: Path to the test image (required)
- `-Bbox`: Array of bounding box coordinates (required, must be multiple of 4)
- `-ServerPort`: Server port (optional, default: 8080)

**Features:**
- Automatically detects WSL IP address
- Colored output for better readability
- Better error handling
- Supports multiple bounding boxes

**Example:**
```powershell
.\test_mobile_server_windows.ps1 -ImagePath "C:\Users\YourName\Pictures\test.jpg" -Bbox 100,100,200,200,300,300,400,400
```

## Troubleshooting

### "Connection refused" or "Connection timeout"

**Most Common Issue:** The script is using the wrong IP address.

The IP `172.21.128.1` is the **Windows host IP** (from WSL's perspective), not the WSL IP itself. You need the **actual WSL VM IP address**.

**Solution:**

1. **Get the correct WSL IP:**
   ```cmd
   REM From Windows
   wsl hostname -I
   ```
   
   Or use the helper script:
   ```cmd
   get_wsl_ip_for_windows.bat
   ```

2. **Check what IP the server is listening on:**
   Look at the server logs in WSL. You should see:
   ```
   * Running on http://172.21.129.92:8080
   ```
   This is the IP you should use from Windows.

3. **Update the test script with the correct IP:**
   ```cmd
   python tools\test_mobile_server.py --server-url http://<CORRECT_WSL_IP>:8080 --image C:\path\to\image.jpg --bbox 100 100 200 200
   ```

**Alternative:** If automatic detection fails, the scripts will prompt you. Use the IP shown in the server logs (the one after "Running on http://...").

### "Could not automatically detect WSL IP address"

The script will prompt you to enter the WSL IP manually. You can also get it manually:

```cmd
wsl hostname -I
```

Or check in WSL:
```bash
# Inside WSL
ip addr show eth0 | grep "inet " | awk '{print $2}' | cut -d/ -f1
```

1. **Check if server is running in WSL:**
   ```bash
   # Inside WSL
   curl http://localhost:8080/health
   ```

2. **Check Windows Firewall:** Make sure port 8080 is not blocked

3. **Verify WSL IP:** The IP might have changed. Get it again:
   ```cmd
   wsl bash -c "ip route show default | grep -oP 'via \K\S+'"
   ```

### "Python is not found"

Make sure Python is installed on Windows and added to PATH:
```cmd
python --version
```

If this fails, install Python from [python.org](https://www.python.org/downloads/) and make sure to check "Add Python to PATH" during installation.

### "Image file not found"

- Use absolute paths: `C:\full\path\to\image.jpg`
- Use forward slashes: `C:/full/path/to/image.jpg`
- Escape spaces: `"C:\path with spaces\image.jpg"`

## Notes

- The server must be running **inside WSL** before testing
- The test client can run from **Windows** or **WSL**
- Bounding boxes are in format: `[x0, y0, x1, y1]` (top-left and bottom-right corners)
- Multiple bounding boxes are supported for tracking multiple objects

