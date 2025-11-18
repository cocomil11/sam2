# Network Setup Guide - Making SAM2 Server Accessible from Other Devices

## Problem

When running the SAM2 server in WSL2, the server is only accessible from the same computer. Other devices on your network cannot connect because WSL2 uses a virtual network adapter.

## Solution Options

### Option 1: Port Forwarding (Recommended for WSL2)

Set up port forwarding from Windows to WSL so external devices can access the server through the Windows host IP.

#### Step 1: Find Your Windows Host IP Address

**From Windows (PowerShell or Command Prompt):**
```powershell
# PowerShell
ipconfig | findstr "IPv4"

# Or more specifically:
ipconfig | findstr /C:"IPv4 Address"
```

**From WSL:**
```bash
# Get Windows host IP (from WSL perspective)
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
```

The Windows host IP will typically look like: `192.168.x.x` or `10.x.x.x`

#### Step 2: Find WSL IP Address

**From WSL:**
```bash
hostname -I
```

This gives you the WSL IP (e.g., `172.21.129.92`)

#### Step 3: Set Up Port Forwarding

**Method A: Using PowerShell (Run as Administrator)**

```powershell
# Replace <WSL_IP> with your WSL IP (e.g., 172.21.129.92)
# Replace <WINDOWS_IP> with your Windows host IP (e.g., 192.168.1.100)
# Replace <PORT> with your server port (default: 8080)

$wslIp = "172.21.129.92"
$port = 8080

# Remove existing rule if it exists
netsh interface portproxy delete v4tov4 listenport=$port listenaddress=0.0.0.0

# Add port forwarding rule
netsh interface portproxy add v4tov4 listenport=$port listenaddress=0.0.0.0 connectport=$port connectaddress=$wslIp

# Verify the rule
netsh interface portproxy show all
```

**Method B: Using netsh (Command Prompt as Administrator)**

```cmd
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=172.21.129.92
```

**Method C: Create a PowerShell Script**

Create `setup_port_forwarding.ps1`:

```powershell
# Run as Administrator
$wslIp = (wsl hostname -I).Trim().Split()[0]
$port = 8080

Write-Host "Setting up port forwarding..."
Write-Host "WSL IP: $wslIp"
Write-Host "Port: $port"

# Remove existing rule
netsh interface portproxy delete v4tov4 listenport=$port listenaddress=0.0.0.0 2>$null

# Add new rule
netsh interface portproxy add v4tov4 listenport=$port listenaddress=0.0.0.0 connectport=$port connectaddress=$wslIp

Write-Host "Port forwarding configured!"
netsh interface portproxy show all
```

Run it:
```powershell
# Right-click PowerShell and "Run as Administrator"
.\setup_port_forwarding.ps1
```

#### Step 4: Configure Windows Firewall

**Allow incoming connections on port 8080:**

```powershell
# PowerShell as Administrator
New-NetFirewallRule -DisplayName "SAM2 Server Port 8080" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow
```

Or use Windows Firewall GUI:
1. Open Windows Defender Firewall
2. Advanced Settings
3. Inbound Rules → New Rule
4. Port → TCP → Specific local ports: 8080
5. Allow the connection
6. Apply to all profiles
7. Name: "SAM2 Server Port 8080"

#### Step 5: Find Your Windows Host IP for External Access

**From Windows:**
```powershell
ipconfig
# Look for "IPv4 Address" under your active network adapter (Wi-Fi or Ethernet)
```

This IP (e.g., `192.168.1.100`) is what other devices should use.

#### Step 6: Test Connection

**From another device on the same network:**
```bash
# Replace with your Windows host IP
curl http://192.168.1.100:8080/health
```

**From your mobile device:**
- Use the Windows host IP (e.g., `http://192.168.1.100:8080`)
- Not the WSL IP (e.g., `http://172.21.129.92:8080`)

### Option 2: Use Windows Host IP Directly (WSL1 or Alternative)

If you're using WSL1 or have a different setup, you might be able to bind directly to the Windows host IP.

**Note:** This typically doesn't work with WSL2 due to network isolation.

### Option 3: Run Server on Windows (Not WSL)

If port forwarding is problematic, you can run the server directly on Windows:

1. Install Python on Windows
2. Install dependencies: `pip install flask flask-cors opencv-python numpy torch`
3. Run the server on Windows
4. Use Windows IP address directly

### Option 4: Use ngrok or Similar Tunneling Service

For testing or development across different networks:

```bash
# Install ngrok
# Download from https://ngrok.com/

# In WSL, expose port 8080
ngrok http 8080

# Use the ngrok URL (e.g., https://abc123.ngrok.io)
```

## Quick Setup Script

Create `setup_network_access.sh` in WSL:

```bash
#!/bin/bash

echo "=== SAM2 Server Network Setup ==="
echo ""

# Get WSL IP
WSL_IP=$(hostname -I | awk '{print $1}')
echo "WSL IP: $WSL_IP"

# Get Windows host IP (from WSL)
WIN_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
echo "Windows Host IP: $WIN_IP"
echo ""

echo "To make server accessible from other devices:"
echo "1. Run this PowerShell command as Administrator on Windows:"
echo ""
echo "   netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=$WSL_IP"
echo ""
echo "2. Allow port 8080 in Windows Firewall:"
echo ""
echo "   New-NetFirewallRule -DisplayName \"SAM2 Server\" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow"
echo ""
echo "3. Find your Windows network IP:"
echo "   (Run 'ipconfig' in Windows Command Prompt)"
echo ""
echo "4. Use the Windows network IP (not WSL IP) from other devices:"
echo "   http://<WINDOWS_NETWORK_IP>:8080"
echo ""
```

Run it:
```bash
chmod +x setup_network_access.sh
./setup_network_access.sh
```

## Verification Steps

### 1. Check Port Forwarding

**From Windows (PowerShell as Admin):**
```powershell
netsh interface portproxy show all
```

Should show:
```
Listen on ipv4:             Connect to ipv4:
Address         Port        Address         Port
--------------- ----------  --------------- ----------
0.0.0.0         8080        172.21.129.92    8080
```

### 2. Check Firewall Rule

**From Windows (PowerShell as Admin):**
```powershell
Get-NetFirewallRule -DisplayName "SAM2*" | Format-Table DisplayName, Enabled, Direction, Action
```

### 3. Test from Windows

```powershell
# Should work
curl http://localhost:8080/health
curl http://127.0.0.1:8080/health
```

### 4. Test from Another Device

**Find Windows network IP:**
```powershell
ipconfig | findstr "IPv4"
```

**From another device on same network:**
```bash
# Replace with your Windows network IP
curl http://192.168.1.100:8080/health
```

### 5. Test from Mobile Device

Use a network scanner app or check your router's connected devices list to find the Windows machine's IP, then test:

```
http://<WINDOWS_NETWORK_IP>:8080/health
```

## Troubleshooting

### Issue: Port forwarding doesn't work

**Solution:**
1. Make sure you ran PowerShell/Command Prompt as Administrator
2. Check if port 8080 is already in use: `netstat -ano | findstr :8080`
3. Try restarting the port forwarding rule
4. Check Windows Firewall logs

### Issue: Connection refused from other devices

**Solutions:**
1. Verify Windows Firewall allows port 8080
2. Check if your router/network allows device-to-device communication
3. Make sure both devices are on the same network (same Wi-Fi)
4. Try disabling Windows Firewall temporarily to test (re-enable after!)

### Issue: WSL IP changes after restart

**Solution:**
Create a script that updates port forwarding automatically:

```powershell
# update_port_forward.ps1
$wslIp = (wsl hostname -I).Trim().Split()[0]
netsh interface portproxy delete v4tov4 listenport=8080 listenaddress=0.0.0.0
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=$wslIp
```

### Issue: Server accessible from Windows but not from mobile

**Check:**
1. Mobile device and Windows PC are on the same Wi-Fi network
2. Using Windows network IP, not WSL IP
3. Windows Firewall allows the connection
4. Router doesn't have AP isolation enabled (some routers isolate devices)

## Updating Swift Integration Guide

After setting up port forwarding, update your Swift app to use the **Windows network IP** instead of the WSL IP:

```swift
// Use Windows network IP (e.g., 192.168.1.100)
// NOT WSL IP (e.g., 172.21.129.92)
let client = SAM2Client(baseURL: "http://192.168.1.100:8080")
```

## Persistent Port Forwarding

To make port forwarding persistent across reboots, you can:

1. **Create a startup script** that runs the port forwarding command
2. **Use Task Scheduler** to run the script on Windows startup
3. **Create a batch file** in Windows Startup folder

Create `C:\Users\<YourUser>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\sam2_port_forward.bat`:

```batch
@echo off
for /f "tokens=*" %%i in ('wsl hostname -I') do set WSL_IP=%%i
netsh interface portproxy delete v4tov4 listenport=8080 listenaddress=0.0.0.0
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=%WSL_IP%
```

## Summary

1. **WSL IP** (172.21.129.92) - Only accessible from Windows host
2. **Windows Network IP** (192.168.x.x) - Accessible from other devices on same network
3. **Port Forwarding** - Maps Windows IP:8080 → WSL IP:8080
4. **Firewall** - Must allow incoming connections on port 8080
5. **Mobile App** - Use Windows network IP, not WSL IP

