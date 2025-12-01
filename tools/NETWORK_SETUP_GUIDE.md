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

## Making Server Accessible from Internet

**⚠️ SECURITY WARNING:** Exposing your server to the internet makes it accessible to anyone. The current server has **NO authentication**. Only do this if:
- You understand the security risks
- You plan to add authentication/authorization
- You're using this for testing/development only
- You have proper network security measures in place

### Prerequisites

1. **Public IP Address**: Your router must have a public IP (most home internet connections do)
2. **Router Access**: You need admin access to your router
3. **Static IP (Optional)**: If your ISP provides a dynamic IP, consider using Dynamic DNS

### Step 1: Find Your Public IP

**From Windows:**
```powershell
# Check your current public IP
curl https://api.ipify.org
# Or visit: https://whatismyipaddress.com/
```

This is the IP address that external devices will use to connect.

### Step 2: Configure Router Port Forwarding

**Important:** Router interfaces vary, but the general steps are:

1. **Access Router Admin Panel**
   - Usually: `http://192.168.1.1` or `http://192.168.0.1`
   - Check router manual for default IP and credentials

2. **Find Port Forwarding/Virtual Server Settings**
   - Look for: "Port Forwarding", "Virtual Server", "NAT", or "Firewall Rules"
   - Common locations: Advanced → NAT, Firewall → Port Forwarding

3. **Add Port Forwarding Rule**
   - **Service Name**: SAM2 Server (or any name)
   - **External Port**: 8080 (or your chosen port)
   - **Internal IP**: Your Windows network IP (e.g., `10.89.232.20`)
   - **Internal Port**: 8080
   - **Protocol**: TCP
   - **Enable**: Yes

4. **Save and Apply**

**Example Router Configuration:**
```
External Port: 8080
Internal IP: 10.89.232.20
Internal Port: 8080
Protocol: TCP
```

### Step 3: Configure Windows Firewall (Already Done)

If you followed the local network setup, Windows Firewall should already allow port 8080. Verify:

```powershell
Get-NetFirewallRule -DisplayName "SAM2*" | Format-Table DisplayName, Enabled, Direction, Action
```

### Step 4: Handle Dynamic IP (If Needed)

If your ISP assigns a dynamic IP that changes, use Dynamic DNS:

**Option A: Use a Dynamic DNS Service**
1. Sign up for a free service: No-IP, DuckDNS, or Dynu
2. Install their client on Windows
3. Get a domain like: `yourname.ddns.net`
4. Use this domain instead of IP address

**Option B: Use ngrok (Easier for Testing)**
```bash
# Install ngrok from https://ngrok.com/
# In WSL or Windows:
ngrok http 8080
# This gives you a public URL like: https://abc123.ngrok.io
```

### Step 5: Test Internet Access

**From Another Network (or use mobile data):**
```bash
# Replace with your public IP or domain
curl http://<YOUR_PUBLIC_IP>:8080/health

# Or if using ngrok:
curl https://abc123.ngrok.io/health
```

**From Mobile Device (on different network):**
```swift
// Use public IP or domain
let client = SAM2Client(baseURL: "http://<YOUR_PUBLIC_IP>:8080")
// Or with ngrok:
let client = SAM2Client(baseURL: "https://abc123.ngrok.io")
```

### Security Recommendations

**⚠️ CRITICAL: The current server has NO authentication!**

Before exposing to internet, consider:

1. **Add Authentication**
   - API keys or tokens
   - Basic HTTP authentication
   - OAuth/JWT tokens

2. **Use HTTPS**
   - Set up SSL/TLS certificate
   - Use reverse proxy (nginx, Caddy) with Let's Encrypt

3. **Rate Limiting**
   - Limit requests per IP
   - Prevent abuse

4. **Firewall Rules**
   - Only allow specific IPs if possible
   - Use fail2ban or similar

5. **Monitor Access**
   - Log all requests
   - Set up alerts for suspicious activity

### Quick Setup with ngrok (Easiest for Testing)

**For quick testing without router configuration:**

```bash
# 1. Download ngrok from https://ngrok.com/
# 2. Extract and add to PATH

# 3. Start your SAM2 server (in WSL)
python tools/sam_mobile_server.py --port 8080 ...

# 4. In another terminal, start ngrok (in WSL or Windows)
ngrok http 8080

# 5. You'll get output like:
# Forwarding  https://abc123.ngrok.io -> http://localhost:8080
# Use this URL in your mobile app
```

**Pros:**
- No router configuration needed
- Works behind NAT/firewall
- HTTPS included
- Quick setup

**Cons:**
- Free tier has limitations (connection limits, random URLs)
- URLs change on restart (unless paid plan)
- Not suitable for production

### Troubleshooting Internet Access

**Issue: Can't connect from internet**

1. **Check Router Port Forwarding**
   - Verify rule is enabled
   - Check internal IP is correct
   - Try different external port (some ISPs block common ports)

2. **Check ISP Restrictions**
   - Some ISPs block incoming connections
   - Some block port 8080
   - Try different port (e.g., 8443, 9000)

3. **Check Windows Firewall**
   ```powershell
   Get-NetFirewallRule -DisplayName "SAM2*"
   ```

4. **Test Locally First**
   ```powershell
   # Should work from Windows
   curl http://localhost:8080/health
   ```

5. **Use Port Checker**
   - Visit: https://www.yougetsignal.com/tools/open-ports/
   - Enter your public IP and port 8080
   - Should show "Open" if forwarding works

6. **Check Router Firewall**
   - Some routers have additional firewall rules
   - May need to allow WAN access

### Alternative: Cloud Deployment

For production use, consider deploying to:
- **AWS EC2 / Lightsail**
- **Google Cloud Platform**
- **Azure**
- **DigitalOcean**
- **Heroku** (with modifications)

These provide:
- Static IP addresses
- Better security
- Scalability
- Managed services

## Summary

### Local Network Access (Current Setup)
1. **WSL IP** (172.21.129.92) - Only accessible from Windows host
2. **Windows Network IP** (192.168.x.x) - Accessible from other devices on same network
3. **Port Forwarding** - Maps Windows IP:8080 → WSL IP:8080
4. **Firewall** - Must allow incoming connections on port 8080
5. **Mobile App** - Use Windows network IP, not WSL IP

### Internet Access (Additional Steps)
1. **Router Port Forwarding** - External Port 8080 → Windows IP:8080
2. **Public IP or Domain** - Use public IP or Dynamic DNS
3. **Security** - Add authentication and HTTPS (recommended)
4. **Mobile App** - Use public IP/domain instead of local IP

