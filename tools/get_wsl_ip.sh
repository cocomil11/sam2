#!/bin/bash
# Get WSL IP address for accessing from Windows

# Method 1: Read from /etc/resolv.conf (most reliable in WSL2)
if [ -f /etc/resolv.conf ]; then
    WSL_IP=$(grep nameserver /etc/resolv.conf | awk '{print $2}' | head -1)
    if [ -n "$WSL_IP" ]; then
        echo "$WSL_IP"
        exit 0
    fi
fi

# Method 2: Use ip route (fallback)
WSL_IP=$(ip route show default | grep -oP 'via \K\S+' | head -1)
if [ -n "$WSL_IP" ]; then
    echo "$WSL_IP"
    exit 0
fi

# Method 3: Use hostname -I
WSL_IP=$(hostname -I | awk '{print $1}')
if [ -n "$WSL_IP" ]; then
    echo "$WSL_IP"
    exit 0
fi

echo "Could not determine WSL IP address"
exit 1

