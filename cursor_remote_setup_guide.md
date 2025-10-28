# Cursor Remote Development Setup Guide

This guide will help you set up SSH access to your Windows development machine so you can use Cursor remotely.

## Prerequisites

- Windows 10/11 with OpenSSH Server
- Cursor installed on both local and remote machines
- Network access between the machines

## Step 1: Set Up SSH Server (Run as Administrator)

1. **Open PowerShell as Administrator**
   - Right-click on PowerShell and select "Run as Administrator"

2. **Run the SSH server setup script**
   ```powershell
   .\setup_ssh_server.ps1
   ```

3. **Run the SSH key setup script**
   ```powershell
   .\setup_ssh_keys.ps1
   ```

## Step 2: Find Your Computer's IP Address

After running the setup scripts, note your computer's IP address from the output. You'll need this to connect from remote machines.

## Step 3: Configure Cursor for Remote Development

### On the Remote Machine (where you want to connect FROM):

1. **Install the Remote-SSH extension**
   - Open Cursor
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Remote - SSH"
   - Install the extension by Microsoft

2. **Configure SSH connection**
   - Press `Ctrl+Shift+P` to open command palette
   - Type "Remote-SSH: Connect to Host"
   - Select "Configure SSH Hosts"
   - Choose your SSH config file (usually `~/.ssh/config`)

3. **Add your Windows machine to SSH config**
   ```
   Host windows-dev
       HostName [YOUR_WINDOWS_IP_ADDRESS]
       User [YOUR_WINDOWS_USERNAME]
       Port 22
       IdentityFile ~/.ssh/id_rsa
   ```

4. **Connect to your Windows machine**
   - Press `Ctrl+Shift+P`
   - Type "Remote-SSH: Connect to Host"
   - Select "windows-dev" (or your configured hostname)
   - Enter your Windows password when prompted

## Step 4: Open Your Project

Once connected:
1. Open the folder containing your Larrak project
2. Cursor will run on the remote machine but display on your local machine
3. All extensions and features will work as if you're working locally

## Security Recommendations

1. **Use SSH keys instead of passwords** (already set up by the scripts)
2. **Consider changing the default SSH port** (22) for additional security
3. **Use a VPN** if connecting over the internet
4. **Enable Windows Firewall** and only allow SSH traffic

## Troubleshooting

### Connection Issues
- Verify Windows Firewall allows SSH (port 22)
- Check that OpenSSH Server is running: `Get-Service sshd`
- Ensure your IP address hasn't changed

### Permission Issues
- Make sure you're running the setup scripts as Administrator
- Check that your user account has proper permissions

### SSH Key Issues
- Verify the public key is in `~/.ssh/authorized_keys` on Windows
- Check file permissions on SSH directories and files

## Useful Commands

### Check SSH service status
```powershell
Get-Service sshd
```

### Restart SSH service
```powershell
Restart-Service sshd
```

### View SSH logs
```powershell
Get-EventLog -LogName "OpenSSH/Operational" -Newest 10
```

### Test SSH connection locally
```powershell
ssh localhost
```

## Next Steps

1. Test the connection from another machine
2. Set up your development environment on the remote machine
3. Configure any additional tools or extensions you need
4. Consider setting up port forwarding for web development if needed

## Project-Specific Notes

Your Larrak project is located at:
`C:\Users\maxed\OneDrive\Desktop\Github Projects\Larrak`

Make sure to open this directory when connecting via Cursor Remote-SSH.



