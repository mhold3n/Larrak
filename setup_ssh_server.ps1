# SSH Server Setup Script for Cursor Remote Development
# Run this script as Administrator

Write-Host "Setting up SSH Server for Cursor Remote Development..." -ForegroundColor Green

# Check if running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "This script requires Administrator privileges. Please run PowerShell as Administrator." -ForegroundColor Red
    exit 1
}

# 1. Install OpenSSH Server if not already installed
Write-Host "Installing OpenSSH Server..." -ForegroundColor Yellow
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# 2. Set SSH service to start automatically
Write-Host "Configuring SSH service..." -ForegroundColor Yellow
Set-Service -Name sshd -StartupType 'Automatic'

# 3. Start the SSH service
Write-Host "Starting SSH service..." -ForegroundColor Yellow
Start-Service sshd

# 4. Configure Windows Firewall
Write-Host "Configuring Windows Firewall..." -ForegroundColor Yellow
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22

# 5. Create SSH configuration directory if it doesn't exist
$sshConfigPath = "$env:ProgramData\ssh"
if (!(Test-Path $sshConfigPath)) {
    New-Item -ItemType Directory -Path $sshConfigPath -Force
}

# 6. Configure SSH server settings
Write-Host "Configuring SSH server settings..." -ForegroundColor Yellow
$sshdConfig = @"
# SSH Server Configuration for Cursor Remote Development
Port 22
Protocol 2
HostKey $env:ProgramData\ssh\ssh_host_rsa_key
HostKey $env:ProgramData\ssh\ssh_host_ecdsa_key
HostKey $env:ProgramData\ssh\ssh_host_ed25519_key

# Authentication
PubkeyAuthentication yes
PasswordAuthentication yes
PermitEmptyPasswords no
ChallengeResponseAuthentication no

# Security settings
MaxAuthTries 3
MaxSessions 10
ClientAliveInterval 60
ClientAliveCountMax 3

# Logging
SyslogFacility LOCAL0
LogLevel INFO

# Allow specific users (add your username here)
AllowUsers $env:USERNAME

# Disable root login
PermitRootLogin no

# Disable X11 forwarding
X11Forwarding no

# Disable agent forwarding
AllowAgentForwarding no

# Disable TCP forwarding
AllowTcpForwarding no

# Disable user environment
PermitUserEnvironment no
"@

$sshdConfig | Out-File -FilePath "$sshConfigPath\sshd_config" -Encoding UTF8

# 7. Restart SSH service to apply configuration
Write-Host "Restarting SSH service..." -ForegroundColor Yellow
Restart-Service sshd

# 8. Check service status
Write-Host "Checking SSH service status..." -ForegroundColor Yellow
Get-Service sshd

# 9. Display connection information
Write-Host "`nSSH Server Setup Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host "Your computer's IP address(es):" -ForegroundColor Cyan
Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*"} | ForEach-Object {
    Write-Host "  - $($_.IPAddress)" -ForegroundColor White
}
Write-Host "`nSSH Connection Details:" -ForegroundColor Cyan
Write-Host "  Host: [Your IP Address]" -ForegroundColor White
Write-Host "  Port: 22" -ForegroundColor White
Write-Host "  Username: $env:USERNAME" -ForegroundColor White
Write-Host "`nTo connect from another computer:" -ForegroundColor Cyan
Write-Host "  ssh $env:USERNAME@[Your IP Address]" -ForegroundColor White
Write-Host "`nFor Cursor Remote Development:" -ForegroundColor Cyan
Write-Host "  1. Install Cursor on the remote computer" -ForegroundColor White
Write-Host "  2. Use 'Remote-SSH' extension" -ForegroundColor White
Write-Host "  3. Connect to: $env:USERNAME@[Your IP Address]" -ForegroundColor White

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Set up SSH key authentication (recommended)" -ForegroundColor White
Write-Host "2. Test the connection from another computer" -ForegroundColor White
Write-Host "3. Configure Cursor for remote development" -ForegroundColor White





