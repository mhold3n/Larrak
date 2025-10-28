# SSH Key Setup Script for Cursor Remote Development
# Run this script as Administrator

Write-Host "Setting up SSH Key Authentication..." -ForegroundColor Green

# Check if running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "This script requires Administrator privileges. Please run PowerShell as Administrator." -ForegroundColor Red
    exit 1
}

# 1. Create .ssh directory for the current user
$userSshPath = "$env:USERPROFILE\.ssh"
if (!(Test-Path $userSshPath)) {
    New-Item -ItemType Directory -Path $userSshPath -Force
    Write-Host "Created .ssh directory at: $userSshPath" -ForegroundColor Yellow
}

# 2. Set proper permissions on .ssh directory
icacls $userSshPath /inheritance:r /grant:r "$env:USERNAME`:F" | Out-Null
Write-Host "Set permissions on .ssh directory" -ForegroundColor Yellow

# 3. Generate SSH key pair if it doesn't exist
$privateKeyPath = "$userSshPath\id_rsa"
$publicKeyPath = "$userSshPath\id_rsa.pub"

if (!(Test-Path $privateKeyPath)) {
    Write-Host "Generating SSH key pair..." -ForegroundColor Yellow
    ssh-keygen -t rsa -b 4096 -f $privateKeyPath -N '""' -C "$env:USERNAME@$env:COMPUTERNAME"
    Write-Host "SSH key pair generated successfully!" -ForegroundColor Green
} else {
    Write-Host "SSH key pair already exists" -ForegroundColor Yellow
}

# 4. Add public key to authorized_keys
$authorizedKeysPath = "$userSshPath\authorized_keys"
if (Test-Path $publicKeyPath) {
    $publicKey = Get-Content $publicKeyPath
    if (!(Test-Path $authorizedKeysPath) -or !(Get-Content $authorizedKeysPath | Where-Object { $_ -eq $publicKey })) {
        Add-Content -Path $authorizedKeysPath -Value $publicKey
        Write-Host "Added public key to authorized_keys" -ForegroundColor Yellow
    } else {
        Write-Host "Public key already in authorized_keys" -ForegroundColor Yellow
    }
}

# 5. Set proper permissions on authorized_keys
if (Test-Path $authorizedKeysPath) {
    icacls $authorizedKeysPath /inheritance:r /grant:r "$env:USERNAME`:F" | Out-Null
    Write-Host "Set permissions on authorized_keys" -ForegroundColor Yellow
}

# 6. Display the public key for copying to remote machines
Write-Host "`nSSH Key Setup Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host "Your public key (copy this to remote machines):" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor White
if (Test-Path $publicKeyPath) {
    Get-Content $publicKeyPath | Write-Host -ForegroundColor White
}
Write-Host "----------------------------------------" -ForegroundColor White

Write-Host "`nTo use this key for remote access:" -ForegroundColor Cyan
Write-Host "1. Copy the public key above" -ForegroundColor White
Write-Host "2. Add it to the remote machine's ~/.ssh/authorized_keys file" -ForegroundColor White
Write-Host "3. Or use: ssh-copy-id -i $publicKeyPath user@remote-host" -ForegroundColor White

Write-Host "`nYour private key is stored at:" -ForegroundColor Cyan
Write-Host "  $privateKeyPath" -ForegroundColor White
Write-Host "`nKeep this private key secure and never share it!" -ForegroundColor Red



