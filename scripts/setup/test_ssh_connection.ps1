# SSH Connection Test Script
# Run this to test your SSH setup

Write-Host "Testing SSH Server Configuration..." -ForegroundColor Green

# 1. Check if SSH service is running
Write-Host "`n1. Checking SSH Service Status:" -ForegroundColor Yellow
$sshService = Get-Service sshd -ErrorAction SilentlyContinue
if ($sshService) {
    Write-Host "   Status: $($sshService.Status)" -ForegroundColor $(if ($sshService.Status -eq 'Running') { 'Green' } else { 'Red' })
    Write-Host "   Startup Type: $($sshService.StartType)" -ForegroundColor White
} else {
    Write-Host "   SSH Service not found!" -ForegroundColor Red
}

# 2. Check if SSH port is listening
Write-Host "`n2. Checking SSH Port (22):" -ForegroundColor Yellow
$sshPort = Get-NetTCPConnection -LocalPort 22 -ErrorAction SilentlyContinue
if ($sshPort) {
    Write-Host "   Port 22 is listening" -ForegroundColor Green
    Write-Host "   State: $($sshPort.State)" -ForegroundColor White
} else {
    Write-Host "   Port 22 is not listening" -ForegroundColor Red
}

# 3. Check Windows Firewall rules
Write-Host "`n3. Checking Windows Firewall:" -ForegroundColor Yellow
$firewallRule = Get-NetFirewallRule -DisplayName "OpenSSH Server (sshd)" -ErrorAction SilentlyContinue
if ($firewallRule) {
    Write-Host "   SSH Firewall rule found" -ForegroundColor Green
    Write-Host "   Enabled: $($firewallRule.Enabled)" -ForegroundColor White
} else {
    Write-Host "   SSH Firewall rule not found" -ForegroundColor Red
}

# 4. Check SSH configuration
Write-Host "`n4. Checking SSH Configuration:" -ForegroundColor Yellow
$sshConfigPath = "$env:ProgramData\ssh\sshd_config"
if (Test-Path $sshConfigPath) {
    Write-Host "   SSH config file exists" -ForegroundColor Green
    $configContent = Get-Content $sshConfigPath
    $portLine = $configContent | Where-Object { $_ -match "^Port\s+" }
    $passwordAuthLine = $configContent | Where-Object { $_ -match "^PasswordAuthentication\s+" }
    $pubkeyAuthLine = $configContent | Where-Object { $_ -match "^PubkeyAuthentication\s+" }
    
    if ($portLine) { Write-Host "   $portLine" -ForegroundColor White }
    if ($passwordAuthLine) { Write-Host "   $passwordAuthLine" -ForegroundColor White }
    if ($pubkeyAuthLine) { Write-Host "   $pubkeyAuthLine" -ForegroundColor White }
} else {
    Write-Host "   SSH config file not found" -ForegroundColor Red
}

# 5. Check SSH keys
Write-Host "`n5. Checking SSH Keys:" -ForegroundColor Yellow
$userSshPath = "$env:USERPROFILE\.ssh"
$privateKeyPath = "$userSshPath\id_rsa"
$publicKeyPath = "$userSshPath\id_rsa.pub"
$authorizedKeysPath = "$userSshPath\authorized_keys"

if (Test-Path $privateKeyPath) {
    Write-Host "   Private key exists" -ForegroundColor Green
} else {
    Write-Host "   Private key not found" -ForegroundColor Red
}

if (Test-Path $publicKeyPath) {
    Write-Host "   Public key exists" -ForegroundColor Green
} else {
    Write-Host "   Public key not found" -ForegroundColor Red
}

if (Test-Path $authorizedKeysPath) {
    Write-Host "   Authorized keys file exists" -ForegroundColor Green
} else {
    Write-Host "   Authorized keys file not found" -ForegroundColor Red
}

# 6. Display connection information
Write-Host "`n6. Connection Information:" -ForegroundColor Yellow
$ipAddresses = Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*"}
if ($ipAddresses) {
    Write-Host "   Your computer's IP addresses:" -ForegroundColor Cyan
    foreach ($ip in $ipAddresses) {
        Write-Host "     - $($ip.IPAddress)" -ForegroundColor White
    }
} else {
    Write-Host "   No valid IP addresses found" -ForegroundColor Red
}

Write-Host "`n   Username: $env:USERNAME" -ForegroundColor White
Write-Host "   Port: 22" -ForegroundColor White

# 7. Test local SSH connection
Write-Host "`n7. Testing Local SSH Connection:" -ForegroundColor Yellow
try {
    $testResult = ssh -o ConnectTimeout=5 -o BatchMode=yes localhost "echo 'SSH test successful'" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   Local SSH connection successful" -ForegroundColor Green
    } else {
        Write-Host "   Local SSH connection failed: $testResult" -ForegroundColor Red
    }
} catch {
    Write-Host "   Local SSH connection failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`nTest Complete!" -ForegroundColor Green
Write-Host "If all checks pass, you should be able to connect from remote machines." -ForegroundColor Cyan










