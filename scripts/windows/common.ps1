# Shared helpers for the TrustFuse Windows scripts (Windows PowerShell 5.1 compatible, ASCII only).

$ErrorActionPreference = "Stop"
$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$script:Project = "trustfuse"
$script:Images = @("trustfuse-api:1.0", "trustfuse-web:1.0")

function Write-Banner([string]$Title) {
    Write-Host ""
    Write-Host "  TrustFuse  " -ForegroundColor White -BackgroundColor DarkGray -NoNewline
    Write-Host "  $Title" -ForegroundColor Gray
    Write-Host ""
}

function Step([string]$Text) { Write-Host "  > $Text" -ForegroundColor Cyan }
function Ok([string]$Text) { Write-Host "    [ok] $Text" -ForegroundColor Green }
function Warn([string]$Text) { Write-Host "    [!]  $Text" -ForegroundColor Yellow }
function Info([string]$Text) { Write-Host "         $Text" -ForegroundColor Gray }
function Fail([string]$Text) {
    Write-Host "    [x]  $Text" -ForegroundColor Red
    exit 1
}

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-DockerUp {
    $null = & docker info --format "{{.ServerVersion}}" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Start-DockerDesktop {
    $candidates = @(
        (Join-Path $Env:ProgramFiles "Docker\Docker\Docker Desktop.exe"),
        (Join-Path ${Env:ProgramFiles(x86)} "Docker\Docker\Docker Desktop.exe"),
        (Join-Path $Env:LOCALAPPDATA "Docker\Docker Desktop.exe")
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) {
            Info "starting Docker Desktop ..."
            Start-Process -FilePath $c | Out-Null
            return $true
        }
    }
    return $false
}

function Ensure-Docker {
    if (-not (Test-Command "docker")) {
        Fail "Docker is not installed. Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
    }
    if (-not (Test-DockerUp)) {
        if (-not (Start-DockerDesktop)) {
            Fail "Docker is installed but not running, and Docker Desktop was not found. Start Docker and retry."
        }
        $deadline = (Get-Date).AddSeconds(180)
        while (-not (Test-DockerUp)) {
            if ((Get-Date) -gt $deadline) { Fail "Docker did not become ready within 3 minutes." }
            Start-Sleep -Seconds 3
            Write-Host "." -NoNewline -ForegroundColor DarkGray
        }
        Write-Host ""
    }
    $null = & docker compose version 2>$null
    if ($LASTEXITCODE -ne 0) { Fail "Docker Compose v2 is required (bundled with current Docker Desktop)." }
    Ok ("Docker " + (& docker info --format "{{.ServerVersion}}") + " is running")
}

function Get-EnvValue([string]$Key, [string]$Default) {
    $envFile = Join-Path $script:RepoRoot ".env"
    if (Test-Path $envFile) {
        foreach ($line in Get-Content $envFile) {
            if ($line -match "^\s*$Key\s*=\s*(.+?)\s*$") { return $Matches[1] }
        }
    }
    return $Default
}

function Test-PortFree([int]$Port) {
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        $listener.Stop()
        return $true
    } catch {
        return $false
    }
}

function Invoke-Compose {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$ComposeArgs)
    Push-Location $script:RepoRoot
    try {
        # Out-Host keeps docker's output on screen and out of the function's return value
        & docker compose @ComposeArgs | Out-Host
        return $LASTEXITCODE
    } finally {
        Pop-Location
    }
}

function Test-ImagesPresent {
    foreach ($img in $script:Images) {
        $null = & docker image inspect $img 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
    }
    return $true
}

function Wait-Healthy([int]$Port, [int]$TimeoutSeconds = 150) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri "http://localhost:$Port/api/health"
            if ($r.StatusCode -eq 200) { Write-Host ""; return $true }
        } catch { }
        Start-Sleep -Seconds 2
        Write-Host "." -NoNewline -ForegroundColor DarkGray
    }
    Write-Host ""
    return $false
}
