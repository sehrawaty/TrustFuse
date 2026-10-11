# .\run : fresh start of the whole stack (previous session state is discarded).
#   -Logs       follow the container logs after start-up
#   -NoBrowser  do not open the browser
#   -Keep       keep the previous session's user runs (do not wipe the runtime volume)
param([switch]$Logs, [switch]$NoBrowser, [switch]$Keep)

. "$PSScriptRoot\common.ps1"
Set-Location $script:RepoRoot
Write-Banner "run (fresh start)"

Step "Checking Docker"
Ensure-Docker
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
if (-not (Test-ImagesPresent)) {
    Warn "Docker images not found. Running .\config-everything first ..."
    & "$PSScriptRoot\config-everything.ps1" -SkipSmoke
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
$port = [int](Get-EnvValue "WEB_PORT" "8080")

Step "Stopping any previous TrustFuse session"
$downArgs = @("down", "--remove-orphans")
if (-not $Keep) { $downArgs += "--volumes" }
$null = Invoke-Compose @downArgs
if (-not $Keep) { Ok "previous containers, network and runtime volume removed" } else { Ok "previous containers removed (runtime kept)" }

if (-not (Test-PortFree $port)) { Fail "Port $port is in use by another program. Change WEB_PORT in .env." }

Step "Starting containers"
if ((Invoke-Compose up -d --force-recreate) -ne 0) { Fail "docker compose up failed" }

Step "Waiting for the API and web server to become healthy"
if (-not (Wait-Healthy $port 180)) {
    $null = Invoke-Compose ps
    $null = Invoke-Compose logs --tail 60
    Fail "TrustFuse did not become healthy. See the logs above."
}
$url = "http://localhost:$port"
Ok "TrustFuse is running at $url"
Info "API docs: $url/api/docs    stop with: .\stop"
if (-not $NoBrowser) { Start-Process $url }
if ($Logs) { $null = Invoke-Compose logs -f }
