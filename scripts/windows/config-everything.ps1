# .\config-everything : one-time setup after cloning (safe to re-run).
#   -NoCache   rebuild the Docker images from scratch
#   -FromRaw   download the original dataset archives and regenerate data/processed
#   -SkipSmoke skip the end-to-end self test
param([switch]$NoCache, [switch]$FromRaw, [switch]$SkipSmoke)

. "$PSScriptRoot\common.ps1"
Set-Location $script:RepoRoot
Write-Banner "configure everything"

Step "Checking prerequisites"
Ensure-Docker
$memBytes = [int64](& docker info --format "{{.MemTotal}}")
$memGb = [math]::Round($memBytes / 1GB, 2)
if ($memBytes -lt 1.2GB) {
    Warn "Docker has $memGb GB of memory. TrustFuse needs about 1.0 GB at peak; 1.5 GB is recommended."
} else {
    Ok "Docker memory: $memGb GB (TrustFuse uses ~0.4 GB idle, < 1 GB peak)"
}
$drive = (Get-Item $script:RepoRoot).PSDrive
if ($drive.Free -lt 4GB) { Warn ("Only " + [math]::Round($drive.Free / 1GB, 1) + " GB free on " + $drive.Name + ": (images need ~2.5 GB)") }
else { Ok ("Disk: " + [math]::Round($drive.Free / 1GB, 1) + " GB free") }

Step "Environment file"
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Ok "created .env from .env.example"
} else {
    Ok ".env present"
}
$port = [int](Get-EnvValue "WEB_PORT" "8080")
$running = (& docker ps --filter "label=com.docker.compose.project=$($script:Project)" -q)
if (-not $running -and -not (Test-PortFree $port)) {
    Warn "Port $port is in use by another program. Change WEB_PORT in .env (e.g. 8090)."
} else {
    Ok "web port $port"
}

Step "Datasets"
if ($FromRaw) {
    $raw = Join-Path $script:RepoRoot "data\raw"
    New-Item -ItemType Directory -Force -Path $raw | Out-Null
    $wesad = Join-Path $raw "WESAD.zip"
    if (-not (Test-Path $wesad)) {
        Info "downloading WESAD (2.25 GB) ..."
        & curl.exe -L -C - --retry 5 -o $wesad "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
        if ($LASTEXITCODE -ne 0) { Fail "WESAD download failed" }
    }
    $dalia = Join-Path $raw "data.zip"
    if (-not (Test-Path $dalia)) {
        $outer = Join-Path $raw "ppg_dalia.zip"
        Info "downloading PPG-DaLiA (2.9 GB) ..."
        & curl.exe -L -C - --retry 5 -o $outer "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
        if ($LASTEXITCODE -ne 0) { Fail "PPG-DaLiA download failed" }
        Expand-Archive -Path $outer -DestinationPath $raw -Force
        Remove-Item $outer
    }
}

$problems = @()
foreach ($ds in @("wesad", "dalia")) {
    $dir = Join-Path $script:RepoRoot "data\processed\$ds"
    $manifest = Join-Path $dir "manifest.json"
    if (-not (Test-Path $manifest)) { $problems += "missing $manifest"; continue }
    $m = Get-Content $manifest -Raw | ConvertFrom-Json
    foreach ($s in $m.subjects) {
        $f = Join-Path $dir $s.file
        if (-not (Test-Path $f)) { $problems += "missing $f"; continue }
        $h = (Get-FileHash -Algorithm SHA256 -Path $f).Hash.ToLower()
        if ($h -ne $s.sha256) { $problems += "checksum mismatch $f" }
    }
    if ($problems.Count -eq 0) { Ok ("{0}: {1} subjects, checksums verified" -f $m.dataset, $m.subjects.Count) }
}
if ($problems.Count -gt 0 -and -not $FromRaw) {
    $problems | Select-Object -First 5 | ForEach-Object { Warn $_ }
    Fail "Processed data is incomplete. Re-clone the repository, or run '.\config-everything -FromRaw'."
}

Step "Building Docker images (first run downloads ~1 GB of base layers and PyTorch)"
$buildArgs = @("build", "--pull")
if ($NoCache) { $buildArgs += "--no-cache" }
if ((Invoke-Compose $buildArgs) -ne 0) { Fail "docker compose build failed" }
Ok "images built: $($script:Images -join ', ')"

if ($FromRaw) {
    Step "Regenerating data/processed from the raw archives"
    $dataDir = Join-Path $script:RepoRoot "data"
    & docker run --rm -u 0 -v "${dataDir}:/app/data" trustfuse-api:1.0 python -m trustfuse prepare | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail "preprocessing failed" }
    Ok "processed data regenerated"
}

if (-not $SkipSmoke) {
    Step "End-to-end smoke test (data checksums, fold models, attacked inference, result sets)"
    $code = Invoke-Compose @("run", "--rm", "--no-deps", "api", "python", "-m", "trustfuse", "smoke")
    if ($code -ne 0) { Fail "smoke test failed (see output above)" }
    Ok "smoke test passed"
}

Write-Host ""
Write-Host "  Everything is configured. Start the application with:  " -NoNewline
Write-Host ".\run" -ForegroundColor White
Write-Host ""
