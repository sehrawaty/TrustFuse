# .\stop : stop everything TrustFuse started.
#   -All    also quit Docker Desktop and shut down WSL (frees all Docker memory)
#   -Purge  also delete the TrustFuse Docker images (re-create them with .\config-everything)
param([switch]$All, [switch]$Purge)

. "$PSScriptRoot\common.ps1"
Set-Location $script:RepoRoot
Write-Banner "stop"

if ((Test-Command "docker") -and (Test-DockerUp)) {
    Step "Stopping TrustFuse containers"
    $null = Invoke-Compose down --remove-orphans --volumes
    $left = & docker ps -a --filter "label=com.docker.compose.project=$($script:Project)" -q
    if ($left) { Warn "some containers are still present: $left" } else { Ok "containers, network and runtime volume removed" }
    if ($Purge) {
        Step "Removing TrustFuse images"
        foreach ($img in $script:Images) { & docker image rm -f $img 2>$null | Out-Null }
        Ok "images removed"
    }
} else {
    Ok "Docker is not running; no TrustFuse containers are active"
}

if ($All) {
    Step "Quitting Docker Desktop and shutting down WSL"
    foreach ($name in @("Docker Desktop", "com.docker.backend", "com.docker.build")) {
        Get-Process -Name $name -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    }
    if (Test-Command "wsl") { & wsl --shutdown 2>$null }
    Ok "Docker Desktop stopped and WSL shut down"
}
Write-Host ""
