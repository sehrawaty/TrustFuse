#!/usr/bin/env bash
# Shared helpers for the TrustFuse macOS / Linux scripts.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT=trustfuse
IMAGES=(trustfuse-api:1.0 trustfuse-web:1.0)
cd "$REPO_ROOT"

if [ -t 1 ]; then C=$'\033[36m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; D=$'\033[90m'; N=$'\033[0m'; else C= G= Y= R= D= N=; fi
banner() { printf '\n  \033[7m TrustFuse \033[0m  %s\n\n' "$1"; }
step() { printf '  %s> %s%s\n' "$C" "$1" "$N"; }
ok() { printf '    %s[ok]%s %s\n' "$G" "$N" "$1"; }
warn() { printf '    %s[!]%s  %s\n' "$Y" "$N" "$1"; }
info() { printf '         %s%s%s\n' "$D" "$1" "$N"; }
fail() { printf '    %s[x]%s  %s\n' "$R" "$N" "$1"; exit 1; }

docker_up() { docker info --format '{{.ServerVersion}}' >/dev/null 2>&1; }

ensure_docker() {
  command -v docker >/dev/null 2>&1 || fail "Docker is not installed (https://docs.docker.com/get-docker/)."
  if ! docker_up; then
    if [ "$(uname)" = "Darwin" ]; then info "starting Docker Desktop ..."; open -a Docker || true
    else fail "Docker is not running. Start it (e.g. 'sudo systemctl start docker') and retry."; fi
    for _ in $(seq 1 60); do docker_up && break; sleep 3; printf '.'; done; echo
    docker_up || fail "Docker did not become ready within 3 minutes."
  fi
  docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required."
  ok "Docker $(docker info --format '{{.ServerVersion}}') is running"
}

env_value() { # key default
  local v=""
  [ -f .env ] && v="$(grep -E "^\s*$1\s*=" .env | tail -n1 | cut -d= -f2- | tr -d '[:space:]')" || true
  echo "${v:-$2}"
}

port_free() { ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

images_present() { for i in "${IMAGES[@]}"; do docker image inspect "$i" >/dev/null 2>&1 || return 1; done; }

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

wait_healthy() { # port timeout
  local deadline=$(( $(date +%s) + ${2:-150} ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    curl -fsS -m 3 "http://localhost:$1/api/health" >/dev/null 2>&1 && { echo; return 0; }
    sleep 2; printf '.'
  done
  echo; return 1
}
