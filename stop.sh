#!/usr/bin/env bash
# ./stop.sh [--all] [--purge] : stop everything TrustFuse started.
#   --all    also quit Docker Desktop (macOS) / print how to stop the Docker service (Linux)
#   --purge  also delete the TrustFuse images
source "$(dirname "$0")/scripts/unix/common.sh"
ALL=0; PURGE=0
for a in "$@"; do case "$a" in --all|-All) ALL=1;; --purge) PURGE=1;; *) fail "unknown option $a";; esac; done
banner "stop"
if command -v docker >/dev/null 2>&1 && docker_up; then
  step "Stopping TrustFuse containers"
  docker compose down --remove-orphans --volumes
  left=$(docker ps -aq --filter "label=com.docker.compose.project=$PROJECT")
  [ -z "$left" ] && ok "containers, network and runtime volume removed" || warn "still present: $left"
  if [ "$PURGE" = 1 ]; then docker image rm -f "${IMAGES[@]}" >/dev/null 2>&1 || true; ok "images removed"; fi
else
  ok "Docker is not running; no TrustFuse containers are active"
fi
if [ "$ALL" = 1 ]; then
  step "Stopping Docker"
  if [ "$(uname)" = "Darwin" ]; then osascript -e 'quit app "Docker"' >/dev/null 2>&1 || true; ok "Docker Desktop quit"
  else info "On Linux the Docker daemon is a system service: sudo systemctl stop docker"; fi
fi
echo
