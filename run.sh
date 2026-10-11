#!/usr/bin/env bash
# ./run.sh [--logs] [--no-browser] [--keep] : fresh start of the whole stack.
source "$(dirname "$0")/scripts/unix/common.sh"
LOGS=0; BROWSER=1; KEEP=0
for a in "$@"; do case "$a" in --logs) LOGS=1;; --no-browser) BROWSER=0;; --keep) KEEP=1;; *) fail "unknown option $a";; esac; done
banner "run (fresh start)"
step "Checking Docker"
ensure_docker
[ -f .env ] || cp .env.example .env
if ! images_present; then warn "Docker images not found; running config-everything first"; "$REPO_ROOT/config-everything.sh" --skip-smoke; fi
port=$(env_value WEB_PORT 8080)

step "Stopping any previous TrustFuse session"
if [ "$KEEP" = 1 ]; then docker compose down --remove-orphans; ok "previous containers removed (runtime kept)"
else docker compose down --remove-orphans --volumes; ok "previous containers, network and runtime volume removed"; fi
port_free "$port" || fail "Port $port is in use. Change WEB_PORT in .env."

step "Starting containers"
docker compose up -d --force-recreate || fail "docker compose up failed"
step "Waiting for the API and web server to become healthy"
if ! wait_healthy "$port" 180; then docker compose ps; docker compose logs --tail 60; fail "TrustFuse did not become healthy"; fi
ok "TrustFuse is running at http://localhost:$port"
info "API docs: http://localhost:$port/api/docs    stop with: ./stop.sh"
if [ "$BROWSER" = 1 ]; then (command -v open >/dev/null && open "http://localhost:$port") || (command -v xdg-open >/dev/null && xdg-open "http://localhost:$port" >/dev/null 2>&1) || true; fi
[ "$LOGS" = 1 ] && docker compose logs -f
exit 0
