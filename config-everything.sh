#!/usr/bin/env bash
# ./config-everything.sh [--no-cache] [--from-raw] [--skip-smoke] : one-time setup after cloning (safe to re-run).
source "$(dirname "$0")/scripts/unix/common.sh"
NO_CACHE=0; FROM_RAW=0; SKIP_SMOKE=0
for a in "$@"; do case "$a" in --no-cache) NO_CACHE=1;; --from-raw) FROM_RAW=1;; --skip-smoke) SKIP_SMOKE=1;; *) fail "unknown option $a";; esac; done
banner "configure everything"

step "Checking prerequisites"
ensure_docker
mem=$(docker info --format '{{.MemTotal}}')
if [ "$mem" -lt 1288490188 ]; then warn "Docker has $((mem / 1048576)) MB of memory; 1.5 GB is recommended."
else ok "Docker memory: $((mem / 1048576)) MB (TrustFuse uses ~0.4 GB idle, < 1 GB peak)"; fi

step "Environment file"
[ -f .env ] || { cp .env.example .env; ok "created .env from .env.example"; }
port=$(env_value WEB_PORT 8080)
if [ -z "$(docker ps -q --filter "label=com.docker.compose.project=$PROJECT")" ] && ! port_free "$port"; then
  warn "Port $port is in use. Change WEB_PORT in .env."
else ok "web port $port"; fi

step "Datasets"
if [ "$FROM_RAW" = 1 ]; then
  mkdir -p data/raw
  [ -f data/raw/WESAD.zip ] || curl -L -C - --retry 5 -o data/raw/WESAD.zip "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
  if [ ! -f data/raw/data.zip ]; then
    curl -L -C - --retry 5 -o data/raw/ppg_dalia.zip "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
    unzip -o -q data/raw/ppg_dalia.zip data.zip -d data/raw && rm data/raw/ppg_dalia.zip
  fi
fi
problems=0
for ds in wesad dalia; do
  man="data/processed/$ds/manifest.json"
  if [ ! -f "$man" ]; then warn "missing $man"; problems=1; continue; fi
  n=0
  # manifest.json is written by trustfuse (one "file"/"sha256" pair per subject, in order)
  while IFS=$'\t' read -r file sum; do
    n=$((n + 1))
    f="data/processed/$ds/$file"
    if [ ! -f "$f" ] || [ "$(sha256 "$f")" != "$sum" ]; then warn "bad or missing $f"; problems=1; fi
  done < <(paste <(grep -o '"file": *"[^"]*"' "$man" | sed 's/.*"\([^"]*\)"$/\1/') \
                 <(grep -o '"sha256": *"[^"]*"' "$man" | sed 's/.*"\([^"]*\)"$/\1/'))
  [ "$problems" = 0 ] && ok "$ds: $n subjects, checksums verified"
done
[ "$problems" = 0 ] || [ "$FROM_RAW" = 1 ] || fail "Processed data is incomplete. Re-clone, or run ./config-everything.sh --from-raw"

step "Building Docker images (first run downloads ~1 GB of base layers and PyTorch)"
args=(build --pull); [ "$NO_CACHE" = 1 ] && args+=(--no-cache)
docker compose "${args[@]}" || fail "docker compose build failed"
ok "images built: ${IMAGES[*]}"

if [ "$FROM_RAW" = 1 ]; then
  step "Regenerating data/processed from the raw archives"
  docker run --rm -u "$(id -u):$(id -g)" -v "$REPO_ROOT/data:/app/data" trustfuse-api:1.0 python -m trustfuse prepare || fail "preprocessing failed"
fi

if [ "$SKIP_SMOKE" = 0 ]; then
  step "End-to-end smoke test (data checksums, fold models, attacked inference, result sets)"
  docker compose run --rm --no-deps api python -m trustfuse smoke || fail "smoke test failed"
  ok "smoke test passed"
fi
printf '\n  Everything is configured. Start the application with:  ./run.sh\n\n'
