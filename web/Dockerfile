# TrustFuse web UI: Vite build -> static files served by nginx (which also proxies /api).
FROM node:22-alpine AS build
WORKDIR /web
COPY web/package.json web/package-lock.json ./
# Optional build secret "extra_ca" (see backend/Dockerfile)
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/extra_ca; fi \
 && npm ci --no-audit --no-fund --loglevel=error
COPY web/ ./
# keep the build inside the Docker memory budget
ENV NODE_OPTIONS=--max-old-space-size=768
RUN npx vite build

FROM nginx:1.27-alpine
COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=10s --timeout=3s --retries=6 CMD wget -qO- http://127.0.0.1/healthz >/dev/null || exit 1
