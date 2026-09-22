#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Install + configure an nginx reverse proxy so the Fermi-LLM web app (running
# unprivileged on 127.0.0.1:8765) is reachable on port 80:
#
#     http://thanos.science.clemson.edu/
#
# Run as root:
#     sudo bash /opt/fermi-llm/deploy/setup_nginx.sh
#
# Idempotent: safe to re-run. Does not touch the app process itself.
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Please run as root:  sudo bash $0" >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PORT="${FERMI_LLM_PORT:-8765}"

echo "[1/5] Installing nginx (if missing)..."
if ! command -v nginx >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y nginx
else
    echo "      nginx already installed: $(nginx -v 2>&1)"
fi

echo "[2/5] Writing site config..."
# WebSocket upgrade map at http scope.
cat > /etc/nginx/conf.d/fermi-llm-upgrade.conf <<'EOF'
map $http_upgrade $fermi_connection_upgrade {
    default upgrade;
    ''      close;
}
EOF

# Server block (proxy to the app). We strip the embedded map from the repo
# copy since it now lives in conf.d above.
mkdir -p /etc/nginx/sites-available /etc/nginx/sites-enabled
cat > /etc/nginx/sites-available/fermi-llm <<EOF
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name thanos.science.clemson.edu _;

    client_max_body_size 50m;

    location / {
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_http_version 1.1;

        proxy_set_header Host              \$host;
        proxy_set_header X-Real-IP         \$remote_addr;
        proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        proxy_set_header Upgrade    \$http_upgrade;
        proxy_set_header Connection \$fermi_connection_upgrade;

        proxy_buffering    off;
        proxy_cache        off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
EOF

echo "[3/5] Enabling site, disabling default..."
ln -sf /etc/nginx/sites-available/fermi-llm /etc/nginx/sites-enabled/fermi-llm
rm -f /etc/nginx/sites-enabled/default

echo "[4/5] Testing config..."
nginx -t

echo "[5/5] Enabling + reloading nginx..."
systemctl enable nginx >/dev/null 2>&1 || true
systemctl restart nginx

echo
echo "Done. The app should now be reachable at:"
echo "    http://thanos.science.clemson.edu/   (proxied to 127.0.0.1:${APP_PORT})"
echo
echo "If it still does not load from off campus, a network/edge firewall may be"
echo "blocking inbound port 80 to this host (separate from this machine)."
