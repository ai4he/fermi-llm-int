#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Add HTTPS to the Fermi-LLM site with a SELF-SIGNED certificate.
#
# Use this when the host is not reachable from the public internet on port 80
# (so Let's Encrypt HTTP-01 fails). Works over the campus network/VPN; browsers
# will show a one-time "not trusted" warning that users accept.
#
# Prerequisite: deploy/setup_nginx.sh has already been run.
#
# Run as root:
#     sudo bash /opt/fermi-llm/deploy/setup_https_selfsigned.sh
#
# Idempotent. To later switch to a trusted cert, just replace the files in
# /etc/ssl/fermi-llm and reload nginx (or re-run setup_https.sh once the
# firewall allows inbound 80).
# ---------------------------------------------------------------------------
set -euo pipefail

DOMAIN="thanos.science.clemson.edu"
APP_PORT="${FERMI_LLM_PORT:-8765}"
CERT_DIR="/etc/ssl/fermi-llm"
CRT="${CERT_DIR}/${DOMAIN}.crt"
KEY="${CERT_DIR}/${DOMAIN}.key"

if [[ $EUID -ne 0 ]]; then
    echo "Please run as root:  sudo bash $0" >&2
    exit 1
fi

echo "[1/4] Generating self-signed certificate for ${DOMAIN}..."
mkdir -p "${CERT_DIR}"
if [[ -f "${CRT}" && -f "${KEY}" ]]; then
    echo "      Existing cert found; regenerating (valid 825 days)."
fi
openssl req -x509 -nodes -newkey rsa:2048 \
    -keyout "${KEY}" -out "${CRT}" \
    -days 825 \
    -subj "/C=US/ST=SC/O=Clemson University/CN=${DOMAIN}" \
    -addext "subjectAltName=DNS:${DOMAIN}"
chmod 600 "${KEY}"

echo "[2/4] Writing nginx config (HTTP redirect + HTTPS proxy)..."
# WebSocket upgrade map (idempotent; also created by setup_nginx.sh).
cat > /etc/nginx/conf.d/fermi-llm-upgrade.conf <<'EOF'
map $http_upgrade $fermi_connection_upgrade {
    default upgrade;
    ''      close;
}
EOF

cat > /etc/nginx/sites-available/fermi-llm <<EOF
# Redirect all plain HTTP to HTTPS.
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name ${DOMAIN} _;
    return 301 https://\$host\$request_uri;
}

# HTTPS reverse proxy to the unprivileged app on 127.0.0.1:${APP_PORT}.
server {
    listen 443 ssl default_server;
    listen [::]:443 ssl default_server;
    server_name ${DOMAIN} _;

    ssl_certificate     ${CRT};
    ssl_certificate_key ${KEY};
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

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

ln -sf /etc/nginx/sites-available/fermi-llm /etc/nginx/sites-enabled/fermi-llm
rm -f /etc/nginx/sites-enabled/default

echo "[3/4] Testing nginx config..."
nginx -t

echo "[4/4] Reloading nginx..."
systemctl reload nginx || systemctl restart nginx

echo
echo "Done. The site is now served over HTTPS:"
echo "    https://${DOMAIN}/"
echo "Plain http:// redirects to https://."
echo
echo "NOTE: the certificate is self-signed, so browsers show a one-time"
echo "'your connection is not private' warning. Users click Advanced ->"
echo "Proceed. Replace ${CRT}/${KEY} with a trusted cert later to remove it."
