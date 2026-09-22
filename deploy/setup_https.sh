#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Add HTTPS to the Fermi-LLM site using a free Let's Encrypt certificate.
#
# Prerequisite: deploy/setup_nginx.sh has already been run (nginx serving the
# app on port 80 for thanos.science.clemson.edu).
#
# Run as root:
#     sudo bash /opt/fermi-llm/deploy/setup_https.sh
#
# What it does:
#   * installs certbot + the nginx plugin
#   * obtains a cert for thanos.science.clemson.edu via the HTTP-01 challenge
#   * lets certbot edit the nginx config to serve TLS on :443 and redirect
#     http -> https
#   * certbot installs a systemd timer that auto-renews the cert
#
# Idempotent: re-running just renews/repairs as needed.
# ---------------------------------------------------------------------------
set -euo pipefail

DOMAIN="thanos.science.clemson.edu"
EMAIL="ai4helab@gmail.com"

if [[ $EUID -ne 0 ]]; then
    echo "Please run as root:  sudo bash $0" >&2
    exit 1
fi

echo "[1/3] Installing certbot + nginx plugin (if missing)..."
if ! command -v certbot >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y certbot python3-certbot-nginx
else
    echo "      certbot already installed: $(certbot --version 2>&1)"
fi

echo "[2/3] Obtaining + installing certificate for ${DOMAIN}..."
echo "      (HTTP-01 challenge: Let's Encrypt must be able to reach this host"
echo "       on port 80 from the public internet.)"
certbot --nginx \
    --non-interactive --agree-tos \
    -m "${EMAIL}" \
    -d "${DOMAIN}" \
    --redirect

echo "[3/3] Verifying renewal timer..."
systemctl list-timers 2>/dev/null | grep -i certbot || \
    systemctl status certbot.timer --no-pager 2>/dev/null | head -3 || true
certbot renew --dry-run || true

echo
echo "Done. The site should now be reachable at:"
echo "    https://${DOMAIN}/"
echo "and http:// will redirect to https://."
