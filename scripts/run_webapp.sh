#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Launch the Fermi-LLM web application.
#
#  Assumes the conda env (default name: fermi-llm) has already been created by
#  scripts/setup.sh and that the configs/ directory contains at least one
#  populated API key file.
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Source the optional environment-variable overrides
if [[ -f configs/env.sh ]]; then
    # shellcheck source=/dev/null
    source configs/env.sh
fi

# Defaults
HOST="${FERMI_LLM_HOST:-0.0.0.0}"
PORT="${FERMI_LLM_PORT:-8765}"
ENV_NAME="${FERMI_LLM_CONDA_ENV:-fermi-llm}"

# Make sure the conda env is active (or at least available)
if ! command -v python >/dev/null 2>&1; then
    echo "[run_webapp] python not found on PATH; activate the conda env first:"
    echo "    conda activate ${ENV_NAME}"
    exit 1
fi

# The platform lives in src/fermi_llm (an installed checkout works too).
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export FERMI_LLM_PROJECT_DIR="${FERMI_LLM_PROJECT_DIR:-${REPO_ROOT}}"

echo "[run_webapp] FERMI_LLM_PROJECT_DIR=${FERMI_LLM_PROJECT_DIR}"
echo "[run_webapp] Listening on ${HOST}:${PORT}"
echo "[run_webapp] Open http://localhost:${PORT}/ in your browser"
echo

exec env HOST="${HOST}" PORT="${PORT}" python -m fermi_llm.app "$@"
