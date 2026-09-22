#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Fermi-LLM testbed bootstrap script.
#
#  Detects what is already installed on the host and installs anything that is
#  missing. Works on Linux (x86_64, aarch64) and macOS (Intel + Apple Silicon).
#
#  Run from the repository root:
#
#      ./scripts/setup.sh                # default: full install
#      ./scripts/setup.sh --no-local-gpu # skip torch/transformers
#      ./scripts/setup.sh --no-conda     # skip the conda env (you provide one)
#      ./scripts/setup.sh --help
#
#  Exit code 0 on success; non-zero on a fatal failure. Missing-but-fixable
#  components are reported and installed automatically.
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------- Defaults & CLI parsing ----------------------------------------
ENV_NAME="${FERMI_LLM_CONDA_ENV:-fermi-llm}"
SKIP_LOCAL_GPU=0
SKIP_CONDA=0
DRY_RUN=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<EOF
Fermi-LLM testbed setup

Usage: $0 [options]

Options:
  --env-name NAME       Conda environment name (default: ${ENV_NAME})
  --no-local-gpu        Skip torch / transformers / accelerate install
  --no-conda            Do not touch conda; assume FermiPy is already available
  --dry-run             Print what would be done, do not run
  --help                Show this help

Components installed (when missing):
  * Conda  (Miniforge by default, falls back to Miniconda)
  * The "fermi-llm" conda env with python 3.11
  * Fermi ScienceTools + FermiPy (from the "fermi" conda channel)
  * Python web-server dependencies (requirements.txt)
  * Optional: PyTorch + transformers (requirements-local.txt)

Configuration files you may want to create after setup:
  configs/gemini_keys.json        (cp configs/gemini_keys.json.example)
  configs/clemson_vllm_key.txt    (cp configs/clemson_vllm_key.txt.example)
  configs/env.sh                  (cp configs/env.sh.example)

EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name) ENV_NAME="$2"; shift 2;;
        --no-local-gpu) SKIP_LOCAL_GPU=1; shift;;
        --no-conda) SKIP_CONDA=1; shift;;
        --dry-run) DRY_RUN=1; shift;;
        --help|-h) usage; exit 0;;
        *) echo "Unknown option: $1"; usage; exit 2;;
    esac
done

# ---------- Logging helpers -----------------------------------------------
CYAN="$(printf '\033[36m')"; GREEN="$(printf '\033[32m')"
YELLOW="$(printf '\033[33m')"; RED="$(printf '\033[31m')"; OFF="$(printf '\033[0m')"
log()  { printf "%b\n" "${CYAN}[fermi-llm-setup]${OFF} $*"; }
ok()   { printf "%b\n" "${GREEN}[ok]${OFF}   $*"; }
warn() { printf "%b\n" "${YELLOW}[warn]${OFF} $*"; }
err()  { printf "%b\n" "${RED}[err]${OFF}  $*"; }

run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        printf "  (dry-run) %s\n" "$*"
    else
        eval "$@"
    fi
}

# ---------- Step 1: Detect OS and architecture ----------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"
log "Detected OS=${OS} ARCH=${ARCH}"

case "$OS" in
    Linux|Darwin) ;;
    *) err "Unsupported OS: $OS. This script supports Linux and macOS."; exit 1;;
esac

# ---------- Step 2: Ensure conda is available -----------------------------
have_conda() { command -v conda >/dev/null 2>&1; }

install_conda() {
    log "Conda not found. Installing Miniforge (community-maintained, conda-forge defaults)..."
    case "$OS" in
        Linux)
            local installer="Miniforge3-Linux-${ARCH}.sh"
            ;;
        Darwin)
            local installer="Miniforge3-MacOSX-${ARCH}.sh"
            ;;
    esac
    local url="https://github.com/conda-forge/miniforge/releases/latest/download/${installer}"
    local target="$HOME/miniforge3"
    log "Downloading ${url}..."
    run "curl -fsSL '${url}' -o '/tmp/${installer}'"
    log "Installing into ${target} (silent)..."
    run "bash '/tmp/${installer}' -b -p '${target}'"
    # Make conda available in the current shell for the rest of this script
    if [[ $DRY_RUN -eq 0 ]]; then
        # shellcheck source=/dev/null
        source "${target}/etc/profile.d/conda.sh"
    fi
    ok "Conda installed to ${target}"
    warn "To use conda in new shells, add this to your shell profile:"
    warn "    source ${target}/etc/profile.d/conda.sh"
}

if [[ $SKIP_CONDA -eq 0 ]]; then
    if have_conda; then
        ok "Conda found: $(conda --version 2>&1)"
        # Make sure 'conda activate' works in this script
        if [[ $DRY_RUN -eq 0 ]]; then
            CONDA_BASE="$(conda info --base 2>/dev/null)"
            # shellcheck source=/dev/null
            source "${CONDA_BASE}/etc/profile.d/conda.sh"
        fi
    else
        install_conda
    fi
fi

# ---------- Step 3: Create / update the fermi-llm conda env --------------
if [[ $SKIP_CONDA -eq 0 ]]; then
    if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        ok "Conda env '${ENV_NAME}' already exists; will update Python deps in it"
        log "Updating env from environment.yml..."
        run "conda env update -n '${ENV_NAME}' -f '${REPO_ROOT}/environment.yml'"
    else
        log "Creating conda env '${ENV_NAME}' from environment.yml..."
        run "conda env create -n '${ENV_NAME}' -f '${REPO_ROOT}/environment.yml'"
    fi
    ok "Conda env ready: ${ENV_NAME}"

    # Activate for the remaining pip installs
    if [[ $DRY_RUN -eq 0 ]]; then
        conda activate "${ENV_NAME}"
    fi
else
    warn "Skipping conda setup. Make sure FermiPy and the Fermi ScienceTools are already on PATH."
fi

# ---------- Step 4: Pip dependencies for the web server -------------------
log "Installing web-server Python dependencies (requirements.txt)..."
run "python -m pip install --upgrade pip"
run "python -m pip install -r '${REPO_ROOT}/requirements.txt'"
ok "Web-server dependencies installed"

# ---------- Step 5: Optional local-model deps -----------------------------
if [[ $SKIP_LOCAL_GPU -eq 0 ]]; then
    log "Installing optional local-GPU dependencies (requirements-local.txt)..."
    log "  (this pulls PyTorch with CUDA 12.4 wheels. Pass --no-local-gpu to skip.)"
    run "python -m pip install -r '${REPO_ROOT}/requirements-local.txt' || true"
    ok "Local-GPU dependencies attempted (failures here are non-fatal on CPU-only machines)"
else
    warn "--no-local-gpu set: skipping torch/transformers/accelerate"
fi

# ---------- Step 6: Verify FermiPy and the Science Tools ------------------
log "Verifying FermiPy install..."
if python -c "import fermipy; print('FermiPy version:', fermipy.__version__)" 2>/dev/null; then
    ok "FermiPy is importable"
else
    warn "FermiPy is not importable from the active environment."
    warn "If you used --no-conda, install FermiPy manually:"
    warn "  conda install -c fermi -c conda-forge fermitools fermipy"
fi

# Check for the binary tools FermiPy relies on
for tool in gtselect gtmktime gtltcube gtsrcmaps; do
    if command -v "${tool}" >/dev/null 2>&1; then
        ok "Found ${tool}"
    else
        warn "${tool} not found on PATH; gta.setup() will not run"
    fi
done

# ---------- Step 7: Bootstrap config skeletons ----------------------------
log "Bootstrapping configuration files..."
for src in configs/gemini_keys.json.example \
           configs/clemson_vllm_key.txt.example \
           configs/env.sh.example; do
    dest="${src%.example}"
    if [[ ! -f "${REPO_ROOT}/${dest}" ]]; then
        run "cp '${REPO_ROOT}/${src}' '${REPO_ROOT}/${dest}'"
        warn "Created ${dest} from template. Edit it before running the server."
    else
        ok "${dest} already exists (left untouched)"
    fi
done

# ---------- Step 8: Sanity-check the data files ---------------------------
log "Sanity-checking data files..."
for f in data/train.json data/test.json \
         lat_data/mrk421_PH.fits lat_data/mrk421_SC.fits \
         lat_data/vela_PH.fits   lat_data/vela_SC.fits \
         lat_data/crab_PH.fits   lat_data/crab_SC.fits; do
    if [[ -f "${REPO_ROOT}/${f}" ]]; then
        ok "${f} present"
    else
        err "${f} missing!"
    fi
done

# ---------- Done ----------------------------------------------------------
cat <<EOF

${GREEN}Setup complete.${OFF}

Next steps:
  1. Edit configs/gemini_keys.json with at least one real Gemini API key
     (free tier works for the demo).
  2. (Optional) edit configs/clemson_vllm_key.txt if you have a Clemson RCD
     vLLM bearer token.
  3. (Optional) edit configs/env.sh to override paths.
  4. Activate the environment and launch the web app:

        conda activate ${ENV_NAME}
        source configs/env.sh    # if you customised any paths
        ./scripts/run_webapp.sh

  5. Open http://localhost:8765 in your browser.

EOF
