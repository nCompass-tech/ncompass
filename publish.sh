#!/usr/bin/env bash
set -e
set -u

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# ==============================================================================
# nCompass PyPI Publishing Script
# ==============================================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

PACKAGE_NAME="ncompass"
SKIP_TESTS="${SKIP_TESTS:-false}"
SKIP_CHECKS="${SKIP_CHECKS:-false}"
DRY_RUN="${DRY_RUN:-false}"

# ==============================================================================
# Helpers
# ==============================================================================

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

check_command() {
    if ! command -v "$1" &>/dev/null; then
        log_error "Required command '$1' not found. Please install it first."
        exit 1
    fi
}

print_help() {
    cat <<EOF
Usage: $(basename "$0") [--test | --prod] [options]

Modes (required):
  --test                Upload to TestPyPI
  --prod                Upload to PyPI

Options via env vars:
  PYPI_TOKEN            PyPI API token (prod). Must start with 'pypi-'
  PYPI_USERNAME         PyPI username (legacy prod auth)
  PYPI_PASSWORD         PyPI password (legacy prod auth)
  TESTPYPI_TOKEN        TestPyPI API token (test). Must start with 'pypi-'
  SKIP_TESTS=true       Skip pytest
  SKIP_CHECKS=true      Skip coverage/pyright checks
  DRY_RUN=true          Build and check, but do not upload

Examples:
  SKIP_TESTS=true $0 --test
  PYPI_TOKEN=... $0 --prod

EOF
}

MODE=""      # 'test' or 'prod'
PYPI_REPO_NAME=""
PYPI_SIMPLE_URL=""
PYPI_PROJECT_URL=""
validate_env() {
    if [[ "$MODE" == "test" ]]; then
        PYPI_REPO_NAME="testpypi"
        PYPI_SIMPLE_URL="https://test.pypi.org/simple/"
        PYPI_PROJECT_URL="https://test.pypi.org/project/${PACKAGE_NAME}/"
        if [[ "${DRY_RUN}" != "true" ]]; then
            if [[ -z "${TESTPYPI_TOKEN:-}" ]]; then
                log_error "TESTPYPI_TOKEN is required for --test"
                exit 1
            fi
            if [[ "${TESTPYPI_TOKEN}" != pypi-* ]]; then
                log_error "TESTPYPI_TOKEN must start with 'pypi-'"
                exit 1
            fi
        fi
    elif [[ "$MODE" == "prod" ]]; then
        PYPI_REPO_NAME="pypi"
        PYPI_SIMPLE_URL="https://pypi.org/simple/"
        PYPI_PROJECT_URL="https://pypi.org/project/${PACKAGE_NAME}/"
        if [[ "${DRY_RUN}" != "true" ]]; then
            if [[ -n "${PYPI_TOKEN:-}" ]]; then
                if [[ "${PYPI_TOKEN}" != pypi-* ]]; then
                    log_error "PYPI_TOKEN must start with 'pypi-'"
                    exit 1
                fi
            elif [[ -n "${PYPI_USERNAME:-}" && -n "${PYPI_PASSWORD:-}" ]]; then
                : # ok
            else
                log_error "Provide PYPI_TOKEN or PYPI_USERNAME/PYPI_PASSWORD for --prod"
                exit 1
            fi
        fi
    else
        log_error "Internal: MODE not set"
        exit 1
    fi
}

get_package_version() {
    python - <<'PY'
import sys
try:
    import tomllib
except Exception:
    print("tomllib not available; require Python 3.11+ or install tomli for CPython<3.11", file=sys.stderr)
    sys.exit(2)
with open("pyproject.toml","rb") as f:
    data = tomllib.load(f)
print(data["project"]["version"])
PY
}

# ==============================================================================
# Arg parsing
# ==============================================================================

if [[ $# -eq 0 ]]; then
    print_help; exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test) MODE="test"; shift ;;
        --prod) MODE="prod"; shift ;;
        -h|--help) print_help; exit 0 ;;
        *)
            log_error "Unknown argument: $1"
            print_help
            exit 1
            ;;
    esac
done

# ==============================================================================
# Pre-flight
# ==============================================================================

log_info "Starting publish workflow for ${PACKAGE_NAME} in mode: ${MODE}"

if [[ ! -f "pyproject.toml" ]]; then
    log_error "pyproject.toml not found. Run from project root."
    exit 1
fi

check_command python
check_command git

PYTHON_VERSION=$(python --version | awk '{print $2}')
log_info "Using Python ${PYTHON_VERSION}"

PACKAGE_VERSION=$(get_package_version)
log_info "Package version: ${PACKAGE_VERSION}"

validate_env
echo ""

# ==============================================================================
# Git status
# ==============================================================================

log_info "Checking git status..."
if [[ -n $(git status --porcelain) ]]; then
    log_warning "Working directory has uncommitted changes"
    read -p "Continue anyway? (y/N) " -n 1 -r; echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted by user"; exit 1
    fi
else
    log_success "Working directory is clean"
fi

if git rev-parse "v${PACKAGE_VERSION}" >/dev/null 2>&1; then
    log_warning "Git tag v${PACKAGE_VERSION} already exists"
    read -p "Continue anyway? (y/N) " -n 1 -r; echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted by user"; exit 1
    fi
fi
echo ""

# ==============================================================================
# Tooling
# ==============================================================================

log_info "Installing/upgrading build tools..."
python -m pip install --upgrade pip build twine setuptools wheel > /dev/null
log_success "Build tools ready"
echo ""

# ==============================================================================
# Tests
# ==============================================================================

if [[ "${SKIP_TESTS}" == "true" ]]; then
    log_warning "Skipping tests (SKIP_TESTS=true)"
else
    log_info "Running test suite..."
    python -m pip install -e ".[dev]" > /dev/null 2>&1 || true
    if python -m pytest tests/ --cov=ncompass --cov-fail-under=80 -q; then
        log_success "All tests passed"
    else
        log_error "Tests failed"; exit 1
    fi
fi
echo ""

# ==============================================================================
# Quality
# ==============================================================================

if [[ "${SKIP_CHECKS}" == "true" ]]; then
    log_warning "Skipping code quality checks (SKIP_CHECKS=true)"
else
    log_info "Running code quality checks..."
    if command -v make &>/dev/null; then
        log_info "  - type-stats"; (cd tools && make type-stats) || { log_error "make type-stats failed"; exit 1; }
        log_info "  - docstring-coverage"; (cd tools && make docstring-coverage) || { log_error "make docstring-coverage failed"; exit 1; }
    else
        log_error "make command not found"; exit 1;
    fi
    log_success "Code quality checks passed"
fi
echo ""

# ==============================================================================
# Build
# ==============================================================================

log_info "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info ncompass.egg-info
log_success "Clean complete"
echo ""

log_info "Building package..."
python -m build
if [[ ! -d "dist" ]] || [[ -z "$(ls -A dist)" ]]; then
    log_error "Build failed - no distribution files created"; exit 1
fi
log_success "Package built"
log_info "Artifacts:"
ls -lh dist/
echo ""

log_info "Checking distribution with twine..."
python -m twine check dist/* || { log_error "twine check failed"; exit 1; }
log_success "Distribution check passed"
echo ""

# ==============================================================================
# Upload
# ==============================================================================

if [[ "${DRY_RUN}" == "true" ]]; then
    log_success "DRY RUN complete. Skipping upload."
    log_info "Artifacts in dist/"; exit 0
fi

log_info "Uploading to ${MODE^^} (${PYPI_REPO_NAME})..."

if [[ "$MODE" == "test" ]]; then
    python -m twine upload \
        --repository "${PYPI_REPO_NAME}" \
        --username __token__ \
        --password "${TESTPYPI_TOKEN}" \
        dist/*
else
    if [[ -n "${PYPI_TOKEN:-}" ]]; then
        python -m twine upload \
            --username __token__ \
            --password "${PYPI_TOKEN}" \
            dist/*
    else
        python -m twine upload \
            --username "${PYPI_USERNAME}" \
            --password "${PYPI_PASSWORD}" \
            dist/*
    fi
fi

log_success "Upload complete"
echo ""

# ==============================================================================
# Summary
# ==============================================================================

log_success "Release complete"
echo ""
echo "=================================================="
echo "  Package: ${PACKAGE_NAME}"
echo "  Version: ${PACKAGE_VERSION}"
echo "  Index:   ${PYPI_SIMPLE_URL}"
echo "  Project: ${PYPI_PROJECT_URL}"
echo "=================================================="
echo ""
if [[ "$MODE" == "test" ]]; then
    log_info "Test install: pip install --index-url ${PYPI_SIMPLE_URL} ${PACKAGE_NAME}"
else
    log_info "Install with: pip install ${PACKAGE_NAME}"
fi
