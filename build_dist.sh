#!/usr/bin/env bash
# ==============================================================================
# Build & Packaging Script for Gemma 4 RAG TUI
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

APP_NAME="gemma-tui"
ENTRYPOINT="rag_app.py"
DIST_DIR="dist"
BUILD_DIR="build"
OUTPUT_DIR="${DIST_DIR}/${APP_NAME}"
TARBALL="${DIST_DIR}/${APP_NAME}-linux.tar.gz"

echo "=================================================="
echo "  Packaging ${APP_NAME} with PyInstaller"
echo "=================================================="

# ------------------------------------------------------------------------------
# 1. Resolve Python environment & Ensure PyInstaller
# ------------------------------------------------------------------------------
echo "==> Step 1/4: Checking Python environment & PyInstaller..."

if [ -n "${VIRTUAL_ENV:-}" ] && [ -f "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
    PIP_BIN="${VIRTUAL_ENV}/bin/pip"
elif [ -f "${HOME}/venv/bin/python" ]; then
    PYTHON_BIN="${HOME}/venv/bin/python"
    PIP_BIN="${HOME}/venv/bin/pip"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    PIP_BIN="$(command -v pip || command -v pip3 || echo "pip")"
else
    echo "ERROR: No Python interpreter found." >&2
    exit 1
fi

echo "    Using Python: ${PYTHON_BIN}"

if ! "${PYTHON_BIN}" -m PyInstaller --version >/dev/null 2>&1; then
    echo "    PyInstaller not found. Installing into active environment..."
    "${PIP_BIN}" install --quiet pyinstaller
else
    echo "    PyInstaller is already installed."
fi

PYINSTALLER_CMD="${PYTHON_BIN} -m PyInstaller"

# ------------------------------------------------------------------------------
# 2. Run PyInstaller build
# ------------------------------------------------------------------------------
echo "==> Step 2/4: Building standalone binary with PyInstaller..."

# Clean previous build artifacts if present
rm -rf "${OUTPUT_DIR}" "${TARBALL}"

${PYINSTALLER_CMD} \
    --name "${APP_NAME}" \
    --onedir \
    --noconfirm \
    --clean \
    --copy-metadata chromadb \
    --copy-metadata tqdm \
    --copy-metadata regex \
    --copy-metadata requests \
    --copy-metadata filelock \
    --copy-metadata textual \
    --collect-data chromadb \
    --collect-data textual \
    --collect-data langchain_community \
    --collect-submodules chromadb \
    --hidden-import "dotenv" \
    --hidden-import "langchain_classic" \
    --hidden-import "langchain_community" \
    --hidden-import "langchain_openai" \
    --hidden-import "pypdf" \
    --hidden-import "docx2txt" \
    --hidden-import "sentence_transformers" \
    "${ENTRYPOINT}"

# ------------------------------------------------------------------------------
# 3. Set up output directory structure & default configuration
# ------------------------------------------------------------------------------
echo "==> Step 3/4: Setting up distribution directory..."

# Create user documents folder
mkdir -p "${OUTPUT_DIR}/docs"

# Create minimal .env.example
cat << 'EOF' > "${OUTPUT_DIR}/.env.example"
# ==============================================================================
# Gemma 4 RAG TUI - Environment Configuration
# ==============================================================================
# Rename this file to .env or enter the key on first launch in the TUI.
# Get a free API key at: https://aistudio.google.com/apikey

GOOGLE_API_KEY=your_google_ai_studio_api_key_here
EOF

# ------------------------------------------------------------------------------
# 4. Create portable distribution tarball
# ------------------------------------------------------------------------------
echo "==> Step 4/4: Creating release tarball..."
tar -czf "${TARBALL}" -C "${DIST_DIR}" "${APP_NAME}"

echo ""
echo "=================================================="
echo "  Build Complete!"
echo "=================================================="
echo "  Binary Folder : ${OUTPUT_DIR}/"
echo "  Release Archive: ${TARBALL}"
echo "=================================================="
echo ""
echo "--- END-USER LAUNCH INSTRUCTIONS ---"
echo ""
echo "1. Extract the tarball (if distributing archive):"
echo "   tar -xzf ${APP_NAME}-linux.tar.gz"
echo "   cd ${APP_NAME}"
echo ""
echo "2. Add your documents:"
echo "   Copy your PDFs, CSVs, or text files into the 'docs/' directory:"
echo "   cp /path/to/my_data.pdf docs/"
echo ""
echo "3. (Optional) Configure API Key:"
echo "   cp .env.example .env"
echo "   # Edit .env and paste your Google AI Studio API key"
echo "   # (Or simply launch the app; it will prompt you securely on first run)"
echo ""
echo "4. Launch the application:"
echo "   ./${APP_NAME}"
echo ""
