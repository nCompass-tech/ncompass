#!/bin/bash
# Quick Coverage Commands Reference for nCompass
# Source: tailored from ChatGPT recommendations
# Location: tools/coverage-commands.sh

echo "========================================="
echo "nCompass Coverage Tools Quick Reference"
echo "========================================="
echo ""
echo "📁 Run all commands from tools/ directory:"
echo "   cd tools && make <command>"
echo ""

# Install dependencies
echo "📦 INSTALL DEPENDENCIES (from project root):"
echo "  pip install -e \".[dev]\""
echo "  # or: pip install -r nc_reqs.txt"
echo ""

# Unit test coverage
echo "✅ 1) UNIT TEST COVERAGE (≥80%):"
echo "  cd tools && make coverage"
echo ""
echo "  # HTML report:"
echo "  cd tools && make coverage-html"
echo "  # View: ../htmlcov/index.html"
echo ""

# Docstring coverage
echo "📝 2) DOCSTRING COVERAGE (≥80%):"
echo "  cd tools && make docstring-coverage"
echo ""

# Type hint coverage
echo "🔤 3) TYPE HINT COVERAGE:"
echo "  cd tools && make type-stats    # Quick stats"
echo "  cd tools && make type-check    # Full checking"
echo "  cd tools && make type-report   # HTML report"
echo ""

# All checks
echo "🚀 RUN ALL CHECKS:"
echo "  cd tools && make all-checks"
echo ""

# Other useful commands
echo "🛠️  OTHER COMMANDS:"
echo "  cd tools && make lint       # Run all linters"
echo "  cd tools && make format     # Auto-format code"
echo "  cd tools && make clean      # Clean generated files"
echo "  cd tools && make help       # Show all targets"
echo ""

echo "========================================="
echo "📁 File Structure:"
echo "  tools/Makefile             - All coverage commands"
echo "  tools/COVERAGE.md          - Detailed documentation"
echo "  tools/coverage-commands.sh - This script"
echo "  tools/pyrightconfig.json   - Pyright config"
echo "  tools/.coveragerc          - Coverage exclusions"
echo ""
echo "  ../pyproject.toml          - Main tool configurations"
echo "========================================="

