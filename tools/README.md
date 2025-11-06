# Development Tools

This directory contains all coverage and quality tooling for the nCompass project.

## 📁 Contents

- **`Makefile`** - All coverage, testing, and quality commands
- **`COVERAGE.md`** - Comprehensive coverage documentation
- **`coverage-commands.sh`** - Quick reference script for all commands

## 🚀 Quick Start

All commands run from this directory:

```bash
cd tools

# Run all quality checks
make all-checks

# Run individual checks
make coverage           # Unit test coverage
make docstring-coverage # Docstring coverage  
make type-stats        # Type hint coverage
make lint              # Run linters
make format            # Auto-format code
```

## 📊 Coverage Types

This project tracks three types of coverage:

1. **Unit Test Coverage** (pytest-cov) - ≥80%
2. **Docstring Coverage** (interrogate) - ≥80%
3. **Type Hint Coverage** (pyright + mypy) - visual metric

## 📖 Documentation

See **`COVERAGE.md`** for complete documentation including:
- Detailed command explanations
- Configuration locations
- CI/CD integration examples
- Tips and best practices

## ⚙️ Configuration Files

Coverage tool configurations:

- `../pyproject.toml` (root) - pytest, interrogate, mypy, ruff settings
- `pyrightconfig.json` (this dir) - pyright configuration
- `.coveragerc` (this dir) - coverage exclusions and reporting

## 🔧 Available Commands

Run `make help` for a complete list of available commands.

