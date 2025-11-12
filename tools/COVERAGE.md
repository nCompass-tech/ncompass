# Coverage Guide for nCompass

This project uses comprehensive coverage tooling to maintain code quality. All coverage tools are configured in the project root's `pyproject.toml` and can be run via the Makefile or directly.

**Location:** All coverage tools and documentation are in the `tools/` directory.

## Quick Start

```bash
# Install all development dependencies (from project root)
pip install -e ".[dev]"

# Or use pinned versions
pip install -r requirements.txt

# Run coverage checks from tools/ directory
cd tools
make all-checks
```

## Coverage Types

### 1. Unit Test Coverage (pytest-cov)

Measures which lines of code are executed during tests.

**Target:** ≥80% coverage

```bash
cd tools  # Navigate to tools directory

# Run with coverage report
make coverage

# Or directly from project root
pytest -q --cov=ncompass --cov-report=term-missing --cov-fail-under=80 --cov-config=tools/.coveragerc

# Generate HTML report
make coverage-html
# View at: ../htmlcov/index.html
```

**Configuration:** 
- `[tool.pytest.ini_options]` in `../pyproject.toml`
- `tools/.coveragerc` for coverage exclusions

### 2. Docstring Coverage (interrogate)

Measures which functions, classes, and modules have docstrings.

**Target:** ≥80% coverage

```bash
cd tools  # Navigate to tools directory

# Check docstring coverage
make docstring-coverage

# Or directly from project root
interrogate ncompass -v --fail-under 80

# See detailed stats
interrogate ncompass -vv
```

**Configuration:** `[tool.interrogate]` in `../pyproject.toml`

### 3. Type Hint Coverage (pyright + mypy)

Measures type annotation completeness and correctness.

**Primary metric:** pyright's "Type completeness" percentage

```bash
cd tools  # Navigate to tools directory

# Quick stats with pyright
make type-stats
# Or from project root: pyright --stats --project tools/pyrightconfig.json

# Full type checking with mypy
make type-check
# Or from project root: mypy ncompass --config-file pyproject.toml

# Generate detailed HTML report
make type-report
# View at: ../.mypy_report/index.html
```

**Configuration:**
- `tools/pyrightconfig.json` for pyright
- `[tool.mypy]` in `../pyproject.toml` for mypy

## Individual Commands

### Linting & Formatting

```bash
cd tools  # Navigate to tools directory

# Run all linters
make lint

# Auto-format code
make format

# Or directly from project root
ruff check ncompass
black ncompass
isort ncompass
```

### Cleaning

```bash
cd tools  # Navigate to tools directory

# Remove all generated files
make clean
```

## CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Install dependencies
  run: pip install -e ".[dev]"

- name: Run all coverage checks
  run: cd tools && make all-checks
```

## Coverage Thresholds

All thresholds are set to **80%**:

| Coverage Type | Tool | Threshold | Config Location |
|---------------|------|-----------|-----------------|
| Unit Tests | pytest-cov | 80% | `pyproject.toml` → `[tool.pytest.ini_options]` |
| Docstrings | interrogate | 80% | `pyproject.toml` → `[tool.interrogate]` |
| Type Hints | pyright | Visual only | `pyrightconfig.json` |

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make help` | Show all available targets |
| `make install-dev` | Install development dependencies |
| `make test` | Run unit tests (no coverage) |
| `make coverage` | Run tests with coverage report |
| `make coverage-html` | Generate HTML coverage report |
| `make docstring-coverage` | Check docstring coverage |
| `make type-check` | Run mypy type checking |
| `make type-stats` | Show pyright type statistics |
| `make type-report` | Generate detailed type report |
| `make lint` | Run all linters |
| `make format` | Auto-format code |
| `make clean` | Clean generated files |
| `make all-checks` | Run all quality checks |

## Tips

1. **Before committing:** Run `make all-checks` to ensure everything passes
2. **Quick feedback:** Run `make test` during development for fast iteration
3. **Type coverage:** Use `pyright --stats` as your headline metric for type completeness
4. **HTML reports:** Generate when you need detailed analysis of what's missing

## File Locations

Coverage tooling is organized as follows:

```
ncompass/
├── pyproject.toml               # Only config at root (Python standard)
└── tools/                       # All dev tools here
    ├── Makefile                 # All coverage commands
    ├── COVERAGE.md              # This documentation
    ├── README.md                # Tools overview
    ├── coverage-commands.sh     # Quick reference
    ├── pyrightconfig.json       # Pyright configuration
    └── .coveragerc              # Coverage exclusions
```

## Exclusions

The following are excluded from coverage:
- `tests/` directory
- `examples/` directory
- `docs/` directory
- `__pycache__` directories
- Abstract methods
- Type checking blocks (`if TYPE_CHECKING:`)
- `__main__` blocks

See `tools/.coveragerc` and `pyproject.toml` for full exclusion configuration.

