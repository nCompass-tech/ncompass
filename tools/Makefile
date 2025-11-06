.PHONY: help test coverage coverage-html docstring-coverage type-check type-stats lint format clean all-checks install-dev

# Navigate to project root for all commands
PROJECT_ROOT := $(shell cd .. && pwd)

help: ## Show this help message
	@echo 'nCompass Coverage & Quality Tools'
	@echo '=================================='
	@echo ''
	@echo 'Usage: cd tools && make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install-dev: ## Install development dependencies
	cd $(PROJECT_ROOT) && pip install -e ".[dev]"

test: ## Run unit tests
	cd $(PROJECT_ROOT) && pytest -q

coverage: ## Run unit tests with coverage report (fail under 80%)
	cd $(PROJECT_ROOT) && pytest -q --cov=ncompass --cov-report=term-missing --cov-fail-under=80 --cov-config=tools/.coveragerc

coverage-html: ## Generate HTML coverage report
	cd $(PROJECT_ROOT) && pytest -q --cov=ncompass --cov-report=html --cov-config=tools/.coveragerc
	@echo "Coverage report generated in htmlcov/index.html"

docstring-coverage: ## Check docstring coverage (fail under 80%)
	cd $(PROJECT_ROOT) && interrogate ncompass -v --fail-under 80

type-check: ## Run type checking with mypy
	cd $(PROJECT_ROOT) && mypy ncompass --config-file pyproject.toml

type-stats: ## Show type coverage statistics with pyright
	cd $(PROJECT_ROOT) && pyright --stats --project pyrightconfig.json

type-report: ## Generate detailed type checking report with mypy
	cd $(PROJECT_ROOT) && mkdir -p .mypy_report
	cd $(PROJECT_ROOT) && mypy ncompass --config-file pyproject.toml --html-report .mypy_report
	@echo "Type check report generated in .mypy_report/index.html"

lint: ## Run all linters (ruff, black check, isort check)
	cd $(PROJECT_ROOT) && ruff check ncompass
	cd $(PROJECT_ROOT) && black --check ncompass
	cd $(PROJECT_ROOT) && isort --check-only ncompass

format: ## Format code with black and isort
	cd $(PROJECT_ROOT) && black ncompass
	cd $(PROJECT_ROOT) && isort ncompass

clean: ## Clean up generated files
	cd $(PROJECT_ROOT) && rm -rf htmlcov .mypy_report .mypy_cache .pytest_cache .ruff_cache
	cd $(PROJECT_ROOT) && rm -rf build dist *.egg-info
	cd $(PROJECT_ROOT) && find . -type d -name __pycache__ -exec rm -rf {} +
	cd $(PROJECT_ROOT) && find . -type f -name "*.pyc" -delete
	cd $(PROJECT_ROOT) && find . -type f -name "*.pyo" -delete
	cd $(PROJECT_ROOT) && find . -type f -name ".coverage" -delete

all-checks: lint type-check type-stats docstring-coverage coverage ## Run all quality checks
	@echo ""
	@echo "✅ All checks passed!"
	@echo ""
	@echo "Summary:"
	@echo "  - Code style: ✓"
	@echo "  - Type checking: ✓"
	@echo "  - Docstring coverage: ✓"
	@echo "  - Unit test coverage: ✓"

