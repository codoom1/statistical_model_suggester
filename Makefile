# Makefile for Statistical Model Suggester Testing

.PHONY: test test-unit test-integration test-coverage test-models test-auth test-main test-admin test-utils clean install-test-deps lint lint-fix

# Default target
test: test-unit test-integration

# Install testing dependencies
install-test-deps:
	pip install pytest pytest-flask pytest-cov flake8

# Run all unit tests
test-unit:
	python -m pytest tests/ -m "not integration" -v

# Run integration tests
test-integration:
	python -m pytest tests/test_integration.py -v

# Run tests with coverage
test-coverage:
	python -m pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

# Run specific test categories
test-models:
	python -m pytest tests/test_models.py -v

test-auth:
	python -m pytest tests/test_auth_routes.py -v

test-main:
	python -m pytest tests/test_main_routes.py -v

test-admin:
	python -m pytest tests/test_admin_routes.py -v

test-utils:
	python -m pytest tests/test_utils.py -v

# Run tests for CI/CD (minimal output)
test-ci:
	python -m pytest tests/ --tb=short -q

# Run fast tests only (skip slow ones)
test-fast:
	python -m pytest tests/ -m "not slow" -v

# Clean up test artifacts and temporary files
clean:
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -name ".DS_Store" -delete
	find . -name "*.log" -delete
	find . -name "*.tmp" -delete
	find . -name "*.bak" -delete

# Linting
lint:
	flake8 routes/ tests/ --max-line-length=88 --extend-ignore=E203,W503

lint-fix:
	@echo "Fixing basic formatting issues..."
	@echo "Note: This fixes only simple whitespace issues. Complex issues need manual fixing."
	# Remove trailing whitespace
	find routes/ tests/ -name "*.py" -exec sed -i '' 's/[[:space:]]*$$//' {} \;
	# Remove blank lines with whitespace
	find routes/ tests/ -name "*.py" -exec sed -i '' '/^[[:space:]]*$$/d' {} \;

# Help
help:
	@echo "Available targets:"
	@echo "  test              - Run all tests"
	@echo "  test-unit         - Run unit tests only"
	@echo "  test-integration  - Run integration tests only"
	@echo "  test-coverage     - Run tests with coverage report"
	@echo "  test-models       - Run model tests"
	@echo "  test-auth         - Run authentication tests"
	@echo "  test-main         - Run main route tests"
	@echo "  test-admin        - Run admin tests"
	@echo "  test-utils        - Run utility tests"
	@echo "  test-ci           - Run tests for CI/CD"
	@echo "  test-fast         - Run fast tests only"
	@echo "  clean             - Clean test artifacts"
	@echo "  install-test-deps - Install testing dependencies"
	@echo "  lint              - Run linters"
	@echo "  lint-fix          - Fix basic linting issues"
	@echo "  help              - Show this help message"
