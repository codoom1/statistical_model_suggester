# Testing Documentation for Statistical Model Suggester

This directory contains comprehensive tests for the Statistical Model Suggester application.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py             # Pytest configuration and fixtures
├── test_models.py          # Model and database tests
├── test_auth_routes.py     # Authentication route tests
├── test_main_routes.py     # Main application route tests
├── test_admin_routes.py    # Admin functionality tests
├── test_utils.py           # Utility function tests
├── test_integration.py     # Integration tests
├── run_tests.py           # Test runner script
└── README.md              # This file
```

## Prerequisites

Install testing dependencies:

```bash
pip install pytest pytest-flask pytest-cov
```

Or install from requirements.txt which includes testing dependencies:

```bash
pip install -r requirements.txt
```

## Running Tests

### Run All Tests

```bash
# From project root
python -m pytest tests/

# Or using the test runner
python tests/run_tests.py
```

### Run Specific Test Categories

```bash
# Run only model tests
python tests/run_tests.py models

# Run only authentication tests
python tests/run_tests.py auth

# Run only main route tests
python tests/run_tests.py main

# Run only admin tests
python tests/run_tests.py admin

# Run only utility tests
python tests/run_tests.py utils
```

### Run Tests with Coverage

```bash
pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
```

This generates an HTML coverage report in `htmlcov/index.html`.

### Run Specific Tests

```bash
# Run a specific test file
pytest tests/test_models.py

# Run a specific test class
pytest tests/test_models.py::TestUser

# Run a specific test method
pytest tests/test_models.py::TestUser::test_user_creation

# Run tests with specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Run only integration tests
```

## Test Categories

### Unit Tests

- **Models** (`test_models.py`): Tests for database models, relationships, and data validation
- **Authentication** (`test_auth_routes.py`): Tests for login, registration, and authentication flows
- **Main Routes** (`test_main_routes.py`): Tests for analysis form submission and model recommendations
- **Admin Routes** (`test_admin_routes.py`): Tests for admin functionality and user management
- **Utilities** (`test_utils.py`): Tests for utility functions, email service, AI service, etc.

### Integration Tests

- **User Workflows** (`test_integration.py`): End-to-end user scenarios
- **Security Integration**: Cross-component security testing
- **Data Integrity**: Database consistency across operations
- **Performance**: Basic performance and load testing

## Test Fixtures

The `conftest.py` file provides several useful fixtures:

- `app`: Flask application instance for testing
- `client`: Test client for making HTTP requests
- `test_user`: A regular test user
- `admin_user`: An admin test user
- `expert_user`: An expert test user
- `authenticated_client`: Pre-authenticated test client
- `admin_client`: Pre-authenticated admin client
- `sample_analysis_data`: Sample data for analysis forms
- `clustering_analysis_data`: Sample data for clustering analysis

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test

```python
def test_user_creation(self, app):
    """Test creating a new user."""
    with app.app_context():
        user = User()
        user.username = 'newuser'
        user.email = 'newuser@example.com'
        user.set_password('password123')
        
        db.session.add(user)
        db.session.commit()
        
        assert user.id is not None
        assert user.username == 'newuser'
        assert user.check_password('password123')
```

### Testing Routes

```python
def test_home_page(self, client):
    """Test that the home page loads."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'statistical' in response.data.lower()
```

### Testing with Authentication

```python
def test_protected_route(self, authenticated_client):
    """Test a route that requires authentication."""
    response = authenticated_client.get('/profile')
    assert response.status_code == 200
```

## Test Data Management

Tests use a temporary SQLite database that is created and destroyed for each test. This ensures:

- Test isolation
- Consistent starting state
- No impact on development/production data

## Continuous Integration

These tests are designed to run in CI/CD environments. Key considerations:

- All external dependencies are mocked
- Tests use temporary databases
- Environment variables are configured for testing
- Tests run without requiring external services

## Coverage Goals

Target coverage areas:

- **Models**: 90%+ coverage of model methods and properties
- **Routes**: 85%+ coverage of route handlers
- **Authentication**: 95%+ coverage of auth flows
- **Admin Functions**: 80%+ coverage of admin features
- **Utilities**: 75%+ coverage of utility functions

## Common Test Scenarios

### Authentication Testing

```python
# Test successful login
def test_successful_login(self, client, test_user):
    response = client.post('/login', data={
        'username': 'testuser',
        'password': 'testpassword'
    }, follow_redirects=True)
    assert response.status_code == 200

# Test failed login
def test_failed_login(self, client):
    response = client.post('/login', data={
        'username': 'nonexistent',
        'password': 'wrongpassword'
    })
    assert b'login failed' in response.data.lower()
```

### Database Testing

```python
# Test model creation
def test_model_creation(self, app):
    with app.app_context():
        model = MyModel()
        model.field = 'value'
        db.session.add(model)
        db.session.commit()
        assert model.id is not None

# Test relationships
def test_model_relationships(self, app, test_user):
    with app.app_context():
        analysis = Analysis()
        analysis.user_id = test_user.id
        db.session.add(analysis)
        db.session.commit()
        assert analysis in test_user.analyses
```

### Form Testing

```python
# Test form submission
def test_form_submission(self, client):
    response = client.post('/form', data={
        'field1': 'value1',
        'field2': 'value2'
    })
    assert response.status_code == 200

# Test form validation
def test_form_validation(self, client):
    response = client.post('/form', data={
        'field1': '',  # Invalid empty field
    })
    assert b'required' in response.data.lower()
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running tests from the project root
2. **Database Errors**: Ensure test database is properly configured in conftest.py
3. **Authentication Errors**: Check that test users are created correctly
4. **Route Errors**: Verify that all required routes are registered

### Debugging Tests

```bash
# Run with verbose output
pytest -v

# Run with debugging output
pytest -s

# Run and drop into debugger on failure
pytest --pdb

# Run specific test with full traceback
pytest tests/test_models.py::TestUser::test_user_creation -vvv
```

## Performance Testing

Basic performance tests are included in `test_integration.py`. For more comprehensive performance testing, consider:

- Load testing with tools like `locust`
- Database performance testing
- Memory usage profiling
- Response time monitoring

## Security Testing

Security tests cover:

- SQL injection prevention
- XSS protection
- Authentication bypass attempts
- CSRF protection
- Access control verification

## Extending Tests

When adding new features:

1. Add unit tests for new models/functions
2. Add route tests for new endpoints
3. Add integration tests for new workflows
4. Update fixtures if needed
5. Add new test markers for categorization
6. Update this documentation

## Best Practices

1. **Test Independence**: Each test should be independent and not rely on other tests
2. **Clear Naming**: Use descriptive test names that explain what is being tested
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Mock External Dependencies**: Don't rely on external services in tests
5. **Test Edge Cases**: Include tests for boundary conditions and error cases
6. **Keep Tests Fast**: Avoid unnecessary delays or complex operations
7. **Use Fixtures**: Leverage pytest fixtures for common setup
8. **Document Complex Tests**: Add comments for complex test logic
