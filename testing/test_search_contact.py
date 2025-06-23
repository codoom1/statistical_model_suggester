import pytest
from app import create_app

@pytest.fixture
def app():
    # Create a test application with a minimal model database
    app = create_app()
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    # Override MODEL_DATABASE for predictable search results
    app.config['MODEL_DATABASE'] = {
        'TestModel': {
            'description': 'A simple test model for unit testing.',
            'analysis_goals': [],
            'dependent_variable': []
        },
        'OtherModel': {
            'description': 'Another model without the keyword.',
            'analysis_goals': [],
            'dependent_variable': []
        }
    }
    return app

@pytest.fixture
def client(app):
    return app.test_client()


def test_search_returns_matching_models(client):
    # Test that searching for 'test' returns only TestModel
    resp = client.get('/search?q=test')
    assert resp.status_code == 200
    content = resp.get_data(as_text=True)
    assert 'Search Results for "test"' in content
    assert 'TestModel' in content
    assert 'OtherModel' not in content


def test_search_no_query_shows_empty(client):
    # Test that empty query shows no models found
    resp = client.get('/search?q=')
    assert resp.status_code == 200
    content = resp.get_data(as_text=True)
    assert 'No models found matching' in content


def test_contact_page_content(client):
    # Test that the Contact Us page loads and includes contact details
    resp = client.get('/contact')
    assert resp.status_code == 200
    content = resp.get_data(as_text=True)
    assert 'Contact Us' in content
    assert 'support@statisticalmodelsuggester.com' in content
    # Check for social links
    assert '@StatModelSuggest' in content
    assert 'GitHub' in content 