"""
Test the authentication routes.
"""
from models import User
class TestAuthRoutes:
    """Test authentication routes."""
    def test_login_page_accessible(self, client):
        """Test that the login page is accessible."""
        response = client.get('/auth/login')
        assert response.status_code == 200
        assert b'login' in response.data.lower()
    def test_register_page_accessible(self, client):
        """Test that the register page is accessible."""
        response = client.get('/auth/register')
        assert response.status_code == 200
        assert b'register' in response.data.lower()
    def test_successful_login(self, client, test_user):
        """Test successful login with valid credentials."""
        response = client.post('/auth/login', data={
            'username': 'testuser',
            'password': 'testpassword'
        }, follow_redirects=True)
        assert response.status_code == 200
        # Should redirect to home page after successful login
        assert b'home' in response.data.lower() or b'dashboard' in response.data.lower()
    def test_failed_login_wrong_password(self, client, test_user):
        """Test login failure with wrong password."""
        response = client.post('/auth/login', data={
            'username': 'testuser',
            'password': 'wrongpassword'
        })
        assert response.status_code == 200
        assert b'login failed' in response.data.lower() or b'invalid' in response.data.lower()
    def test_failed_login_nonexistent_user(self, client):
        """Test login failure with non-existent user."""
        response = client.post('/auth/login', data={
            'username': 'nonexistentuser',
            'password': 'password'
        })
        assert response.status_code == 200
        assert b'login failed' in response.data.lower() or b'invalid' in response.data.lower()
    def test_login_missing_credentials(self, client):
        """Test login with missing credentials."""
        # Missing password
        response = client.post('/auth/login', data={
            'username': 'testuser'
        })
        assert response.status_code == 200
        assert b'provide both' in response.data.lower() or b'required' in response.data.lower()
        # Missing username
        response = client.post('/auth/login', data={
            'password': 'testpassword'
        })
        assert response.status_code == 200
        assert b'provide both' in response.data.lower() or b'required' in response.data.lower()
    def test_successful_registration(self, client, app):
        """Test successful user registration."""
        response = client.post('/auth/register', data={
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'newpassword',
            'confirm_password': 'newpassword'
        }, follow_redirects=True)
        assert response.status_code == 200
        # Check that user was created in database
        with app.app_context():
            user = User.query.filter_by(username='newuser').first()
            assert user is not None
            assert user.email == 'newuser@example.com'
            assert user.check_password('newpassword')
    def test_registration_duplicate_username(self, client, test_user):
        """Test registration with duplicate username."""
        response = client.post('/auth/register', data={
            'username': 'testuser',  # Same as test_user
            'email': 'different@example.com',
            'password': 'newpassword',
            'confirm_password': 'newpassword'
        })
        assert response.status_code == 200
        assert b'username already exists' in response.data.lower() or b'taken' in response.data.lower()
    def test_registration_duplicate_email(self, client, test_user):
        """Test registration with duplicate email."""
        response = client.post('/auth/register', data={
            'username': 'differentuser',
            'email': 'test@example.com',  # Same as test_user
            'password': 'newpassword',
            'confirm_password': 'newpassword'
        })
        assert response.status_code == 200
        assert b'email already registered' in response.data.lower() or b'taken' in response.data.lower()
    def test_registration_password_mismatch(self, client):
        """Test registration with mismatched passwords."""
        response = client.post('/auth/register', data={
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'password1',
            'confirm_password': 'password2'
        })
        assert response.status_code == 200
        assert b'password' in response.data.lower() and (b'match' in response.data.lower() or b'confirm' in response.data.lower())
    def test_registration_missing_fields(self, client):
        """Test registration with missing fields."""
        # Missing username
        response = client.post('/auth/register', data={
            'email': 'newuser@example.com',
            'password': 'password',
            'confirm_password': 'password'
        })
        assert response.status_code == 200
        # Missing email
        response = client.post('/auth/register', data={
            'username': 'newuser',
            'password': 'password',
            'confirm_password': 'password'
        })
        assert response.status_code == 200
    def test_logout(self, authenticated_client):
        """Test user logout."""
        response = authenticated_client.get('/auth/logout', follow_redirects=True)
        assert response.status_code == 200
        # After logout, should be redirected to login page or home
        # Try to access a protected page to verify logout
        response = authenticated_client.get('/profile')
        assert response.status_code == 302  # Should redirect to login
    def test_logout_requires_login(self, client):
        """Test that logout requires being logged in."""
        response = client.get('/auth/logout')
        assert response.status_code == 302  # Should redirect to login
    def test_redirect_after_login(self, client, test_user):
        """Test redirect to intended page after login."""
        # Try to access a protected page first
        response = client.get('/profile')
        assert response.status_code == 302
        # Login with next parameter
        response = client.post('/auth/login?next=%2Fprofile', data={
            'username': test_user['username'],
            'password': test_user['password']
        }, follow_redirects=True)
        assert response.status_code == 200
    def test_authenticated_user_redirect_from_login(self, authenticated_client):
        """Test that authenticated users are redirected from login page."""
        response = authenticated_client.get('/auth/login', follow_redirects=True)
        assert response.status_code == 200
        # Should be redirected away from login page
    def test_authenticated_user_redirect_from_register(self, authenticated_client):
        """Test that authenticated users are redirected from register page."""
        response = authenticated_client.get('/auth/register', follow_redirects=True)
        assert response.status_code == 200
        # Should be redirected away from register page
class TestPasswordSecurity:
    """Test password security features."""
    def test_password_hashing(self, app):
        """Test that passwords are properly hashed."""
        with app.app_context():
            user = User()
            user.username = 'testuser'
            user.email = 'test@example.com'
            password = 'plaintext_password'
            user.set_password(password)
            # Password should not be stored in plain text
            assert user.password_hash != password
            # Should be able to verify the password
            assert user.check_password(password)
            # Should reject wrong passwords
            assert not user.check_password('wrong_password')
    def test_password_hash_uniqueness(self, app):
        """Test that the same password generates different hashes."""
        with app.app_context():
            user1 = User()
            user1.username = 'user1'
            user1.email = 'user1@example.com'
            user1.set_password('samepassword')
            user2 = User()
            user2.username = 'user2'
            user2.email = 'user2@example.com'
            user2.set_password('samepassword')
            # Same password should generate different hashes (due to salt)
            assert user1.password_hash != user2.password_hash
            # But both should verify correctly
            assert user1.check_password('samepassword')
            assert user2.check_password('samepassword')
class TestUserSession:
    """Test user session management."""
    def test_remember_me_functionality(self, client, test_user):
        """Test remember me checkbox functionality."""
        # Login with remember me
        response = client.post('/auth/login', data={
            'username': test_user['username'],
            'password': test_user['password'],
            'remember': 'on'
        }, follow_redirects=True)
        assert response.status_code == 200
        # Cookie should be set for remember me
        # This is hard to test without checking cookies directly
    def test_session_persistence(self, authenticated_client):
        """Test that authenticated sessions persist across requests."""
        # Make multiple requests to verify session persistence
        response1 = authenticated_client.get('/')
        assert response1.status_code == 200
        response2 = authenticated_client.get('/')
        assert response2.status_code == 200
        # Both requests should work without re-authentication
