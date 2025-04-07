from app import create_app
from models import db, User

app = create_app()

with app.app_context():
    # Query all users
    all_users = User.query.all()
    print(f"Total users in database: {len(all_users)}")

    # Query admin users specifically
    admin_users = User.query.filter_by(_is_admin=True).all()
    print(f"Admin users count: {len(admin_users)}")

    # Print details of admin users
    print("\nAdmin Users Details:")
    for user in admin_users:
        print(f"ID: {user.id}")
        print(f"Username: {user.username}")
        print(f"Email: {user.email}")
        print(f"Admin property access: {user.is_admin}")
        print(f"Raw _is_admin field: {user._is_admin}")
        print(f"Created at: {user.created_at}")
        print("-" * 40)

    # Check for users with username 'admin'
    admin_by_name = User.query.filter_by(username='admin').first()
    if admin_by_name:
        print(f"User with username 'admin' exists:")
        print(f"ID: {admin_by_name.id}")
        print(f"Admin property access: {admin_by_name.is_admin}")
        print(f"Raw _is_admin field: {admin_by_name._is_admin}")
    else:
        print("No user with username 'admin' found") 