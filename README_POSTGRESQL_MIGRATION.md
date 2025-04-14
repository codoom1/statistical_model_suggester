# PostgreSQL Migration Guide

This guide will help you migrate your data from a local SQLite database to PostgreSQL on Render.

## Prerequisites

- Your application deployed on Render with PostgreSQL
- Access to your local development environment with SQLite database
- Python 3.8+ installed

## Migration Steps

### 1. Set Up Your Environment

Make sure your `.env` file is configured correctly with your PostgreSQL connection string:

```
DATABASE_URL=postgresql://statistical_model_suggester_db_user:ivTzIBDnhZuza5GEFMwzuMLVjnHnBM5H@dpg-cvu7142dbo4c739eumcg-a.oregon-postgres.render.com/statistical_model_suggester_db
```

### 2. Run the Migration Script

We've provided a migration script that handles the data transfer process. Run it with:

```bash
python migrate_data_to_postgres.py
```

This script will:
- Export all data from your local SQLite database to a JSON file
- Connect to your PostgreSQL database on Render
- Import the data into PostgreSQL

### 3. Verify the Migration

After the migration, verify that all your data was correctly transferred:

1. Log in to your application on Render
2. Check that all users, analyses, and other data are present
3. Visit the `/system/check-db` endpoint to confirm database connectivity

## Troubleshooting

### Common Issues

1. **Connection Errors**: If you see connection errors, verify your PostgreSQL URL is correct and the database is accessible.

2. **Primary Key Conflicts**: If you see errors about duplicate primary keys, you may need to clear the destination tables before importing.

3. **Data Type Mismatches**: Some fields might have different data types between SQLite and PostgreSQL. The script tries to handle these automatically, but you may need to manually fix some data.

### Manual Database Management

For direct access to your Render PostgreSQL database:

1. Install the PostgreSQL client:
   ```bash
   # MacOS
   brew install postgresql
   
   # Ubuntu/Debian
   sudo apt-get install postgresql-client
   ```

2. Connect to your database:
   ```bash
   psql "postgresql://statistical_model_suggester_db_user:ivTzIBDnhZuza5GEFMwzuMLVjnHnBM5H@dpg-cvu7142dbo4c739eumcg-a.oregon-postgres.render.com/statistical_model_suggester_db"
   ```

3. Common PostgreSQL commands:
   - List tables: `\dt`
   - Examine table structure: `\d table_name`
   - Delete all rows from a table: `DELETE FROM table_name;`
   - Count rows in a table: `SELECT COUNT(*) FROM table_name;`

## Security Note

The migration script contains your database credentials. Make sure to:
- Keep your `.env` file secure
- Don't commit the database export file to version control
- Consider deleting the export file after successful migration

## Additional Resources

- [Render PostgreSQL Documentation](https://render.com/docs/databases)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/) 