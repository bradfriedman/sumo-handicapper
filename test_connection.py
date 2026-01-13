"""
Test script to verify database connection
Tests both proxy and Cloud SQL Connector methods
"""
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core.db_connector import get_connection, close_connector

def test_connection():
    """Test database connection"""
    print("Testing database connection...")
    print(f"CLOUD_SQL_CONNECTION_NAME: {os.getenv('CLOUD_SQL_CONNECTION_NAME', 'Not set (using proxy)')}")
    print(f"DB_HOST: {os.getenv('DB_HOST', 'Not set')}")
    print(f"DB_PORT: {os.getenv('DB_PORT', 'Not set')}")
    # Redact credentials path for security
    creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')
    if creds_path != 'Not set':
        creds_path = '***REDACTED***'
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {creds_path}")
    print()

    try:
        conn = get_connection()
        print("✓ Successfully connected to database!")

        # Run a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM boi_ozumobout")
        count = cursor.fetchone()[0]
        print(f"✓ Query successful: Found {count:,} bouts in database")

        cursor.close()
        conn.close()
        print("✓ Connection closed successfully")

        return True

    except Exception as e:
        print(f"✗ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    try:
        success = test_connection()
    finally:
        # Clean up the global connector instance
        close_connector()
    sys.exit(0 if success else 1)
