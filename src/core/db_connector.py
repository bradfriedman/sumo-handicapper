"""
Database connection utilities for Cloud SQL
Supports both local proxy connection and Cloud SQL Python Connector
"""
import os
import json
import tempfile
import pymysql
from typing import Optional, Callable, Any
from google.cloud.sql.connector import Connector


# Global connector instance (reused across connections)
_connector: Optional[Connector] = None

# Default database user
DEFAULT_DB_USER = 'dewsweeper'
DEFAULT_DB_NAME = 'dewsweeper3'


def _get_streamlit_secret(key: str, default: Any = None) -> Any:
    """
    Get a value from Streamlit secrets, or return default if not available.

    Args:
        key: The secret key to retrieve
        default: Default value if secret not found

    Returns:
        The secret value or default
    """
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            return st.secrets.get(key, default)
    except ImportError:
        pass
    return default


def _setup_credentials_for_streamlit():
    """
    Set up Google Cloud credentials from Streamlit secrets if available.
    Streamlit stores secrets in st.secrets, so we need to convert them
    to a format that google-cloud libraries can use.

    Note: The temporary file created here is intentionally not cleaned up,
    as it needs to persist for the lifetime of the Streamlit app. The OS
    will clean it up when the container/process terminates.
    """
    try:
        import streamlit as st

        # Check if we have GCP service account in secrets
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            # Create a temporary file with the service account JSON
            # The google-cloud libraries expect a file path or environment variable

            # Convert st.secrets dict to JSON
            service_account_info = dict(st.secrets['gcp_service_account'])

            # Write to a temporary file (not deleted - needs to persist for app lifetime)
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
            json.dump(service_account_info, temp_file)
            temp_file.close()

            # Set the environment variable to point to this file
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file.name

            return True
    except ImportError:
        # Streamlit not available (local environment)
        pass
    except Exception as e:
        # Log but don't fail - maybe credentials are already set via env var
        print(f"Warning: Could not set up Streamlit credentials: {e}")

    return False


def get_connector() -> Connector:
    """Get or create the global connector instance"""
    global _connector
    if _connector is None:
        # Disable GCE metadata service to prevent timeout on non-GCE environments
        os.environ['GCE_METADATA_HOST'] = 'metadata.google.internal.invalid'

        # Try to set up Streamlit credentials first
        _setup_credentials_for_streamlit()

        # Check if we have credentials from Streamlit secrets or env var
        # If so, create connector with explicit credentials to avoid metadata server
        credentials = None
        if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            from google.oauth2 import service_account

            # Load credentials with explicit universe_domain to prevent metadata lookup
            credentials = service_account.Credentials.from_service_account_file(
                os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            )

            # Explicitly set universe_domain on the credentials object
            # to prevent it from trying to fetch from metadata service
            credentials = credentials.with_universe_domain("googleapis.com")

        if credentials:
            # Pass credentials with universe domain already set
            _connector = Connector(credentials=credentials)
        else:
            # No explicit credentials, let it auto-detect (for local development)
            _connector = Connector()
    return _connector


def get_connection_via_connector(
    instance_connection_name: str,
    user: str,
    password: str,
    database: str
) -> pymysql.Connection:
    """
    Create a database connection using Cloud SQL Python Connector

    Args:
        instance_connection_name: Format "PROJECT:REGION:INSTANCE"
        user: Database user
        password: Database password
        database: Database name

    Returns:
        pymysql.Connection object
    """
    connector = get_connector()

    conn = connector.connect(
        instance_connection_name,
        "pymysql",
        user=user,
        password=password,
        db=database
    )

    return conn


def get_connection_via_proxy(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str
) -> pymysql.Connection:
    """
    Create a database connection via Cloud SQL Proxy (local development)

    Args:
        host: Database host (usually 127.0.0.1 for proxy)
        port: Database port (usually 3307 for proxy)
        user: Database user
        password: Database password
        database: Database name

    Returns:
        pymysql.Connection object
    """
    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database
    )

    return conn


def get_connection() -> pymysql.Connection:
    """
    Get a database connection using environment variables or Streamlit secrets

    Supports both environment variables and Streamlit secrets.
    Priority: Streamlit secrets > Environment variables

    Environment variables (Cloud SQL Connector - for production):
        CLOUD_SQL_CONNECTION_NAME: Format "PROJECT:REGION:INSTANCE"
        DB_USER: Database user
        DB_PASSWORD: Database password
        DB_NAME: Database name

    Environment variables (Proxy - for local development):
        DB_HOST: Database host (127.0.0.1)
        DB_PORT: Database port (3307)
        DB_USER: Database user
        DB_PASSWORD: Database password
        DB_NAME: Database name

    Streamlit secrets (for Streamlit Cloud):
        CLOUD_SQL_CONNECTION_NAME: Format "PROJECT:REGION:INSTANCE"
        DB_USER: Database user
        DB_PASSWORD: Database password
        DB_NAME: Database name
        [gcp_service_account]: Service account JSON as TOML section

    Returns:
        pymysql.Connection object
    """
    # Try to get config from Streamlit secrets first, then fall back to environment variables
    instance_connection_name = _get_streamlit_secret('CLOUD_SQL_CONNECTION_NAME') or \
                                os.getenv('CLOUD_SQL_CONNECTION_NAME')
    db_user = _get_streamlit_secret('DB_USER') or \
              os.getenv('DB_USER', DEFAULT_DB_USER)
    db_password = _get_streamlit_secret('DB_PASSWORD') or \
                  os.getenv('DB_PASSWORD')
    db_name = _get_streamlit_secret('DB_NAME') or \
              os.getenv('DB_NAME', DEFAULT_DB_NAME)

    if instance_connection_name:
        # Use Cloud SQL Python Connector (production/Streamlit Cloud)
        return get_connection_via_connector(
            instance_connection_name=instance_connection_name,
            user=db_user,
            password=db_password,
            database=db_name
        )
    else:
        # Use proxy connection (local development)
        host = _get_streamlit_secret('DB_HOST') or os.getenv('DB_HOST', '127.0.0.1')
        port = int(_get_streamlit_secret('DB_PORT') or os.getenv('DB_PORT', '3307'))

        return get_connection_via_proxy(
            host=host,
            port=port,
            user=db_user,
            password=db_password,
            database=db_name
        )


def get_connection_params() -> dict:
    """
    Get connection parameters as a dict (for compatibility with existing code)

    Note: This returns a callable 'creator' function for SQLAlchemy-style usage
    or traditional connection params for pymysql.connect()
    """
    instance_connection_name = os.getenv('CLOUD_SQL_CONNECTION_NAME')

    if instance_connection_name:
        # Return a creator function for Cloud SQL Connector
        def creator() -> pymysql.Connection:
            return get_connection()

        return {
            'creator': creator,
            'use_connector': True
        }
    else:
        # Return traditional connection params
        return {
            'host': os.getenv('DB_HOST', '127.0.0.1'),
            'port': int(os.getenv('DB_PORT', '3307')),
            'user': os.getenv('DB_USER', DEFAULT_DB_USER),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME', DEFAULT_DB_NAME),
            'use_connector': False
        }


def close_connector():
    """Close the global connector instance (call on application shutdown)"""
    global _connector
    if _connector is not None:
        _connector.close()
        _connector = None
