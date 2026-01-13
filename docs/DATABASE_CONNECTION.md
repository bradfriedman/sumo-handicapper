# Database Connection Methods

The Sumo Handicapper app supports two methods for connecting to Cloud SQL:

## Method 1: Cloud SQL Proxy (Local Development)

Best for local development on your machine.

**Setup:**
```bash
# Start the Cloud SQL Proxy
./cloud-sql-proxy PROJECT:REGION:INSTANCE --port=3307

# Set environment variables in .env
DB_HOST=127.0.0.1
DB_PORT=3307
DB_USER=dewsweeper
DB_PASSWORD=your_password
DB_NAME=dewsweeper3
```

**Pros:**
- Simple to set up
- Works with any database client
- No code changes needed
- Good for debugging

**Cons:**
- Requires separate proxy process
- IP address still changes (but proxy handles it)

## Method 2: Cloud SQL Python Connector (Production)

Best for Streamlit Cloud and production deployments.

**Setup:**
```bash
# Set environment variables
CLOUD_SQL_CONNECTION_NAME=PROJECT:REGION:INSTANCE
DB_USER=dewsweeper
DB_PASSWORD=your_password
DB_NAME=dewsweeper3
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

**Pros:**
- No separate proxy process needed
- No IP whitelisting required
- Built-in connection pooling
- Automatic SSL/TLS encryption
- IAM authentication support

**Cons:**
- Requires service account setup
- Python-only (can't use with other tools)

## How the App Chooses

The connection logic in `src/core/db_connector.py` automatically detects which method to use:

```python
if os.getenv('CLOUD_SQL_CONNECTION_NAME'):
    # Use Cloud SQL Python Connector
    use_connector = True
else:
    # Use proxy connection (DB_HOST/DB_PORT)
    use_connector = False
```

## Quick Command Reference

### Local Development
```bash
# Start proxy
./cloud-sql-proxy PROJECT:REGION:INSTANCE --port=3307

# Run app (in another terminal)
uv run streamlit run src/prediction/streamlit_app.py
```

### Production (Streamlit Cloud)
Set these secrets in Streamlit Cloud dashboard:
```toml
CLOUD_SQL_CONNECTION_NAME = "PROJECT:REGION:INSTANCE"
DB_USER = "dewsweeper"
DB_PASSWORD = "your_password"
DB_NAME = "dewsweeper3"

[gcp_service_account]
# Paste service account JSON here
```

## Migration Guide

If you're migrating from proxy-only to supporting both methods:

1. ✅ Install dependency: `cloud-sql-python-connector[pymysql]`
2. ✅ Create `src/core/db_connector.py` module
3. ✅ Update all `pymysql.connect()` calls to use `get_connection()`
4. ✅ Set up service account for production
5. ✅ Configure Streamlit Cloud secrets
6. ✅ Test both connection methods

## Troubleshooting

### "Can't connect to MySQL server"
- Check if proxy is running (local) or service account is configured (production)
- Verify connection name format: `PROJECT:REGION:INSTANCE` (not instance ID)

### "Could not refresh access token"
- Check service account key is valid
- Verify `GOOGLE_APPLICATION_CREDENTIALS` points to correct file
- Ensure Cloud SQL Admin API is enabled

### "Access denied for user"
- Check database password is correct
- Verify user has permissions on the database
- Try connecting with mysql client to test credentials

## Environment Variable Reference

| Variable | Required | Used By | Description |
|----------|----------|---------|-------------|
| `CLOUD_SQL_CONNECTION_NAME` | No | Connector | Format: `PROJECT:REGION:INSTANCE` |
| `DB_HOST` | No | Proxy | Usually `127.0.0.1` |
| `DB_PORT` | No | Proxy | Usually `3307` |
| `DB_USER` | Yes | Both | Database username |
| `DB_PASSWORD` | Yes | Both | Database password |
| `DB_NAME` | Yes | Both | Database name |
| `GOOGLE_APPLICATION_CREDENTIALS` | No | Connector | Path to service account JSON |

## See Also

- [Streamlit Cloud Setup Guide](./STREAMLIT_CLOUD_SETUP.md) - Full deployment guide
- [Cloud SQL Python Connector Docs](https://github.com/GoogleCloudPlatform/cloud-sql-python-connector)
- [Cloud SQL Proxy Docs](https://cloud.google.com/sql/docs/mysql/sql-proxy)
