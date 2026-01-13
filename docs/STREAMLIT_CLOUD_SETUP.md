# Streamlit Cloud Deployment Guide

This guide walks you through deploying the Sumo Handicapper app to Streamlit Cloud with Cloud SQL access.

## Overview

The app now uses **Cloud SQL Python Connector** to connect to your Cloud SQL database without needing to whitelist IP addresses. This is the recommended approach for Streamlit Cloud deployments.

## Prerequisites

1. A Google Cloud Project with Cloud SQL instance
2. A service account with Cloud SQL Client permissions
3. A Streamlit Cloud account
4. Your database credentials

## Step 1: Create a Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **IAM & Admin > Service Accounts**
3. Click **Create Service Account**
4. Name it (e.g., `streamlit-cloud-sql-connector`)
5. Grant it the **Cloud SQL Client** role
6. Click **Done**

## Step 2: Create and Download Service Account Key

1. Click on the service account you just created
2. Go to the **Keys** tab
3. Click **Add Key > Create New Key**
4. Choose **JSON** format
5. Click **Create** - this downloads the key file

## Step 3: Prepare Streamlit Cloud Secrets

Streamlit Cloud uses a TOML-based secrets management system. You'll need to configure your secrets with:

1. The service account JSON content
2. Your Cloud SQL connection details
3. Your database credentials

Create a secrets configuration in the following format:

```toml
# Cloud SQL Connection Configuration
CLOUD_SQL_CONNECTION_NAME = "PROJECT_ID:REGION:INSTANCE_NAME"
DB_USER = "dewsweeper"
DB_PASSWORD = "your_database_password"
DB_NAME = "dewsweeper3"

# Google Cloud credentials (paste the ENTIRE contents of your service account JSON file)
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "key-id-here"
private_key = "-----BEGIN PRIVATE KEY-----\nYour private key here...\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "123456789"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
universe_domain = "googleapis.com"
```

### Finding Your Connection Name

Your `CLOUD_SQL_CONNECTION_NAME` has the format: `PROJECT_ID:REGION:INSTANCE_NAME`

To find it:
1. Go to [Cloud SQL Instances](https://console.cloud.google.com/sql/instances)
2. Click on your instance
3. Look for **Connection name** (e.g., `my-project:us-central1:dewsweeper-db`)

## Step 4: Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click **New app**
4. Select your repository and branch
5. Set the main file path: `src/prediction/streamlit_app.py`
6. Click **Advanced settings**
7. Under **Secrets**, paste your secrets configuration from Step 3
8. Click **Deploy**

## Step 5: Verify Environment Variables Are Set

The app will automatically detect the `CLOUD_SQL_CONNECTION_NAME` environment variable and use the Cloud SQL Python Connector instead of the proxy method.

The connection logic in `src/core/db_connector.py` checks for:
- If `CLOUD_SQL_CONNECTION_NAME` is set → Use Cloud SQL Python Connector (production)
- If not set → Use proxy connection with `DB_HOST` and `DB_PORT` (local development)

## Local Development Setup

For local development, you can continue using the Cloud SQL Proxy:

### Option 1: Using Cloud SQL Proxy (Recommended for Local)

1. Start the Cloud SQL Proxy:
   ```bash
   ./cloud-sql-proxy PROJECT:REGION:INSTANCE --port=3307
   ```

2. Set environment variables in `.env`:
   ```env
   DB_HOST=127.0.0.1
   DB_PORT=3307
   DB_USER=dewsweeper
   DB_PASSWORD=your_password
   DB_NAME=dewsweeper3
   ```

3. Run the app:
   ```bash
   uv run streamlit run src/prediction/streamlit_app.py
   ```

### Option 2: Using Cloud SQL Python Connector (Same as Production)

1. Set environment variables:
   ```env
   CLOUD_SQL_CONNECTION_NAME=PROJECT:REGION:INSTANCE
   DB_USER=dewsweeper
   DB_PASSWORD=your_password
   DB_NAME=dewsweeper3
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   ```

2. Run the app:
   ```bash
   uv run streamlit run src/prediction/streamlit_app.py
   ```

## Troubleshooting

### Connection Errors

If you see connection errors:

1. **Check service account permissions**: Ensure the service account has the "Cloud SQL Client" role
2. **Verify connection name format**: Should be `PROJECT:REGION:INSTANCE` (not the instance ID)
3. **Check secrets are set correctly**: In Streamlit Cloud, go to Settings > Secrets to verify
4. **Check Cloud SQL API is enabled**: Go to [APIs & Services](https://console.cloud.google.com/apis/dashboard) and enable "Cloud SQL Admin API"

### Authentication Errors

If you see authentication errors:

1. **Check the service account JSON**: Ensure you copied the entire JSON content correctly
2. **Verify private key format**: The private key should include `\n` for newlines in TOML format
3. **Check GOOGLE_APPLICATION_CREDENTIALS**: The connector looks for this environment variable

### Performance Issues

If connections are slow:

1. **Connection pooling**: The connector automatically pools connections
2. **Choose nearby region**: Deploy your Streamlit app in the same region as your Cloud SQL instance if possible

## Security Best Practices

1. **Never commit service account keys to Git**
   - Add `*.json` to `.gitignore` for service account keys
   - Always use Streamlit Cloud secrets for production

2. **Rotate keys regularly**
   - Delete and recreate service account keys periodically
   - Update Streamlit Cloud secrets when rotating

3. **Use least privilege**
   - Only grant "Cloud SQL Client" role (not "Cloud SQL Admin")
   - Create read-only database users for prediction endpoints if possible

4. **Enable Cloud SQL Auth Proxy fallback**
   - Keep the proxy method working for local development
   - Document both connection methods for team members

## How It Works

The app uses a unified connection module (`src/core/db_connector.py`) that:

1. Checks for `CLOUD_SQL_CONNECTION_NAME` environment variable
2. If set:
   - Uses Cloud SQL Python Connector
   - Authenticates via service account (from `GOOGLE_APPLICATION_CREDENTIALS` or Streamlit secrets)
   - Creates encrypted connection without IP whitelisting
3. If not set:
   - Falls back to traditional `pymysql.connect()` with `DB_HOST`/`DB_PORT`
   - Works with Cloud SQL Proxy for local development

This approach provides:
- ✅ No IP whitelisting needed
- ✅ Seamless local/production switching
- ✅ Automatic SSL/TLS encryption
- ✅ IAM-based authentication
- ✅ Connection pooling built-in

## Reference Links

- [Cloud SQL Python Connector Documentation](https://github.com/GoogleCloudPlatform/cloud-sql-python-connector)
- [Streamlit Cloud Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)
- [Cloud SQL IAM Authentication](https://cloud.google.com/sql/docs/mysql/authentication)
- [Service Account Key Management](https://cloud.google.com/iam/docs/creating-managing-service-account-keys)
