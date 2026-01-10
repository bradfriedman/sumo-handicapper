# Streamlit Cloud Deployment Guide

Complete guide for deploying the Sumo Handicapper app to Streamlit Cloud.

## Prerequisites Completed ✅

1. ✅ Removed Git LFS - model file now tracked directly in git (23MB)
2. ✅ Updated code to support Streamlit secrets for database configuration
3. ✅ Cloud SQL instance has public IP enabled

## Cloud SQL Configuration

**Instance Details:**
- Connection name: `dewsweeper3:us-central1:dewsweeper3-instance`
- Public IP: `35.194.54.36`
- Port: `3306` (standard MySQL port)
- Database: `dewsweeper3`
- User: `dewsweeper`

## Deployment Steps

### 1. Authorize Streamlit Cloud IPs in Cloud SQL

Go to your Cloud SQL instance → Connections → Networking → Authorized networks

**Option A: Allow All IPs (Simple, less secure)**
Add this network:
```
Name: Streamlit Cloud (all)
Network: 0.0.0.0/0
```

**Option B: Specific IPs (More secure, recommended)**
Contact Streamlit Support to get current IP ranges and add them individually.

### 2. Push Code to GitHub

```bash
git push origin main
```

### 3. Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Connect your GitHub repository
4. Configure:
   - **Repository**: `[your-github-username]/sumo-handicapper`
   - **Branch**: `main`
   - **Main file path**: `src/prediction/streamlit_app.py`

### 4. Configure Secrets in Streamlit Cloud

In your app's dashboard, go to **Settings → Secrets** and add:

```toml
DB_HOST = "35.194.54.36"
DB_PORT = "3306"
DB_USER = "dewsweeper"
DB_PASSWORD = "dewsweeper_password123"
DB_NAME = "dewsweeper3"
```

⚠️ **IMPORTANT**: Replace `dewsweeper_password123` with your actual production password!

### 5. Deploy

Click "Deploy" and wait for the app to start.

## Testing the Deployment

Once deployed, test these features:

1. **Model Loading**: Verify the model loads successfully (check sidebar for model info)
2. **Database Connection**: Search for a wrestler by name (e.g., "Hakuho")
3. **Predictions**: Run a single bout prediction
4. **Live Data**: Verify head-to-head records are displayed

## Troubleshooting

### "Cannot connect to database"
- Verify secrets are configured correctly in Streamlit Cloud
- Check that Streamlit Cloud IPs are in Cloud SQL authorized networks
- Verify Cloud SQL instance has public IP enabled

### "Model file not found"
- Ensure you pushed the commit that removed LFS tracking
- Check that `models/sumo_predictor_production.joblib` exists in your repo
- Wait for Streamlit Cloud to fully rebuild the app

### "Module not found" errors
- Streamlit Cloud uses `requirements.txt` for dependencies
- File is already configured in the repo
- Check logs to see which dependency is missing

### Slow first load
- First prediction loads the 23MB model file - this is normal
- Subsequent predictions use cached model and are much faster

## Local Development vs Production

**Local Development:**
- Uses Cloud SQL Proxy on `127.0.0.1:3307`
- Runs with: `uv run streamlit run src/prediction/streamlit_app.py`
- No secrets configuration needed (uses defaults)

**Streamlit Cloud Production:**
- Direct connection to Cloud SQL on `35.194.54.36:3306`
- Uses secrets from dashboard
- Automatic deployment on git push

## Security Best Practices

1. ✅ Use strong database passwords (not the default)
2. ⚠️ Consider restricting authorized networks to specific Streamlit IPs
3. ✅ Enable SSL connections in Cloud SQL (optional but recommended)
4. ✅ Never commit passwords to git - use secrets only

## Cost Considerations

**Streamlit Cloud:**
- Free tier: 1 app, community support
- Paid tiers: More apps, more resources, priority support

**Cloud SQL:**
- Charged for instance runtime and storage
- Consider stopping instance when not in use for development
- Production instances should stay running

## Updating the App

To deploy updates:

```bash
git add .
git commit -m "Your update message"
git push origin main
```

Streamlit Cloud will automatically detect the push and redeploy.

## Support

- **Streamlit Cloud Issues**: https://discuss.streamlit.io/
- **Cloud SQL Issues**: Google Cloud Console support
- **App Issues**: Check Streamlit Cloud logs in the dashboard
