# Deployment

## Secrets

Do not commit real API keys to GitHub.

For local runs, copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` and set:

```toml
LOSTARK_API_KEY = "your-api-key"
```

For Streamlit Community Cloud, add the same key in the app's Secrets settings.

The app also reads these environment variables:

- `LOSTARK_API_KEY`
- `LOA_API_TOKEN`

## Before Push

Check that no real key remains in the repository:

```powershell
rg "eyJ|LOSTARK_API_KEY|LOA_API_TOKEN|secret|token" app.py modules configs .streamlit
```

`LOSTARK_API_KEY` references in code and examples are expected. Real JWT values must not appear.
