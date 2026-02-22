# XIRANG Demo Deployment (Streamlit Community Cloud)

This guide gives you a public URL that works even when your laptop is off.

## 1) Prepare GitHub repo

Upload these files/folders to one GitHub repository:

- `xirang_demo_app.py`
- `Modules/CRM_module.py`
- `Datasets/` (keep current structure)
- `requirements.txt`
- `.streamlit/config.toml`

## 2) Deploy on Streamlit Community Cloud

1. Open [https://share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub.
3. Click **New app**.
4. Select your repository and branch.
5. Set **Main file path** to:

```text
xirang_demo_app.py
```

6. Click **Deploy**.

After deployment, Streamlit gives you a fixed URL like:

```text
https://your-app-name.streamlit.app
```

You can share that link directly.

## 3) Notes

- If first load is slow, wait 20-60 seconds for cold start.
- If app fails, open **Manage app -> Logs** and check missing files.
- If data path errors occur, make sure `Datasets/` is at repo root.

