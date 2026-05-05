# 🔍 Fake News Detector — Deployment Guide

## Option 1: Streamlit Community Cloud (FREE — Recommended)
**Result:** `https://your-app-name.streamlit.app` — shareable with anyone

### Steps (10 minutes):
1. Push this folder to a GitHub repo (public or private)
2. Go to https://share.streamlit.io
3. Click **"New app"** → connect your GitHub account
4. Select your repo, branch `main`, file `app.py`
5. Click **Deploy** — done! Share the URL with teammates and professors.

---

## Option 2: Render.com (FREE — always-on server)
**Result:** `https://your-app.onrender.com`

### Additional file needed — create `Procfile`:
```
web: streamlit run app.py --server.port $PORT --server.headless true
```

### Steps:
1. Push to GitHub
2. Go to https://render.com → New → Web Service
3. Connect GitHub repo
4. Set: Build Command = `pip install -r requirements.txt`
5. Set: Start Command = `streamlit run app.py --server.port $PORT --server.headless true`
6. Deploy → share URL

---

## Local Run (for testing)
```bash
pip install -r requirements.txt
streamlit run app.py
# Opens at http://localhost:8501
```

## Project Structure
```
fakenews_deploy/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .streamlit/
    └── config.toml     # Streamlit theme config
```
