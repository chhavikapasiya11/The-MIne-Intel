# Mine-Intel

## Backend
The Flask API that serves the CatBoost prediction lives in `backend/app.py`.

```bash
cd backend
pip install -r requirements.txt
python app.py
```

## Streamlit Frontend
A richer, structured frontend is available under `frontend/streamlit_app`.

```bash
cd frontend/streamlit_app
pip install -r requirements.txt  # or ensure streamlit + requests installed
streamlit run app.py
```

Set `MINE_INTEL_API_URL` if the backend is not running locally on port 5000.

## Training scripts
`My_mine.py` is the main experiment script. Use `scripts/train_xgb.py` for XGBoost.
    