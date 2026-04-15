# Used Car Price Estimator

Machine-learning powered web app that predicts the selling price of a used car based on real-world listing data.

The project includes:
- Random Forest regression model (scikit-learn)
- FastAPI backend for model inference (Dockerized)
- HTML + JavaScript frontend
- Backend deployed on Fly.io (frontend can be hosted via GitHub Pages)

---

## Tech Stack

- Python, pandas, numpy, scikit-learn  
- FastAPI, Uvicorn, Docker, Fly.io  
- HTML, CSS, Vanilla JavaScript

---

## Structure
Backend: FastAPI app, Docker + Fly deploy files
Frontend: Web UI (index.html, app.js, styles.css)

The trained model file (`car_price_model.joblib`) is kept locally and not committed.

This project was built as a learning + portfolio project to practice ML modeling, API deployment, Docker, and frontend integration. This repo also includes a jupyter notebook with a model built around synthetic data.

---

## Local Development

### Option 1: npm live reload (recommended)

1. Install dependencies:
	npm install
2. Run the frontend with live reload:
	npm run dev
3. Open:
	http://127.0.0.1:5173

Extra script:
- npm run dev:docs serves the docs folder on port 5174

### Option 2: simplest no-npm preview

If you only need a quick local preview without live reload:

1. Run:
	python3 -m http.server 5173 --directory frontend
2. Open:
	http://127.0.0.1:5173
