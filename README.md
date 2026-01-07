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
