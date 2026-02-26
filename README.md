## Improving Risk Management with Regime-Switching Models

**Case Study: Tesla (TSLA) in Q1 2025**

This project wraps your Jupyter notebook analysis into a small, reusable codebase and a Streamlit dashboard.

### Project Structure

- `TeslalokanaShankara.ipynb` – your original research notebook.
- `data_loader.py` – fetches and caches TSLA price data.
- `models.py` – baseline GARCH and Markov regime-switching models.
- `risk.py` – VaR computation and backtesting (Kupiec and Christoffersen tests).
- `pipeline.py` – end-to-end TSLA Q1 2025 pipeline; writes outputs for the dashboard.
- `app.py` – Streamlit dashboard for visualization and decision support.
- `requirements.txt` – Python dependencies.
- `ENVIRONMENT_SETUP.md` – instructions to recreate the environment.

### How to Run

1. Create and activate a virtual environment (see `ENVIRONMENT_SETUP.md`).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the analysis pipeline to generate outputs:

```bash
python pipeline.py
```

4. Launch the dashboard:

```bash
streamlit run app.py
```

### How This Matches the Capstone Brief

- **Model**: Uses a baseline GARCH and a regime-switching (HMM-based) volatility model.
- **Focus**: TSLA risk and volatility in **Q1 2025**, with an estimation window before 2025-01-01.
- **Risk Management**: Computes VaR, counts violations, and applies Kupiec/Christoffersen tests.
- **Productization**: Provides a live dashboard that reports current regime, VaR, and a simple risk recommendation for TSLA.

You can now plug any additional code or figures from your notebook into these modules while keeping the overall structure and interface stable for your presentation.

