## Environment Setup

This project is designed to run in a standard Python 3.10+ environment.

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS / Linux
# .venv\Scripts\activate   # on Windows
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Running the project

**Run the analysis pipeline (TSLA, Q1 2025):**

```bash
python pipeline.py
```

**Run the Streamlit dashboard:**

```bash
streamlit run app.py
```

Make sure you always have this virtual environment activated when running any of the project scripts.

