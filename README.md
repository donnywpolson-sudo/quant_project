# 1) Clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2) Create & activate env, install requirements

# Powershell (line by line)
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Codespaces
python -m venv .venv && \
source .venv/bin/activate && \
pip install --upgrade pip && \
pip install -r requirements.txt

# 3) Run pipeline / backtest
python -m <package>.pipeline

# 4) Tests
pytest -q