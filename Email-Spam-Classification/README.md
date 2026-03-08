# Email Spam Classification (Flask UI + API)

A simple spam/ham classifier trained with **TF‑IDF** + **Logistic Regression**, with a single-page web UI served by a Flask backend.

## What’s inside

- `train_spam_model.py` — trains the model and saves `spam_model.pkl` and `vectorizer.pkl`
- `run_spam.py` — CLI inference using the saved model/vectorizer
- `app.py` — Flask server (serves UI + JSON API)
- `templates/index.html` — single-file frontend UI
- `spam_env/` — included virtual environment (recommended)

## Quick start (Windows / PowerShell)

From this folder:

```powershell
# 1) Activate the included environment
.\spam_env\Scripts\Activate.ps1

# 2) (Optional) If you don’t have model files yet
python train_spam_model.py

# 3) Start the web app
python app.py
```

Open:

- http://127.0.0.1:5000/

## Using the UI

The UI supports multiple input styles:

- **Quick Text** — paste message content
- **Subject + Body** — paste structured email fields
- **Raw Email** — paste full email text (headers + body)

## API

### Health

`GET /api/health`

### Predict

`POST /api/predict` (JSON)

Supported payload shapes:

```json
{ "text": "..." }
```

```json
{ "subject": "...", "body": "..." }
```

```json
{ "rawEmail": "..." }
```

Response example:

```json
{
  "label": "Spam",
  "isSpam": true,
  "spamProbability": 0.77,
  "inputLength": 38,
  "model": "LogisticRegression + TF-IDF"
}
```

## CLI usage

```powershell
.\spam_env\Scripts\Activate.ps1
python run_spam.py "Congratulations! You won a free prize"
```

## Notes

- Default server address is `http://127.0.0.1:5000/`.
- You can change the port with:

```powershell
$env:PORT = "5050"; python app.py
```
