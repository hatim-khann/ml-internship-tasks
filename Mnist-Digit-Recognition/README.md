# MNIST Digit Recognition (Flask UI + API)

A MNIST digit recognizer using a **CNN** model (`mnist_cnn_model.h5`) with a single-page web UI served by a Flask backend.

## What’s inside

- `train_mnist_digit.py` — trains the CNN and saves `mnist_cnn_model.h5` (optional)
- `run_mnist_digit.py` — CLI inference on an image path
- `app.py` — Flask server (serves UI + upload API)
- `templates/index.html` — single-file frontend UI (drag & drop / paste / select)
- `mnist_env/` — included virtual environment (recommended)

## Quick start (Windows / PowerShell)

From this folder:

```powershell
# 1) Activate the included environment
.\mnist_env\Scripts\Activate.ps1

# 2) (Optional) Train if the model file is missing
python train_mnist_digit.py

# 3) Start the web app
python app.py
```

Open:

- http://127.0.0.1:5000/

## Using the UI

- Drag & drop an image, paste an image from clipboard, or choose a file.
- Click **Recognize** to get the predicted digit and confidence.

## API

### Health

`GET /api/health`

### Predict

`POST /api/predict` (`multipart/form-data`)

- Form field name: `image`

Example (PowerShell):

```powershell
.\mnist_env\Scripts\Activate.ps1
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:5000/api/predict" -Form @{ image = Get-Item ".\digit.png" } |
  ConvertTo-Json -Depth 5 | Out-String
```

Response example:

```json
{
  "digit": 7,
  "confidence": 0.98,
  "top3": [
    {"digit": 7, "confidence": 0.98},
    {"digit": 1, "confidence": 0.01},
    {"digit": 9, "confidence": 0.00}
  ]
}
```

## CLI usage

```powershell
.\mnist_env\Scripts\Activate.ps1
python run_mnist_digit.py .\digit.png
```

## Notes

- Preprocessing matches `run_mnist_digit.py` (invert → threshold → crop → resize 20×20 → pad to 28×28).
- Default server address is `http://127.0.0.1:5000/`.
- You can change the port with:

```powershell
$env:PORT = "5050"; python app.py
```
