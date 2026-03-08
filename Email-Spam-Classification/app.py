from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


try:
    MODEL = _load_pickle(MODEL_PATH)
    VECTORIZER = _load_pickle(VECTORIZER_PATH)
except FileNotFoundError:
    MODEL = None
    VECTORIZER = None

app = Flask(__name__)


def _coerce_text(payload: dict[str, Any]) -> str:
    # Supported inputs:
    # - {"text": "..."}
    # - {"subject": "...", "body": "..."}
    # - {"rawEmail": "..."}
    text = (payload.get("text") or "").strip()
    raw_email = (payload.get("rawEmail") or "").strip()
    subject = (payload.get("subject") or "").strip()
    body = (payload.get("body") or "").strip()

    if raw_email:
        return raw_email

    if subject or body:
        joined = (subject + "\n\n" + body).strip()
        return joined

    return text


def _predict_spam(text: str) -> dict[str, Any]:
    if MODEL is None or VECTORIZER is None:
        raise RuntimeError(
            "Model files not found. Run train_spam_model.py to generate spam_model.pkl and vectorizer.pkl"
        )

    features = VECTORIZER.transform([text])
    pred = int(MODEL.predict(features)[0])

    spam_probability = None
    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(features)[0]
        # proba[1] corresponds to label 1 (spam) in the training script
        spam_probability = float(proba[1])

    label = "Spam" if pred == 1 else "Not Spam"

    return {
        "label": label,
        "isSpam": pred == 1,
        "spamProbability": spam_probability,
        "inputLength": len(text),
        "model": "LogisticRegression + TF-IDF",
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "modelLoaded": MODEL is not None and VECTORIZER is not None,
            "modelPath": str(MODEL_PATH),
            "vectorizerPath": str(VECTORIZER_PATH),
        }
    )


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = _coerce_text(payload)

    if not text:
        return jsonify({"error": "Please provide text, or subject/body, or a raw email."}), 400

    # Simple guardrails: keep requests reasonable
    max_chars = 50_000
    if len(text) > max_chars:
        return jsonify({"error": f"Input too large. Please keep it under {max_chars} characters."}), 400

    try:
        result = _predict_spam(text)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


if __name__ == "__main__":
    # For local development only
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
