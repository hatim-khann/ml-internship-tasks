from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mnist_cnn_model.h5"

app = Flask(__name__)


def _load() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    return load_model(str(MODEL_PATH))


MODEL = _load()


def _get_primary_input_shape(model: Any) -> tuple[int, ...] | None:
    shape = getattr(model, "input_shape", None)
    if shape is None:
        return None
    if isinstance(shape, list) and shape:
        shape = shape[0]
    if not isinstance(shape, tuple):
        return None
    # Remove batch dim
    return tuple(int(x) for x in shape[1:] if x is not None)


def _preprocess_flat(image: Image.Image) -> np.ndarray:
    # Legacy: grayscale -> 28x28 -> normalize -> flatten(784)
    img = image.convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 784)


def _preprocess_cnn(image: Image.Image) -> np.ndarray:
    # Match run_mnist_digit.py preprocessing exactly:
    # open -> grayscale -> invert -> threshold -> crop bbox -> resize 20x20 -> pad to 28x28 -> normalize
    img = image.convert("L")
    img = ImageOps.invert(img)

    arr = np.array(img)
    arr = np.where(arr > 128, 255, 0).astype(np.uint8)

    coords = np.argwhere(arr)
    if coords.shape[0] == 0:
        padded = np.zeros((28, 28), dtype=np.float32)
        return padded.reshape(1, 28, 28, 1)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cropped = arr[y0 : y1 + 1, x0 : x1 + 1]

    pil = Image.fromarray(cropped)
    resample = getattr(Image, "Resampling", None)
    lanczos = resample.LANCZOS if resample else Image.LANCZOS
    pil = pil.resize((20, 20), lanczos)

    img20 = np.array(pil)
    padded = np.pad(img20, ((4, 4), (4, 4)), "constant", constant_values=0)
    padded = (padded / 255.0).astype(np.float32)
    return padded.reshape(1, 28, 28, 1)


def _preprocess_for_model(image: Image.Image) -> np.ndarray:
    shape = _get_primary_input_shape(MODEL)
    # Expected either (784,) or (28,28,1)
    if shape == (28, 28, 1):
        return _preprocess_cnn(image)
    if shape == (784,):
        return _preprocess_flat(image)
    # Fallback: prefer CNN preprocessing if unclear
    return _preprocess_cnn(image)


def _predict(x: np.ndarray) -> dict[str, Any]:
    probs = MODEL.predict(x, verbose=0)[0]

    digit = int(np.argmax(probs))
    confidence = float(np.max(probs))

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [{"digit": int(i), "confidence": float(probs[i])} for i in top3_idx]

    return {"digit": digit, "confidence": confidence, "top3": top3}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "modelPath": str(MODEL_PATH)})


@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing file field: image"}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    try:
        data = file.read()
        if not data:
            return jsonify({"error": "Empty file"}), 400

        image = Image.open(io.BytesIO(data))
        x = _preprocess_for_model(image)
        result = _predict(x)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Could not process image: {exc}"}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
