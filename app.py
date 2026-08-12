"""
app.py
------
Flask backend for the Speech Emotion Detection web app.

Routes:
    GET  /          -> Renders the single-page UI (templates/index.html)
    POST /predict    -> Accepts an uploaded audio file, runs it through the
                        fine-tuned Wav2Vec2 model, and returns JSON with the
                        predicted emotion, emoji, and confidence score.
"""

import os
import uuid
from flask import Flask, request, jsonify, render_template

from predict import EmotionPredictor

UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}
MAX_CONTENT_LENGTH_MB = 15

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
os.makedirs(UPLOAD_DIR, exist_ok=True)

# The model is loaded once at startup rather than per-request, since loading
# a transformer checkpoint from disk is relatively expensive.
predictor = None
model_load_error = None
try:
    predictor = EmotionPredictor()
except FileNotFoundError as exc:
    # Allow the server to still start (so the UI is viewable) even if the
    # model hasn't been trained yet -- /predict will report a clear error.
    model_load_error = str(exc)
    print(f"[WARNING] {model_load_error}")


def allowed_file(filename: str) -> bool:
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Render the main (and only) page of the app."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept an uploaded audio file and return the predicted emotion.

    Expects a multipart/form-data request with the file under the key
    'audio_file'. Returns JSON:
        {
            "filename": str,
            "emotion": str,
            "confidence": float,
            "emoji": str,
            "all_scores": {emotion: confidence, ...}
        }
    """
    if predictor is None:
        return jsonify({
            "error": "Model not available. Please run train.py to fine-tune "
                     "and save a model before making predictions."
        }), 503

    if "audio_file" not in request.files:
        return jsonify({"error": "No audio file was uploaded."}), 400

    file = request.files["audio_file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed types: "
                     f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    # Save with a unique name to avoid collisions between concurrent users,
    # while keeping the original name to display back in the UI.
    original_filename = file.filename
    extension = original_filename.rsplit(".", 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{extension}"
    saved_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        file.save(saved_path)
        result = predictor.predict(saved_path)
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the UI
        return jsonify({"error": f"Failed to process audio: {exc}"}), 500
    finally:
        # Clean up the uploaded file -- no persistent storage is needed.
        if os.path.exists(saved_path):
            os.remove(saved_path)

    return jsonify({
        "filename": original_filename,
        "emotion": result["emotion"].capitalize(),
        "confidence": result["confidence"],
        "emoji": result["emoji"],
        "all_scores": result["all_scores"],
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
