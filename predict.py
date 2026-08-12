"""
predict.py
----------
Loads the fine-tuned Wav2Vec2 emotion classification model and runs
inference on a single audio file, returning the predicted emotion label
and a confidence score.

Used by app.py (Flask backend). Can also be run standalone for quick
command-line testing:

    python predict.py path/to/audio.wav
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

from preprocess import preprocess_audio, extract_features

SAVED_MODEL_DIR = os.path.join("model", "saved_model")

# Emoji shown alongside each predicted emotion in the UI response.
EMOTION_EMOJIS = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😮",
    "disgust": "🤢",
    "neutral": "😐",
}


class EmotionPredictor:
    """
    Wraps the fine-tuned Wav2Vec2 model + processor and exposes a single
    `predict()` method. Instantiated once at Flask app startup so the model
    is loaded into memory a single time, not on every request.
    """

    def __init__(self, model_dir: str = SAVED_MODEL_DIR):
        if not os.path.isdir(model_dir) or not os.listdir(model_dir):
            raise FileNotFoundError(
                f"No fine-tuned model found at '{model_dir}'. "
                "Run train.py first to fine-tune and save the model."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # id2label comes from the saved model config (set during training).
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, file_path: str) -> dict:
        """
        Run the full pipeline (preprocess -> feature extraction -> model
        forward pass -> softmax) on a single audio file.

        Args:
            file_path: Path to the uploaded audio file.

        Returns:
            dict with keys: emotion, confidence (0-100, 2dp), emoji,
            and all_scores (per-class confidence breakdown).
        """
        waveform = preprocess_audio(file_path)
        input_values = extract_features(waveform, self.processor)

        input_tensor = torch.tensor(input_values, dtype=torch.float32) \
            .unsqueeze(0).to(self.device)

        logits = self.model(input_values=input_tensor).logits
        probabilities = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        predicted_id = int(probabilities.argmax())
        predicted_label = self.id2label[predicted_id]
        confidence = float(probabilities[predicted_id]) * 100

        all_scores = {
            self.id2label[i]: round(float(prob) * 100, 2)
            for i, prob in enumerate(probabilities)
        }

        return {
            "emotion": predicted_label,
            "confidence": round(confidence, 2),
            "emoji": EMOTION_EMOJIS.get(predicted_label, ""),
            "all_scores": all_scores,
        }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_audio_file>")
        sys.exit(1)

    predictor = EmotionPredictor()
    result = predictor.predict(sys.argv[1])
    print(f"Predicted emotion: {result['emoji']} {result['emotion'].capitalize()}")
    print(f"Confidence: {result['confidence']}%")
    print(f"All scores: {result['all_scores']}")
