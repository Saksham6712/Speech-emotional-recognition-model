import argparse
import os
import glob
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from preprocess import (
    preprocess_audio,
    extract_features,
    get_processor,
    BASE_MODEL_CHECKPOINT,
)

# Fixed, ordered label set. The index of each label is its class id, and this
# exact order must match what app.py / predict.py use at inference time.
EMOTION_LABELS = ["happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"]
LABEL2ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

SAVED_MODEL_DIR = os.path.join("model", "saved_model")


def normalize_label(raw_name: str) -> str:
    """
    Map a raw TESS folder/file token (e.g. "OAF_Pleasant_surprise",
    "YAF_Sad") to one of our canonical EMOTION_LABELS.
    """
    raw = raw_name.lower()
    if "angry" in raw:
        return "angry"
    if "disgust" in raw:
        return "disgust"
    if "fear" in raw:
        return "fear"
    if "happy" in raw:
        return "happy"
    if "neutral" in raw:
        return "neutral"
    if "sad" in raw:
        return "sad"
    if "surprise" in raw or "pleasant" in raw:
        return "surprise"
    return None


def collect_tess_files(data_dir: str):

    samples = []
    wav_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)

    for path in wav_files:
        # The emotion is reliably present in either the parent folder name
        # or the filename itself, so check both.
        folder_name = os.path.basename(os.path.dirname(path))
        file_name = os.path.basename(path)
        label = normalize_label(folder_name) or normalize_label(file_name)
        if label is None:
            continue  # skip files we can't confidently label
        samples.append((path, LABEL2ID[label]))

    if not samples:
        raise RuntimeError(
            f"No labeled TESS audio files found under '{data_dir}'. "
            "Check TESS_DATA_DIR / --data_dir points to the dataset root."
        )
    return samples


class TESSDataset(Dataset):
    """
    PyTorch Dataset that lazily preprocesses TESS audio files into
    Wav2Vec2 input feature vectors.
    """

    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label_id = self.samples[idx]
        waveform = preprocess_audio(file_path)
        input_values = extract_features(waveform, self.processor)
        return {
            "input_values": torch.tensor(input_values, dtype=torch.float32),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


def compute_metrics(eval_pred):
    """Simple accuracy metric used during validation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 on TESS")
    parser.add_argument("--data_dir", type=str, default="TESS_DATA_DIR",
                         help="Path to the root folder of the TESS dataset")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--val_split", type=float, default=0.15)
    args = parser.parse_args()

    print(f"Loading TESS files from: {args.data_dir}")
    samples = collect_tess_files(args.data_dir)
    print(f"Found {len(samples)} labeled audio samples across "
          f"{len(EMOTION_LABELS)} emotion classes.")

    labels_only = [label for _, label in samples]
    train_samples, val_samples = train_test_split(
        samples, test_size=args.val_split, random_state=42, stratify=labels_only
    )
    print(f"Train samples: {len(train_samples)} | Validation samples: {len(val_samples)}")

    processor = get_processor(BASE_MODEL_CHECKPOINT)

    train_dataset = TESSDataset(train_samples, processor)
    val_dataset = TESSDataset(val_samples, processor)

    # Load the pretrained Wav2Vec2 encoder and attach a fresh classification
    # head sized for our 7 emotion classes. This is the transfer-learning step.
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        BASE_MODEL_CHECKPOINT,
        num_labels=len(EMOTION_LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    # Freeze the convolutional feature encoder (the low-level acoustic
    # feature layers) and only fine-tune the transformer encoder layers and
    # the classification head. This trains faster and works well on the
    # comparatively small TESS dataset.
    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Running final validation...")
    metrics = trainer.evaluate()
    print(f"Final validation accuracy: {metrics.get('eval_accuracy'):.4f}")

    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    trainer.save_model(SAVED_MODEL_DIR)
    processor.save_pretrained(SAVED_MODEL_DIR)
    print(f"Fine-tuned model and processor saved to: {SAVED_MODEL_DIR}")


if __name__ == "__main__":
    main()
