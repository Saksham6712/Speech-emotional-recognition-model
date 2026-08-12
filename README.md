# Speech Emotion Detection using Wav2Vec2 Transformer

A clean, minimal web app that detects the emotion in a spoken audio clip.
A pretrained **Wav2Vec2** transformer (`facebook/wav2vec2-base`) is
fine-tuned via transfer learning on the **TESS** (Toronto Emotional Speech
Set) dataset to classify audio into 7 emotions, then served through a
Flask backend with a simple white-and-blue Bootstrap frontend.

No CNN. No LSTM. Just transfer learning on top of Wav2Vec2.

## Emotion classes

| Emotion  | Emoji |
|----------|-------|
| Happy    | 😊 |
| Sad      | 😢 |
| Angry    | 😠 |
| Fear     | 😨 |
| Disgust  | 🤢 |
| Surprise | 😮 |
| Neutral  | 😐 |

## Project structure

```
SpeechEmotionDetection/
├── app.py                # Flask backend (routes: / and /predict)
├── train.py               # Fine-tunes Wav2Vec2 on TESS, saves the model
├── predict.py              # Loads the fine-tuned model, runs inference
├── preprocess.py            # Shared audio preprocessing + feature extraction
├── requirements.txt
├── model/
│   └── saved_model/        # Fine-tuned model + processor land here after training
├── uploads/                # Temporary storage for uploaded audio (auto-cleaned)
├── templates/
│   └── index.html          # Single-page UI
├── static/
│   ├── css/style.css
│   └── js/script.js
└── README.md
```

## 1. Setup

**Requirements:** Python 3.10+ recommended.

```bash
cd SpeechEmotionDetection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Get the TESS dataset

Download the TESS dataset (e.g. from Kaggle: "Toronto emotional speech set
(TESS)") and unzip it locally. Its default layout looks like:

```
TESS_DATA_DIR/
    OAF_angry/
        OAF_back_angry.wav
        ...
    OAF_disgust/
    OAF_Fear/
    OAF_happy/
    OAF_neutral/
    OAF_Pleasant_surprise/
    OAF_Sad/
    YAF_angry/
    YAF_disgust/
    ...
```

`train.py` automatically infers the emotion label from folder/file names
(handling variants like `Pleasant_surprise` → `surprise`), so you don't
need to reorganize the files — just point `--data_dir` at the dataset root.

## 3. Fine-tune the model

```bash
python train.py --data_dir /path/to/TESS_DATA_DIR --epochs 8 --batch_size 4
```

This will:
1. Walk the dataset and label every `.wav` file.
2. Preprocess audio (mono, 16 kHz, normalized, silence-trimmed).
3. Fine-tune `facebook/wav2vec2-base` with a 7-class classification head
   (the pretrained feature encoder is frozen; the transformer layers and
   classification head are trained).
4. Evaluate on a held-out validation split.
5. Save the fine-tuned model + processor to `model/saved_model/`.

Training on CPU works but is slow — a GPU is strongly recommended.

## 4. Run the web app

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

- Upload a `.wav`, `.mp3`, `.flac`, `.ogg`, or `.m4a` file (≤ 15 MB).
- Click **Detect Emotion**.
- The predicted emotion, matching emoji, and confidence score (plus a
  full per-class breakdown) will be displayed.

> If `model/saved_model/` is empty, the server still starts, but
> `/predict` will return a clear message asking you to run `train.py`
> first.

## Quick command-line test

You can also test the fine-tuned model directly, without the web UI:

```bash
python predict.py path/to/sample.wav
```

## Notes on preprocessing

Every audio file — during training and during inference — goes through
the same pipeline in `preprocess.py`:

1. **Mono conversion** — multi-channel audio is downmixed.
2. **Resampling to 16 kHz** — matches Wav2Vec2's pretraining rate.
3. **Silence trimming** — removes leading/trailing dead air.
4. **Normalization** — peak-normalizes the waveform to [-1, 1].
5. **Feature extraction** — the Wav2Vec2 processor converts the waveform
   into the normalized input tensor the model expects.

This consistency between training and inference preprocessing is what
keeps predictions accurate.
