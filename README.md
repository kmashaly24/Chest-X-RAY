# Chest X-Ray Disease Classification (Vision Transformer)

This project classifies frontal chest X-rays into 14 pathologies using a
Vision Transformer (**ViT-B/16**) fine-tuned on the NIH **ChestX-ray14** dataset.
It is a clean conversion of `Chest_X_RAY.ipynb` into plain Python scripts with a
few changes.


---

## Project structure

| File | Purpose |
| --- | --- |
| `config.py` | Shared constants (label order, transforms, image size, device). |
| `model.py` | `ChestViT` (ViT-B/16 + MLP head) and `AsymmetricLoss`. |
| `data_utils.py` | Kaggle download, `Data_Entry_2017.csv` parsing, patient-level split, dataset/DataLoaders. |
| `train.py` | Download -> train -> save best model + F1 thresholds. |
| `predictor.py` | Loads the saved model + thresholds and runs inference. |
| `app.py` | Gradio web UI for interactive prediction. |
| `requirements.txt` | Python dependencies. |
| `setup_env.bat` | Creates the conda environment and installs requirements. |
| `chest_xray_run/` | Saved artifacts from the notebook (best model + thresholds). |

---

## Setup (new environment)

A fresh conda environment named `chest_xray` (Python 3.11) is recommended.
Run the helper script (Windows):

```bat
setup_env.bat
```

or do it manually:

```bat
conda create -n chest_xray python=3.11 pip -y
conda activate chest_xray
pip install -r requirements.txt
```

**GPU note:** on Linux, `pip install torch torchvision` installs CUDA wheels by
default, so the code uses the GPU automatically when available
(`torch.cuda.is_available()`). On Windows the default PyPI wheels are CPU-only.
To force a specific CUDA build, install PyTorch first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## Downloading the data

The dataset is downloaded from Kaggle: [NIH Chest X-rays (ChestX-ray14)](https://www.kaggle.com/datasets/nih-chest-xrays/data) (`nih-chest-xrays/data`, ~4.4 GB compressed). It requires Kaggle credentials, provided either as environment
variables (`KAGGLE_USERNAME`, `KAGGLE_KEY`) or a `~/.kaggle/kaggle.json` file.

The download happens automatically inside `train.py`. The images are **streamed
directly from `data.zip`** (no unpacking), which avoids expanding the ~45 GB of
raw images to disk.

---

## Training

```bash
python train.py --download-path data --output-dir checkpoints --epochs 15
```

Key options:

| Flag | Description |
| --- | --- |
| `--download-path` | Folder that will hold `data.zip` and the CSV (default `data`). |
| `--output-dir` | Where `run_best.pth`, `run_checkpoint.pth` and `f1_thresholds.npy` are saved. |
| `--epochs` | Number of epochs (default 15). |
| `--batch-size` | Batch size (default 64). |
| `--workers` | DataLoader workers (default 0; use 2+ on Linux/GPU). |
| `--patience` | Early-stop patience (default 3). |
| `--skip-download` | Skip downloading (data already present). |
| `--eval-test` | Also evaluate the held-out test set at the end. |

What the script does:

1. Downloads `data.zip` and extracts `Data_Entry_2017.csv`.
2. Maps `Image Index` -> full zip path.
3. Splits by **patient ID** (70/15/15) to avoid data leakage.
4. Builds a **ViT-B/16** model (last 4 encoder blocks + final LayerNorm
   unfrozen) with an MLP head, trained with **AsymmetricLoss**,
   **AdamW** (discriminative LR: head 1e-3 / backbone 1e-4), **mixed
   precision**, gradient clipping, and a **warmup -> cosine** LR schedule.
5. Saves the best checkpoint (by validation mAP) and, after training,
   computes per-class **F1-maximizing thresholds** on the validation set and
   saves them to `f1_thresholds.npy`.

> **Important:** training is GPU-oriented. On CPU it is extremely slow (the
> full dataset has ~112k images). The Gradio app below does **not** require
> retraining - it uses the already-saved model in `chest_xray_run/`.

## Inference (Gradio)

Start the app (uses the saved best model + thresholds):

```bash
python app.py
```

Then open http://127.0.0.1:7860 and upload a chest X-ray. The model returns a
list of diseases (with probabilities) whose score is **at or above** each
class's tuned threshold. If every disease is below its threshold, it prints
**No Finding (no disease detected)**.

You can also point the app at different artifacts:

```bash
python app.py --model-path checkpoints/run_best.pth --thresholds-path checkpoints/f1_thresholds.npy
```

---

## Model and thresholds (from the notebook)

**Download the saved artifacts:** the model and checkpoint files are too large for Git. Download them from [Google Drive](https://drive.google.com/drive/folders/1DzHvjISfvvSFqyhJKyE3Qgz8qH8wBe9v?usp=drive_link) and place them in the `chest_xray_run/` folder.

- Model: `chest_xray_run/run2_best.pth` (best by validation mAP).
- Thresholds: `chest_xray_run/f1_thresholds.npy` (per-class F1-maximizing).
- Best checkpoint info: **epoch 7**, validation **mAP 0.2088**.


---

## Results on the test set (from the notebook)

Evaluating the best model on the held-out test set with the validation-derived
thresholds gives:

- **Test mAP**: 0.2203
- **Test macro-AUC**: 0.8081
- **Micro-F1**: 0.3443 | **Macro-F1**: 0.2779
- **Micro-Precision**: 0.2851 | **Micro-Recall**: 0.4345

Per-class AUC / Average Precision on the test set (sorted by AP):

| Label | AUC | AP | Positives |
| --- | --- | --- | --- |
| Effusion | 0.868318 | 0.489327 | 1892 |
| Infiltration | 0.698229 | 0.333465 | 2932 |
| Emphysema | 0.884203 | 0.307962 | 454 |
| Atelectasis | 0.787162 | 0.304273 | 1677 |
| Pneumothorax | 0.851639 | 0.301490 | 725 |
| Cardiomegaly | 0.886218 | 0.266367 | 439 |
| Mass | 0.783771 | 0.245198 | 727 |
| Nodule | 0.712154 | 0.167779 | 945 |
| Edema | 0.883759 | 0.149719 | 374 |
| Consolidation | 0.792008 | 0.140407 | 730 |
| Hernia | 0.887613 | 0.136102 | 27 |
| Pleural_Thickening | 0.758813 | 0.113502 | 484 |
| Fibrosis | 0.791253 | 0.094723 | 271 |
| Pneumonia | 0.728448 | 0.034520 | 214 |