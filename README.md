# Academy Object Detection

A minimal PyTorch baseline for detecting warehouse objects in the
[LOCO dataset](https://github.com/tum-fml/loco).

## Project objective

Improve the baseline detector while balancing three goals:

- increase mAP@0.5;
- reduce the total number of model parameters;
- reduce GFLOPs per image.

Your final model should be Pareto-optimal: it is dominated when another submission has no lower
mAP@0.5, no higher total parameter count or GFLOPs, and is strictly better on at least one axis.

## Rules

- Train and fine-tune only on subsets 2, 3, and 5. Make a fixed validation holdout from them for tuning and checkpoint selection.
- Reserve subsets 1 and 4 together for final evaluation. Never train, fine-tune, or tune on them.
- Pretrained models are allowed only when fine-tuned.
- Your submitted repository must reproduce your final reported metrics when cloned and rerun.
- The submitted code must pass the repository's unit tests and Ruff checks.
- Do not modify, bypass, or otherwise tamper with the existing evaluation metrics.

## Setup

```bash
git clone https://github.com/malek-wahidi/AcademyObjectDetection.git
cd AcademyObjectDetection
uv sync
bash scripts/download_loco.sh
```

The download script stores the dataset in `dataset/`, which is the default location in
`config.yaml`. To use another location, pass it to the script and update `data.raw_dir`:

```bash
bash scripts/download_loco.sh /path/to/data
```

The data directory must contain the `rgb/` annotations and `subset-*/` image directories.

For a hosted GPU, upload `AcademyObjectDetection.ipynb` to Colab or Kaggle and follow the
notebook setup.

## Train

```bash
uv run train.py
```

Choose your final checkpoint using the validation holdout. The baseline reserves one image in every five
from each training subset for validation and saves its best checkpoint to `runs/baseline/weights/best.pt`.

## Final submission: E12 YOLO11n

The final model is `e12_sharpen_20`, a COCO-pretrained YOLO11 nano detector fine-tuned at
640 pixels. Its checkpoint was selected only with the fixed validation holdout from subsets 2, 3,
and 5, then frozen before its final evaluation on subsets 1 and 4.

E12 is a controlled ablation of E11: the only change is increasing the probability of mild
sharpening from 0.12 to 0.20. The hypothesis was that occasional sharpening would improve
robustness to blur and variable image quality without permanently altering the dataset or removing
useful colour cues. The allowed validation metric improved from 0.6467 for E11 to 0.6619 for E12.

| Model | Validation mAP@0.5 | Locked test mAP@0.5 | Parameters | GFLOPs |
| ----- | ------------------: | --------------------: | ---------: | -----: |
| Instructor baseline | - | 0.2547 | 18,950,729 | 23.825 |
| E11 YOLO11n | 0.6467 | 0.2817 | 2,624,080 | 6.7 |
| **E12 YOLO11n (final)** | **0.6619** | **0.2914** | **2,624,080** | **6.7** |

E12 improves locked-test mAP@0.5 by 0.0367 over the baseline while using about 86% fewer
parameters and 72% fewer GFLOPs. The complete aggregate and per-class record is versioned in
`results/e12_sharpen_20.json`. The locked checkpoint is identified by SHA-256:

```text
b9d38bf788ff08911d56f8e37220d78a6d2f6cf623d14fa1a26deb5c09be25ca
```

Train or resume the final experiment:

```bash
uv run train_yolo.py --config configs/e12_sharpen_20.yaml
```

Place the selected checkpoint at `runs/e12_sharpen_20/weights/best.pt`, verify its SHA-256, and
reproduce the development evaluation with:

```bash
shasum -a 256 runs/e12_sharpen_20/weights/best.pt

uv run eval_yolo.py \
  --config configs/e12_sharpen_20.yaml \
  --weights runs/e12_sharpen_20/weights/best.pt \
  --split validation \
  --class-metrics \
  --batch-size 2 \
  --num-workers 0
```

The checkpoint is intentionally not stored as a normal Git object. The repository ignores `*.pt`
files to avoid bloating its history. Authorized repository collaborators can download the exact
checkpoint from the private
[E12 final checkpoint release](https://github.com/JonathanAchkouti/inmindAcademyDetector/releases/tag/e12-final):

```bash
mkdir -p runs/e12_sharpen_20/weights
gh release download e12-final \
  --repo JonathanAchkouti/inmindAcademyDetector \
  --pattern e12_sharpen_20_best.pt \
  --dir runs/e12_sharpen_20/weights
mv runs/e12_sharpen_20/weights/e12_sharpen_20_best.pt \
  runs/e12_sharpen_20/weights/best.pt
shasum -a 256 runs/e12_sharpen_20/weights/best.pt
```

## COCO-pretrained YOLO experiments

`e11_yolo11n_domain_aug` established the accuracy-efficiency candidate. It fine-tunes the
COCO-pretrained YOLO11 nano detector at 640 pixels with aspect-ratio-preserving letterboxing. Its
domain-generalization pipeline randomly uses grayscale, Gaussian or motion blur, mild sharpening,
gamma changes, brightness/contrast changes, HSV jitter, mosaic, and small geometric perturbations.
It never permanently removes colour or attempts to reconstruct information from every blurry image.

Build a lightweight YOLO view and verify that the default manifest contains only subsets 2, 3,
and 5:

```bash
uv run yolo_data.py --raw-dir dataset --output-dir /tmp/loco-yolo
```

The Colab configuration copies permitted development images to the runtime's fast local disk.
Checkpoints remain persistent in Drive, but repeated training reads do not pay the mounted-Drive
latency penalty.

Train or resume E11:

```bash
uv run train_yolo.py --config configs/e11_yolo11n_domain_aug.yaml
```

Ultralytics saves resumable and best checkpoints under
`runs/e11_yolo11n_domain_aug/weights/`. Evaluate the selected checkpoint with the repository's
unchanged Torchmetrics mAP@0.5 implementation:

```bash
uv run eval_yolo.py \
  --config configs/e11_yolo11n_domain_aug.yaml \
  --weights runs/e11_yolo11n_domain_aug/weights/best.pt \
  --split validation \
  --class-metrics \
  --batch-size 2 \
  --num-workers 0
```

Use `YOLODomainGeneralization.ipynb` in Colab for the guarded end-to-end workflow. Its training
manifest excludes subsets 1 and 4, and its locked test cell is disabled by default.

## Evaluate

```bash
uv run eval.py --weights runs/baseline/weights/best.pt
```

After selecting a checkpoint, run evaluation once on subsets 1 and 4. It reports mAP@0.5, parameter counts, GFLOPs per image, and batch-one latency. Submissions
are compared using mAP@0.5, total parameters, and GFLOPs. Latency is informational because it
depends on the hardware.

## Inspect predictions

```bash
uv run predict.py --weights runs/baseline/weights/best.pt
```

Prediction images are saved to `runs/predictions/`.

## Baseline results

Stock `fasterrcnn_mobilenet_v3_large_fpn` with LOCO's five classes plus background:

| mAP@0.5 | Parameters | GFLOPs |
| ------- | ---------- | ------ |
| 0.2547  | 18,950,729 | 23.825 |

Pareto comparison requires final-evaluation mAP@0.5 at least as high as the recorded baseline mAP.

## Checks

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
```
