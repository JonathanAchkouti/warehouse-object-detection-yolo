# Academy Object Detection

A minimal PyTorch baseline for detecting warehouse objects in the
[LOCO dataset](https://github.com/tum-fml/loco).

> **Current training candidate:** use `train_yolo.py`, `eval_yolo.py`, and
> `configs/e13_yolo26s_896.yaml` for the unmeasured YOLO26s high-resolution experiment. E12 remains
> the last locked and reported model until E13 has been selected on development validation. The
> template's `train.py` and `eval.py` remain available only for the earlier SSDLite experiments.

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
git clone https://github.com/JonathanAchkouti/inmindAcademyDetector.git
cd inmindAcademyDetector
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

## E13 candidate: YOLO26s at 896 pixels

E13 is the next development-only candidate. It replaces YOLO11n with COCO-pretrained YOLO26s and
raises the inference size from 640 to 896 pixels. It otherwise retains E12's data split, metric,
confidence/NMS limits, and domain-oriented augmentation recipe. Batch size is reduced to 8 and the
multi-scale range to 0.15 so the largest randomized training batches remain practical on a T4.

E13 has no claimed validation or locked-test result yet. Do not add it to the experiment ledger or
run subsets 1 and 4 until its checkpoint and settings have been selected using only the fixed
development validation split.

Before training, run the conservative budget gate. It uses a blank 1920x1080 probe and the larger
80-class COCO head; fine-tuning replaces that head with the smaller five-class LOCO head. The command
downloads `yolo26s.pt` on first use and fails if the deployed model exceeds either instructor baseline
budget:

```bash
uv run preflight_yolo.py --config configs/e13_yolo26s_896.yaml
```

With Ultralytics 8.4.121, the conservative COCO-head preflight measured 9,496,140 parameters and
23.478239 GFLOPs, so the 896-pixel candidate passed both limits. These are architecture-gate
figures, not final submission measurements; repeat the unchanged official efficiency evaluation on
the selected five-class checkpoint.

Then run an isolated one-epoch smoke test and start or resume the full experiment:

```bash
uv run train_yolo.py --config configs/e13_yolo26s_896.yaml --smoke
uv run train_yolo.py --config configs/e13_yolo26s_896.yaml
```

Select the checkpoint only from validation on subsets 2, 3, and 5:

```bash
uv run eval_yolo.py \
  --config configs/e13_yolo26s_896.yaml \
  --weights runs/e13_yolo26s_896/weights/best.pt \
  --split validation \
  --class-metrics \
  --batch-size 2 \
  --num-workers 0
```

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
| E11 YOLO11n | 0.6467 | 0.2817 | 2,583,127 | - |
| **E12 YOLO11n (final)** | **0.6619** | **0.2914** | **2,583,127** | **3.811** |

E12 improves locked-test mAP@0.5 by 0.0367 over the baseline while using about 86% fewer
parameters and 84% fewer GFLOPs. Efficiency is measured with the repository's unchanged
`count_parameters` and `measure_gflops` functions after deployment-time batch-normalization
fusion, using the first locked-test image as the fixed probe. The complete aggregate and per-class record is versioned in
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

Reproduce only the graded efficiency figures on the locked probe without recomputing test accuracy:

```bash
uv run measure_yolo_efficiency.py \
  --config configs/e12_sharpen_20.yaml \
  --weights runs/e12_sharpen_20/weights/best.pt \
  --split test
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

### Domain-generalization result

The same-warehouse validation holdout substantially overestimates performance on unseen
warehouses: overall mAP@0.5 falls from 0.6619 to 0.2914. This is the expected LOCO domain shift,
not an image-level split leak; checkpoint selection used only subsets 2, 3, and 5.

| Class | Validation AP@0.5 | Locked test AP@0.5 | Change |
| ----- | ----------------: | -----------------: | -----: |
| small load carrier | 0.5657 | 0.2454 | -0.3203 |
| forklift | 0.7673 | 0.1435 | -0.6238 |
| pallet | 0.6004 | 0.5279 | -0.0725 |
| stillage | 0.6994 | 0.3863 | -0.3131 |
| pallet truck | 0.6765 | 0.1542 | -0.5223 |

Forklift and pallet-truck AP fall most because they are rare and their appearance and frequency
shift sharply across warehouses. The final result therefore measures cross-warehouse robustness,
while the validation result should be interpreted only as an in-domain model-selection signal.

### Experiment ledger

The committed [experiment ledger](results/ledger.csv) records every run for which metrics were
retained. Configuration files without ledger rows were prepared candidate ablations, not completed
runs, and are not presented as empirical evidence.

| Run | Main question | Validation mAP@0.5 | Locked test mAP@0.5 | Decision |
| --- | ------------- | ------------------: | --------------------: | -------- |
| E03 | Can tuned P3+SSDLite learn at 512 px in 15 epochs? | 0.1860 | - | Extend training and improve robustness. |
| E09 | Do a longer schedule, augmentation, and rare-image repetition generalize? | 0.3301 | 0.1320 | Reject: severe warehouse-domain failure. |
| E11 | Does a COCO-pretrained nano YOLO improve the accuracy-efficiency frontier? | 0.6467 | 0.2817 | Accept as the first baseline-beating candidate. |
| E12 | Does increasing mild sharpening probability from 0.12 to 0.20 help? | 0.6619 | 0.2914 | Select as final from validation, then lock. |

E12 changes one variable relative to E11 and improves validation by 0.0152 and locked test by
0.0097. E09 is retained in the ledger because its large validation-to-test reversal motivated the
move to stronger COCO pretraining and domain-oriented image augmentation.

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

## Third-party model

This academic project fine-tunes Ultralytics YOLO11 and YOLO26 checkpoints. Ultralytics publishes
them under [AGPL-3.0 and Enterprise licensing options](https://docs.ultralytics.com/models/yolo26/).
