# Academy Object Detection

A minimal PyTorch baseline for detecting warehouse objects in the
[LOCO dataset](https://github.com/tum-fml/loco).

> **Official submission: E12 YOLO11n.** Use `train_yolo.py`, `eval_yolo.py`, and
> `configs/e12_sharpen_20.yaml`. The template's `train.py` and `eval.py` remain available only for
> the earlier SSDLite experiments and do not construct the submitted YOLO model. E13 is documented
> below only as a later, non-submission follow-up and does not replace E12.

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

## Non-submission follow-up: E13 YOLO26s at 896 pixels

E13 was trained and evaluated after E12 had already been selected as the official submission. It
is retained for research completeness, but **E12 remains the submitted model and the default-branch
reproduction target**.

The follow-up replaced YOLO11n with COCO-pretrained YOLO26s, raised the input size from 640 to 896,
reduced the batch size from 16 to 8, increased patience from 20 to 25, and reduced the multi-scale
range from 0.25 to 0.15. It otherwise retained E12's deterministic split, inference thresholds,
and domain-oriented augmentation recipe. Training still used only subsets 2, 3, and 5; subsets 1
and 4 remained locked until the checkpoint was fixed.

The complete implementation and provenance record are isolated on the
[`codex/yolo26s-highres`](https://github.com/JonathanAchkouti/inmindAcademyDetector/tree/codex/yolo26s-highres)
branch. See the
[`E13_FINAL_MODEL_SUMMARY.md`](https://github.com/JonathanAchkouti/inmindAcademyDetector/blob/52ce6d21c6aac95497c5076b9a19064be106ebcd/results/E13_FINAL_MODEL_SUMMARY.md)
record for the full configuration, dataset manifest, augmentation details, hashes, runtime, and
domain-gap analysis.

### E13 aggregate results

| Model | Role | Validation mAP@0.5 | Locked test mAP@0.5 | Parameters | GFLOPs |
| --- | --- | ---: | ---: | ---: | ---: |
| Instructor baseline | Reference | - | 0.2547 | 18,950,729 | 23.825 |
| **E12 YOLO11n** | **Official submission** | **0.6619** | **0.2914** | **2,583,127** | **3.811** |
| E13 YOLO26s | Non-submission follow-up | 0.7498 | 0.3725 | 9,467,115 | 23.298 |

E13 improved locked-test mAP@0.5 by 0.0811 over E12, an 8.11 percentage-point or 27.83%
relative gain. It came within 0.0275 of 0.40 while remaining below both instructor budgets. The
larger model uses 3.66 times E12's parameters and 6.11 times its GFLOPs, so it represents a
different accuracy-efficiency tradeoff rather than a replacement for the submitted E12 result.

### Accuracy-efficiency trade-off

E12 and E13 intentionally illustrate two ends of the assignment's requested trade-off. E12 is the
efficiency-oriented extreme: it minimizes parameters and GFLOPs while still improving on the
instructor baseline. E13 is the accuracy-oriented extreme: it spends almost the full permitted
compute budget to maximize locked-test mAP. Neither model dominates the other because E12 is much
smaller and cheaper, while E13 is substantially more accurate.

An intermediate sweet spot may exist between these two endpoints, but its location is
implementation-dependent rather than a fixed property of the model names. Architecture scale,
input resolution, training recipe, augmentation, and validation-selected post-processing can all
shift the balance between mAP, parameter count, and GFLOPs. For example, YOLO26n could be evaluated
as a smaller intermediate candidate: it may retain part of YOLO26s's accuracy gain while requiring
fewer parameters and GFLOPs. A controlled sweep of YOLO26n resolutions and training settings could
trace the Pareto frontier and select a model with more accuracy than E12 at considerably less cost
than E13. That experiment was outside the submitted scope, so YOLO26n is presented only as a
plausible future candidate, while E12 and E13 remain the measured endpoints.

### E13 per-class results

| Class | Validation AP@0.5 | Locked test AP@0.5 | Test change from E12 |
| --- | ---: | ---: | ---: |
| small load carrier | 0.6959 | 0.3070 | +0.0616 |
| forklift | 0.8212 | 0.1392 | -0.0043 |
| pallet | 0.7204 | 0.6135 | +0.0856 |
| stillage | 0.7565 | 0.5403 | +0.1540 |
| pallet truck | 0.7549 | 0.2623 | +0.1081 |
| **Mean** | **0.7498** | **0.3725** | **+0.0811** |

The validation-to-test drop is 0.3773, confirming that warehouse-domain shift remains the main
limitation. Pallet and stillage transfer best, while rare forklift examples still generalize
poorly. No E13 setting was selected from the locked result, and no post-test tuning was performed.

The selected E13 checkpoint and configuration are identified by SHA-256:

```text
checkpoint: d57a4a6374ab9cd207156196c105c1b4c75a25e059f50d609a75c1cb601c0873
config:     a853788b0baf1e8fccd43b6a8899b1980472cdb5accb58ffe5b6e98042f9a45a
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

## Third-party model

The official E12 submission fine-tunes Ultralytics YOLO11, while the isolated E13 follow-up branch
fine-tunes YOLO26s. Ultralytics publishes its models under AGPL-3.0 and Enterprise licensing
options.
