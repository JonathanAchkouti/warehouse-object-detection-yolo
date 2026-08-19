# E13 Final Model Summary: YOLO26s at 896 Pixels

## 1. Executive summary

E13 is a COCO-pretrained YOLO26s detector fine-tuned on the permitted LOCO development subsets at
an input size of 896 pixels.

| Metric | E13 result |
| --- | ---: |
| Official validation mAP@0.5 | 0.7498 |
| Locked-test mAP@0.5 | **0.3725** |
| Total deployed parameters | 9,467,115 |
| GFLOPs per image | 23.298 |
| Batch-one latency on Tesla T4 | 63.36 ms |

E13 improves the previous E12 locked-test result from 0.2914 to 0.3725:

- absolute improvement: **+0.0811 mAP**;
- percentage-point improvement: **+8.11 points**;
- relative improvement: **+27.83%**;
- distance from the stated 40% target: **2.75 percentage points**.

E13 also exceeds the instructor baseline test mAP of 0.2547 by 0.1178 while using fewer
parameters and slightly fewer GFLOPs. It therefore dominates the instructor baseline across
accuracy, parameter count, and computation.

## 2. Model identity and provenance

| Item | Value |
| --- | --- |
| Experiment | `e13_yolo26s_896` |
| Architecture | YOLO26s |
| Pretrained initialization | `yolo26s.pt`, pretrained on COCO |
| Fine-tuned classes | Five LOCO warehouse-object classes |
| Input size | 896 pixels |
| Selected checkpoint | `runs/e13_yolo26s_896/weights/best.pt` |
| Checkpoint size | 20,381,189 bytes |
| Checkpoint SHA-256 | `d57a4a6374ab9cd207156196c105c1b4c75a25e059f50d609a75c1cb601c0873` |
| Configuration SHA-256 | `a853788b0baf1e8fccd43b6a8899b1980472cdb5accb58ffe5b6e98042f9a45a` |
| Git branch | `codex/yolo26s-highres` |
| Git commit used for training | `415163dc954e75b7d469d7efb552ae1601c46dec` |

Persistent Colab artifacts are stored under:

```text
/content/drive/MyDrive/inmindAcademyDetector/runs/e13_yolo26s_896/
```

The consolidated Colab record is:

```text
/content/drive/MyDrive/inmindAcademyDetector/runs/e13_yolo26s_896/final_model_record.json
```

## 3. Data split and leakage prevention

Training and model selection used only LOCO subsets 2, 3, and 5.

| Subset | Training images | Validation images | Total |
| --- | ---: | ---: | ---: |
| 2 | 856 | 214 | 1,070 |
| 3 | 698 | 175 | 873 |
| 5 | 701 | 176 | 877 |
| **Total** | **2,255** | **565** | **2,820** |

The split is deterministic: one image in every five from each development subset is reserved for
validation. The remaining four images are used for training.

The locked test consists of subsets 1 and 4, containing 2,277 images and approximately 64,492
annotated objects. These subsets were excluded from dataset preparation, training, augmentation,
checkpoint selection, and hyperparameter selection. The training manifest explicitly recorded
`include_test: false`.

The first locked-test execution finished but its Colab output was not persisted. The exact same
checkpoint, configuration, and evaluator were rerun solely to save the output. No model or
hyperparameter changes were made between the executions, and the test result was not used for
further tuning.

## 4. Development-set class distribution

The training and validation subsets contain 86,936 valid objects.

| Class | Objects | Share |
| --- | ---: | ---: |
| pallet | 68,148 | 78.389% |
| small load carrier | 13,240 | 15.230% |
| stillage | 3,400 | 3.911% |
| pallet truck | 1,674 | 1.926% |
| forklift | 474 | 0.545% |
| **Total** | **86,936** | **100%** |

This distribution is highly imbalanced. Forklift represents only 0.545% of development
annotations, while pallet represents more than 78%. However, official mAP macro-averages the five
class AP values, so every class contributes equally to the final score.

## 5. Architecture and implementation

E13 replaced E12's YOLO11n model with YOLO26s and increased inference resolution from 640 to 896
pixels. The higher resolution was intended to improve detection of small, distant, blurred, and
partially occluded warehouse objects.

The model was initialized from the COCO-pretrained `yolo26s.pt` checkpoint. During fine-tuning, the
original 80-class COCO detection head was replaced with a five-class LOCO head.

The implementation preserves image aspect ratio. Images are converted to RGB tensors and passed
to Ultralytics prediction, which applies aspect-ratio-preserving letterboxing rather than
stretching every image into a square.

YOLO class indices 0-4 are converted back to the repository's official foreground labels 1-5.
Label 0 remains reserved for background in the Torchvision/TorchMetrics interface.

Invalid boxes, crowd annotations, and boxes with zero or negative dimensions are excluded.
COCO-format boxes are clipped to image boundaries and converted into normalized YOLO
center-width-height format for training.

## 6. Inference configuration

| Setting | Value |
| --- | ---: |
| Input size | 896 |
| Confidence threshold | 0.001 |
| NMS IoU threshold | 0.70 |
| Maximum YOLO detections per image | 300 |
| Official metric IoU threshold | 0.50 |
| Evaluation batch size | 2 |
| Evaluation workers | 0 |

The low confidence threshold preserves low-confidence candidates across the precision-recall
curve. The evaluator then computes AP from their confidence ranking.

YOLO may retain up to 300 detections before reporting. The unchanged TorchMetrics evaluator retains
its standard maximum of 100 detections for the official mAP calculation.

## 7. Training configuration

| Setting | Value |
| --- | ---: |
| Requested epochs | 100 |
| Completed epochs | 100 |
| Batch size | 8 |
| Workers | 2 |
| Random seed | 0 |
| Patience | 25 |
| Resumed | false |
| Automatic mixed precision | true |
| Deterministic training | true |
| Optimizer | Ultralytics `auto` |
| Learning-rate schedule | cosine |
| Mosaic disabled for final | 10 epochs |
| Multi-scale range | 0.15 |
| Checkpoint selection | Development validation only |

The batch size was reduced from E12's 16 to 8 because of the larger model and 896-pixel input. The
multi-scale range was reduced from 0.25 to 0.15 to keep the largest randomized batches practical
on a Tesla T4.

Training took approximately 13,463 seconds, or about 3 hours and 44 minutes. The run began from
COCO-pretrained weights and did not resume from an earlier E13 checkpoint.

## 8. Standard augmentation

| Augmentation | Configuration |
| --- | --- |
| Horizontal flip | Probability 0.50 |
| HSV hue | 0.03 |
| HSV saturation | 0.40 |
| HSV value | 0.40 |
| Rotation | Up to +/-3 degrees |
| Translation | 0.10 |
| Scale | 0.40 |
| Shear | Up to 1 degree |
| Perspective | 0.0005 |
| Mosaic | Probability 0.50 |
| MixUp | Probability 0.05 |
| Multi-scale | 0.15 |

Mosaic was closed during the final ten epochs so that the model finished training on ordinary
image layouts.

## 9. Domain-generalization augmentation

E13 inherited E12's complete domain-oriented augmentation pipeline.

| Augmentation | Probability | Parameters |
| --- | ---: | --- |
| Grayscale | 0.20 | Occasional removal of color information |
| Gaussian or motion blur | 0.18 | Kernel 3-7; Gaussian sigma 0.1-2.0 |
| Mild sharpening | 0.20 | Alpha 0.10-0.30; lightness 0.8-1.2 |
| Random gamma | 0.20 | Gamma range 70-140 |
| Brightness/contrast | 0.30 | Limits +/-0.30 |

These transformations are probabilistic rather than permanent. The model therefore continues to
see normal RGB images while learning robustness to warehouse changes in color, lighting,
exposure, blur, motion, and sharpness.

## 10. Native Ultralytics training metrics

The best epoch recorded in `results.csv` was epoch 79.

| Native training metric | Best logged value |
| --- | ---: |
| Precision | 0.77909 |
| Recall | 0.69100 |
| mAP@0.5 | 0.75924 |
| mAP@0.5-0.95 | 0.38912 |

Ultralytics' post-training evaluation of the selected `best.pt` checkpoint reported:

| Metric | Value |
| --- | ---: |
| Precision | 0.774340 |
| Recall | 0.697361 |
| mAP@0.5 | 0.759263 |
| mAP@0.5-0.95 | 0.389441 |
| Fitness | 0.389441 |

The small difference between the epoch CSV and the post-training result comes from the final
evaluation of the saved best checkpoint.

The assignment's graded metric is not Ultralytics mAP@0.5-0.95. The graded result is the
repository's unchanged TorchMetrics mAP@0.5.

## 11. Official validation and locked-test metrics

| Class | E12 validation | E13 validation | Validation change | E12 test | E13 test | Test change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small load carrier | 0.5657 | 0.6959 | +0.1302 | 0.2454 | 0.3070 | +0.0616 |
| Forklift | 0.7673 | 0.8212 | +0.0539 | 0.1435 | 0.1392 | -0.0043 |
| Pallet | 0.6004 | 0.7204 | +0.1200 | 0.5279 | 0.6135 | +0.0856 |
| Stillage | 0.6994 | 0.7565 | +0.0571 | 0.3863 | 0.5403 | +0.1540 |
| Pallet truck | 0.6765 | 0.7549 | +0.0784 | 0.1542 | 0.2623 | +0.1081 |
| **Mean mAP@0.5** | **0.6619** | **0.7498** | **+0.0879** | **0.2914** | **0.3725** | **+0.0811** |

E13 improved four of five locked-test classes relative to E12. The largest improvements were:

- stillage: +0.1540;
- pallet truck: +0.1081;
- pallet: +0.0856;
- small load carrier: +0.0616.

Forklift declined slightly by 0.0043. Its development set contains only 474 annotations, and its
appearance and frequency change sharply between warehouses.

## 12. Cross-warehouse domain gap

| Class | E13 validation | E13 test | Change | Test retention |
| --- | ---: | ---: | ---: | ---: |
| Small load carrier | 0.6959 | 0.3070 | -0.3889 | 44.12% |
| Forklift | 0.8212 | 0.1392 | -0.6820 | 16.95% |
| Pallet | 0.7204 | 0.6135 | -0.1069 | 85.16% |
| Stillage | 0.7565 | 0.5403 | -0.2162 | 71.42% |
| Pallet truck | 0.7549 | 0.2623 | -0.4926 | 34.75% |
| **Mean** | **0.7498** | **0.3725** | **-0.3773** | **49.68%** |

The same-warehouse validation score substantially overestimates unseen-warehouse performance. E13
retains approximately half of its validation mAP on the locked warehouses.

Pallet transfers best because it is abundant across the development warehouses. Forklift
transfers worst because it is extremely rare and has a particularly large warehouse-frequency
shift.

The aggregate E13 domain drop of 0.3773 is similar to E12's drop of approximately 0.3705. The
larger model and higher input resolution raise both validation and test performance, but they do
not eliminate the underlying warehouse-domain shift.

## 13. Efficiency and assignment-budget compliance

| Efficiency metric | E13 | Instructor limit | Remaining margin |
| --- | ---: | ---: | ---: |
| Total parameters | 9,467,115 | 18,950,729 | 9,483,614 |
| GFLOPs per image | 23.298 | 23.825 | 0.527 |

E13 uses:

- 50.04% fewer parameters than the instructor baseline;
- 2.21% fewer GFLOPs than the instructor baseline;
- 3.66 times E12's parameters;
- 6.11 times E12's GFLOPs.

The model is therefore close to the assignment's computation ceiling but remains compliant.

The pre-training architecture gate used the larger 80-class COCO head and measured 9,496,140
parameters and 23.478239 GFLOPs. After fine-tuning, the five-class LOCO head reduced the final
deployed network to 9,467,115 parameters and 23.298 GFLOPs.

The reported `trainable: 0` value is a post-inference deployment artifact. Ultralytics fuses batch
normalization into convolutions and places the deployed network in evaluation state. It does not
mean that zero parameters were trained; the detector was fully fine-tuned before deployment
fusion. The assignment grades total deployed parameters.

## 14. Latency

| Split used for probe | Median batch-one latency |
| --- | ---: |
| Validation | 63.52 ms |
| Locked test | 63.36 ms |

Latency was measured on a Tesla T4 using CUDA, after two warm-up passes, as the median of ten timed
batch-one passes. Latency is hardware-dependent and is not part of the assignment grade.

## 15. Evaluation implementation

Official accuracy uses the repository's unchanged TorchMetrics `MeanAveragePrecision`
implementation with IoU fixed at 0.5.

A prediction counts as correct only when:

1. its class matches the ground-truth class; and
2. its intersection-over-union with a ground-truth box is at least 0.5.

AP is computed independently for each of the five classes and then macro-averaged. This prevents
the abundant pallet class from numerically overwhelming rare classes such as forklift.

GFLOPs are measured by tracing the real deployed prediction path with PyTorch `FlopCounterMode`.
The first fixed split image is used as the probe, and aspect-ratio-preserving letterboxing produces
a rectangular inference shape. This is why the official number differs from generic square-input
YOLO estimates.

## 16. Runtime environment

| Component | Version |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| Ultralytics | 8.4.123 |
| CUDA | 12.8 |
| GPU | Tesla T4 |

The repository requires Ultralytics 8.4.121 or newer within major version 8. The actual training
runtime used 8.4.123.

## 17. Reproducibility and verification

The implementation includes:

- a dedicated E13 configuration inheriting E12's validated data and augmentation recipe;
- a conservative pre-training parameter/GFLOP gate;
- a Colab workflow that keeps checkpoints and results in Google Drive;
- deterministic splitting and training settings;
- an adapter between Ultralytics output and the repository's official detection interface;
- official aggregate and per-class evaluation;
- checkpoint and configuration SHA-256 hashes;
- local unit, formatting, and lint checks;
- passing GitHub Actions verification.

The implementation branch passed all 27 repository tests, Ruff formatting, and Ruff linting.

## 18. Final conclusion

E13 is the strongest model evaluated in this project. It increases locked-test mAP@0.5 from 0.2914
to 0.3725 while remaining below both instructor efficiency limits.

It falls 0.0275 short of an exact 0.40 result but clearly reaches the intended "around 40%" range.
The higher-resolution YOLO26s architecture substantially improves pallet, stillage, pallet-truck,
and small-load-carrier detection on unseen warehouses.

The main remaining weakness is forklift generalization. This is consistent with severe class
imbalance and warehouse-frequency shift rather than a failure to fit the development data:
forklift validation AP is 0.8212, but locked-test AP is only 0.1392.

E13 should now be treated as the final frozen model. No further architecture, augmentation,
threshold, or hyperparameter decisions should be made using subsets 1 and 4.
