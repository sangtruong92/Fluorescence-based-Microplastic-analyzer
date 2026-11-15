# Benchmark Evaluation Before and After the Repair

## 1. Executive conclusion

The legacy benchmark reported **F1 = 0.9837** for YOLO26m, but its formula compared only the number of predictions and Ground Truth (GT) objects in each image. A prediction at the wrong location or with the wrong class could still be counted as a true positive.

After the repair, each prediction is matched to at most one unused GT object of the same class. Predictions are processed in descending confidence order and must satisfy **IoU ≥ 0.50**. With the same dataset, model, confidence threshold, and prediction count, the valid ML F1 score is **0.6090**.

| ML metric | Before: count-only | After: class + IoU | Difference |
|---|---:|---:|---:|
| TP | 9,914 | 6,138 | −3,776 |
| FP | 243 | 4,019 | +3,776 |
| FN | 86 | 3,862 | +3,776 |
| Precision | 0.9761 | 0.6043 | −0.3718 |
| Recall | 0.9914 | 0.6138 | −0.3776 |
| F1 | **0.9837** | **0.6090** | **−0.3747** |

The **37.47 percentage-point** decrease in F1 does not mean that the model became worse. The model was not trained or fine-tuned in this experiment. The repaired benchmark measures localization, bounding-box size, duplicate-detection, and class errors that the old formula concealed.

![Comparison of ML metrics before and after the evaluator repair](../benchmark_results/evidence/after_fix_20260825_231305/before_after_overall_metrics.png)

## 2. Controlled comparison conditions

| Component | Locked value |
|---|---|
| Dataset | `benchmark_results/dataset/20260824` |
| Images / GT files | 200 / 200 |
| Total valid GT objects with bounding boxes | 10,000 |
| Microbead/Pellet | 3,302 |
| Fiber/Filament | 3,364 |
| Irregular, including Fragment | 3,334 |
| Dataset manifest SHA-256 | `d5d55d44aa9a8cff6f39a292a7f99e186887a15b8c0c5ac300399b8e5f4d5a23` |
| Model | `src/ml/Yolo26m/best.pt` |
| Model SHA-256 | `c308593934437df75a9ea34ef0a6cca11337fcbe66159b2293d295be9807da5c` |
| Confidence threshold | 0.25 |
| After-fix IoU threshold | 0.50 |
| Raw YOLO predictions | 10,157 |

The runner checked the dataset, model, and baseline hashes before and after execution. The sidecar confirms 200 image records, 200 ML lineage records, and spatial metrics with `available=true`.

The legacy baseline HTML did not store a prediction sidecar, so the exact `001406` inference cannot be replayed. The after-fix result therefore comes from a new inference using the same dataset, model, and confidence threshold. Stability was checked because the count-only formula applied to the new snapshot reproduced the unrounded precision, recall, and F1 values of the legacy baseline.

## 3. What was wrong with the legacy benchmark

### 3.1 The count-only formula fabricated true positives

The legacy code used this logic:

```python
tp = min(detected, ground_truth)
fp = max(0, detected - ground_truth)
fn = max(0, ground_truth - detected)
```

It did not consider the spatial relationship between bounding boxes. If an image contained 50 predictions and 50 GT objects, the formula returned TP=50, FP=0, and FN=0 regardless of prediction locations.

The following evidence contains four images with 50 predictions and 50 GT objects each. Count-only evaluation reports F1=1.000, while bounding-box matching finds only 18–20 true positives per image.

![Images with perfect count F1 but mismatched bounding boxes](../benchmark_results/evidence/pre_fix_20260825_001406/spatial_failure_contact_sheet.png)

Overlay legend: green indicates matched GT, yellow indicates missed GT, cyan indicates a matched prediction, and red indicates a false positive.

![Bounding-box overlay for synthetic_002](../benchmark_results/evidence/pre_fix_20260825_001406/overlay_synthetic_002.png)

### 3.2 The report mixed YOLO classes with post-processing morphology

The model has three classes, but the legacy report used `feature['shape']` produced by `ShapeAnalyzer` inside each ROI. As a result, the ML chart contained both `Fragment` and `Irregular`, which did not match the YOLO output ontology.

After the repair:

- ML detection metrics and class distributions use raw YOLO predictions.
- GT or heuristic `Fragment` labels are normalized to `Irregular` in the benchmark ontology.
- Post-processing morphology is no longer presented as the model's output class.

### 3.3 Color and area could be misinterpreted

YOLO26m does not predict color. Color is produced by `ColorAnalyzer` inside each ROI. The legacy report omitted `Unknown/Other`, causing the total ML color count to be much lower than the prediction count without explaining coverage.

The repaired report correctly calls this result **ROI color post-processing**, retains `Unknown`, and records `color_coverage`. The area chart is identified as an unmatched descriptive distribution and must not be used to infer per-object area error.

### 3.4 Averages were truncated to integers

10,157 / 200 = **50.785 predictions per image**, but the legacy report displayed `50`. The repaired payload retains the decimal value and also stores `detected_total=10157`.

### 3.5 The legacy baseline lacked replay provenance

The `001406` HTML does not contain raw bounding boxes, input order, model and dataset hashes, or evaluator configuration. After the repair, each report has a `.benchmark.json` sidecar containing:

- HTML, model, input-image, and GT hashes;
- input order;
- confidence and IoU thresholds;
- parsed GT data;
- raw YOLO predictions, post-processing features, and rejected predictions;
- matching results by image and class; and
- count-only results marked `diagnostic_only`, not as a quality metric.

## 4. Exact benchmark repairs

1. Parse and validate GT bounding boxes in pixel `xywh` format. Empty, negative, out-of-image boxes or unsupported classes cause an explicit benchmark failure.
2. Normalize the ontology to three classes: `Microbead/Pellet`, `Fiber/Filament`, and `Irregular`.
3. For each image, sort predictions by descending confidence with a stable original-index tie-break.
4. Match a prediction only to the unused GT object of the same class with the highest IoU, requiring IoU ≥ 0.50.
5. Count unmatched predictions as FP and unmatched GT objects as FN. Aggregate TP/FP/FN across the dataset before calculating micro precision, recall, and F1.
6. Report per-class metrics and explicitly describe the evaluation contract.
7. If complete GT bounding boxes are unavailable, return `available=false` and render `N/A`; never substitute count agreement.
8. Persist the report and evidence snapshot atomically. A report is not considered replayable if sidecar creation fails.

Automated tests cover perfect matches, wrong locations, wrong classes, duplicates, empty sets, GT parsing, invalid boxes, unavailable status, single-inference lineage, and `N/A` report rendering.

## 5. After-fix results by class

| Class | GT | Predictions | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Microbead/Pellet | 3,302 | 3,413 | 2,674 | 739 | 628 | 0.7835 | 0.8098 | **0.7964** |
| Fiber/Filament | 3,364 | 3,465 | 399 | 3,066 | 2,965 | 0.1152 | 0.1186 | **0.1169** |
| Irregular | 3,334 | 3,279 | 3,065 | 214 | 269 | 0.9347 | 0.9193 | **0.9270** |

![After-fix ML metrics by class](../benchmark_results/evidence/after_fix_20260825_231305/after_fix_ml_per_class.png)

The most important observation is that the Fiber/Filament prediction count (3,465) is close to the GT count (3,364), but only 399 predictions match the correct class and IoU requirement. This clearly demonstrates that **similar aggregate distributions do not imply correct object detection**.

## 6. Why after-fix F1 0.609 differs from the earlier independent audit F1 0.526

Both evaluations require the same class, one-to-one matching, and IoU ≥ 0.50, but they use different greedy ordering policies:

- The earlier independent pre-fix audit prioritized the highest-IoU pair across the entire image and produced F1=0.5257.
- The production after-fix evaluator prioritizes predictions by descending confidence and then selects the unused GT with the highest IoU, producing F1=0.6090.

In crowded images or where boxes overlap, an earlier match changes which GT objects remain available, so the number of true positives can differ. The production policy was selected because it evaluates detections according to confidence ranking and is explicitly recorded in the sidecar. These two values must not be combined in one table without identifying the matching policy.

## 7. Supported and unsupported conclusions

Supported conclusions:

- The legacy F1=0.984 was inflated and was not a spatial object-detection F1 score.
- The repaired benchmark represents localization and class errors more accurately and is auditable.
- Fiber/Filament is the weakest class and requires investigation of annotations, bounding-box geometry, and model behavior.

Unsupported conclusions:

- The model quality decreased by 37.47 points. The model did not change; the evaluator changed.
- Fine-tuning must be the next step. Fiber bounding boxes, annotations, and failure cases must be audited first.
- The model has poor area or color performance based only on the histograms. These are post-processing outputs and are not object-matched metrics.
- The timing results provide a fair comparison between runs. The experiment did not include performance-benchmark controls such as warm-up and repetitions.

## 8. Official artifacts

| Artifact | Purpose |
|---|---|
| `benchmark_results/ml_benchmark_200images_20260825_001406.html` | Pre-fix baseline HTML |
| `benchmark_results/evidence/pre_fix_20260825_001406/audit_summary.json` | Numerical evidence for the legacy evaluator defect |
| `benchmark_results/evidence/pre_fix_20260825_001406/spatial_failure_contact_sheet.png` | Visual evidence of correct counts but incorrect boxes |
| `benchmark_results/evidence/after_fix_20260825_231305/after_fix_final.html` | Final after-fix report |
| `benchmark_results/evidence/after_fix_20260825_231305/after_fix_final.benchmark.json` | Replay/audit snapshot for 200 images with the manifest hash |
| `benchmark_results/evidence/after_fix_20260825_231305/run_record.json` | Run seals, invariants, and aggregate metrics |
| `benchmark_results/evidence/after_fix_20260825_231305/final_validation.json` | Manifest-order, hash, and report-disclosure validation |
| `benchmark_results/evidence/after_fix_20260825_231305/before_after_comparison.json` | Machine-readable comparison |

## 9. Status

- [x] Locked and validated the 200-image dataset with 10,000 GT bounding boxes.
- [x] Saved evidence demonstrating the legacy metric defect.
- [x] Replaced the evaluator with class-aware one-to-one IoU matching.
- [x] Corrected the ontology, report, and evidence snapshot.
- [x] Ran the after-fix benchmark with the same dataset, model, and confidence threshold.
- [x] Passed 14/14 unit and integration tests, `compileall`, and system information checks.
- [x] Generated before/after comparison tables and figures.
- [ ] Audit Fiber/Filament failure cases before deciding whether to fine-tune.

**Current stopping point:** the objective of comparing the benchmark before and after the repair is complete. The appropriate next step is to analyze Fiber/Filament FP/FN cases stored in the evidence, not to modify the evaluator further to increase the score.
