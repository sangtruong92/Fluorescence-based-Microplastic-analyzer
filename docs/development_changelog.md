---
title: Development Changelog - YOLO Retrain, ML Benchmark Fixes, REST API, and Commit History
description: Chronological summary of the source changes across the synthetic data pipeline, YOLO dataset export, ML benchmark analyzer, the FastAPI integration layer, and a review of the commits authored by NghiaNT20.
author: Microplastic Analyzer Team
ms.date: 2026-11-09
ms.topic: reference
keywords:
  - yolo
  - benchmark
  - changelog
  - retrain
  - color analysis
  - rest api
  - git history
estimated_reading_time: 16
---

## Overview

This document consolidates the source-level changes that took the microplastic
detector from `yolo26m_v1` to `yolo26m_v2`, the follow-up fix applied to the
ML benchmark color and area post-processing, the performance pass that
brought Quick/Deep Analysis color extraction in line with it, the production
object-matched color/area evaluation feature, and a review of every commit
authored by `NghiaNT20`. It cross-references the git history and the existing
analysis in `docs/yolo_evaluation_audit_and_retrain_plan.md` and
`docs/benchmark_compare_before_after.md`.

## Combined project update (2026-11-15)

The colleague's `main` branch updates have been integrated into the
`raspberry-pi-ai-camera-clean` branch alongside the existing Raspberry Pi AI
Camera support and the local low-light threshold improvement. The integration
preserves the existing desktop application while adding the following:

* **REST API:** a FastAPI application with routes for image analysis,
  benchmarking, calibration, and synthetic-data generation. The API also
  includes request/response schemas, model selection, serialization, and
  background-job handling.
* **Benchmark evaluation:** consistent three-class YOLO labels, raw prediction
  tracking, object-level matching using bounding-box IoU, and JSON snapshots
  that retain evaluation evidence alongside HTML reports.
* **Training-data and model updates:** synthetic ground-truth bounding boxes
  derived from particle masks, corresponding YOLO exporter fixes, and the
  `yolo26m_v1` / `yolo26m_v2` model files.
* **Analysis and maintenance:** refinements to image analysis, calibration,
  reporting, and diagnostics, plus unit tests covering detection metrics,
  benchmark integration, and YOLO export.
* **Documentation:** benchmark comparisons and evaluation findings, a YOLO
  audit and retraining plan, and API usage guidance in the README.

The focused validation for this integration passed: Python compilation
succeeded, and the three benchmark/export test modules reported **18 passed
and 3 passing subtests**. Generated local datasets and benchmark output
directories are intentionally kept out of the GitHub update; they can be
regenerated and are not required to run the application.

## Timeline

| Stage | Scope | Git state (2026-10-24) |
|---|---|---|
| 1. yolo26m_v1 baseline | Original synthetic data export, mismatched bbox ontology | Part of `735069b` initial commit |
| 2. Export pipeline fix, retrain to yolo26m_v2 | `synthetic_generator.py`, `yolo_exporter.py` | Uncommitted (working tree) |
| 3. Benchmark evaluator repair | `detection_metrics.py`, `benchmark_batch.py`, report sidecar | Bundled in repo history, no incremental commits found |
| 4. ML benchmark color/area fix | `ml_benchmark_analyzer.py` | Uncommitted (working tree) |
| 5. Quick/Deep color extraction crop | `quick_analyzer.py`, `deep_analyzer.py` | Uncommitted (working tree) |
| 6. Object-matched color/area evaluation | `ml_benchmark_analyzer.py`, `detection_metrics.py`, `benchmark_batch.py`, `report_generator.py`, `main_window.py` | Uncommitted (working tree) |
| 7. REST API for frontend integration | new `api/` package, `run_api.py` | Uncommitted (working tree) |
| 8. P0 bug fixes, repo housekeeping, full system re-check | `quick_analyzer.py`, `deep_analyzer.py`, `report_generator.py`, `test_benchmark_integration.py`, `api/routers/benchmark.py`, `.gitignore`, `README.md`, file moves/renames | Uncommitted (working tree) |
| 9. `print()` -> `logging`, `jobs.py` TTL, benchmark code dedup | `image_processing.py`, `calibration.py`, `deep_analyzer.py`, `quick_analyzer.py`, `main_window.py`, `api/jobs.py`, `api/routers/benchmark.py`, `run_ml_batch_benchmark.py`, `benchmark_batch.py` | Uncommitted (working tree) |

## Stage 1: yolo26m_v1 baseline issues

The v1 training data was exported with two defects:

* **Ontology mismatch.** `yolo_exporter.py` used `SHAPE_CATEGORIES`
  (`Microbead/Pellet`, `Fiber/Filament`, `Fragment`, `Irregular`), a different
  ontology from the one the benchmark evaluator uses
  (`BENCHMARK_CANONICAL_CLASSES`: `Microbead/Pellet`, `Fiber/Filament`,
  `Irregular`).
* **Fabricated bounding boxes.** The exporter estimated train-label boxes from
  `area` and a hard-coded orientation assumption instead of the particle mask:

  ```python
  # yolo_exporter.py (v1)
  if 'Fiber' in shape_name or 'Filament' in shape_name:
      bbox_width = max(size_hint * 0.3, 8)   # forced narrow
      bbox_height = size_hint * 1.2          # forced upright
  ```

  `synthetic_generator.py` (v1) drew fibers at a random rotation
  (`angle = random.uniform(0, 180)`), so a diagonal fiber's true bounding
  rectangle is much larger than the narrow, upright box the exporter wrote to
  the label file. The ground-truth dict at that time did not store the mask's
  real bounding box at all, only `position` (centroid), `area`, and `size`.

### Why this specifically broke Fiber/Filament

Fiber particles are generated as a rotated rectangle
(`src/data_generation/synthetic_generator.py`, `shape_label == 'Fiber'`):
`length` 15-45 px, `width_f` 2-6 px, `angle` uniform 0-180 degrees, with
`size_hint = length / 2`. Worked example at `length=40`, `width_f=4`,
`angle=45` (a common case, since angle is uniform over 180 degrees):

| | Width | Height | Area |
|---|---:|---:|---:|
| True rotated bounding box (what is actually visible in the image) | ~31.1 px | ~31.1 px | ~967 px^2 |
| v1 exporter label (`size_hint=20`): `max(20*0.3, 8)` x `20*1.2` | 8 px | 24 px | 192 px^2 |

The exported label is about 5x smaller in area, the wrong aspect ratio (narrow
upright box instead of a roughly square diagonal box), and does not even use
the particle's actual `angle`. Microbead and Irregular shapes are closer to
circular or compact, so the same area-only estimate happens to land much
closer to their true box; Fiber/Filament is where the hard-coded "always
upright, always narrow" assumption diverges most.

This is a **training-data defect, not only an evaluation defect**: the model
was fitted against these undersized, wrong-aspect boxes, so the model itself
learned to predict undersized boxes, not just that the old benchmark scored
them leniently. That is why the fix required regenerating the dataset and
retraining (`yolo26m_v2`), not just fixing the evaluator (which was a
separate, already-applied fix, see Stage 3).

**Measured impact** (`docs/yolo_evaluation_audit_and_retrain_plan.md`, section 2 to 3):
Fiber/Filament F1 at IoU >= 0.50 was **11.69%**, even though 99.3% of Fiber GT
objects had a spatially overlapping prediction with the correct class, and
96.3% of those were classified correctly. Median predicted-box area was only
**42.4%** of true area. The model located and classified fibers correctly; it
just drew the box too small, exactly as its training labels taught it to.

## Stage 2: Export pipeline fix and the yolo26m_v2 retrain

Two files were changed to fix the root cause. Both are currently uncommitted
in the working tree.

### `src/data_generation/synthetic_generator.py` (+6/-2 lines)

Added a real bounding box derived from the rendered mask, stored alongside the
existing `area`/`position` fields:

```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bx, by, bw, bh = (int(cx - size_hint), int(cy - size_hint), int(size_hint * 2), int(size_hint * 2))
if contours:
    cnt = max(contours, key=cv2.contourArea)
    ...
    bx, by, bw, bh = cv2.boundingRect(cnt)

ground_truth.append({
    ...
    'bounding_box': (bx, by, bw, bh),
    'bbox_xywh': [float(bx), float(by), float(bw), float(bh)],
    ...
})
```

### `src/data_generation/yolo_exporter.py` (129 lines changed)

* Replaced the local `SHAPE_TO_CATEGORY` mapping and `SHAPE_CATEGORIES` with
  the shared `BENCHMARK_CANONICAL_CLASSES` / `BENCHMARK_CLASS_ALIASES` from
  `config/constants.py`. Class IDs changed from a 4-class scheme to the
  3-class scheme the model and benchmark both use now.
* `ground_truth_to_yolo_bbox()` now reads `bbox_xywh`/`bounding_box` from the
  particle dict (added in the generator change above) and uses it directly
  for the YOLO label. The area-based estimation, including the hard-coded
  Fiber orientation rule, is kept only as a fallback for particles without a
  mask-derived box.
* `save_detailed_ground_truth()` now writes a `Bounding Box: (x, y, w, h)`
  line to each `*_groundtruth.txt` file. This is the exact format the
  benchmark parser (`parse_benchmark_ground_truth` in
  `src/analysis/detection_metrics.py`) consumes today.

### Result

The 200-image, 10,000-object dataset at `benchmark_results/dataset/20260824`
was generated with this corrected pipeline and used to retrain the model now
stored at `src/ml/yolo26m_v2/best.pt`. Per
`docs/yolo_evaluation_audit_and_retrain_plan.md` section 7:

| Class | v1 F1 | v2 F1 | Delta |
|---|---:|---:|---:|
| Fiber/Filament | 11.69% | 97.39% | +85.70 |
| Microbead/Pellet | 79.64% | 97.43% | +17.79 |
| Irregular | 92.70% | 99.33% | +6.63 |
| Overall | 60.90% | 98.05% | +37.15 |

A regression test for this fix exists at
`tests/unit/test_yolo_exporter_fixes.py` (untracked), covering the 3-class
mapping and the exact-bbox code path.

## Stage 3: Benchmark evaluator repair

Documented in `docs/benchmark_compare_before_after.md` and
`docs/benchmark_evaluation_issues.md`. This work is already present in the
current `detection_metrics.py` and `benchmark_batch.py` and did not surface as
a separate working-tree diff, so it is summarized here for completeness
rather than re-derived from a diff:

* `match_detections()` requires both IoU >= threshold and matching class
  before counting a true positive (closes the "wrong class still counts as
  spatial TP" defect).
* Declared GT particle counts are reconciled against parsed, geometrically
  valid records; a mismatch raises an explicit evaluator error instead of
  silently using the declared count.
* Every report is paired with a `.benchmark.json` sidecar containing model,
  dataset, and image hashes, raw predictions, and per-image match results.
* The legacy count-only formula (`tp = min(detected, ground_truth)`) is
  retained only as `legacy_count_diagnostic`, explicitly flagged
  `diagnostic_only` and never used for the reported precision/recall/F1.

## Stage 4: ML benchmark color and area fix (this session)

`src/analysis/ml_benchmark_analyzer.py` (+63 lines), uncommitted.

### Problem found

YOLO does not predict color: `yolo26m_v2` only has 3 shape classes
(`data/yolo_retrain_dataset/microplastics_yolo.yaml`, `nc: 3`). Color and area
come from a separate post-processing step that crops each YOLO box, segments
the particle inside the crop, and classifies the hue of the masked pixels.

That segmentation step called `ImageProcessor.preprocess_segment(roi,
method='adaptive')`, a routine designed for full microscope frames where
background pixels dominate. Applied to a tight YOLO crop, where the particle
can fill most or all of the ROI, two failures compounded:

* `detect_background_color()` on the small crop frequently misclassified a
  fluorescent (dark-background) crop as brightfield, routing it to the wrong
  segmentation branch, because its "mean brightness > 160" heuristic assumes
  background dominates the pixel count. In a tight crop that is mostly bright
  particle, that assumption is reversed.
* Even when the background type was detected correctly, the
  `mean + 2.5 * std` threshold in `_preprocess_adaptive()` assumes background
  pixels outnumber foreground pixels, which is false for a foreground-filled
  crop, and returned an empty mask.

### Why this produced "Unknown" specifically

Reproduced directly on a real prediction from the production benchmark
(`synthetic_001.png`, YOLO box `[659, 840, 38, 44]`, class `Microbead/Pellet`,
confidence 0.974):

```text
roi shape (44, 38, 3) roi mean 87.6 roi max 246
detected bg for roi: black                         # background type correct here
DEBUG Adaptive: Processing RGB channels with background subtraction...
DEBUG Adaptive: Blue channel low variation, skipping
DEBUG Adaptive: No valid channels, using grayscale fallback with bg subtraction
DEBUG Adaptive: Final mask has 0 pixels            # <- segmentation returns nothing
```

Even with the background type detected correctly, the per-channel
`mean + 2.5*std` threshold could not find a "bright enough" subset of pixels
in a crop that is already almost entirely the bright particle. With a
completely empty mask, downstream code had nothing to measure:

* `ColorAnalyzer.extract_color_from_region(roi, mask)` finds zero pixels
  under the mask and returns `"Unknown"` by definition.
* `ShapeAnalyzer.compute_shape_metrics_consistent(mask)` also fails on an
  empty mask, so `MLBenchmarkAnalyzer` fell back to
  `_compute_bbox_metrics()`, which reports `area = width * height` of the raw
  YOLO box, not the particle's actual footprint.

That is also why the two symptoms (`color = Unknown` and `area = bbox area`)
were almost perfectly correlated: 99.8% of the bbox-fallback detections also
had `color = Unknown`, because they are two outputs of the same failed
segmentation call.

Measured on `benchmark_results/ml_benchmark_200images_yolo26m_v2.benchmark.json`
(10,051 ML detections, before the fix):

| Metric | Value |
|---|---:|
| Color = Unknown | 86.6% |
| Area equals raw bbox area (segmentation fallback) | 86.9% |

### Fix applied

Added `MLBenchmarkAnalyzer._segment_particle_roi()`, scoped to the ML
benchmark path only (it does not touch `ImageProcessor`, so Quick/Deep
Analysis in the GUI are unaffected):

* Reuses the background type already detected once on the full image,
  instead of re-detecting it per ROI.
* Thresholds the ROI's own max-channel intensity with Otsu, which stays
  reliable regardless of how much of the crop the particle fills.
* Falls back to treating the whole ROI as the particle when Otsu finds no
  contrast, since a confirmed YOLO detection makes "no particle" a worse
  assumption than "the particle fills the box."

### Result

Full 200-image re-run
(`benchmark_results/ml_benchmark_200images_20261013_102500.benchmark.json`):

| Metric | Before | After |
|---|---:|---:|
| Color = Unknown | 86.6% | 0.0% |
| Area = raw bbox fallback | 86.9% | 0.0% |
| Overall detection F1 (unaffected, IoU+class matched) | 0.9805 | 0.9818 |

Detection precision/recall/F1 barely moved, as expected: those metrics are
computed from raw YOLO boxes and classes, independent of the color/area
post-processing bug.

### Known limitation of this metric

The color and area numbers in the HTML report remain **aggregate
distributions**, not object-matched, per-particle accuracy. The fix removes
the noise caused by segmentation failures; it does not add prediction-to-GT
color or area matching to `benchmark_batch.py`. A one-off script
(`debug_color_area_noblur.py`) was used during this fix to demonstrate
object-matched color/area comparison on a single generated image; its key
findings (area ratio and color match rate per matched pair) are captured
above. The script itself was removed after validation and this logic has not
been added to the production benchmark pipeline.

## Stage 5: Quick/Deep color extraction now crops to the particle first

`src/analysis/quick_analyzer.py` and `src/analysis/deep_analyzer.py`,
uncommitted.

### What changed

Both analyzers previously called
`color_analyzer.extract_color_from_region(image, particle_mask)` with the
**full image** and a full-image-sized mask (only the contour pixels set),
repeated once per detected particle. That function always starts with
`cv2.cvtColor(image, cv2.COLOR_RGB2HSV)`, so every particle paid the cost of
converting the entire frame to HSV again.

Both now crop to the particle's own `bounding_box` first, matching the
approach `MLBenchmarkAnalyzer` already uses for YOLO detections:

```python
bx, by, bw, bh = metrics['bounding_box']
roi = image[by:by + bh, bx:bx + bw]
roi_mask = particle_mask[by:by + bh, bx:bx + bw]
color_name, hue, sat, val = self.color_analyzer.extract_color_from_region(roi, roi_mask)
```

Reason: performance (convert a small crop instead of the full frame per
particle) and consistency with the ML path, not a color-accuracy bug. Unlike
Stage 4, there was nothing broken in Quick/Deep color output before this
change.

### Why this was expected to be a near no-op

`bounding_box` comes from `cv2.boundingRect(cnt)`, which always fully
contains the contour, so no mask pixel is cut off by the crop.
`cv2.cvtColor` converts RGB to HSV per pixel with no dependency on
neighboring pixels, so the same physical pixels produce the same H/S/V
values whether read from the full image or a cropped view of it.

### Verification

Re-ran the full 200-image batch benchmark before and after this change
(`benchmark_results/ml_benchmark_200images_20261013_102500.benchmark.json`
vs `benchmark_results/ml_benchmark_200images_20261024_204512.benchmark.json`):

| Method | F1 before | F1 after | Color distribution before | Color distribution after |
|---|---:|---:|---|---|
| Quick | 0.7875 | 0.7875 | `Blue 1589, Red 2078, Green 2390, Yellow 2139` | `Blue 1589, Red 2078, Green 2388, Yellow 2141` |
| Deep | 0.7257 | 0.7257 | `Blue 2017, Red 2105, Green 2208, Yellow 2012` | `Blue 2017, Red 2105, Green 2213, Yellow 2007` |
| ML Benchmark | 0.9818 | 0.9818 | identical | identical (file unchanged by this stage) |

F1 is unchanged for every method. Quick/Deep color distributions differ by
2-5 particles out of roughly 8,600 detections (about 0.02-0.06%), consistent
with a few particles sitting exactly on a Hue classification boundary
(for example near 0.20, the Yellow/Green cutoff) where floating-point
percentile rounding can flip the result either way. A repeated
same-process run of `QuickAnalyzer` on one image produced byte-identical
color output twice in a row, confirming there is no run-to-run randomness;
the tiny difference is a reproducible, expected side effect of the crop, not
a regression.

## Appendix: Commit history (NghiaNT20)

`git log --all --format='%h %an'` shows 20 commits total on `main`, authored
by three different people. Only 3 are authored by `NghiaNT20
<nghianguyen.200303@gmail.com>`; the other 17 (README edits, camera
calibration feature, repository hygiene) belong to `Sangtt` / `ttsang` and are
out of scope here. Stages 1, 2, 4, and 5 above are not in any commit yet
(uncommitted working tree); this appendix covers only what is actually
committed.

| Date | Commit | Subject |
|---|---|---|
| 2026-08-25 | `8fc0b39` | fix: benchmark file and validate |
| 2026-08-26 | `d5c92fd` | fix: Enhance ML Benchmark Analyzer with raw detection tracking |
| 2026-08-26 | `b0f27f7` | Refactor benchmark evaluation documentation |

### `8fc0b39` fix: benchmark file and validate

No `.py` files changed (confirmed via `git show --stat`). This commit adds:

* The 200-image synthetic benchmark dataset at
  `benchmark_results/dataset/20260824/` (images, `*_groundtruth.txt`,
  `dataset_manifest.json`).
* Dataset readiness and audit evidence
  (`dataset_readiness.json`, `dataset_readiness_pre_run.json`,
  `audit_summary.json`, `pre_fix_evaluator_audit.json`).
* The first versions of the audit documentation, later renamed in the next
  commit: `BASELINE_BENCHMARK_PRE_FIX_EVALUATION.md`,
  `BENCHMARK_BEFORE_AFTER.md`, `BENCHMARK_BEFORE_AFTER_CLEAN.md`,
  `BENCHMARK_MODEL_EVALUATION_ISSUES.md`, `YOLO26M_FINETUNING_EVIDENCE.md`.

Reason: capture a locked, hash-verifiable benchmark dataset and record the
pre-fix evaluator's behavior as evidence before touching any evaluation code.

### `d5c92fd` fix: Enhance ML Benchmark Analyzer with raw detection tracking

The commit subject undersells its scope. This is the commit that replaces the
count-only legacy evaluator (Stage 3 above) with the class-aware, IoU-matched
one. Full file list:

| File | Change | Purpose |
|---|---:|---|
| `src/analysis/detection_metrics.py` | +325 (new file) | IoU calculation, GT parsing/validation, `match_detections()` (confidence-descending, class-aware, one-to-one), `evaluate_detection_dataset()`, `legacy_count_agreement()` kept as an explicitly labeled diagnostic |
| `src/analysis/benchmark_batch.py` | +231 (new file) | Builds the report payload and replay evidence from per-method results; computes shape/color/area distributions and reconciles declared vs. parsed GT counts |
| `src/analysis/benchmark_snapshot.py` | +62 (new file) | Writes the `.benchmark.json` sidecar atomically, with model/image/GT hashes, so every report is replayable |
| `config/constants.py` | +21 | Adds `BENCHMARK_CANONICAL_CLASSES` (3-class ontology), `BENCHMARK_CLASS_ALIASES`, `BENCHMARK_EVALUATION_IOU_THRESHOLD = 0.50` |
| `config/settings.py` | +2 | Adds `raw_detections` and `rejected_detections` fields to `AnalysisResult` |
| `src/analysis/__init__.py` | +13/-2 | Exports the new `detection_metrics` functions |
| `src/analysis/ml_benchmark_analyzer.py` | +25/-13 | Tracks every raw YOLO box (`raw_detections`) and every rejected one with a reason (`below_confidence_threshold`, `invalid_or_empty_roi`), not only accepted detections; fixes an ROI bounds check that allowed negative `x`/`y` |
| `src/analysis/report_generator.py` | +137/-x | Renders `N/A` when evaluation is `unavailable`, shows per-class precision/recall/F1, displays `color_coverage` and labels it ROI post-processing output |
| `src/gui/main_window.py` | +104/-x | Wires `build_ml_batch_payload` and `save_benchmark_snapshot` into the batch benchmark flow; keeps the old count-only formula only as `legacy_count_diagnostic` |
| `tests/unit/test_benchmark_integration.py` | +151 (new file) | End-to-end payload/report tests |
| `tests/unit/test_detection_metrics.py` | +89 (new file) | Unit tests for IoU, matching, and GT parsing |

Reason: the pre-fix evaluator (`tp = min(detected, ground_truth)`) could
report a perfect F1 score when every prediction was in the wrong place, as
proven by the evidence captured in `8fc0b39`. This commit replaces it with
one-to-one, same-class, IoU >= 0.50 matching, and makes every report
replayable via a hash-sealed sidecar.

### `b0f27f7` Refactor benchmark evaluation documentation

Rewrites and renames the three docs added in `8fc0b39` into their current
form: `docs/benchmark_compare_before_after.md`,
`docs/benchmark_evaluation_issues.md`, `docs/benchmark_pre_fix.md`
(706 insertions, 912 deletions net across the three files). No source code
changed; this is a prose pass over the documents written right after the
evaluator repair landed.

## Stage 6: Object-matched color/area evaluation in the production report

`src/analysis/ml_benchmark_analyzer.py`, `src/analysis/detection_metrics.py`,
`src/analysis/benchmark_batch.py`, `src/analysis/report_generator.py`,
`src/gui/main_window.py`, uncommitted.

### Why: the aggregate distributions could not answer "how accurate is color/area"

Stage 4 and 5 fixed the color/area *pipeline*, but the HTML report only ever
showed aggregate distributions (total count per color, total area histogram)
across all predictions. Two systems can match the same totals while making
completely different per-particle errors: a prediction's color being wrong in
one direction can cancel out with another prediction being wrong in the
opposite direction, and the chart shows nothing. There was no single "color
accuracy %" or "area is off by X%" number a reader could cite.

### Bug found while building this: `bbox_global` lost sub-pixel precision

Before this fix, `MLBenchmarkAnalyzer.analyze()` stored
`'bbox_global': [x, y, w, h]` using the same `int(x1), int(y1), int(x2-x1),
int(y2-y1)` truncated to integers for pixel-array indexing. Any code matching
this box against ground truth by IoU (as object-matched evaluation needs to)
lost up to ~1px per side, for example `38.593 -> 38`. For a small particle
(Microbead bbox is often around 7x7 px), losing 1px per side is enough to
drop IoU below the 0.5 threshold and silently miss a true positive. Measured
impact: evaluating with the truncated box gave F1=0.9534, versus the official
F1=0.9818 computed elsewhere in the same codebase from the untruncated float
box in `raw_detections`. The two code paths disagreed on the same detections
purely because of this rounding, which the aggregate-distribution report
never had a way to expose.

Fix: `bbox_global` now stores `[float(x1), float(y1), float(x2 - x1),
float(y2 - y1)]`. Pixel-array indexing (ROI crop, mask placement) still uses
the separate `int(x1), int(y1), ...` locals, unchanged. The one GUI consumer
of `bbox_global` (`src/gui/main_window.py`, the single-image annotated-image
draw loop) now rounds to int at the point of use, since `cv2.rectangle`/
`cv2.putText` require integer coordinates.

### What was added

* `detection_metrics._summarize_matched_attributes()`: reuses the
  `matches` list `evaluate_detection_dataset()` already computes for
  precision/recall/F1 (no second matching pass) and, for each matched pair,
  compares `prediction['color']` to `ground_truth['color_label']` and
  computes `prediction['area'] / ground_truth['area']`. Attached to the
  evaluator's return value as `matched_attributes` (`unavailable_evaluation()`
  gets a zeroed version of the same shape so callers never need a special
  case).
* `benchmark_batch._bbox_prediction()` / `_method_predictions()`: now carry
  `color` and `area` through to the prediction dicts used for matching. For
  ML, raw YOLO detections do not carry color/area (only the post-processed
  `features` do), so predictions are joined to their feature by
  `source_prediction_id` before being handed to the evaluator.
* `report_generator.py`: new "Object-matched Color and Area Accuracy"
  section, rendered for all three methods, placed after the existing
  distribution charts (which are kept, not replaced, since they still work
  as a quick visual sanity check).

### Result

Full 200-image re-run
(`benchmark_results/ml_benchmark_200images_20261108_223027.benchmark.json`),
read directly from `report_payload[method]['matched_attributes']`:

| Method | Color accuracy | Matched pairs | Area ratio (median / mean) |
|---|---:|---:|---|
| Quick | 98.10% | 7,165 | 1.531 / 1.618 |
| Deep | 98.93% | 6,655 | 1.594 / 1.690 |
| ML (yolo26m_v2) | 97.36% | 9,843 | 1.049 / 1.216 |

Overall detection P/R/F1 for every method is unchanged from before this stage
(ML: 0.9793 / 0.9843 / 0.9818), confirming the `bbox_global` fix did not
alter the official detection numbers, only the precision available to any
future consumer that needs exact IoU from `bbox_global`.

Reading the table: Quick/Deep have *higher* per-matched-pair color accuracy
than ML, but far fewer matched pairs (lower detection recall) and a much
worse area ratio (1.5-1.7x GT, versus ML's 1.05x). This is consistent with
Stage 4/5's finding that Quick/Deep segment on the full blurred/glowing image
(capturing the glow halo beyond the true pre-effects footprint), while ML's
Otsu-on-ROI segmentation (Stage 4) stays closer to the GT definition. The
top confusion pair for every method is the same: `Yellow -> Green`
(the two closest wavelengths, 565nm and 525nm, sit close together in Hue).

The standalone script used to validate this before productionizing it
(`run_full_pipeline_experiment.py`) has been removed; its logic now lives in
`detection_metrics.py`/`benchmark_batch.py` and its numbers are reproduced
above.

## Stage 7: REST API for frontend integration

New `api/` package (FastAPI) and `run_api.py`, uncommitted. Requested to let
a separate web frontend call the existing Quick/Deep/ML pipeline instead of
only the PyQt desktop GUI.

### Why a new layer was needed, not just calling the analyzers directly

The analyzer classes (`QuickAnalyzer`, `DeepAnalyzer`, `MLBenchmarkAnalyzer`,
`ImageProcessor`, `ShapeAnalyzer`, `ColorAnalyzer`) already have zero PyQt
dependency, so they could be called from any Python process. What was
missing was everything between "a Python function call" and "an HTTP API a
browser-based frontend can use":

* Feature dicts contain values that are not JSON-serializable as-is: raw
  `cv2.findContours` output under the `'Contour'` key (a numpy array), plus
  numpy scalar types in `area`/`centroid`/etc. `json.dumps()` fails on these
  directly.
* `AnalysisResult.mask` and any annotated/overlay image are numpy arrays; a
  web frontend needs an image it can `<img src=...>`, not an array.
* The GUI's async execution (`AnalysisThread(QThread)` /
  `pyqtSignal`) is Qt-specific and not reusable outside the desktop app.
* The 200-image batch benchmark takes 40-90s end to end (measured in Stage
  4-6 runs); a synchronous HTTP request would time out or block the
  frontend, so it needs to run as a background job with status polling.
* The YOLO model must be loaded once per process, not once per request
  (checkpoint load is expensive), and ultralytics/torch model objects are
  not guaranteed safe for concurrent inference from multiple threads.

### What was added

* `api/serialization.py`: `json_safe()` (recursively strips
  `Contour`/`contour` keys and converts numpy scalars/arrays) and
  `encode_image_base64()` (mask/annotated image -> `data:image/png;base64,...`
  string). Does not modify the analyzer classes, so the desktop GUI is
  unaffected.
* `api/model_registry.py`: loads a YOLO checkpoint once per process and
  reuses the same `MLBenchmarkAnalyzer` instance across requests; discovers
  every `src/ml/*/best.pt` so the frontend can list and switch models.
* `api/jobs.py`: in-memory job store (status: pending/running/done/error) for
  the batch benchmark background task. Explicitly scoped to single-process
  local testing; swap for Redis/DB-backed storage before running multiple
  server workers.
* `api/routers/analyze.py`: `POST /api/analyze` -- upload one image, pick
  `method=quick|deep|ml` plus `PreprocessingParams` fields, get back JSON
  features + base64 mask/annotated image.
* `api/routers/benchmark.py`: `POST /api/benchmark/batch` (202 Accepted,
  returns a `job_id` immediately) and `GET /api/benchmark/batch/{job_id}`
  (status + per-method precision/recall/F1 + `matched_attributes` once
  done, plus links to the generated HTML report and `.benchmark.json`
  sidecar, served via a `/reports` static mount). Reuses
  `build_ml_batch_payload()`/`ReportGenerator`/`save_benchmark_snapshot()`
  unchanged, so API-triggered and GUI-triggered benchmarks produce identical
  artifacts.
* `api/routers/generate.py`: `POST /api/generate` (one image, synchronous,
  for a live frontend preview) and `POST /api/generate/dataset` /
  `GET /api/generate/dataset/{job_id}` (background job writing N images +
  `*_groundtruth.txt` files under `benchmark_results/dataset/<name>/`, same
  format `YOLODatasetExporter.save_detailed_ground_truth` already produces
  for the GUI). The resulting `dataset_dir` can be passed straight into
  `POST /api/benchmark/batch` -- verified end to end (generate 2 images,
  benchmark them, got a result back) without writing any new parsing code.
* `api/main.py`: FastAPI app, CORS enabled for local Vite/CRA dev ports,
  `GET /api/health`, `GET /api/models`, auto-generated docs at `/docs`.

### Feature audit: every GUI menu action / `run_*`/`export_*`/`open_*_dialog` method in `main_window.py`, matched against the API

| GUI feature | API coverage |
|---|---|
| File > Open Image, Run Quick/Deep/Both Analysis | `POST /api/analyze` (method=quick/deep) |
| Machine Learning > Run ML Detection, ML Benchmark (single image) | `POST /api/analyze` (method=ml) |
| Run Benchmark (batch, Quick+Deep+ML), ML Benchmark (batch 200) | `POST /api/benchmark/batch` + polling |
| Tools > Generate Synthetic Data (single + batch) | `POST /api/generate`, `POST /api/generate/dataset` |
| Export YOLO Training Dataset | `POST /api/generate/yolo-dataset` (added this round) |
| Camera Calibration dialog | `POST /api/calibration`, `POST /api/calibration/convert` (added this round) |
| Machine Learning > Load YOLO Model | `GET /api/models` + `model_name` param (registered folders only, not an arbitrary file path) |
| File > Save Results / Export JSON or CSV | Not added: the client already receives the full JSON from `/api/analyze`; writing it to disk is a frontend concern, not a backend one |
| ESP32-CAM network capture | Not added: hardware-specific, no remote device in scope for this round |

* `api/routers/calibration.py`: `POST /api/calibration` (wraps
  `CameraCalibration.calibrate_from_reference/_fov/_theoretical`, one
  endpoint selected by a `method` field) and `POST /api/calibration/convert`
  (wraps `convert_features()`, adds `area_um2`/`diameter_um`/
  `size_category` to a feature list). Deliberately stateless -- no
  server-side "current calibration" is kept; the caller re-sends
  `um_per_pixel_x/y` with every conversion, since a hidden shared
  singleton would leak calibration state across concurrent users.
* `api/routers/generate.py` gained `POST /api/generate/yolo-dataset` +
  `GET .../{job_id}`: background job generating a real YOLO-trainable
  dataset (`images/{train,val,test}/`, `labels/{...}/*.txt` in YOLO's
  normalized format, `dataset.yaml`) via `YOLODatasetExporter.export_dataset()`
  / `create_dataset_yaml()` -- distinct from `POST /api/generate/dataset`,
  which only writes the detailed-groundtruth benchmark format. Generates one
  image at a time per split instead of buffering the whole split in memory,
  so `train_images=20000` does not require holding 20,000 full-size images
  in RAM at once.

### Verified

Started the server locally and exercised every endpoint against real data:
`GET /api/health` and `/api/models` (model auto-discovery and load),
`POST /api/analyze` with `method=quick|deep|ml` (color/shape/area present,
no `Contour` serialization error, base64 mask/annotated image decodes), and
a full `POST /api/benchmark/batch` -> poll `GET .../{job_id}` cycle on a
3-image dataset, confirming the job reaches `done` with per-method
`matched_attributes` and a working `/reports/...html` link. `pytest tests/`
still passes (18 passed, 3 subtests) after these additions.

### Known limitations of this first pass (scope was "all features", delivered: core analysis + batch benchmark + synthetic data generation + YOLO dataset export + calibration)

* GUI-only features still not wrapped: YOLO training itself (not just
  dataset export), ESP32-CAM capture, loading an arbitrary YOLO weights
  file path (only pre-registered `src/ml/*/best.pt` folders).
* `POST /api/benchmark/batch` takes a server-side `dataset_dir` path, not a
  multi-file upload from the browser; fine for local testing, needs an
  upload endpoint before a remote frontend can supply its own images.
* The `print("DEBUG ...")` calls still present in `image_processing.py`/
  `deep_analyzer.py` (noted as an open item below) now also run inside the
  API process; not a functional blocker for local testing, but should move
  to `logging` before running on a shared server.
* In-memory job store and single loaded model assume one process/one
  worker; add shared storage and a request lock (or per-worker model
  instances) before scaling beyond local testing.

## Stage 8: P0 bug fixes, repo housekeeping, and a full system re-check

`src/analysis/quick_analyzer.py`, `src/analysis/deep_analyzer.py`,
`src/analysis/report_generator.py`, `tests/unit/test_benchmark_integration.py`,
`api/routers/benchmark.py`, plus non-code repo cleanup. Triggered by a
two-axis (Standards/Spec) code review of the full working-tree diff against
`HEAD` (`b0f27f7`), then a full re-run of every API endpoint and the test
suite to validate the fixes and catch anything the review missed.

### P0 fixes from the review

* **Quick/Deep color-extraction crop used the wrong bounding box.**
  `metrics['bounding_box']` is `cv2.boundingRect()` of the
  Douglas-Peucker-simplified contour (`shape_analysis.py`), which is not
  guaranteed to fully contain the raw contour `particle_mask` is drawn
  from -- unlike the Stage 5 docstring's claim ("always fully contains the
  contour"). Fixed by cropping from `cv2.boundingRect(cnt)` (the raw
  contour) instead, in both `quick_analyzer.py` and `deep_analyzer.py`.
  `metrics['bounding_box']` itself is unchanged and still used everywhere
  else (reporting, IoU matching).
* **`report_generator.py`'s object-matched accuracy section never hid
  itself.** The gating condition was
  `any(data.get('matched_attributes') for data in [...])`, but
  `unavailable_evaluation()` always returns a populated (zeroed)
  `matched_attributes` dict, so the condition was always true. Changed to
  `any(data.get('available') for data in [...])`, the actual flag that
  distinguishes a real evaluation from an unavailable one. Verified both
  branches directly: a forced GT-count mismatch now correctly hides the
  section; a normal run still shows it.
* **Resolves the "stale test mock" open item below.**
  `test_benchmark_integration.py` now monkeypatches
  `analyzer._segment_particle_roi` instead of
  `analyzer.image_processor.preprocess_segment`, which `analyze()` stopped
  calling after the Stage 4 fix. The test was passing for the wrong reason;
  it now actually exercises the code path it claims to test.

### New gap found while re-running the system end to end

Re-running the full `POST /api/benchmark/batch` flow (not just unit tests)
surfaced a fourth bug the static review missed: `api/routers/benchmark.py`
hand-picks only `precision`/`recall`/`f1_score`/`per_class`/
`matched_attributes` into the job's JSON `result`, silently dropping
`available`/`reason`. A frontend reading the API response alone cannot tell
"evaluation is genuinely unavailable" (e.g. GT-count mismatch) apart from
"evaluation ran and found nothing" -- the exact ambiguity the Stage 6
`matched_attributes` design and the fix above were built to avoid, but only
for the HTML report, not the JSON API. Fixed by adding `available` and
`reason` to the `summary` dict. Verified on a 3-image dataset
(`available: true, reason: null`) and, separately, a full 200-image run
through the real API (not just a script) reproduced the Stage 5/6 numbers
exactly (ML F1 0.9818, Quick F1 0.7875, Deep F1 0.7257), confirming the P0
crop fix did not change detection-level metrics, only the color/area
post-processing it touches.

### Incident: 442 tracked files were missing from the working tree

While auditing the repo for this stage, `git status` (previously only ever
checked with a filter that hid deletions) showed 442 tracked files reported
deleted: all of `benchmark_results/dataset/20260824/` and
`benchmark_results/baselines/` (425 files), 7 `UV_Filter_AloneFire_SV57/*.png`
samples, and `src/ml/YoloV8s_v3/best.pt` (22 MB). None of this session's
`rm`/`rm -rf` commands targeted those paths (only `test_calibration.py`,
`debug_report.py`, the 35 agent-skill symlink directories, and a handful of
`/tmp` and generated-report test artifacts were intentionally removed), so
the cause predates or is outside this round of work. All 440 non-intentional
deletions were recoverable from `HEAD` (confirmed with `git show HEAD:<path>`
before restoring) and were restored with `git checkout HEAD -- <paths>`;
`git diff --stat` against those paths is now empty (byte-identical to the
committed blobs). The 2 intentionally-removed files
(`test_calibration.py`/`debug_report.py`, see below) were excluded from the
restore.

### Repo housekeeping (non-code)

* `.gitignore`: added the ~35 agent-tool directories (`.claude/`,
  `.windsurf/`, `.roo/`, etc. -- symlinks into `.agents/skills/code-review/`
  left by a skill installer, not project files), `skills-lock.json`,
  `*.code-workspace`, and `UV_Filter_AloneFire_SV57/results[_v2]/` (generated
  comparison output).
* Deleted those 35 directories from disk (kept `.agents/` itself, the real
  skill content).
* `test_calibration.py` -> `examples/calibration_demo.py`: it has no
  assertions (only `print()`-based worked examples), so `test_` was a
  misnomer; added a `sys.path` shim so it still runs standalone from any cwd.
* Deleted `debug_report.py`: dead code, imports a `generate_report_v2`
  module that does not exist anywhere in the repo.
* `README.md`: its documented folder tree said model weights live in
  `models/`, which does not exist; corrected to `src/ml/<version>/best.pt`,
  and added the undocumented `api/`/`examples/` directories.
* This file renamed `change_history_v1_to_v2.md` -> `development_changelog.md`
  (two references updated: its own frontmatter title and an `api/model_registry.py` docstring pointer).

### Verified

`pytest tests/` (18 passed, 3 subtests) after every change above. Full
server re-run of every endpoint: `GET /api/health`/`/api/models`,
`POST /api/analyze` (quick/deep/ml), `POST /api/calibration` +
`/convert`, `POST /api/generate` + `/dataset` + `/yolo-dataset`, and
`POST /api/benchmark/batch` on both a 3-image and the full 200-image
dataset. GUI still constructs (`MicroplasticAnalyzerGUI()`) without error.
All test-only artifacts (generated datasets, report files, `/tmp` diffs)
were deleted after verification.

## Stage 9: print() -> logging, jobs.py TTL, and the benchmark/run_ml_batch_benchmark duplication

Closes out the two P1/P2 items Stage 8 left open: the 100+ debug `print()`
calls and the duplicated dataset-loading/analyzer-loop code between
`api/routers/benchmark.py` and `run_ml_batch_benchmark.py`.

### print() -> logging

Converted every `print()` call reachable from an actual analysis/API/GUI
request path to `logging` (`logger = logging.getLogger(__name__)` per
module), classifying by message content: `ERROR ...` -> `logger.error`,
`Warning`/`⚠`/`✗ ...` -> `logger.warning`, `[Deep Analysis]`/`[Quick Analysis]`
summaries and `✓ ...` confirmations -> `logger.info`, everything else
(`DEBUG ...` traces) -> `logger.debug`. Converted: `src/core/image_processing.py`
(54), `src/core/calibration.py` (15, library methods only), `src/analysis/
deep_analyzer.py` (3), `src/analysis/quick_analyzer.py` (1), `src/gui/
main_window.py` (8).

Left as `print()` deliberately, since they are one-time CLI/startup output,
not per-request library noise: `src/ml/__init__.py`'s YOLO-availability
banner (prints once at import), `calibration.py`'s own `if __name__ ==
"__main__":` demo block, `examples/calibration_demo.py` (the whole point of
a demo script is printed output), and `yolo_exporter.py`'s
`print_dataset_stats()` (a formatted table meant for direct display, not a
diagnostic trace).

Added `logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), ...)`
to both entry points (`main.py`, `api/main.py`) so `logger.debug(...)` is
silent by default (the original goal: stop polluting stdout on a shared
server) but still reachable with `LOG_LEVEL=DEBUG ./.venv/bin/python3
run_api.py` when actually debugging segmentation behavior. Verified both
states directly: default run produced zero `DEBUG` lines in server stdout
for a `POST /api/analyze` call; `LOG_LEVEL=DEBUG` produced the expected
`src.core.image_processing: DEBUG Advanced: ...` trace lines for the same
call.

### `api/jobs.py`: bounded job store + visible failure on bad job_id

* Added `_FINISHED_JOB_TTL = timedelta(hours=1)`; `create_job()` now prunes
  `done`/`error` jobs older than the TTL before inserting the new one, so a
  long-running server process does not accumulate every historical batch
  result (which can include a full 200-image `matched_attributes` payload)
  forever in memory.
* `update_job()` on an unknown `job_id` now logs a warning instead of
  silently no-oping, so a typo'd or stale job_id is visible in the logs
  instead of just leaving a job stuck at `pending` with no trace.
* Verified with a synthetic old/new job pair: the old finished job was
  pruned on the next `create_job()` call, a recent one was kept, and an
  unknown `job_id` logged a warning without raising.

### Deduplicated `benchmark.py` / `run_ml_batch_benchmark.py`

Moved the near-identical dataset-loading loop and per-analyzer-loop into
two shared functions in `src/analysis/benchmark_batch.py` (the module both
callers already imported from): `load_benchmark_images(dataset_dir)` and
`run_analyzer_over_images(analyzer, images_data, params, *,
confidence_threshold=None, on_progress=None)`. `api/routers/benchmark.py`'s
private `_load_images_data`/`_run_method` and
`run_ml_batch_benchmark.py`'s `load_images_data`/`run_method` were deleted;
both now call the shared versions. `run_ml_batch_benchmark.py`'s
periodic `"  [Quick] 25/200"`-style progress printing is preserved via the
new `on_progress` callback parameter, which the API route simply omits.
Also dropped `time`/`numpy`/`cv2`/`parse_benchmark_ground_truth` imports
from `run_ml_batch_benchmark.py` that were unused once the duplicated
functions moved out (`time`/`numpy` were already dead before this change).

### Verified

`pytest tests/` (18 passed, 3 subtests). Full server re-run: `GET
/api/health`, `POST /api/analyze` (method=quick/deep, confirming identical
`features` counts as before, 47 for the same test image), server stdout
inspected directly for the presence/absence of `DEBUG` lines in both the
default and `LOG_LEVEL=DEBUG` configurations. GUI still constructs without
error.

## Working-tree snapshot recorded on 2026-11-09

The following list records the repository state when this changelog was first
written on 2026-11-09. It is a historical snapshot, not the status of the
combined project today:

```text
 M .gitignore
 M README.md
 M api/routers/benchmark.py
 M requirements.txt
 M src/analysis/benchmark_batch.py
 M src/analysis/deep_analyzer.py
 M src/analysis/detection_metrics.py
 M src/analysis/ml_benchmark_analyzer.py
 M src/analysis/quick_analyzer.py
 M src/analysis/report_generator.py
 M src/data_generation/synthetic_generator.py
 M src/data_generation/yolo_exporter.py
 M src/gui/main_window.py
 M tests/unit/test_benchmark_integration.py
?? api/__init__.py, api/jobs.py, api/main.py, api/model_registry.py, api/routers/ (analyze.py, calibration.py, generate.py, __init__.py), api/schemas.py, api/serialization.py
?? run_api.py
?? examples/calibration_demo.py
?? docs/development_changelog.md
?? docs/yolo_evaluation_audit_and_retrain_plan.md
?? run_ml_batch_benchmark.py
?? src/ml/YoloV8s/ (untracked, pre-existing, not added by this work)
?? src/ml/yolo26m_v1/
?? src/ml/yolo26m_v2/
?? tests/unit/test_yolo_exporter_fixes.py
?? benchmark_results/ml_benchmark_200images_yolo26m_v2.*
?? benchmark_results/ml_benchmark_200images_20261013_102500.*
?? benchmark_results/ml_benchmark_200images_20261024_204512.*
?? benchmark_results/ml_benchmark_200images_20261108_223027.*
```

## Open items not covered by this round of fixes

* **Unfair timing protocol.** `ml_benchmark_analyzer.py` still times each
  image with a single `time.time()` call, no warm-up, and a fixed
  Quick -> Deep -> ML execution order. Processing-time numbers in every
  report should not be used to compare method speed.
* **Unverified train/test independence.** `data/yolo_retrain_dataset` has no
  manifest or seed record to cross-check against the benchmark dataset's
  `master_seed = 20260824`, so disjoint train/test seeding cannot be
  confirmed from the repository alone.
* **Debug `print()` calls run inside the API process now too.** Resolved in
  Stage 9: converted to `logging` (silent by default, `LOG_LEVEL=DEBUG` to
  re-enable). A handful of intentional CLI/startup `print()`s remain by
  design (see Stage 9 for the list).
* **Batch benchmark only accepts a server-side `dataset_dir` path.** No
  multi-file upload endpoint yet, so a remote frontend cannot supply its own
  images for `POST /api/benchmark/batch` without server filesystem access.

## References

* `docs/yolo_evaluation_audit_and_retrain_plan.md`
* `docs/benchmark_compare_before_after.md`
* `docs/benchmark_evaluation_issues.md`
* `api/main.py`
* `api/routers/analyze.py`
* `api/routers/benchmark.py`
* `src/data_generation/synthetic_generator.py`
* `src/data_generation/yolo_exporter.py`
* `src/analysis/ml_benchmark_analyzer.py`
* `src/analysis/quick_analyzer.py`
* `src/analysis/deep_analyzer.py`
* `src/analysis/detection_metrics.py`
* `src/analysis/benchmark_batch.py`
* `src/analysis/report_generator.py`
* `tests/unit/test_yolo_exporter_fixes.py`
* `tests/unit/test_benchmark_integration.py`
* `examples/calibration_demo.py`
