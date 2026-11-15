import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from config.settings import PreprocessingParams
from src.analysis.benchmark_batch import build_ml_batch_payload
from src.analysis.benchmark_snapshot import save_benchmark_snapshot, sha256_file
from src.analysis.ml_benchmark_analyzer import MLBenchmarkAnalyzer
from src.analysis.report_generator import ReportGenerator


class ArrayValue:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeBox:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = [ArrayValue(xyxy)]
        self.conf = [ArrayValue(confidence)]
        self.cls = [ArrayValue(class_id)]


class FakeResult:
    names = {0: "Irregular"}

    def __init__(self):
        self.boxes = [FakeBox((1, 1, 8, 8), 0.9, 0), FakeBox((9, 9, 15, 15), 0.8, 0)]


class FakeModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        return [FakeResult()]


class BenchmarkIntegrationTests(unittest.TestCase):
    def test_ml_analyzer_captures_single_inference_lineage_and_rejection(self):
        model = FakeModel()
        analyzer = MLBenchmarkAnalyzer(model)
        analyzer.image_processor.detect_background_color = lambda _image: "black"
        analyzer._segment_particle_roi = lambda roi, _bg_color, min_size=3: np.ones(roi.shape[:2], dtype=np.uint8) * 255
        analyzer.shape_analyzer.compute_shape_metrics_consistent = lambda _mask, **_kwargs: {
            "area": 49.0, "perimeter": 28.0, "circularity": 0.7, "eccentricity": 0.1,
            "aspect_ratio": 1.0, "rectangularity": 1.0, "solidity": 1.0,
            "centroid": (3.5, 3.5), "bounding_box": (0, 0, 7, 7),
        }
        analyzer.shape_analyzer.classify_shape_unified = lambda _metrics: "Irregular"
        analyzer.color_analyzer.extract_color_from_region = lambda _roi, _mask: ("Red", 0, 0, 0)
        result = analyzer.analyze(np.zeros((10, 10, 3), dtype=np.uint8), PreprocessingParams())
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(result.raw_detections), 2)
        self.assertEqual(len(result.features), 1)
        self.assertEqual(result.features[0]["source_prediction_id"], 1)
        self.assertEqual(result.rejected_detections[0]["prediction_id"], 2)
        self.assertEqual(result.rejected_detections[0]["rejection_reason"], "invalid_or_empty_roi")

    def test_snapshot_is_atomic_json_safe_and_hashes_html(self):
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory) / "report.html"
            html.write_text("<html>ok</html>", encoding="utf-8")
            output = save_benchmark_snapshot(html, {"value": np.int64(3), "path": Path("input.png")})
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["benchmark_snapshot_schema_version"], 1)
            self.assertEqual(saved["value"], 3)
            self.assertEqual(saved["report"]["sha256"], sha256_file(html))

    def test_batch_payload_scores_raw_yolo_boxes_and_labels_legacy_formula(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "best.pt"
            model.write_bytes(b"model")
            truth = {
                "id": 1, "shape": "Irregular", "class_name": "Irregular",
                "bounding_box": [0, 0, 10, 10], "bbox_xywh": [0, 0, 10, 10],
                "color_label": "Red", "area": 100.0,
            }
            images = [{
                "name": "sample", "ground_truth": 1, "ground_truth_data": [truth],
                "ground_truth_error": None, "source_path": None, "ground_truth_path": None,
            }]
            feature = {
                "id": 1, "shape": "Irregular", "ml_class": "Irregular",
                "bounding_box": [0, 0, 10, 10], "bbox_global": [0, 0, 10, 10],
                "color": "Red", "area": 100.0,
            }
            traditional = SimpleNamespace(features=[feature])
            ml = SimpleNamespace(
                features=[feature],
                raw_detections=[{
                    "prediction_id": 1, "class_name": "Irregular", "confidence": 0.9,
                    "bbox_xywh": [0, 0, 10, 10],
                }],
                rejected_detections=[],
            )
            entry = lambda result: [{"result": result, "time": 0.1}]
            payload, evidence = build_ml_batch_payload(
                images,
                {
                    "quick_analysis": entry(traditional),
                    "deep_analysis": entry(traditional),
                    "ml_benchmark": entry(ml),
                },
                timestamp="2026-08-25 00:00:00",
                model_path=str(model),
                confidence_threshold=0.25,
            )
            self.assertEqual(payload["ml_benchmark"]["f1_score"], 1.0)
            self.assertEqual(payload["ml_benchmark"]["detected_total"], 1)
            self.assertTrue(payload["ml_benchmark"]["legacy_count_diagnostic"]["diagnostic_only"])
            self.assertEqual(len(evidence["ml_lineage"]), 1)

    def test_report_renders_unavailable_metrics_as_na(self):
        unavailable = {
            "detected": 0.0, "ground_truth": 1.0, "processing_time": 0.0,
            "precision": None, "recall": None, "f1_score": None,
            "color_coverage": 0.0, "color_evaluated_count": 0, "detected_total": 0,
        }
        html = ReportGenerator()._build_html(
            {
                "num_images": 1,
                "quick_analysis": unavailable,
                "deep_analysis": unavailable,
                "ml_benchmark": unavailable,
                "evaluation_method": {
                    "availability": "unavailable", "iou_threshold": 0.5,
                    "classes": ["Microbead/Pellet", "Fiber/Filament", "Irregular"],
                    "unavailable_reasons": ["missing bbox"],
                },
            },
            {"color_comparison": "data:image/png;base64,test"},
        )
        self.assertIn("Precision: N/A", html)
        self.assertIn("missing bbox", html)
        self.assertIn("ROI color post-processing coverage", html)


if __name__ == "__main__":
    unittest.main()
