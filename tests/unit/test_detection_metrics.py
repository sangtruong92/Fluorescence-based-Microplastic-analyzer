import unittest

from src.analysis.detection_metrics import (
    bbox_iou_xywh,
    evaluate_detection_dataset,
    legacy_count_agreement,
    match_detections,
    normalize_benchmark_class,
    parse_benchmark_ground_truth,
    unavailable_evaluation,
)


def detection(class_name="Irregular", bbox=(0, 0, 10, 10), confidence=0.9, prediction_id=1):
    return {"class_name": class_name, "bbox_xywh": bbox, "confidence": confidence, "prediction_id": prediction_id}


def truth(class_name="Irregular", bbox=(0, 0, 10, 10), particle_id=1):
    return {"class_name": class_name, "bbox_xywh": bbox, "id": particle_id}


class DetectionMetricsTests(unittest.TestCase):
    def test_class_aliases_match_three_class_ontology(self):
        self.assertEqual(normalize_benchmark_class("Fragment"), "Irregular")
        self.assertEqual(normalize_benchmark_class("Fiber"), "Fiber/Filament")
        self.assertEqual(normalize_benchmark_class("Pellet"), "Microbead/Pellet")
        self.assertIsNone(normalize_benchmark_class("Unknown"))

    def test_iou_and_perfect_match(self):
        self.assertEqual(bbox_iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        result = match_detections([detection()], [truth()])
        self.assertEqual((result["true_positives"], result["false_positives"], result["false_negatives"]), (1, 0, 0))
        self.assertEqual(result["f1_score"], 1.0)

    def test_wrong_location_is_fp_and_fn(self):
        result = match_detections([detection(bbox=(20, 20, 10, 10))], [truth()])
        self.assertEqual((result["true_positives"], result["false_positives"], result["false_negatives"]), (0, 1, 1))

    def test_wrong_class_is_fp_and_fn(self):
        result = match_detections([detection(class_name="Fiber/Filament")], [truth()])
        self.assertEqual((result["true_positives"], result["false_positives"], result["false_negatives"]), (0, 1, 1))

    def test_duplicate_prediction_matches_only_once(self):
        result = match_detections([detection(prediction_id=1), detection(confidence=0.8, prediction_id=2)], [truth()])
        self.assertEqual((result["true_positives"], result["false_positives"], result["false_negatives"]), (1, 1, 0))
        self.assertEqual(result["matches"][0]["prediction_id"], 1)

    def test_empty_cases_are_not_treated_as_perfect_detection(self):
        missed = match_detections([], [truth()])
        extra = match_detections([detection()], [])
        self.assertEqual((missed["recall"], missed["false_negatives"]), (0.0, 1))
        self.assertEqual((extra["precision"], extra["false_positives"]), (0.0, 1))

    def test_dataset_aggregate_and_per_class_metrics(self):
        result = evaluate_detection_dataset(
            [[detection()], [detection(class_name="Fiber/Filament")]],
            [[truth()], [truth(class_name="Fiber/Filament", bbox=(50, 50, 10, 10))]],
            image_ids=["a", "b"],
        )
        self.assertEqual((result["true_positives"], result["false_positives"], result["false_negatives"]), (1, 1, 1))
        self.assertEqual(result["per_class"]["Irregular"]["f1_score"], 1.0)
        self.assertEqual(result["per_class"]["Fiber/Filament"]["f1_score"], 0.0)

    def test_ground_truth_parser_requires_bbox_and_normalizes_alias(self):
        parsed = parse_benchmark_ground_truth(
            """Total Particles: 1\n\nParticle 1:\n  Shape: Fragment\n  Color: Red\n  Bounding Box: (1, 2, 3, 4)\n  Area: 12.0 px2\n""",
            image_size=(20, 20),
        )
        self.assertEqual(parsed[0]["class_name"], "Irregular")
        self.assertEqual(parsed[0]["bbox_xywh"], [1.0, 2.0, 3.0, 4.0])
        with self.assertRaisesRegex(ValueError, "Bounding Box"):
            parse_benchmark_ground_truth("Total Particles: 1\n\nParticle 1:\n  Shape: Fiber\n")

    def test_malformed_bbox_is_rejected(self):
        for bbox in ((0, 0, 0, 1), (-1, 0, 1, 1), (0, 0, float("nan"), 1)):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                match_detections([detection(bbox=bbox)], [truth()])

    def test_unavailable_state_and_legacy_diagnostic_are_explicit(self):
        unavailable = unavailable_evaluation("missing bbox")
        legacy = legacy_count_agreement([[1]], [[1]])
        self.assertFalse(unavailable["available"])
        self.assertIsNone(unavailable["precision"])
        self.assertTrue(legacy["diagnostic_only"])
        self.assertEqual(legacy["method"], "legacy_count_agreement_not_detection_quality")


if __name__ == "__main__":
    unittest.main()
