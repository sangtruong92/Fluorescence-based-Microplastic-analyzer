import unittest
import tempfile
from pathlib import Path

from config.settings import SyntheticImageParams
from src.data_generation.synthetic_generator import SyntheticImageGenerator
from src.data_generation.yolo_exporter import YOLODatasetExporter
from src.analysis.detection_metrics import parse_benchmark_ground_truth


class YoloExporterFixesTests(unittest.TestCase):
    def test_canonical_classes_and_mapping(self):
        self.assertEqual(
            YOLODatasetExporter.CANONICAL_CLASSES,
            ['Microbead/Pellet', 'Fiber/Filament', 'Irregular']
        )
        self.assertEqual(YOLODatasetExporter.get_class_id('Microbead'), 0)
        self.assertEqual(YOLODatasetExporter.get_class_id('Pellet'), 0)
        self.assertEqual(YOLODatasetExporter.get_class_id('Fiber'), 1)
        self.assertEqual(YOLODatasetExporter.get_class_id('Filament'), 1)
        self.assertEqual(YOLODatasetExporter.get_class_id('Fragment'), 2)
        self.assertEqual(YOLODatasetExporter.get_class_id('Irregular'), 2)

    def test_synthetic_generator_outputs_exact_bounding_box(self):
        gen = SyntheticImageGenerator()
        params = SyntheticImageParams(
            num_particles=10,
            shape_type='Mixed',
            image_width=640,
            image_height=640,
            enable_blur=False,
            enable_glow=False
        )
        img, gt = gen.generate(params)
        self.assertEqual(len(gt), 10)
        for p in gt:
            self.assertIn('bounding_box', p)
            self.assertIn('bbox_xywh', p)
            bx, by, bw, bh = p['bounding_box']
            self.assertGreater(bw, 0)
            self.assertGreater(bh, 0)
            self.assertEqual(p['bbox_xywh'], [float(bx), float(by), float(bw), float(bh)])

    def test_ground_truth_to_yolo_bbox_uses_exact_bbox(self):
        particle = {
            'shape': 'Fiber',
            'position': (100, 200),
            'bounding_box': (80, 170, 40, 60),
            'bbox_xywh': [80.0, 170.0, 40.0, 60.0]
        }
        cls_id, xc, yc, w, h = YOLODatasetExporter.ground_truth_to_yolo_bbox(particle, 1000, 1000)
        self.assertEqual(cls_id, 1)
        self.assertAlmostEqual(xc, (80 + 20) / 1000.0)
        self.assertAlmostEqual(yc, (170 + 30) / 1000.0)
        self.assertAlmostEqual(w, 40 / 1000.0)
        self.assertAlmostEqual(h, 60 / 1000.0)

    def test_detailed_ground_truth_can_be_parsed_by_benchmark(self):
        particles = [
            {
                'id': 1,
                'shape': 'Fiber',
                'color_label': 'Green',
                'wavelength_nm': 525,
                'position': (100.0, 200.0),
                'bounding_box': (80, 170, 40, 60),
                'bbox_xywh': [80.0, 170.0, 40.0, 60.0],
                'area': 240.0,
                'size': 30.0
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = Path(tmpdir) / 'test_gt.txt'
            YOLODatasetExporter.save_detailed_ground_truth(gt_path, particles)
            content = gt_path.read_text()
            parsed = parse_benchmark_ground_truth(content)
            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0]['class_name'], 'Fiber/Filament')
            self.assertEqual(parsed[0]['bbox_xywh'], [80.0, 170.0, 40.0, 60.0])


if __name__ == '__main__':
    unittest.main()
