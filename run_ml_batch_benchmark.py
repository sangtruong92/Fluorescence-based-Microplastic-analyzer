"""
Standalone (GUI-free) replica of `_run_batch_ml_benchmark` in src/gui/main_window.py.

Runs Quick / Deep / ML Benchmark (yolo26m_v2) over benchmark_results/dataset/20260824
and writes the same HTML + .benchmark.json sidecar artifacts the GUI would produce,
so the corrected color/area pipeline can be validated on the full 200-image dataset.

Run with: ./.venv/bin/python3 run_ml_batch_benchmark.py
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from config.settings import PreprocessingParams, MLModelConfig
from src.analysis import QuickAnalyzer, DeepAnalyzer, ReportGenerator
from src.analysis.ml_benchmark_analyzer import MLBenchmarkAnalyzer
from src.analysis.benchmark_batch import build_ml_batch_payload, load_benchmark_images, run_analyzer_over_images
from src.analysis.benchmark_snapshot import save_benchmark_snapshot

DATASET_DIR = Path('benchmark_results/dataset/20260824')
MODEL_PATH = 'src/ml/yolo26m_v2/best.pt'

# Mirrors GUI widget defaults in src/gui/main_window.py (get_analysis_params)
QUICK_PARAMS = PreprocessingParams(method='basic', min_area=20, max_area=10000, blur=5, distance_thresh=0.5)
DEEP_PARAMS = PreprocessingParams(
    method='advanced', min_area=20, max_area=10000, blur=3, distance_thresh=0.4,
    marker_size=5, brightness_threshold=0, min_blur_score=0.0, adaptive_c_value=2,
)


def main():
    print(f"Loading dataset from {DATASET_DIR} ...")
    images_data = load_benchmark_images(DATASET_DIR)
    print(f"Loaded {len(images_data)} images "
          f"({sum(1 for d in images_data if d['ground_truth'] > 0)} with ground truth)")

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    ml_confidence_threshold = MLModelConfig().confidence_threshold

    def progress(label):
        def on_progress(i, total):
            if i % 25 == 0 or i == total:
                print(f"  [{label}] {i}/{total}")
        return on_progress

    print("\n[1/3] Quick Analysis...")
    quick_results_list = run_analyzer_over_images(QuickAnalyzer(), images_data, QUICK_PARAMS, on_progress=progress('Quick'))

    print("\n[2/3] Deep Analysis...")
    deep_results_list = run_analyzer_over_images(DeepAnalyzer(), images_data, DEEP_PARAMS, on_progress=progress('Deep'))

    print("\n[3/3] ML Benchmark (yolo26m_v2, with color/area fix)...")
    ml_results_list = run_analyzer_over_images(
        MLBenchmarkAnalyzer(model), images_data, DEEP_PARAMS,
        confidence_threshold=ml_confidence_threshold, on_progress=progress('ML'),
    )

    report_timestamp = datetime.now()
    batch_data, evidence = build_ml_batch_payload(
        images_data,
        {
            'quick_analysis': quick_results_list,
            'deep_analysis': deep_results_list,
            'ml_benchmark': ml_results_list,
        },
        timestamp=report_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        model_path=MODEL_PATH,
        confidence_threshold=ml_confidence_threshold,
    )

    report_gen = ReportGenerator()
    timestamp = report_timestamp.strftime('%Y%m%d_%H%M%S')
    output_dir = Path('benchmark_results')
    output_dir.mkdir(exist_ok=True)

    html_path = output_dir / f'ml_benchmark_{len(images_data)}images_{timestamp}.html'
    report_gen.generate_benchmark_report(batch_data, str(html_path))
    snapshot_path = save_benchmark_snapshot(html_path, evidence)

    print(f"\nHTML report  -> {html_path}")
    print(f"Sidecar JSON -> {snapshot_path}")

    ml_eval = batch_data['ml_benchmark']
    print("\nML Benchmark (yolo26m_v2) overall metrics:")
    print(f"  Precision: {ml_eval.get('precision')}")
    print(f"  Recall:    {ml_eval.get('recall')}")
    print(f"  F1:        {ml_eval.get('f1_score')}")
    color_dist = ml_eval.get('color_distribution', {})
    total_colors = sum(color_dist.values()) or 1
    print(f"  Color distribution: {color_dist}")
    print(f"  Unknown color rate: {100 * color_dist.get('Unknown', 0) / total_colors:.1f}%")


if __name__ == '__main__':
    main()
