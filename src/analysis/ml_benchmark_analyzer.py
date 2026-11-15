"""
ML Benchmark Analyzer
Combines YOLO detection with traditional shape/color analysis for comparison
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Optional

from config.settings import PreprocessingParams, AnalysisResult
from src.core import ImageProcessor, ShapeAnalyzer, ColorAnalyzer


class MLBenchmarkAnalyzer:
    """
    ML-based analyzer that combines YOLO detection with shape/color analysis.
    Provides comparable metrics to Quick and Deep analysis for benchmarking.
    """
    
    def __init__(self, yolo_model):
        """
        Initialize ML Benchmark Analyzer
        
        Args:
            yolo_model: Loaded YOLO model instance
        """
        self.yolo_model = yolo_model
        self.shape_analyzer = ShapeAnalyzer()
        self.color_analyzer = ColorAnalyzer()
        self.image_processor = ImageProcessor()
    
    def analyze(self, image: np.ndarray, params: PreprocessingParams = None, 
                confidence_threshold: float = 0.25) -> AnalysisResult:
        """
        Perform ML-based detection and extract detailed metrics.
        
        Args:
            image: Input RGB image
            params: Preprocessing parameters (for compatibility)
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            AnalysisResult with ML detections and computed metrics
        """
        start_time = time.time()
        
        if image is None or image.size == 0:
            return AnalysisResult(
                features=[],
                mask=np.zeros((1, 1), dtype=np.uint8),
                processing_time=0,
                params=params or PreprocessingParams(),
                analysis_type='ml_benchmark'
            )
        
        # Detect background color
        bg_color = self.image_processor.detect_background_color(image)
        
        # Run YOLO detection
        results = self.yolo_model(image, conf=confidence_threshold, verbose=False)
        
        # Create combined mask from all detections
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        features = []
        raw_detections = []
        rejected_detections = []
        prediction_id = 0
        
        for result in results:
            boxes = result.boxes
            for idx, box in enumerate(boxes):
                prediction_id += 1
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                class_name = result.names[cls] if cls < len(result.names) else f"Class_{cls}"
                raw_detection = {
                    'prediction_id': prediction_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox_xywh': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                }
                raw_detections.append(raw_detection)
                
                # Skip low confidence detections
                if conf < confidence_threshold:
                    rejected_detections.append({**raw_detection, 'rejection_reason': 'below_confidence_threshold'})
                    continue
                
                # Create mask for this detection
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                
                # Extract region of interest
                roi = image[y:y+h, x:x+w] if x >= 0 and y >= 0 and y+h <= image.shape[0] and x+w <= image.shape[1] else None
                
                if roi is None or roi.size == 0:
                    rejected_detections.append({**raw_detection, 'rejection_reason': 'invalid_or_empty_roi'})
                    continue
                
                # Create particle mask using segmentation on ROI.
                # preprocess_segment()/detect_background_color() assume the
                # background dominates the frame, which is false for a tight
                # YOLO crop -- use a crop-aware segmenter instead.
                roi_mask = self._segment_particle_roi(roi, bg_color, min_size=3)
                
                # Place roi_mask in the full image mask
                if roi_mask.shape == (h, w):
                    mask[y:y+h, x:x+w] = np.maximum(mask[y:y+h, x:x+w], roi_mask)
                
                # Compute detailed shape metrics from the segmented region
                metrics = self.shape_analyzer.compute_shape_metrics_consistent(
                    roi_mask,
                    compute_solidity=True
                )

                # Track whether segmentation produced valid metrics
                segmentation_ok = metrics is not None
                bbox_fallback = False

                if not segmentation_ok:
                    # Fallback: use original bounding box metrics
                    metrics = self._compute_bbox_metrics(x, y, w, h)
                    bbox_fallback = True
                else:
                    # Adjust centroid and bounding box to global coordinates.
                    cx, cy = metrics['centroid']
                    metrics['centroid'] = (cx + x, cy + y)
                    bx, by, bw, bh = metrics['bounding_box']
                    metrics['bounding_box'] = (bx + x, by + y, bw, bh)

                # Classify shape based on metrics
                shape_class = self.shape_analyzer.classify_shape_unified(metrics)

                # Extract color from ROI using roi_mask
                roi_binary = np.zeros_like(roi_mask)
                roi_binary[roi_mask > 0] = 255
                color_name, h_val, s_val, v_val = self.color_analyzer.extract_color_from_region(
                    roi, roi_binary
                )
                
                # Build feature dictionary and include diagnostics
                features.append({
                    'id': idx + 1,
                    'source_prediction_id': prediction_id,
                    'shape': shape_class,
                    'color': color_name,
                    'ml_class': class_name,
                    'ml_confidence': conf,
                    'area': metrics['area'],
                    'perimeter': metrics['perimeter'],
                    'circularity': metrics['circularity'],
                    'eccentricity': metrics['eccentricity'],
                    'aspect_ratio': metrics['aspect_ratio'],
                    'rectangularity': metrics['rectangularity'],
                    'solidity': metrics.get('solidity', 0),
                    'centroid': metrics['centroid'],
                    'bounding_box': metrics['bounding_box'],
                    # Full-precision box (not int(x,y,w,h) used for array
                    # indexing above), so IoU matching against GT does not
                    # silently lose up to ~1px per side on small particles.
                    'bbox_global': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                })
        
        # Diagnostics summary for ML detections
        total_detections = len(features)
        segmentation_ok_count = sum(1 for f in features if f.get('segmentation_ok'))
        bbox_fallback_count = sum(1 for f in features if f.get('bbox_fallback'))
        avg_roi_pixels = (sum(f.get('roi_mask_pixels', 0) for f in features) / total_detections) if total_detections > 0 else 0
        print(f"[ML Benchmark] Detections={total_detections}, segmentation_ok={segmentation_ok_count}, bbox_fallback={bbox_fallback_count}, avg_roi_mask_pixels={avg_roi_pixels:.1f}")
        
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            features=features,
            mask=mask,
            processing_time=processing_time,
            params=params or PreprocessingParams(),
            analysis_type='ml_benchmark',
            num_detections=len(features),
            background_color=bg_color,
            raw_detections=raw_detections,
            rejected_detections=rejected_detections
        )
    
    def _segment_particle_roi(self, roi: np.ndarray, bg_color: str, min_size: int = 3) -> np.ndarray:
        """
        Segment a single YOLO-cropped ROI into a particle mask.

        Uses an Otsu split on the ROI's own intensity histogram instead of
        the full-frame mean/std heuristics in ImageProcessor, since a tight
        detection crop is often mostly particle rather than mostly background.
        Falls back to treating the whole ROI as the particle when Otsu finds
        no usable contrast -- an empty mask is a worse estimate than the box
        itself, given YOLO already confirmed a particle is present.

        Args:
            roi: RGB crop for one detection
            bg_color: 'white', 'black', or 'unknown' background type detected
                on the full image (not re-detected per-ROI)
            min_size: Minimum connected-component size to keep

        Returns:
            Binary mask (uint8, 0/255) matching roi.shape[:2]
        """
        roi_shape = roi.shape[:2]
        # Max-channel intensity stays bright for any single-hue fluorescent
        # particle (pure blue/red/green all peak on one RGB channel).
        intensity = np.max(roi, axis=2).astype(np.uint8)
        if bg_color == 'white':
            intensity = 255 - intensity

        total = intensity.size
        if total == 0:
            return np.zeros(roi_shape, dtype=np.uint8)

        full_roi_mask = np.full(roi_shape, 255, dtype=np.uint8)
        try:
            _, binary = cv2.threshold(intensity, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except cv2.error:
            return full_roi_mask

        foreground = int(np.count_nonzero(binary))
        # Otsu degenerates (near-empty or near-full) when the crop lacks real
        # contrast; assume the whole ROI is the particle rather than nothing.
        if foreground < min_size or foreground > total - min_size:
            return full_roi_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned_bool = ImageProcessor._remove_small_components(cleaned > 0, min_size)
        final_mask = (cleaned_bool * 255).astype(np.uint8)

        if np.count_nonzero(final_mask) == 0:
            return full_roi_mask
        return final_mask

    def _compute_bbox_metrics(self, x: int, y: int, w: int, h: int) -> Dict:
        """
        Compute basic metrics from bounding box when segmentation fails.
        
        Args:
            x, y, w, h: Bounding box coordinates
            
        Returns:
            Basic metrics dictionary
        """
        area = w * h
        perimeter = 2 * (w + h)
        circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        aspect_ratio = max(w, h) / max(min(w, h), 1e-6)
        rectangularity = 1.0  # Perfect rectangle
        
        # Estimate eccentricity from aspect ratio
        # For rectangles: ecc ≈ sqrt(1 - (min/max)^2)
        ratio = min(w, h) / max(w, h)
        eccentricity = np.sqrt(max(0, 1 - ratio * ratio))
        
        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': min(circularity, 1.0),
            'eccentricity': min(eccentricity, 1.0),
            'aspect_ratio': aspect_ratio,
            'rectangularity': rectangularity,
            'solidity': 1.0,  # Assume solid
            'centroid': (x + w/2, y + h/2),
            'bounding_box': (x, y, w, h)
        }
