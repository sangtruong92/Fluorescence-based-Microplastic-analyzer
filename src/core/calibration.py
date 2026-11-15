"""
Camera calibration and real-world size conversion for microplastic analysis

This module provides tools to convert pixel measurements to real-world dimensions
using camera calibration data.

Example usage:
    # Calibrate with reference object
    calib = CameraCalibration()
    calib.calibrate_from_reference(reference_size_um=100, measured_pixels=150)
    
    # Convert measurements
    real_area = calib.pixel_area_to_um2(particle_area_pixels)
    diameter = calib.calculate_real_diameter(particle_area_pixels)
"""

import logging
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class CameraCalibration:
    """
    Handles camera calibration and pixel-to-real-world conversion
    
    Attributes:
        um_per_pixel_x (float): Micrometers per pixel in X direction
        um_per_pixel_y (float): Micrometers per pixel in Y direction
        is_calibrated (bool): Whether calibration has been performed
    """
    
    def __init__(self, um_per_pixel=None):
        """
        Initialize calibration
        
        Parameters:
            um_per_pixel (float, tuple, or None): Micrometers per pixel
                - None: Not calibrated (defaults to 1.0)
                - float: Same calibration for both axes
                - tuple: (x, y) different calibration per axis
        """
        if um_per_pixel is None:
            # Default: not calibrated
            self.um_per_pixel_x = 1.0
            self.um_per_pixel_y = 1.0
            self.is_calibrated = False
        elif isinstance(um_per_pixel, (tuple, list)):
            self.um_per_pixel_x = float(um_per_pixel[0])
            self.um_per_pixel_y = float(um_per_pixel[1])
            self.is_calibrated = True
        else:
            self.um_per_pixel_x = float(um_per_pixel)
            self.um_per_pixel_y = float(um_per_pixel)
            self.is_calibrated = True
    
    def pixel_area_to_um2(self, area_pixels):
        """
        Convert pixel area to square micrometers
        
        Parameters:
            area_pixels (float): Area in square pixels
            
        Returns:
            float: Area in square micrometers (μm²)
        """
        return area_pixels * self.um_per_pixel_x * self.um_per_pixel_y
    
    def um2_to_pixel_area(self, area_um2):
        """
        Convert square micrometers to pixel area
        
        Parameters:
            area_um2 (float): Area in square micrometers
            
        Returns:
            float: Area in square pixels
        """
        return area_um2 / (self.um_per_pixel_x * self.um_per_pixel_y)
    
    def pixel_length_to_um(self, length_pixels, axis='x'):
        """
        Convert pixel length to micrometers
        
        Parameters:
            length_pixels (float): Length in pixels
            axis (str): 'x' or 'y' axis
            
        Returns:
            float: Length in micrometers
        """
        if axis.lower() == 'x':
            return length_pixels * self.um_per_pixel_x
        else:
            return length_pixels * self.um_per_pixel_y
    
    def um_to_pixel_length(self, length_um, axis='x'):
        """
        Convert micrometers to pixel length
        
        Parameters:
            length_um (float): Length in micrometers
            axis (str): 'x' or 'y' axis
            
        Returns:
            float: Length in pixels
        """
        if axis.lower() == 'x':
            return length_um / self.um_per_pixel_x
        else:
            return length_um / self.um_per_pixel_y
    
    def calculate_real_diameter(self, area_pixels):
        """
        Calculate equivalent circular diameter in micrometers
        
        Parameters:
            area_pixels (float): Area in square pixels
            
        Returns:
            float: Equivalent circular diameter in micrometers
        """
        area_um2 = self.pixel_area_to_um2(area_pixels)
        diameter_um = 2 * np.sqrt(area_um2 / np.pi)
        return diameter_um
    
    def calculate_dimensions(self, width_pixels, height_pixels):
        """
        Calculate real-world dimensions from pixel measurements
        
        Parameters:
            width_pixels (float): Width in pixels
            height_pixels (float): Height in pixels
            
        Returns:
            tuple: (width_um, height_um) in micrometers
        """
        width_um = width_pixels * self.um_per_pixel_x
        height_um = height_pixels * self.um_per_pixel_y
        return width_um, height_um
    
    def calibrate_from_reference(self, reference_size_um, measured_pixels, axis='both'):
        """
        Calibrate using a reference object
        
        Parameters:
            reference_size_um (float): Known size of reference object in micrometers
            measured_pixels (float): Measured size in pixels
            axis (str): 'x', 'y', or 'both' - which axis to calibrate
        """
        calibration_factor = reference_size_um / measured_pixels
        
        if axis.lower() == 'both':
            self.um_per_pixel_x = calibration_factor
            self.um_per_pixel_y = calibration_factor
        elif axis.lower() == 'x':
            self.um_per_pixel_x = calibration_factor
        elif axis.lower() == 'y':
            self.um_per_pixel_y = calibration_factor
        
        self.is_calibrated = True
        logger.info(f"✓ Calibration complete: {calibration_factor:.3f} μm/pixel ({axis} axis)")
    
    def calibrate_from_fov(self, fov_width_mm, fov_height_mm, image_width_px, image_height_px):
        """
        Calibrate from known field of view dimensions
        
        Parameters:
            fov_width_mm (float): Field of view width in millimeters
            fov_height_mm (float): Field of view height in millimeters
            image_width_px (int): Image width in pixels
            image_height_px (int): Image height in pixels
        """
        self.um_per_pixel_x = (fov_width_mm * 1000) / image_width_px
        self.um_per_pixel_y = (fov_height_mm * 1000) / image_height_px
        self.is_calibrated = True
        
        logger.info(f"✓ Calibration from FOV:")
        logger.debug(f"  X-axis: {self.um_per_pixel_x:.3f} μm/pixel")
        logger.debug(f"  Y-axis: {self.um_per_pixel_y:.3f} μm/pixel")
    
    def calibrate_theoretical(self, sensor_width_mm, fov_degrees, working_distance_mm, 
                             resolution_width_px, aspect_ratio=16/9):
        """
        Theoretical calibration based on camera specifications
        
        Parameters:
            sensor_width_mm (float): Physical sensor width in mm
            fov_degrees (float): Diagonal field of view in degrees
            working_distance_mm (float): Distance to sample in mm
            resolution_width_px (int): Image width in pixels
            aspect_ratio (float): Width/height ratio (default 16:9)
        """
        import math
        
        # Calculate sensor height from aspect ratio
        sensor_height_mm = sensor_width_mm / aspect_ratio
        sensor_diagonal_mm = math.sqrt(sensor_width_mm**2 + sensor_height_mm**2)
        
        # Calculate focal length
        focal_length_mm = sensor_diagonal_mm / (2 * math.tan(math.radians(fov_degrees / 2)))
        
        # Calculate FOV at working distance
        fov_width_mm = (sensor_width_mm * working_distance_mm) / focal_length_mm
        fov_height_mm = (sensor_height_mm * working_distance_mm) / focal_length_mm
        
        # Calculate resolution
        resolution_height_px = int(resolution_width_px / aspect_ratio)
        
        self.um_per_pixel_x = (fov_width_mm * 1000) / resolution_width_px
        self.um_per_pixel_y = (fov_height_mm * 1000) / resolution_height_px
        self.is_calibrated = True
        
        logger.info(f"✓ Theoretical calibration:")
        logger.debug(f"  Focal length: {focal_length_mm:.2f} mm")
        logger.debug(f"  FOV: {fov_width_mm:.2f} × {fov_height_mm:.2f} mm")
        logger.debug(f"  X-axis: {self.um_per_pixel_x:.3f} μm/pixel")
        logger.debug(f"  Y-axis: {self.um_per_pixel_y:.3f} μm/pixel")
    
    def get_size_category(self, diameter_um):
        """
        Categorize particle by size (for microplastics)
        
        Parameters:
            diameter_um (float): Diameter in micrometers
            
        Returns:
            str: Size category
        """
        if diameter_um < 10:
            return "Nanoplastic (<10 μm)"
        elif diameter_um < 100:
            return "Small Microplastic (10-100 μm)"
        elif diameter_um < 1000:
            return "Medium Microplastic (100-1000 μm)"
        elif diameter_um < 5000:
            return "Large Microplastic (1-5 mm)"
        else:
            return "Macroplastic (>5 mm)"
    
    def convert_features(self, features_dict):
        """
        Add calibrated measurements to a feature dictionary
        
        Parameters:
            features_dict (dict): Feature dictionary with pixel measurements
                                 Must contain 'area' key
            
        Returns:
            dict: Updated dictionary with real-world measurements
        """
        if not self.is_calibrated:
            logger.warning("⚠ Warning: Camera not calibrated. Using pixel values.")
            return features_dict
        
        # Create copy to avoid modifying original
        features = features_dict.copy()
        
        # Convert area
        if 'area' in features:
            features['area_um2'] = self.pixel_area_to_um2(features['area'])
            features['diameter_um'] = self.calculate_real_diameter(features['area'])
            features['size_category'] = self.get_size_category(features['diameter_um'])
        
        # Convert perimeter
        if 'perimeter' in features:
            features['perimeter_um'] = self.pixel_length_to_um(features['perimeter'])
        
        # Convert bounding box dimensions
        if 'bbox_width' in features and 'bbox_height' in features:
            width_um, height_um = self.calculate_dimensions(
                features['bbox_width'], 
                features['bbox_height']
            )
            features['bbox_width_um'] = width_um
            features['bbox_height_um'] = height_um
        
        # Convert major/minor axis
        if 'major_axis' in features:
            features['major_axis_um'] = self.pixel_length_to_um(features['major_axis'])
        
        if 'minor_axis' in features:
            features['minor_axis_um'] = self.pixel_length_to_um(features['minor_axis'])
        
        return features
    
    def save_calibration(self, filepath='calibration.json'):
        """
        Save calibration to JSON file
        
        Parameters:
            filepath (str): Path to save calibration file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'um_per_pixel_x': self.um_per_pixel_x,
            'um_per_pixel_y': self.um_per_pixel_y,
            'is_calibrated': self.is_calibrated,
            'note': 'Micrometers per pixel calibration for microplastic analysis'
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        
        logger.info(f"✓ Calibration saved to {filepath}")
    
    def load_calibration(self, filepath='calibration.json'):
        """
        Load calibration from JSON file
        
        Parameters:
            filepath (str): Path to calibration file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"✗ Calibration file not found: {filepath}")
            return False
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.um_per_pixel_x = data['um_per_pixel_x']
        self.um_per_pixel_y = data['um_per_pixel_y']
        self.is_calibrated = data['is_calibrated']
        
        logger.info(f"✓ Calibration loaded from {filepath}")
        logger.debug(f"  X-axis: {self.um_per_pixel_x:.3f} μm/pixel")
        logger.debug(f"  Y-axis: {self.um_per_pixel_y:.3f} μm/pixel")
        
        return True
    
    def __str__(self):
        """String representation of calibration"""
        if self.is_calibrated:
            return (f"CameraCalibration(calibrated=True, "
                   f"x={self.um_per_pixel_x:.3f} μm/px, "
                   f"y={self.um_per_pixel_y:.3f} μm/px)")
        else:
            return "CameraCalibration(calibrated=False)"
    
    def __repr__(self):
        return self.__str__()


# Example usage and testing
if __name__ == "__main__":
    print("=== Camera Calibration Module Demo ===\n")
    
    # Method 1: Calibrate with reference object
    print("Method 1: Reference object calibration")
    calib = CameraCalibration()
    calib.calibrate_from_reference(reference_size_um=100, measured_pixels=40)
    
    # Convert measurements
    particle_area_pixels = 500
    real_area = calib.pixel_area_to_um2(particle_area_pixels)
    diameter = calib.calculate_real_diameter(particle_area_pixels)
    
    print(f"  Particle area: {particle_area_pixels} px² → {real_area:.1f} μm²")
    print(f"  Equivalent diameter: {diameter:.1f} μm")
    print(f"  Size category: {calib.get_size_category(diameter)}\n")
    
    # Method 2: Theoretical calibration (your camera setup)
    print("Method 2: Theoretical calibration (your setup)")
    calib2 = CameraCalibration()
    calib2.calibrate_theoretical(
        sensor_width_mm=5.03,      # 1/2.5" sensor
        fov_degrees=94.5,          # Your camera FOV
        working_distance_mm=5.0,   # Water layer depth
        resolution_width_px=3840   # Your resolution
    )
    
    real_area2 = calib2.pixel_area_to_um2(particle_area_pixels)
    diameter2 = calib2.calculate_real_diameter(particle_area_pixels)
    print(f"  Same particle: {real_area2:.1f} μm², {diameter2:.1f} μm\n")
    
    # Method 3: Convert feature dictionary
    print("Method 3: Convert feature dictionary")
    features = {
        'area': 500,
        'perimeter': 80,
        'bbox_width': 30,
        'bbox_height': 25,
        'circularity': 0.78
    }
    
    features_calibrated = calib.convert_features(features)
    print(f"  Original: {features['area']} px²")
    print(f"  Calibrated: {features_calibrated['area_um2']:.1f} μm²")
    print(f"  Diameter: {features_calibrated['diameter_um']:.1f} μm")
    print(f"  Category: {features_calibrated['size_category']}\n")
    
    # Save calibration
    print("Method 4: Save/load calibration")
    calib.save_calibration('calibration.json')
    
    # Load it back
    new_calib = CameraCalibration()
    new_calib.load_calibration('calibration.json')
    
    print(f"\n{calib}")
