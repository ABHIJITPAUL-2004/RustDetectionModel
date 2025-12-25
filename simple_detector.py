import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

class RustDetector:
    def __init__(self):
        # HSV thresholds for rust detection
        self.rust_lower = np.array([5, 50, 50])
        self.rust_upper = np.array([25, 255, 200])
    
    def segment_rails(self, image: np.ndarray) -> np.ndarray:
        """Stage 1: Advanced metal detection - isolate only rail metal surfaces"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Metal detection using texture and intensity
        # Rails are typically darker, smoother, and more uniform than ballast
        
        # 1. Intensity-based filtering (rails are darker)
        _, dark_regions = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        
        # 2. Texture analysis - rails have less texture variation
        kernel = np.ones((5,5), np.float32) / 25
        smooth = cv2.filter2D(gray, -1, kernel)
        texture_diff = cv2.absdiff(gray, smooth)
        _, low_texture = cv2.threshold(texture_diff, 15, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Edge-based rail detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect long straight lines (rail edges)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        line_mask = np.zeros_like(gray)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Only keep near-vertical or near-horizontal lines (rail orientation)
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if angle < 30 or angle > 150:  # Vertical-ish lines
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 8)
        
        # 4. Combine all metal detection criteria
        metal_mask = cv2.bitwise_and(dark_regions, low_texture)
        metal_mask = cv2.bitwise_and(metal_mask, cv2.dilate(line_mask, np.ones((5,5), np.uint8)))
        
        # 5. Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small objects (pebbles, noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_OPEN, kernel)
        
        return (metal_mask > 0).astype(np.uint8)
    
    def refine_rail_mask(self, mask: np.ndarray) -> np.ndarray:
        """Stage 2: Refine rail mask using morphological operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Close gaps in rails
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def extract_rail_only(self, image: np.ndarray, rail_mask: np.ndarray) -> np.ndarray:
        """Stage 3: Extract only rail regions, remove ballast/pebbles"""
        rail_only = image.copy()
        rail_only[rail_mask == 0] = [0, 0, 0]  # Black out non-rail areas
        return rail_only
    
    def detect_rust_hsv(self, rail_image: np.ndarray, rail_mask: np.ndarray) -> np.ndarray:
        """Stage 4: Detect rust using HSV color space"""
        hsv = cv2.cvtColor(rail_image, cv2.COLOR_BGR2HSV)
        
        # Create rust mask based on color
        rust_mask = cv2.inRange(hsv, self.rust_lower, self.rust_upper)
        
        # Apply only to rail regions
        rust_mask = cv2.bitwise_and(rust_mask, rail_mask * 255)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_CLOSE, kernel)
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)
        
        return (rust_mask > 0).astype(np.uint8)
    
    def calculate_rust_severity(self, rust_mask: np.ndarray, rail_mask: np.ndarray) -> Dict:
        """Stage 5: Calculate rust severity metrics"""
        rail_pixels = np.sum(rail_mask)
        rust_pixels = np.sum(rust_mask)
        
        if rail_pixels == 0:
            return {"percentage": 0, "severity": "No Rails Detected", "regions": 0}
        
        rust_percentage = (rust_pixels / rail_pixels) * 100
        
        # Find rust regions
        contours, _ = cv2.findContours(rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_regions = len(contours)
        
        # Classify severity
        if rust_percentage < 5:
            severity = "Healthy"
        elif rust_percentage < 15:
            severity = "Moderate Rust"
        else:
            severity = "Critical"
        
        return {
            "percentage": round(rust_percentage, 2),
            "severity": severity,
            "regions": num_regions,
            "rail_pixels": rail_pixels,
            "rust_pixels": rust_pixels
        }
    
    def process_image(self, image_path: str) -> Dict:
        """Complete pipeline: Process railway image and detect rust"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print("Stage 1: Segmenting rails...")
        rail_mask = self.segment_rails(image)
        
        print("Stage 2: Refining rail mask...")
        rail_mask_refined = self.refine_rail_mask(rail_mask)
        
        print("Stage 3: Extracting rail regions...")
        rail_only = self.extract_rail_only(image, rail_mask_refined)
        
        print("Stage 4: Detecting rust...")
        rust_mask = self.detect_rust_hsv(rail_only, rail_mask_refined)
        
        print("Stage 5: Calculating severity...")
        severity_metrics = self.calculate_rust_severity(rust_mask, rail_mask_refined)
        
        # Generate intermediate processing images
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        
        # Texture analysis
        kernel = np.ones((5,5), np.float32) / 25
        smooth = cv2.filter2D(gray, -1, kernel)
        texture_diff = cv2.absdiff(gray, smooth)
        _, low_texture = cv2.threshold(texture_diff, 15, 255, cv2.THRESH_BINARY_INV)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # HSV analysis
        hsv = cv2.cvtColor(rail_only, cv2.COLOR_BGR2HSV)
        
        # Morphological operations
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(rail_mask_refined * 255, cv2.MORPH_OPEN, kernel_morph)
        
        return {
            "original_image": image,
            "binary_mask": binary_mask,
            "texture_mask": low_texture,
            "edges": edges,
            "rail_mask": rail_mask_refined,
            "morphological": opened,
            "rail_only": rail_only,
            "hsv_image": hsv,
            "rust_mask": rust_mask,
            "metrics": severity_metrics
        }
    
    def visualize_results(self, results: Dict, save_path: str = None):
        """Visualize detection results with green rail lines"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(results["original_image"], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Rail mask
        axes[0, 1].imshow(results["rail_mask"], cmap='gray')
        axes[0, 1].set_title("Rail Segmentation")
        axes[0, 1].axis('off')
        
        # Rail only
        axes[0, 2].imshow(cv2.cvtColor(results["rail_only"], cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Rails Only (Ballast Removed)")
        axes[0, 2].axis('off')
        
        # Rust mask
        axes[1, 0].imshow(results["rust_mask"], cmap='Reds')
        axes[1, 0].set_title("Rust Detection")
        axes[1, 0].axis('off')
        
        # Overlay with green rail lines and red rust
        overlay = results["original_image"].copy()
        
        # Draw green lines for detected rails
        rail_contours, _ = cv2.findContours(results["rail_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in rail_contours:
            if cv2.contourArea(contour) > 1000:  # Only draw significant rail areas
                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 3)  # Green outline
        
        # Highlight rust areas in red
        overlay[results["rust_mask"] == 1] = [0, 0, 255]
        
        axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Green=Rails, Red=Rust")
        axes[1, 1].axis('off')
        
        # Metrics
        metrics = results["metrics"]
        axes[1, 2].text(0.1, 0.8, f"Rust Percentage: {metrics['percentage']}%", fontsize=12)
        axes[1, 2].text(0.1, 0.6, f"Severity: {metrics['severity']}", fontsize=12)
        axes[1, 2].text(0.1, 0.4, f"Rust Regions: {metrics['regions']}", fontsize=12)
        axes[1, 2].text(0.1, 0.2, f"Rail Pixels: {metrics['rail_pixels']}", fontsize=10)
        axes[1, 2].set_title("Analysis Results")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    detector = RustDetector()
    
    try:
        results = detector.process_image("railway_image.jpg")
        detector.visualize_results(results, "rust_analysis_results.png")
        
        print("\n=== RUST DETECTION RESULTS ===")
        print(f"Rust Percentage: {results['metrics']['percentage']}%")
        print(f"Severity Level: {results['metrics']['severity']}")
        print(f"Number of Rust Regions: {results['metrics']['regions']}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please provide a valid railway image path")