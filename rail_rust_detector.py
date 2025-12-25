import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

class RailSegmentationModel(nn.Module):
    """Lightweight U-Net for rail segmentation"""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(3, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        
        # Decoder
        self.dec3 = self._conv_block(256, 128)
        self.dec2 = self._conv_block(128, 64)
        self.dec1 = nn.Conv2d(64, 2, 1)  # 2 classes: rail, background
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        
        # Decoder
        d3 = self.dec3(nn.Upsample(scale_factor=2)(e3))
        d2 = self.dec2(nn.Upsample(scale_factor=2)(d3))
        return torch.softmax(self.dec1(d2), dim=1)

class RustDetector:
    def __init__(self):
        self.rail_model = RailSegmentationModel()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rail_model.to(self.device)
        
        # HSV thresholds for rust detection
        self.rust_lower = np.array([5, 50, 50])
        self.rust_upper = np.array([25, 255, 200])
        
        # Transform for model input
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def segment_rails(self, image: np.ndarray) -> np.ndarray:
        """Stage 1: Segment rails from background using semantic segmentation"""
        # Convert to PIL for transforms
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.rail_model(input_tensor)
            rail_mask = output[0, 1].cpu().numpy()  # Get rail class
        
        # Resize back to original size
        rail_mask = cv2.resize(rail_mask, (image.shape[1], image.shape[0]))
        return (rail_mask > 0.5).astype(np.uint8)
    
    def refine_rail_mask(self, mask: np.ndarray) -> np.ndarray:
        """Stage 2: Refine rail mask using morphological operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Close gaps in rails
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Keep only largest connected components (actual rails)
            areas = [cv2.contourArea(c) for c in contours]
            if areas:
                max_area = max(areas)
                mask_refined = np.zeros_like(mask)
                for contour in contours:
                    if cv2.contourArea(contour) > max_area * 0.1:  # Keep significant components
                        cv2.fillPoly(mask_refined, [contour], 1)
                return mask_refined
        
        return mask
    
    def extract_rail_only(self, image: np.ndarray, rail_mask: np.ndarray) -> np.ndarray:
        """Stage 3: Extract only rail regions, remove ballast/pebbles"""
        rail_only = image.copy()
        rail_only[rail_mask == 0] = [0, 0, 0]  # Black out non-rail areas
        return rail_only
    
    def detect_rust_hsv(self, rail_image: np.ndarray, rail_mask: np.ndarray) -> np.ndarray:
        """Stage 4A: Detect rust using HSV color space (baseline method)"""
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
        
        return {
            "original_image": image,
            "rail_mask": rail_mask_refined,
            "rail_only": rail_only,
            "rust_mask": rust_mask,
            "metrics": severity_metrics
        }
    
    def visualize_results(self, results: Dict, save_path: str = None):
        """Visualize detection results"""
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
        
        # Overlay
        overlay = results["original_image"].copy()
        overlay[results["rust_mask"] == 1] = [0, 0, 255]  # Red for rust
        axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Rust Overlay")
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

# Example usage and testing
if __name__ == "__main__":
    detector = RustDetector()
    
    # Process an image (replace with actual image path)
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