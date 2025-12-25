import cv2
import numpy as np
import matplotlib.pyplot as plt
from simple_detector import RustDetector

def create_synthetic_railway_image():
    """Create a synthetic railway image for testing"""
    # Create base image
    img = np.ones((400, 600, 3), dtype=np.uint8) * 120  # Gray background
    
    # Add ballast/pebbles texture
    for _ in range(1000):
        x, y = np.random.randint(0, 600), np.random.randint(0, 400)
        size = np.random.randint(3, 8)
        color = np.random.randint(80, 140, 3)
        cv2.circle(img, (x, y), size, color.tolist(), -1)
    
    # Add railway tracks
    rail_width = 40
    rail_gap = 120
    
    # Left rail
    left_rail_x = 150
    cv2.rectangle(img, (left_rail_x, 0), (left_rail_x + rail_width, 400), (80, 80, 80), -1)
    
    # Right rail  
    right_rail_x = left_rail_x + rail_gap
    cv2.rectangle(img, (right_rail_x, 0), (right_rail_x + rail_width, 400), (80, 80, 80), -1)
    
    # Add some rust patches
    rust_color = (30, 60, 120)  # Brownish-red in BGR
    
    # Rust on left rail
    cv2.ellipse(img, (left_rail_x + 20, 100), (15, 25), 0, 0, 360, rust_color, -1)
    cv2.ellipse(img, (left_rail_x + 10, 250), (20, 30), 0, 0, 360, rust_color, -1)
    
    # Rust on right rail
    cv2.ellipse(img, (right_rail_x + 25, 180), (18, 20), 0, 0, 360, rust_color, -1)
    
    # Add some noise and texture
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def test_rust_detector():
    """Test the rust detection system"""
    print("Testing Railway Rust Detection System")
    print("=" * 50)
    
    # Create synthetic test image
    print("Creating synthetic railway image...")
    test_image = create_synthetic_railway_image()
    cv2.imwrite("test_railway.jpg", test_image)
    
    # Initialize detector
    print("Initializing rust detector...")
    detector = RustDetector()
    
    # Process the test image
    print("\nProcessing test image through pipeline...")
    try:
        results = detector.process_image("test_railway.jpg")
        
        print("\nProcessing completed successfully!")
        print("\n=== DETECTION RESULTS ===")
        metrics = results['metrics']
        print(f"Rust Percentage: {metrics['percentage']}%")
        print(f"Severity Level: {metrics['severity']}")
        print(f"Number of Rust Regions: {metrics['regions']}")
        print(f"Rail Pixels: {metrics['rail_pixels']:,}")
        print(f"Rust Pixels: {metrics['rust_pixels']:,}")
        
        # Visualize results
        print("\nGenerating visualization...")
        detector.visualize_results(results, "test_results.png")
        print("Results saved as 'test_results.png'")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

if __name__ == "__main__":
    print("Railway Rust Detection System - Test Suite")
    print("=" * 60)
    
    # Run full system test
    success = test_rust_detector()
    
    if success:
        print("\nAll tests passed!")
        print("\nGenerated files:")
        print("- test_railway.jpg (synthetic test image)")
        print("- test_results.png (detection results)")
    else:
        print("\nSome tests failed. Check the error messages above.")