import os
import glob
from simple_detector import RustDetector

def process_dataset():
    """Process all images from dataset folder"""
    detector = RustDetector()
    dataset_path = r"C:\Users\abhih\Desktop\Track investigative bot\rusty2\dataset"
    
    # Find all image files
    image_files = glob.glob(os.path.join(dataset_path, "*.png"))
    
    print(f"Found {len(image_files)} images to process")
    
    # Create results folder
    results_folder = os.path.join(dataset_path, "results")
    os.makedirs(results_folder, exist_ok=True)
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            results = detector.process_image(image_path)
            metrics = results['metrics']
            
            # Save visualization
            result_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(results_folder, f"{result_name}_analysis.png")
            detector.visualize_results(results, save_path)
            
            print(f"  Rust: {metrics['percentage']}% - {metrics['severity']} - {metrics['regions']} regions")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nResults saved in: {results_folder}")

if __name__ == "__main__":
    process_dataset()