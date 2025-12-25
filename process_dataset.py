import os
import glob
from simple_detector import RustDetector

def process_user_dataset():
    """Process all images from user dataset folder"""
    detector = RustDetector()
    
    # Get dataset folder from user
    dataset_path = input("Enter dataset folder path: ").strip()
    
    if not os.path.exists(dataset_path):
        print(f"Error: Path {dataset_path} does not exist")
        return
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(dataset_path, ext)))
        image_files.extend(glob.glob(os.path.join(dataset_path, ext.upper())))
    
    if not image_files:
        print("No image files found in dataset folder")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Create results folder
    results_folder = os.path.join(dataset_path, "rust_analysis_results")
    os.makedirs(results_folder, exist_ok=True)
    
    # Process each image
    results_summary = []
    
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            results = detector.process_image(image_path)
            metrics = results['metrics']
            
            # Save visualization
            result_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(results_folder, f"{result_name}_analysis.png")
            detector.visualize_results(results, save_path)
            
            # Store summary
            results_summary.append({
                'image': os.path.basename(image_path),
                'rust_percentage': metrics['percentage'],
                'severity': metrics['severity'],
                'regions': metrics['regions']
            })
            
            print(f"  Rust: {metrics['percentage']}% - {metrics['severity']}")
            
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
    
    # Save summary report
    report_path = os.path.join(results_folder, "summary_report.txt")
    with open(report_path, 'w') as f:
        f.write("Railway Rust Detection - Dataset Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results_summary:
            f.write(f"Image: {result['image']}\n")
            f.write(f"Rust Percentage: {result['rust_percentage']}%\n")
            f.write(f"Severity: {result['severity']}\n")
            f.write(f"Regions: {result['regions']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\nProcessing complete! Results saved in: {results_folder}")
    print(f"Summary report: {report_path}")

if __name__ == "__main__":
    process_user_dataset()