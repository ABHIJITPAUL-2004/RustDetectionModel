# 🚂 Railway Track Rust Detection System

A multi-stage computer vision pipeline for detecting rust on railway tracks from images taken at any angle, height, or lighting condition.

## 🎯 System Architecture

```
User Image (Any Angle/Height/Lighting)
           ↓
Stage 1: Rail & Background Segmentation (U-Net)
           ↓
Stage 2: Rail Mask Refinement (Morphological Ops)
           ↓
Stage 3: Rail Extraction (Remove Ballast/Pebbles)
           ↓
Stage 4: Rust Detection (HSV + ML)
           ↓
Stage 5: Severity Assessment & Reporting
```

## ✨ Key Features

- **🎯 Semantic Segmentation**: Isolates rails from background regardless of camera angle
- **🧹 Noise Removal**: Automatically removes ballast, pebbles, and stones
- **🔍 Rust Detection**: HSV-based color analysis with morphological processing
- **📊 Severity Assessment**: Calculates rust percentage and severity levels
- **🌐 Web Interface**: Easy-to-use Streamlit interface for image upload
- **📈 Visualization**: Complete pipeline visualization and results

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the System

```bash
python test_system.py
```

This creates synthetic railway images and tests the complete pipeline.

### 3. Run Web Interface

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` and upload railway images.

## 📋 Usage

### Command Line Usage

```python
from rail_rust_detector import RustDetector

detector = RustDetector()
results = detector.process_image("railway_image.jpg")

print(f"Rust Percentage: {results['metrics']['percentage']}%")
print(f"Severity: {results['metrics']['severity']}")
```

### Web Interface

1. Launch: `streamlit run app.py`
2. Upload railway track image (JPG, PNG, BMP)
3. Click "Analyze Rust"
4. View results and download report

## 🔧 System Components

### Stage 1: Rail Segmentation (`RailSegmentationModel`)
- **Purpose**: Separate rails from background/ballast
- **Method**: Lightweight U-Net semantic segmentation
- **Handles**: Any camera angle, height, perspective distortion

### Stage 2: Mask Refinement
- **Purpose**: Clean up segmentation noise
- **Methods**: Morphological operations (closing, opening)
- **Result**: Clean rail-only mask

### Stage 3: Rail Extraction
- **Purpose**: Remove all non-rail elements
- **Method**: Apply mask to original image
- **Result**: Image containing only rail metal surfaces

### Stage 4: Rust Detection
- **Purpose**: Identify rust regions on rails
- **Method**: HSV color space thresholding
- **HSV Range**: H(5-25), S(50-255), V(50-200)

### Stage 5: Severity Assessment
- **Metrics**: Rust percentage, region count, severity classification
- **Levels**: 
  - Healthy (0-5%)
  - Moderate (5-15%)
  - Critical (>15%)

## 📊 Output Metrics

```python
{
    "percentage": 12.5,           # Rust coverage percentage
    "severity": "Moderate Rust",  # Severity classification
    "regions": 3,                 # Number of rust regions
    "rail_pixels": 15420,         # Total rail pixels
    "rust_pixels": 1928           # Rust pixels detected
}
```

## 🎨 Visualization

The system generates comprehensive visualizations:

1. **Original Image**: Input railway image
2. **Rail Segmentation**: Binary mask showing detected rails
3. **Rails Only**: Image with background removed
4. **Rust Detection**: Rust regions highlighted
5. **Final Overlay**: Original image with rust areas marked

## 🔬 Technical Details

### Why Multi-Stage Pipeline?

❌ **Single-stage approaches fail because:**
- Fixed cropping doesn't handle perspective changes
- Direct rust detection includes ballast/pebbles
- No separation between rail and non-rail surfaces

✅ **Multi-stage pipeline succeeds because:**
- Semantic segmentation handles any camera angle
- Rail isolation removes noise sources
- Focused rust detection on clean rail surfaces

### HSV Color Space for Rust

Rust characteristics in HSV:
- **Hue**: 5-25° (reddish-brown range)
- **Saturation**: 50-255 (moderate to high)
- **Value**: 50-200 (medium brightness)

### Model Architecture

```python
RailSegmentationModel (U-Net):
├── Encoder: 3→64→128→256 channels
├── Decoder: 256→128→64→2 classes
└── Output: [Background, Rail] segmentation
```

## 📁 File Structure

```
rusty2/
├── rail_rust_detector.py    # Main detection system
├── app.py                   # Streamlit web interface
├── test_system.py          # Testing and demo script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🛠 Customization

### Adjust Rust Detection Sensitivity

```python
detector = RustDetector()
# More sensitive (detects lighter rust)
detector.rust_lower = np.array([3, 30, 30])
detector.rust_upper = np.array([30, 255, 220])
```

### Modify Severity Thresholds

```python
# In calculate_rust_severity method:
if rust_percentage < 3:      # More strict
    severity = "Healthy"
elif rust_percentage < 10:   # Lower threshold
    severity = "Moderate Rust"
```

## 🚀 Deployment Options

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Cloud Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **AWS/GCP**: Container deployment
- **Heroku**: Web app deployment

## 🔍 Troubleshooting

### Common Issues

1. **No rails detected**
   - Ensure image contains visible railway tracks
   - Check image quality and lighting
   - Verify rail segmentation model performance

2. **False rust detection**
   - Adjust HSV thresholds for lighting conditions
   - Check for shadows or discoloration
   - Refine morphological operations

3. **Performance issues**
   - Resize large images before processing
   - Use GPU acceleration if available
   - Optimize model inference

## 📈 Future Enhancements

1. **Advanced ML Models**
   - YOLOv8-Seg for real-time detection
   - DeepLabV3+ for better segmentation
   - Custom rust classification models

2. **Additional Features**
   - Crack detection
   - Wear analysis
   - Bolt condition assessment
   - Track gauge measurement

3. **Production Features**
   - Batch processing
   - API endpoints
   - Database integration
   - Automated reporting

## 📄 License

This project is open source. Feel free to modify and distribute.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make improvements
4. Submit pull request

## 📞 Support

For issues or questions:
- Create GitHub issue
- Check troubleshooting section
- Review code documentation