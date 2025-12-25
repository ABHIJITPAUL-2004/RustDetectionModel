import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from rail_rust_detector import RustDetector

def main():
    st.set_page_config(
        page_title="Railway Rust Detection System",
        page_icon="🚂",
        layout="wide"
    )
    
    st.title("🚂 Railway Track Rust Detection System")
    st.markdown("Upload a railway track image to detect rust and assess severity")
    
    # Initialize detector
    @st.cache_resource
    def load_detector():
        return RustDetector()
    
    detector = load_detector()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a railway track image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image of railway tracks from any angle or height"
    )
    
    if uploaded_file is not None:
        # Convert uploaded file to opencv format
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3:
            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_array
        
        # Save temporarily for processing
        temp_path = "temp_railway_image.jpg"
        cv2.imwrite(temp_path, image_cv)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Railway Image", use_column_width=True)
        
        with col2:
            st.subheader("Processing Pipeline")
            
            # Process button
            if st.button("🔍 Analyze Rust", type="primary"):
                with st.spinner("Processing image through detection pipeline..."):
                    try:
                        # Process the image
                        results = detector.process_image(temp_path)
                        
                        # Display results
                        st.success("Analysis Complete!")
                        
                        # Metrics display
                        metrics = results['metrics']
                        
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric(
                                "Rust Percentage", 
                                f"{metrics['percentage']}%",
                                delta=None
                            )
                        
                        with col_b:
                            st.metric(
                                "Severity Level", 
                                metrics['severity']
                            )
                        
                        with col_c:
                            st.metric(
                                "Rust Regions", 
                                metrics['regions']
                            )
                        
                        # Display pipeline stages
                        st.subheader("Detection Pipeline Results")
                        
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "Rail Segmentation", 
                            "Rails Only", 
                            "Rust Detection", 
                            "Final Overlay"
                        ])
                        
                        with tab1:
                            st.image(
                                results['rail_mask'], 
                                caption="Stage 1: Rail vs Background Segmentation",
                                use_column_width=True,
                                clamp=True
                            )
                            st.info("White areas = detected rails, Black = background/ballast")
                        
                        with tab2:
                            rail_only_rgb = cv2.cvtColor(results['rail_only'], cv2.COLOR_BGR2RGB)
                            st.image(
                                rail_only_rgb,
                                caption="Stage 2: Isolated Rail Tracks (Ballast Removed)",
                                use_column_width=True
                            )
                            st.info("Only rail metal regions, background removed")
                        
                        with tab3:
                            st.image(
                                results['rust_mask'],
                                caption="Stage 3: Rust Detection on Rails",
                                use_column_width=True,
                                clamp=True
                            )
                            st.info("Red areas indicate detected rust regions")
                        
                        with tab4:
                            # Create overlay
                            overlay = results['original_image'].copy()
                            overlay[results['rust_mask'] == 1] = [0, 0, 255]
                            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                            
                            st.image(
                                overlay_rgb,
                                caption="Final Result: Rust Regions Highlighted",
                                use_column_width=True
                            )
                            st.info("Original image with rust areas highlighted in red")
                        
                        # Detailed analysis
                        st.subheader("Detailed Analysis")
                        
                        col_detail1, col_detail2 = st.columns(2)
                        
                        with col_detail1:
                            st.write("**Detection Statistics:**")
                            st.write(f"• Total rail pixels: {metrics['rail_pixels']:,}")
                            st.write(f"• Rust pixels detected: {metrics['rust_pixels']:,}")
                            st.write(f"• Rust coverage: {metrics['percentage']}%")
                            st.write(f"• Number of rust regions: {metrics['regions']}")
                        
                        with col_detail2:
                            st.write("**Severity Assessment:**")
                            if metrics['percentage'] < 5:
                                st.success("✅ Track condition: Healthy")
                                st.write("Minimal rust detected. Regular maintenance schedule.")
                            elif metrics['percentage'] < 15:
                                st.warning("⚠️ Track condition: Moderate Rust")
                                st.write("Increased monitoring recommended.")
                            else:
                                st.error("🚨 Track condition: Critical")
                                st.write("Immediate maintenance required!")
                        
                        # Download results
                        st.subheader("Download Results")
                        
                        # Create downloadable report
                        report = f"""
Railway Rust Detection Report
============================

Image Analysis Results:
- Rust Percentage: {metrics['percentage']}%
- Severity Level: {metrics['severity']}
- Number of Rust Regions: {metrics['regions']}
- Total Rail Pixels: {metrics['rail_pixels']:,}
- Rust Pixels: {metrics['rust_pixels']:,}

Recommendations:
"""
                        if metrics['percentage'] < 5:
                            report += "- Continue regular inspection schedule\n- No immediate action required"
                        elif metrics['percentage'] < 15:
                            report += "- Increase inspection frequency\n- Plan maintenance within 3-6 months"
                        else:
                            report += "- URGENT: Schedule immediate maintenance\n- Consider track replacement if rust is severe"
                        
                        st.download_button(
                            label="📄 Download Analysis Report",
                            data=report,
                            file_name=f"rust_analysis_report_{metrics['percentage']:.1f}percent.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                        st.info("Please ensure the image contains visible railway tracks")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About This System")
        st.write("""
        This railway rust detection system uses a multi-stage pipeline:
        
        **Stage 1:** Rail Segmentation
        - Separates rails from background
        - Handles any camera angle/height
        
        **Stage 2:** Noise Removal  
        - Removes ballast, pebbles, stones
        - Isolates only metal rail surfaces
        
        **Stage 3:** Rust Detection
        - HSV color space analysis
        - Morphological processing
        
        **Stage 4:** Severity Assessment
        - Calculates rust percentage
        - Classifies severity level
        - Counts rust regions
        """)
        
        st.header("Supported Images")
        st.write("""
        ✅ Any camera angle
        ✅ Various heights/distances  
        ✅ Different lighting conditions
        ✅ Curved or straight tracks
        ✅ Partial track visibility
        
        📁 Formats: JPG, PNG, BMP
        """)
        
        st.header("Severity Levels")
        st.write("""
        🟢 **Healthy** (0-5% rust)
        - Normal condition
        - Regular maintenance
        
        🟡 **Moderate** (5-15% rust)  
        - Increased monitoring
        - Plan maintenance
        
        🔴 **Critical** (>15% rust)
        - Immediate action required
        - Safety concern
        """)

if __name__ == "__main__":
    main()