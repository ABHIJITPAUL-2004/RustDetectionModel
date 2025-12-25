import streamlit as st
import cv2
import numpy as np
import os
import glob
from simple_detector import RustDetector
from PIL import Image
import matplotlib.pyplot as plt
import io

def main():
    st.set_page_config(
        page_title="Railway Dataset Analysis",
        page_icon="🚂",
        layout="wide"
    )
    
    st.title("Railway Dataset Analysis - Interactive Viewer")
    
    # Initialize detector
    @st.cache_resource
    def load_detector():
        return RustDetector()
    
    detector = load_detector()
    
    # Dataset path
    dataset_path = r"C:\Users\abhih\Desktop\Track investigative bot\rusty2\dataset"
    image_files = glob.glob(os.path.join(dataset_path, "*.png"))
    
    if not image_files:
        st.error("No images found in dataset folder")
        return
    
    # Initialize session state
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("Previous", disabled=(st.session_state.current_index == 0)):
            st.session_state.current_index -= 1
            st.rerun()
    
    with col2:
        st.write(f"Image {st.session_state.current_index + 1} of {len(image_files)}")
    
    with col3:
        if st.button("Next", disabled=(st.session_state.current_index == len(image_files) - 1)):
            st.session_state.current_index += 1
            st.rerun()
    
    # Current image
    current_image_path = image_files[st.session_state.current_index]
    image_name = os.path.basename(current_image_path)
    
    st.subheader(f"Analyzing: {image_name}")
    
    # Process current image
    with st.spinner("Processing image..."):
        try:
            results = detector.process_image(current_image_path)
            metrics = results['metrics']
            
            # Display metrics with better styling
            st.markdown("### Analysis Results")
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Rust Percentage", f"{metrics['percentage']}%", 
                         delta=f"{metrics['percentage'] - 5:.1f}% vs healthy" if metrics['percentage'] > 5 else None)
            
            with col_b:
                severity_color = "🟢" if metrics['severity'] == "Healthy" else "🟡" if "Moderate" in metrics['severity'] else "🔴"
                st.metric("Severity", f"{severity_color} {metrics['severity']}")
            
            with col_c:
                st.metric("Rust Regions", metrics['regions'])
            
            with col_d:
                st.metric("Rail Pixels", f"{metrics['rail_pixels']:,}")
            
            # Display images with all processing stages
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "Original", "Binary Mask", "Texture Analysis", "Edge Detection", "Metal Detection", "Morphological", "Rust Detection", "Final Result"
            ])
            
            with tab1:
                st.image(cv2.cvtColor(results['original_image'], cv2.COLOR_BGR2RGB), 
                        caption="Original Railway Image", use_column_width=True)
                st.info("Input image from dataset")
            
            with tab2:
                st.image(results['binary_mask'], caption="Binary Thresholding", 
                        use_column_width=True, clamp=True)
                st.info("Binary mask isolating darker regions (potential metal surfaces)")
            
            with tab3:
                st.image(results['texture_mask'], caption="Texture Analysis", 
                        use_column_width=True, clamp=True)
                st.info("Low texture regions (smooth metal vs rough ballast)")
            
            with tab4:
                st.image(results['edges'], caption="Edge Detection (Canny)", 
                        use_column_width=True, clamp=True)
                st.info("Rail boundaries and edges detected")
            
            with tab5:
                st.image(results['rail_mask'], caption="Final Metal Detection", 
                        use_column_width=True, clamp=True)
                st.info("Combined result: metal rails isolated from ballast")
            
            with tab6:
                st.image(results['morphological'], caption="Morphological Operations", 
                        use_column_width=True, clamp=True)
                st.info("Noise removal and gap filling using opening/closing")
            
            with tab7:
                col_rust1, col_rust2 = st.columns(2)
                
                with col_rust1:
                    st.image(results['hsv_image'], caption="HSV Color Space", use_column_width=True)
                    st.info("HSV representation for rust color analysis")
                
                with col_rust2:
                    st.image(results['rust_mask'], caption="Rust Detection Mask", 
                            use_column_width=True, clamp=True)
                    st.info("Final rust regions (H:5-25°, S:50-255, V:50-200)")
            
            with tab8:
                # Create overlay with green rail lines and red rust
                overlay = results['original_image'].copy()
                
                # Draw green lines for detected rails
                rail_contours, _ = cv2.findContours(results['rail_mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in rail_contours:
                    if cv2.contourArea(contour) > 1000:  # Only draw significant rail areas
                        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 3)  # Green outline
                
                # Highlight rust areas in red
                overlay[results['rust_mask'] == 1] = [0, 0, 255]
                
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 
                        caption="Final Analysis: Green=Rails, Red=Rust", use_column_width=True)
                st.info("Green lines show detected rail tracks, red areas show rust")
            
            # Severity assessment with better visibility
            st.markdown("---")
            st.subheader("Track Condition Assessment")
            
            if metrics['percentage'] < 5:
                st.success("✅ **HEALTHY TRACK** - Regular maintenance schedule")
                st.markdown("**Recommendation:** Continue normal inspection routine")
            elif metrics['percentage'] < 15:
                st.warning("⚠️ **MODERATE RUST DETECTED** - Increased monitoring needed")
                st.markdown("**Recommendation:** Schedule maintenance within 3-6 months")
            else:
                st.error("🚨 **CRITICAL CONDITION** - Immediate maintenance required!")
                st.markdown("**Recommendation:** URGENT - Schedule immediate repair work")
            
            # Additional details
            st.markdown(f"**Detailed Analysis:**")
            st.markdown(f"- Rust Coverage: **{metrics['percentage']}%**")
            st.markdown(f"- Severity Level: **{metrics['severity']}**")
            st.markdown(f"- Number of Rust Regions: **{metrics['regions']}**")
            st.markdown(f"- Total Rail Pixels Analyzed: **{metrics['rail_pixels']:,}**")
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
    
    # Dataset summary
    with st.sidebar:
        st.header("Dataset Overview")
        st.write(f"Total Images: {len(image_files)}")
        st.write(f"Current: {st.session_state.current_index + 1}")
        
        # Technical Details
        st.subheader("Detection Techniques")
        st.markdown("""
        **Multi-Stage Pipeline:**
        1. **Binary Thresholding** - Isolate dark regions (potential metal)
        2. **Texture Analysis** - Detect smooth surfaces vs rough ballast
        3. **Edge Detection** - Find rail boundaries using Canny
        4. **Line Detection** - Hough transform for rail edges
        5. **Morphological Ops** - Remove noise, fill gaps
        6. **HSV Analysis** - Color-based rust detection
        
        **HSV Rust Parameters:**
        - Hue: 5-25° (reddish-brown)
        - Saturation: 50-255 (moderate to high)
        - Value: 50-200 (medium brightness)
        """)
        
        # Quick navigation
        st.subheader("Quick Jump")
        selected_index = st.selectbox(
            "Go to image:",
            range(len(image_files)),
            index=st.session_state.current_index,
            format_func=lambda x: f"{x+1}. {os.path.basename(image_files[x])}"
        )
        
        if selected_index != st.session_state.current_index:
            st.session_state.current_index = selected_index
            st.rerun()

if __name__ == "__main__":
    main()