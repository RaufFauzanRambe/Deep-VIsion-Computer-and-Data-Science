"""
================================================================================
COMPUTER VISION & DATA SCIENCE WEB APPLICATION
================================================================================
A unified web application combining:
1. Smart Image Upload with YOLO Object Detection
2. Automated Analytics Dashboard with Plotly Visualizations
3. Real-Time Camera Stream with Live Detection

100% FREE TO RUN - NO PAID API KEYS REQUIRED!

Author: AI Engineer
Version: 1.0.0

================================================================================
HOW THIS APP WORKS WITHOUT ANY API COSTS:
================================================================================

1. YOLO MODEL LOADING (Local & Free):
   - The app uses Ultralytics YOLOv8, which downloads pre-trained models
     automatically on first run (~6MB for yolov8n.pt)
   - Models are cached locally at: ~/.config/ultralytics/weights/
   - All inference runs on YOUR local CPU/GPU - NO cloud API calls!
   - After first download, no internet connection needed

2. IMAGE PROCESSING:
   - All image operations use OpenCV (runs locally)
   - No external API calls for any image processing

3. DATA VISUALIZATION:
   - Plotly charts render entirely in your browser
   - No external services or APIs required

4. WEBCAM STREAMING:
   - Uses your local webcam via OpenCV
   - All processing happens on your machine
   - No data sent to any external server

================================================================================
"""

# ==============================================================================
# IMPORTS - All Free and Open Source Libraries
# ==============================================================================

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import tempfile
import os
import time
from io import BytesIO
from datetime import datetime

# ==============================================================================
# YOLO MODEL IMPORT AND LOADING
# ==============================================================================
# The ultralytics library provides YOLO models that run ENTIRELY LOCALLY.
# No API keys, no cloud calls - everything runs on your machine!

from ultralytics import YOLO

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="CV & Data Science Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM CSS STYLING
# ==============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #42A5F5;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1E88E5 0%, #42A5F5 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1565C0 0%, #1E88E5 100%);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
# Session state is used to persist data across Streamlit app reruns.
# This allows us to accumulate detection results throughout the user session.

if 'detection_history' not in st.session_state:
    st.session_state['detection_history'] = []

if 'total_detections' not in st.session_state:
    st.session_state['total_detections'] = 0

if 'camera_running' not in st.session_state:
    st.session_state['camera_running'] = False

# ==============================================================================
# MODEL LOADING WITH CACHING
# ==============================================================================
# @st.cache_resource ensures the model is loaded ONLY ONCE and cached.
# This prevents reloading the model on every Streamlit rerun, saving time and memory.

@st.cache_resource
def load_yolo_model(model_name="yolov8n.pt"):
    """
    Load YOLO model locally - 100% FREE, No API Required!
    
    ========================================================================
    HOW LOCAL MODEL LOADING WORKS (No API Costs):
    ========================================================================
    
    1. FIRST RUN:
       - YOLO() checks if the model exists locally
       - If not found, it downloads from Ultralytics servers (one-time)
       - Model is saved to: ~/.config/ultralytics/weights/yolov8n.pt
       - Download size: ~6MB for nano model, ~22MB for small model
    
    2. SUBSEQUENT RUNS:
       - Model is loaded from local cache (instant load)
       - No internet connection required!
       - No API calls - all inference is local
    
    3. AVAILABLE MODELS (All Free & Pre-trained on COCO Dataset):
       - yolov8n.pt - Nano (fastest, 6MB, 80 classes)
       - yolov8s.pt - Small (22MB, more accurate)
       - yolov8m.pt - Medium (52MB)
       - yolov8l.pt - Large (83MB)
       - yolov8x.pt - Extra Large (130MB, most accurate)
    
    4. COCO DATASET CLASSES (80 Object Categories):
       Person, bicycle, car, motorcycle, airplane, bus, train, truck, 
       traffic light, fire hydrant, stop sign, parking meter, bench, 
       bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, 
       giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, 
       skis, snowboard, sports ball, kite, baseball bat, baseball glove,
       skateboard, surfboard, tennis racket, bottle, wine glass, cup,
       fork, knife, spoon, bowl, banana, apple, sandwich, orange,
       broccoli, carrot, hot dog, pizza, donut, cake, chair, couch,
       potted plant, bed, dining table, toilet, tv, laptop, mouse,
       remote, keyboard, cell phone, microwave, oven, toaster, sink,
       refrigerator, book, clock, vase, scissors, teddy bear, hair dryer,
       toothbrush
    
    Parameters:
    -----------
    model_name : str
        Name of the YOLO model to load. Default is 'yolov8n.pt' (nano).
    
    Returns:
    --------
    YOLO model instance ready for inference
    """
    
    try:
        # The YOLO class automatically handles model download and caching
        # On first run: downloads model to ~/.config/ultralytics/weights/
        # On subsequent runs: loads from local cache
        model = YOLO(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ==============================================================================
# OBJECT DETECTION FUNCTION
# ==============================================================================

def detect_objects(model, image, confidence_threshold=0.25):
    """
    Perform object detection on an image using YOLO.
    
    All processing happens LOCALLY on your machine - no API calls!
    
    Parameters:
    -----------
    model : YOLO model instance
        The loaded YOLO model for inference
    image : numpy.ndarray
        Input image in BGR format (OpenCV format)
    confidence_threshold : float
        Minimum confidence score for detections (0-1)
    
    Returns:
    --------
    tuple: (annotated_image, detections_list)
        - annotated_image: Image with bounding boxes drawn
        - detections_list: List of detected objects with details
    """
    
    if model is None:
        return image, []
    
    # Perform inference - ALL LOCAL, no API calls!
    # The model processes the image on your CPU/GPU
    results = model(image, conf=confidence_threshold, verbose=False)
    
    detections = []
    
    # Process detection results
    for result in results:
        # Get bounding boxes
        boxes = result.boxes
        
        if boxes is not None:
            for box in boxes:
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence score
                confidence = box.conf[0].cpu().numpy()
                
                # Get class ID and name
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                # Store detection info
                detections.append({
                    'class': class_name,
                    'confidence': float(confidence),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
    
    # Draw annotations on image
    # The plot() method draws bounding boxes and labels automatically
    annotated_image = results[0].plot() if results else image
    
    return annotated_image, detections

# ==============================================================================
# DRAW BOUNDING BOXES MANUALLY (Alternative Method)
# ==============================================================================

def draw_boxes_manual(image, detections, color_map=None):
    """
    Draw bounding boxes manually with custom styling.
    
    This gives you full control over the visualization style.
    All drawing operations use OpenCV - runs locally, no API needed.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    detections : list
        List of detection dictionaries with 'class', 'confidence', 'bbox'
    color_map : dict, optional
        Mapping of class names to BGR colors
    
    Returns:
    --------
    numpy.ndarray: Image with drawn bounding boxes
    """
    
    if color_map is None:
        # Default colors for common objects (BGR format)
        color_map = {
            'person': (0, 255, 0),      # Green
            'car': (255, 0, 0),          # Blue
            'bicycle': (0, 255, 255),    # Yellow
            'dog': (0, 165, 255),        # Orange
            'cat': (255, 0, 255),        # Purple
            'bird': (0, 255, 255),       # Cyan
            'default': (0, 255, 0)       # Green
        }
    
    annotated = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        
        # Get color for this class
        color = color_map.get(class_name, color_map['default'])
        
        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label = f"{class_name}: {confidence:.2f}"
        
        # Calculate text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw label background
        cv2.rectangle(
            annotated,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 5, y1),
            color,
            -1  # Filled rectangle
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White text
            2
        )
    
    return annotated

# ==============================================================================
# ANALYTICS FUNCTIONS
# ==============================================================================

def create_detection_dataframe(detections):
    """
    Create a pandas DataFrame from detection results.
    
    Parameters:
    -----------
    detections : list
        List of detection dictionaries
    
    Returns:
    --------
    pandas.DataFrame: Structured detection data
    """
    
    if not detections:
        return pd.DataFrame(columns=['Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
    
    data = []
    for det in detections:
        data.append({
            'Class': det['class'],
            'Confidence': f"{det['confidence']:.3f}",
            'X1': det['bbox'][0],
            'Y1': det['bbox'][1],
            'X2': det['bbox'][2],
            'Y2': det['bbox'][3]
        })
    
    return pd.DataFrame(data)

def create_bar_chart(detection_counts):
    """
    Create an interactive bar chart using Plotly.
    
    All rendering happens in your browser - no external API calls!
    
    Parameters:
    -----------
    detection_counts : pandas.Series
        Counts of each detected object class
    
    Returns:
    --------
    plotly.graph_objects.Figure: Interactive bar chart
    """
    
    fig = go.Figure(data=[
        go.Bar(
            x=detection_counts.index,
            y=detection_counts.values,
            marker=dict(
                color=detection_counts.values,
                colorscale='Viridis',
                showscale=True
            ),
            text=detection_counts.values,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Object Detection Count by Class',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1E88E5'}
        },
        xaxis_title='Object Class',
        yaxis_title='Count',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_pie_chart(detection_counts):
    """
    Create an interactive pie chart using Plotly.
    
    All rendering happens in your browser - no external API calls!
    
    Parameters:
    -----------
    detection_counts : pandas.Series
        Counts of each detected object class
    
    Returns:
    --------
    plotly.graph_objects.Figure: Interactive pie chart
    """
    
    fig = go.Figure(data=[
        go.Pie(
            labels=detection_counts.index,
            values=detection_counts.values,
            hole=0.4,  # Donut chart
            textinfo='label+percent',
            textposition='outside',
            marker=dict(
                colors=px.colors.qualitative.Set3
            ),
            pull=[0.05] * len(detection_counts)  # Slight pull on all slices
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Object Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1E88E5'}
        },
        template='plotly_white',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# ==============================================================================
# CSV EXPORT FUNCTION
# ==============================================================================

def get_csv_download_link(df, filename="detection_report.csv"):
    """
    Generate a CSV download link for the detection report.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Detection data to export
    filename : str
        Name for the downloaded file
    
    Returns:
    --------
    str: CSV data as string for download
    """
    
    return df.to_csv(index=False).encode('utf-8')

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    """
    Main application function that builds the Streamlit web interface.
    """
    
    # ========================================================================
    # HEADER SECTION
    # ========================================================================
    
    st.markdown('<h1 class="main-header">🔬 Computer Vision & Data Science Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>🆓 100% FREE - NO API KEYS REQUIRED!</strong><br>
        All AI processing runs locally on your machine using open-source models.
        No cloud APIs, no hidden costs, unlimited usage!
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR - Model Selection and Settings
    # ========================================================================
    
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Model selection with explanation
        st.markdown("""
        ### 🤖 YOLO Model Selection
        
        **How it works (No API Cost):**
        
        1. Models are downloaded **ONCE** to your local machine
        2. All inference runs on **YOUR CPU/GPU**
        3. **No internet needed** after first download
        4. **Zero API costs** - unlimited detections!
        """)
        
        model_option = st.selectbox(
            "Select YOLO Model:",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
            index=0,
            help="""
            - yolov8n: Nano (fastest, 6MB)
            - yolov8s: Small (balanced, 22MB)
            - yolov8m: Medium (52MB)
            - yolov8l: Large (83MB)
            - yolov8x: Extra Large (most accurate, 130MB)
            """
        )
        
        # Confidence threshold slider
        confidence_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for detections. Higher = fewer but more confident detections."
        )
        
        # Load model button
        if st.button("🔄 Load Model", use_container_width=True):
            with st.spinner("Loading model (first run downloads ~6MB)..."):
                st.session_state['model'] = load_yolo_model(model_option)
            st.success("✅ Model loaded successfully!")
        
        # Load model on first run
        if 'model' not in st.session_state:
            with st.spinner("Loading model for the first time..."):
                st.session_state['model'] = load_yolo_model(model_option)
        
        st.divider()
        
        # Session statistics
        st.markdown("### 📊 Session Statistics")
        st.metric("Total Detections", st.session_state['total_detections'])
        st.metric("Images Processed", len(st.session_state['detection_history']))
        
        # Clear history button
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state['detection_history'] = []
            st.session_state['total_detections'] = 0
            st.success("History cleared!")
    
    # ========================================================================
    # MAIN CONTENT AREA - Three Feature Tabs
    # ========================================================================
    
    tab1, tab2, tab3 = st.tabs([
        "🖼️ Smart Image Upload",
        "📊 Analytics Dashboard",
        "📹 Real-Time Camera"
    ])
    
    # ========================================================================
    # TAB 1: SMART IMAGE UPLOAD
    # ========================================================================
    
    with tab1:
        st.markdown('<h2 class="sub-header">Smart Image Upload with YOLO Detection</h2>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>How it works:</strong> Upload an image and YOLO will detect objects 
            with bounding boxes and labels. All processing happens locally on your machine!
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Supported formats: JPG, JPEG, PNG, BMP, WEBP"
        )
        
        if uploaded_file is not None:
            # Create two columns for original and processed image
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📷 Original Image")
                
                # Read and display original image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Convert BGR to RGB for display
                original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                st.image(original_rgb, use_container_width=True)
            
            with col2:
                st.markdown("### 🔍 Detection Results")
                
                with st.spinner("Running YOLO detection..."):
                    # Perform object detection
                    model = st.session_state.get('model')
                    annotated_image, detections = detect_objects(
                        model, 
                        original_image, 
                        confidence_threshold
                    )
                
                # Convert annotated image for display
                annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)
                
                # Display detection results
                st.markdown("### 📋 Detection Summary")
                
                if detections:
                    # Create metrics row
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Objects Found", len(detections))
                    metric_cols[1].metric("Unique Classes", len(set(d['class'] for d in detections)))
                    avg_conf = sum(d['confidence'] for d in detections) / len(detections)
                    metric_cols[2].metric("Avg Confidence", f"{avg_conf:.2%}")
                    
                    # Update session statistics
                    st.session_state['total_detections'] += len(detections)
                    st.session_state['detection_history'].extend(detections)
                    
                    # Display detection table
                    st.markdown("#### Detected Objects")
                    detection_df = create_detection_dataframe(detections)
                    st.dataframe(detection_df, use_container_width=True, hide_index=True)
                    
                    # Download processed image button
                    st.markdown("### 💾 Download Results")
                    
                    # Convert annotated image to bytes for download
                    _, img_buffer = cv2.imencode('.png', annotated_image)
                    img_bytes = img_buffer.tobytes()
                    
                    dl_col1, dl_col2 = st.columns(2)
                    
                    with dl_col1:
                        st.download_button(
                            label="📥 Download Annotated Image",
                            data=img_bytes,
                            file_name=f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    with dl_col2:
                        st.download_button(
                            label="📥 Download CSV Report",
                            data=get_csv_download_link(detection_df),
                            file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")
    
    # ========================================================================
    # TAB 2: ANALYTICS DASHBOARD
    # ========================================================================
    
    with tab2:
        st.markdown('<h2 class="sub-header">Automated Analytics Dashboard</h2>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>Session Analytics:</strong> View aggregated statistics from all images 
            processed during this session. Charts update automatically as you upload more images!
        </div>
        """, unsafe_allow_html=True)
        
        # Get detection history
        detection_history = st.session_state['detection_history']
        
        if detection_history:
            # Create summary statistics
            all_classes = [d['class'] for d in detection_history]
            detection_counts = pd.Series(all_classes).value_counts()
            
            # Top metrics
            metric_cols = st.columns(4)
            metric_cols[0].metric("Total Objects Detected", len(detection_history))
            metric_cols[1].metric("Unique Object Types", len(detection_counts))
            metric_cols[2].metric("Most Common", detection_counts.index[0] if len(detection_counts) > 0 else "N/A")
            metric_cols[3].metric("Session Duration", f"{time.time() - st.session_state.get('start_time', time.time()):.0f}s")
            
            # Charts section
            st.markdown("### 📈 Interactive Visualizations")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("#### Bar Chart - Object Counts")
                bar_fig = create_bar_chart(detection_counts)
                st.plotly_chart(bar_fig, use_container_width=True)
            
            with chart_col2:
                st.markdown("#### Pie Chart - Object Distribution")
                pie_fig = create_pie_chart(detection_counts)
                st.plotly_chart(pie_fig, use_container_width=True)
            
            # Confidence distribution histogram
            st.markdown("### 📊 Confidence Score Distribution")
            
            confidences = [d['confidence'] for d in detection_history]
            conf_fig = go.Figure(data=[
                go.Histogram(
                    x=confidences,
                    nbinsx=20,
                    marker=dict(
                        color='#1E88E5',
                        line=dict(color='#1565C0', width=1)
                    ),
                    opacity=0.75
                )
            ])
            
            conf_fig.update_layout(
                title='Detection Confidence Distribution',
                xaxis_title='Confidence Score',
                yaxis_title='Count',
                template='plotly_white',
                height=300
            )
            
            st.plotly_chart(conf_fig, use_container_width=True)
            
            # Detailed data table
            st.markdown("### 📋 Complete Detection History")
            
            full_df = create_detection_dataframe(detection_history)
            st.dataframe(full_df, use_container_width=True, hide_index=True)
            
            # Download full report
            st.markdown("### 💾 Export Full Report")
            
            report_col1, report_col2 = st.columns(2)
            
            with report_col1:
                st.download_button(
                    label="📥 Download Full CSV Report",
                    data=get_csv_download_link(full_df),
                    file_name=f"full_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with report_col2:
                # Summary statistics as text
                summary_text = f"""
Detection Report Summary
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Total Detections: {len(detection_history)}
Unique Object Types: {len(detection_counts)}

Object Counts:
{detection_counts.to_string()}

Average Confidence: {sum(confidences)/len(confidences):.2%}
Minimum Confidence: {min(confidences):.2%}
Maximum Confidence: {max(confidences):.2%}
"""
                st.download_button(
                    label="📥 Download Summary Report",
                    data=summary_text,
                    file_name=f"detection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>📊 No Data Yet!</strong><br>
                Upload images in the "Smart Image Upload" tab to start building your analytics data.
                All detection results will be aggregated here for analysis.
            </div>
            """, unsafe_allow_html=True)
            
            # Show placeholder visualizations
            st.markdown("### 📈 Sample Visualization Preview")
            
            sample_data = {
                'person': 15,
                'car': 8,
                'dog': 3,
                'bicycle': 5,
                'bird': 2
            }
            sample_series = pd.Series(sample_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("*Sample Bar Chart (will update with real data)*")
                st.plotly_chart(create_bar_chart(sample_series), use_container_width=True)
            
            with col2:
                st.markdown("*Sample Pie Chart (will update with real data)*")
                st.plotly_chart(create_pie_chart(sample_series), use_container_width=True)
    
    # ========================================================================
    # TAB 3: REAL-TIME CAMERA STREAM
    # ========================================================================
    
    with tab3:
        st.markdown('<h2 class="sub-header">Real-Time Camera Stream</h2>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>Live Detection:</strong> Use your webcam for real-time object detection.
            All processing happens locally on your CPU - no data sent to any server!
        </div>
        """, unsafe_allow_html=True)
        
        # Camera controls
        control_col1, control_col2, control_col3 = st.columns([1, 1, 2])
        
        with control_col1:
            start_camera = st.button("▶️ Start Camera", use_container_width=True, type="primary")
        
        with control_col2:
            stop_camera = st.button("⏹️ Stop Camera", use_container_width=True)
        
        with control_col3:
            st.markdown("""
            <div class="warning-box" style="margin: 0; padding: 0.5rem;">
                <strong>💡 Tip:</strong> For best performance, use yolov8n.pt (nano model).
                Detection runs on CPU - close other apps if you experience lag.
            </div>
            """, unsafe_allow_html=True)
        
        # Camera feed placeholder
        camera_placeholder = st.empty()
        
        # Status indicator
        status_placeholder = st.empty()
        
        # Detection stats for camera
        camera_stats = st.empty()
        
        # Handle camera start/stop
        if start_camera:
            st.session_state['camera_running'] = True
        
        if stop_camera:
            st.session_state['camera_running'] = False
        
        # Run camera loop
        if st.session_state['camera_running']:
            status_placeholder.markdown("""
            <div class="success-box">
                <strong>🟢 Camera Active</strong> - Detecting objects in real-time...
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize video capture
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not open webcam. Please check your camera connection.")
                st.session_state['camera_running'] = False
            else:
                # Camera settings for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                frame_count = 0
                detection_count = 0
                
                while st.session_state['camera_running']:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.warning("⚠️ Failed to read from camera.")
                        break
                    
                    # Process every other frame for better performance
                    if frame_count % 2 == 0:
                        model = st.session_state.get('model')
                        annotated_frame, detections = detect_objects(
                            model, 
                            frame, 
                            confidence_threshold
                        )
                        
                        # Update detection count
                        if detections:
                            detection_count += len(detections)
                            st.session_state['total_detections'] += len(detections)
                            st.session_state['detection_history'].extend(detections)
                    else:
                        annotated_frame = frame
                    
                    # Convert to RGB for display
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Add FPS and detection count overlay
                    cv2.putText(
                        annotated_rgb,
                        f"Frame: {frame_count} | Objects: {len(detections) if frame_count % 2 == 0 else 0}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    # Display frame
                    camera_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
                    
                    # Update stats
                    camera_stats.markdown(f"""
                    **📊 Camera Statistics:**
                    - Frames Processed: {frame_count}
                    - Total Objects Detected: {detection_count}
                    - Current Frame Objects: {len(detections) if frame_count % 2 == 0 else 'Processing...'}
                    """)
                    
                    frame_count += 1
                    
                    # Small delay for stability
                    time.sleep(0.03)  # ~30 FPS target
                
                # Release camera when stopped
                cap.release()
                
                status_placeholder.markdown("""
                <div class="warning-box">
                    <strong>🔴 Camera Stopped</strong> - Click "Start Camera" to resume.
                </div>
                """, unsafe_allow_html=True)
        else:
            # Show placeholder when camera is off
            camera_placeholder.markdown("""
            <div style="background-color: #f5f5f5; padding: 4rem; text-align: center; 
                        border-radius: 0.5rem; border: 2px dashed #ccc;">
                <h3 style="color: #666;">📷 Camera Preview</h3>
                <p style="color: #888;">Click "Start Camera" to begin real-time detection</p>
            </div>
            """, unsafe_allow_html=True)
            
            status_placeholder.markdown("""
            <div class="info-box">
                <strong>📷 Camera Ready</strong> - Click "Start Camera" to begin live detection.
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.divider()
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <p style="color: #666;">
            🔬 <strong>Computer Vision & Data Science Dashboard</strong><br>
            Built with Streamlit, YOLOv8, OpenCV, and Plotly<br>
            <em>100% Free & Open Source - No API Keys Required!</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Initialize session start time
    if 'start_time' not in st.session_state:
        st.session_state['start_time'] = time.time()
    
    # Run main application
    main()
