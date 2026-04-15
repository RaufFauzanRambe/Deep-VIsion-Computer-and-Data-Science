# 🔬 Computer Vision & Data Science Dashboard

A unified web application combining:
- **Smart Image Upload** with YOLO Object Detection
- **Automated Analytics Dashboard** with Plotly Visualizations  
- **Real-Time Camera Stream** with Live Detection

## 🆓 100% FREE - NO API KEYS REQUIRED!

All AI processing runs locally on your machine using open-source models.
No cloud APIs, no hidden costs, unlimited usage!

---

## 📋 Features

### 1. Smart Image Upload (Option 1)
- Upload images in JPG, JPEG, PNG, BMP, or WEBP formats
- Automatic object detection using YOLOv8
- Bounding boxes and labels drawn on detected objects
- Download annotated images and CSV reports

### 2. Automated Analytics Dashboard (Option 2)
- Session-based detection statistics
- Interactive bar charts showing object counts
- Interactive pie charts showing object distribution
- Confidence score distribution histogram
- Export full reports as CSV

### 3. Real-Time Camera Stream (Option 3)
- Live webcam feed with object detection
- Start/Stop toggle for camera control
- Real-time frame processing on local CPU
- Detection statistics overlay

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd cv_dashboard
pip install -r requirements.txt
```

### Step 2: Run the Application

```bash
streamlit run app.py
```

### Step 3: Open in Browser

The app will automatically open at `http://localhost:8501`

---

## 🤖 How YOLO Works Locally (No API Cost)

### Model Loading Process

1. **First Run**: 
   - YOLO downloads the pre-trained model to your local machine
   - Model is cached at `~/.config/ultralytics/weights/`
   - Download size: ~6MB for yolov8n.pt

2. **Subsequent Runs**: 
   - Model loads from local cache (instant)
   - No internet connection required

3. **Inference**: 
   - All object detection runs on YOUR CPU/GPU
   - No cloud API calls - everything is local

### Available Models (All Free)

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov8n.pt | 6MB | Fastest | Good |
| yolov8s.pt | 22MB | Fast | Better |
| yolov8m.pt | 52MB | Medium | Great |
| yolov8l.pt | 83MB | Slow | Excellent |
| yolov8x.pt | 130MB | Slowest | Best |

### COCO Dataset Classes (80 Categories)

The pre-trained models can detect 80 object types including:
- **People & Animals**: person, dog, cat, bird, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Household Items**: chair, couch, bed, dining table, toilet, tv, laptop, mouse, keyboard, cell phone
- **Food & Kitchen**: bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- **And many more!**

---

## 📁 Project Structure

```
cv_dashboard/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies (all free)
└── README.md          # This documentation
```

---

## ⚙️ Configuration

### Confidence Threshold

Adjust the detection confidence threshold in the sidebar:
- **Lower (0.1-0.3)**: More detections, but may include false positives
- **Higher (0.5-0.9)**: Fewer detections, but more confident results

### Model Selection

Choose different YOLO models based on your needs:
- **For speed**: Use `yolov8n.pt` (recommended for real-time camera)
- **For accuracy**: Use `yolov8x.pt` (slower but more accurate)

---

## 💡 Tips for Best Performance

### For Image Upload
- Use larger models (yolov8l or yolov8x) for better accuracy
- Adjust confidence threshold based on your needs

### For Real-Time Camera
- Use yolov8n.pt for smooth frame rates
- Close other applications to free CPU resources
- Ensure good lighting for better detection

### General
- The model downloads automatically on first run
- After download, no internet is needed
- Clear detection history periodically for better performance

---

## 🛠️ Troubleshooting

### Camera Not Working
1. Check if your webcam is connected
2. Close other applications using the camera
3. Try refreshing the page

### Slow Detection
1. Switch to yolov8n.pt (fastest model)
2. Increase confidence threshold to reduce detections
3. Close other CPU-intensive applications

### Model Download Issues
1. Check your internet connection
2. Try downloading manually:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')  # Will download automatically
   ```

---

## 📦 Dependencies

All libraries are free and open-source:

| Library | Purpose |
|---------|---------|
| streamlit | Web dashboard framework |
| ultralytics | YOLO object detection |
| opencv-python | Image processing |
| numpy | Numerical computing |
| pandas | Data manipulation |
| plotly | Interactive charts |
| Pillow | Image handling |

---

## 📄 License

This project uses open-source libraries:
- **Streamlit**: Apache 2.0 License
- **Ultralytics YOLO**: AGPL-3.0 License
- **OpenCV**: Apache 2.0 License
- **Plotly**: MIT License

---

## 🙏 Credits

- **YOLO**: Ultralytics team for the amazing object detection library
- **Streamlit**: For the intuitive web framework
- **OpenCV**: For computer vision capabilities
- **Plotly**: For beautiful interactive visualizations

---

**Enjoy free, unlimited object detection! 🎉**
