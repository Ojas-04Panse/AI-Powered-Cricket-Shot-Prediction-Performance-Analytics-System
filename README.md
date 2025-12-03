# AI-Powered-Cricket-Shot-Prediction-Performance-Analytics-System

# Cricket Highlight Event Detection (Computer Vision + Machine Learning)

## 🏏 Project Overview
This project focuses on detecting key events in cricket highlight videos — such as boundaries, wickets, and celebrations — using a Machine Learning approach. The dataset was **manually labeled** from YouTube cricket highlight footage.  

We apply **Computer Vision techniques** to extract useful features from frames and then train a **Random Forest classifier** to automatically identify highlight-worthy moments.

---

## 🎯 Objective
- To detect important cricket events automatically from broadcast highlight videos.
- To build a system capable of supporting sports analytics and automated highlight generation.

---

## 📂 Dataset
- Videos sourced from YouTube cricket highlights.
- Frames extracted from the videos at regular intervals.
- Each frame **manually labeled** into event classes:
  - Boundary
  - Wicket
  - Normal Play
  - Celebration (optional)

---

## 🧠 Methodology

### 1️⃣ Data Collection
- Downloaded YouTube cricket highlights.
- Extracted video frames for analysis.

### 2️⃣ Manual Labeling
- Each frame was tagged based on visible event:
  - e.g., "4", "6", "Wicket", or "None"

### 3️⃣ Feature Extraction (Computer Vision)
- Applied methods like:
  - Histogram of Oriented Gradients (HOG)
  - Motion and intensity variation
  - Visual patterns from scoreboard/players

> These extracted values serve as mathematical features for ML training.

### 4️⃣ Model Training (Machine Learning)
- Used **Random Forest Classifier**
- Split into Train/Test datasets
- Evaluated based on accuracy and performance metrics

---

## 🧪 Tech Stack
| Component | Tools Used |
|----------|------------|
| Programming | Python |
| CV Techniques | OpenCV |
| ML Algorithm | Random Forest |
| Visualization | Matplotlib, Seaborn |
| Annotation | Manual CSV Labeling |

---

## 🚀 How It Works
1. Input: Cricket highlight video
2. Frames extracted and features generated
3. Random Forest model predicts event type for each frame
4. Output: Detected highlight frames + timestamps

This can be extended into automatic highlight video generation.

---

## 📊 Results
- The trained model successfully identifies highlight events with promising accuracy.
- Boundary and wicket events are detected more reliably due to strong visual cues.

(You can add your actual metrics here: Accuracy, Precision, Recall, Confusion Matrix, etc.)

---

## 🔮 Future Improvements
- Include audio cues (crowd cheer, commentary spike)
- Use deep learning (CNN/LSTM) for sequence prediction
- Automate labeling with weak supervision
- Detect more advanced cricket actions

---

## 👥 Authors
- Team Members  
(Replace with your names)

---

## 📘 Academic Note
This project was developed for academic learning and experimentation in:
- Computer Vision
- Machine Learning
- Video Event Detection

Not intended for commercial use.

