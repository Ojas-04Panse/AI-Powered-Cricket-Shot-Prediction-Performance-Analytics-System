# 🏏 AI-Powered Cricket Shot Prediction & Performance Analytics System

A Computer Vision and Machine Learning based system that classifies cricket shots from video clips, aimed at enabling deeper sports analytics and coaching feedback.

---

## 📌 Project Overview

- Collected and curated cricket footage from **YouTube highlights**
- **Manually labeled** each video clip based on the type of batting stroke
- Extracted motion-based features using **OpenCV** and mathematical logic
- Trained a **Random Forest Classifier** for robust prediction
- Developed an interactive dashboard for:
  - Predicted shot category
  - Confidence score
  - Top-3 class probabilities
  - Video replay and swing-arc visualization

---

## 🔍 Shot Classes

The following cricket shot types were used for classification:

- Pull
- Cut
- Cover Drive
- Backfoot Punch
- Defensive
- Lofted Shot
- Other / Miscellaneous Shots

---

## 🛠️ Tech Stack

| Role | Technology |
|------|------------|
| Machine Learning | Random Forest |
| Feature Extraction | OpenCV, Motion & Pose Logic |
| Development | Python |
| Interface | Custom Dashboard (Video & Probability Insights) |

---

## ⚙️ Workflow

1. **Data Collection** – Sourced cricket highlights from YouTube
2. **Annotation** – Manually labeled each shot clip
3. **Feature Engineering** – Extracted trajectory, pose transitions, body motion
4. **Model Training** – Implemented Random Forest model for classification
5. **Evaluation & Deployment** – Dashboard integrated for user interaction

---

## 📊 Sample Output

> *<img width="2235" height="1397" alt="Screenshot 2025-09-23 133710" src="https://github.com/user-attachments/assets/6f081b2a-5e6e-4ad5-8a6d-60959b6cc3b5" />
*
> *<img width="2129" height="1260" alt="Screenshot 2025-09-23 133740" src="https://github.com/user-attachments/assets/525de7e3-f0ff-4516-91d2-fc14dc9d15b6" />
*

The dashboard displays:
- Predicted shot with description
- Confidence % bar
- Probability breakdown
- Bat swing arc and replay visuals

---

## 🚀 Future Enhancements

- Upgrade classifier to **CNN + LSTM** for sequential video learning
- Real-time detection from live camera feed
- Expand dataset with more player scenarios
- Performance scoring & insight analytics

---

## 👤 Author

**Ojas Panse**  
mail: ojaspanse200@gmail.com
LinkedIN: https://www.linkedin.com/in/ojas-panse-2a8a80286/
---

