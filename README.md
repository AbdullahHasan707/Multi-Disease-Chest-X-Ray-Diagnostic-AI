# Multi-Disease Chest X-Ray Diagnostic AI

A deep learning computer vision system that analyzes chest X-ray images and classifies them into **four diagnostic categories**:

- **Normal**
- **Pneumonia**
- **COVID-19**
- **Tuberculosis (TB)**

Built with **DenseNet121** (Transfer Learning), **TensorFlow/Keras**, and deployed as an interactive **Streamlit** web application.

---

## 🔍 Project Overview

This project uses a pretrained **DenseNet121** model fine-tuned on a multi-class chest X-ray dataset from Kaggle. The system provides real-time diagnosis with confidence scores and class probability breakdowns through a clean clinical dashboard.

### Key Features
- 4-class chest X-ray classification
- Transfer Learning with DenseNet121
- Real-time inference via Streamlit
- Confidence score + probability visualization
- Patient session logging

---

## 🧠 Model Details

| Item                    | Value                          |
|-------------------------|--------------------------------|
| Architecture            | DenseNet121 (ImageNet weights) |
| Input Size              | 224 × 224                      |
| Classes                 | Normal, Pneumonia, COVID-19, TB |
| Training Epochs         | 10                             |
| Peak Validation Accuracy| **92.19%**                     |
| Final Test Accuracy     | **83.79%**                     |
| Framework               | TensorFlow / Keras             |

---

## 📁 Dataset

- **Source**: [Kaggle – Chest X-Ray (Pneumonia, COVID-19, Tuberculosis)](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis)
- **Classes**: Normal | Pneumonia | COVID-19 | Tuberculosis
- Images were resized, normalized, and augmented (rotation, zoom, flip)

---

## 🖥️ Live Demo

🔗 **Streamlit App**: https://multi-disease-chest-x-ray-diagnostic.streamlit.app/

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/AbdullahHasan707/Multi-Disease-Chest-X-Ray-Diagnostic-AI.git
cd Multi-Disease-Chest-X-Ray-Diagnostic-AI
