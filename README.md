# Multi-Disease Chest X-Ray Diagnostic AI

A deep learning computer vision system that analyzes chest X-ray images and classifies them into **four diagnostic categories**:

- **Normal**
- **Pneumonia**
- **COVID-19**
- **Tuberculosis (TB)**

Built with **DenseNet121** (Transfer Learning), **TensorFlow/Keras**, and deployed as an interactive **Streamlit** web application.

Includes **Grad-CAM** visual explainability.

---

## Project Overview

This project uses a pretrained **DenseNet121** model fine-tuned on a multi-class chest X-ray dataset from Kaggle. The system provides real-time diagnosis with confidence scores, class probability breakdowns, and Grad-CAM heatmaps (regions the model focused on).

### Key Features
- 4-class chest X-ray classification (Normal / Pneumonia / COVID-19 / TB)
- Transfer Learning with DenseNet121
- **Grad-CAM** heatmap for visual explainability
- Real-time inference via Streamlit
- Confidence score + probability visualization
- Patient session logging

---

## Model Details

| Item                     | Value                           |
|--------------------------|---------------------------------|
| Architecture             | DenseNet121 (ImageNet weights)  |
| Input Size               | 224 x 224                       |
| Classes                  | Normal, Pneumonia, COVID-19, TB |
| Training Epochs          | 10                              |
| Final Test Accuracy      | **83.79%**                      |
| Test Loss                | 0.3801                          |
| Framework                | TensorFlow / Keras              |
| Explainability           | Grad-CAM                        |
| Deployment               | Streamlit Community Cloud       |

---

## Test Set Performance

| Class         | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| COVID-19      | 0.94      | 0.80   | 0.87     | 106     |
| Normal        | 0.90      | 0.64   | 0.75     | 234     |
| Pneumonia     | 0.79      | 0.96   | 0.86     | 390     |
| Tuberculosis  | 0.90      | 0.93   | 0.92     | 41      |
| **Accuracy**  |           |        | **0.84** | 771     |
| **Macro avg** | 0.89      | 0.83   | 0.85     | 771     |
| **Weighted avg** | 0.85   | 0.84   | 0.83     | 771     |

---

## Dataset

- **Source**: [Kaggle – Chest X-Ray (Pneumonia, COVID-19, Tuberculosis)](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis)
- **Classes**: Normal | Pneumonia | COVID-19 | Tuberculosis
- Images resized to 224x224, normalized, and augmented (rotation, zoom, flip)

---

## Live Demo

Streamlit App: https://multi-disease-chest-x-ray-diagnostic.streamlit.app/

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/AbdullahHasan707/Multi-Disease-Chest-X-Ray-Diagnostic-AI.git
cd Multi-Disease-Chest-X-Ray-Diagnostic-AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Model file
Ensure the trained model exists at:
```text
model/multi_disease_cnn_final.h5
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## App Flow

1. Upload chest X-ray
2. Click **RUN MULTI-DISEASE DEEP SCAN**
3. See:
   - Predicted class + confidence
   - Class probability bar chart
   - **Grad-CAM heatmap** (model attention regions)

---

## Deployment

This project is deployed on **Streamlit Community Cloud**.

- Repository: public GitHub repo
- Main file: `app.py`

---

## Disclaimer

This system is for **research and educational / decision-support** purposes only. It is **not** a final medical diagnosis. Always correlate with clinical findings and expert radiologist review.

---

## Author

**Abdullah Hasan Shah**  
BS Computer Science  
University of Technology Nowshera, Pakistan  

GitHub: [AbdullahHasan707](https://github.com/AbdullahHasan707)

---

## License

Open-source for educational and research purposes.
