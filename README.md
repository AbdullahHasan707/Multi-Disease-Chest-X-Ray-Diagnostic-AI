# Multi-Disease Chest X-Ray Diagnostic AI

A deep learning computer vision system that analyzes chest X-ray images and classifies them into **four diagnostic categories**:

- **Normal**
- **Pneumonia**
- **COVID-19**
- **Tuberculosis (TB)**

Built with **DenseNet121** (Transfer Learning), **TensorFlow/Keras**, and deployed as an interactive **Streamlit** web application with **Grad-CAM** visual explainability.

---

## Project Overview

This project uses a pretrained **DenseNet121** model fine-tuned on a multi-class chest X-ray dataset. The system provides real-time diagnosis with confidence scores, class probability breakdowns, and Grad-CAM heatmaps showing the regions the model focused on.

### Key Features
- 4-class chest X-ray classification (Normal / Pneumonia / COVID-19 / TB)
- Transfer Learning with DenseNet121 (ImageNet pretrained)
- Data augmentation + class weighting for imbalance handling
- Early stopping and learning rate scheduling
- **Grad-CAM** heatmap for visual explainability
- Real-time inference via Streamlit
- Confidence score + probability visualization

---

## Model Details

| Item                     | Value                              |
|--------------------------|------------------------------------|
| Architecture             | DenseNet121 (ImageNet weights)     |
| Input Size               | 224 × 224                          |
| Classes                  | COVID-19, Normal, Pneumonia, TB    |
| Training Epochs          | 16 (with Early Stopping)           |
| **Final Test Accuracy**  | **87.81%**                         |
| Test Loss                | 0.3378                             |
| Framework                | TensorFlow / Keras                 |
| Explainability           | Grad-CAM                           |
| Deployment               | Streamlit Community Cloud          |

---

## Test Set Performance

| Class         | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| COVID-19      | 0.95      | 0.89   | 0.92     | 106     |
| Normal        | 0.80      | 0.87   | 0.83     | 234     |
| Pneumonia     | 0.91      | 0.87   | 0.89     | 390     |
| Tuberculosis  | 0.93      | 0.95   | 0.94     | 41      |
| **Accuracy**  |           |        | **0.88** | 771     |
| **Macro avg** | 0.90      | 0.90   | 0.90     | 771     |
| **Weighted avg** | 0.88   | 0.88   | 0.88     | 771     |

---

## Dataset

- **Source**: [Kaggle – Chest X-Ray (Pneumonia, COVID-19, Tuberculosis)](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis)
- **Classes**: COVID-19 | Normal | Pneumonia | Tuberculosis
- Images resized to 224×224, normalized, and augmented (rotation, shift, zoom, horizontal flip)
- Class weights applied to handle data imbalance

---

## Live Demo

**Streamlit App**: [https://multi-disease-chest-x-ray-diagnostic.streamlit.app/](https://multi-disease-chest-x-ray-diagnostic.streamlit.app/)

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

## Training Notebook

The full training pipeline is available in:

- `DenseNet121_MultiClass_CXR_Classification.ipynb`

It includes data loading, augmentation, class weights, model training, evaluation, confusion matrix, and training curves.

---

## App Flow

1. Upload a chest X-ray image
2. Click **RUN MULTI-DISEASE DEEP SCAN**
3. View:
   - Predicted class + confidence score
   - Class probability bar chart
   - **Grad-CAM heatmap** (model attention regions)

---

## Disclaimer

This system is for **research and educational / decision-support** purposes only. It is **not** a final medical diagnosis. Always correlate results with clinical findings and expert radiologist review.

---

## Author

**Abdullah Hasan Shah**  
BS Computer Science  
University of Technology Nowshera, Pakistan  

GitHub: [AbdullahHasan707](https://github.com/AbdullahHasan707)

---

## License

Open-source for educational and research purposes.
