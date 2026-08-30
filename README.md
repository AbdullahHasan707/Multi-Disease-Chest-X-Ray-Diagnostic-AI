# Multi-Disease Chest X-Ray Diagnostic AI

A deep learning computer vision system that analyzes chest X-ray images and classifies them into **four diagnostic categories**:

- **Normal**
- **Pneumonia**
- **COVID-19**
- **Tuberculosis (TB)**

Built with **DenseNet121** (Transfer Learning), **TensorFlow/Keras**, and deployed as an interactive **Streamlit** web application.

**New:** Grad-CAM visual explainability + optional Alibaba **Qwen** natural-language explanation.

---

## Project Overview

This project uses a pretrained **DenseNet121** model fine-tuned on a multi-class chest X-ray dataset from Kaggle. The system provides real-time diagnosis with confidence scores, class probability breakdowns, Grad-CAM heatmaps (where the model looked), and optional AI-generated clinical-style explanations via Alibaba Qwen.

### Key Features
- 4-class chest X-ray classification (Normal / Pneumonia / COVID-19 / TB)
- Transfer Learning with DenseNet121
- **Grad-CAM** heatmap for visual explainability
- Optional **Qwen** (Alibaba DashScope) text explanation
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
| Peak Validation Accuracy | **92.19%**                      |
| Final Test Accuracy      | **83.79%**                      |
| Framework                | TensorFlow / Keras              |
| Explainability           | Grad-CAM                        |
| Optional LLM             | Alibaba Qwen (DashScope)        |

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
Ensure `models/multi_disease_cnn_final.h5` exists in the repo.

### 4. (Optional) Qwen API key
For AI text explanation, set your Alibaba DashScope API key:
```bash
export DASHSCOPE_API_KEY="your_key_here"
```
Or add it in Streamlit secrets / environment variables.

### 5. Run the app
```bash
streamlit run app.py
```

---

## Grad-CAM + Qwen Flow

1. Upload chest X-ray
2. Click **RUN MULTI-DISEASE DEEP SCAN**
3. See:
   - Predicted class + confidence
   - Class probability bar chart
   - **Grad-CAM heatmap** (model attention regions)
   - **Qwen explanation** (if API key is set)

---

## Deploy on Alibaba Cloud (Hackathon)

### Quick path (ECS)
1. Create Ubuntu ECS (2 vCPU / 4GB+ RAM)
2. Install Python, pip, git
3. Clone repo and install requirements
4. Place model file under `models/`
5. Run:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```
6. Open security group port **8501**
7. Access: `http://YOUR_PUBLIC_IP:8501`

### Qwen on Alibaba
- Enable **DashScope / Model Studio**
- Create API key for `qwen-turbo` or `qwen-plus`
- Set `DASHSCOPE_API_KEY` on the server

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
