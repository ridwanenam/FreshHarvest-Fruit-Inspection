# 🍎 FreshHarvest AI — Automated Fruit Freshness Inspection System

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Model Backbone](https://img.shields.io/badge/Backbone-ResNet50%20Transfer%20Learning-green.svg)](https://keras.io/api/applications/resnet50/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end Computer Vision & Deep Learning solution developed for **FreshHarvest Logistics** (Cold Storage & Warehousing, California) to automate produce quality control, sort fresh vs. spoiled produce, and eliminate manual conveyor inspection bottlenecks.

---

## 📌 Executive Summary & Business Context

In high-throughput cold storage facilities, inspecting fruits manually is slow, error-prone, and labor-intensive. Spoilage in even a single fruit can cross-contaminate entire pallets during refrigerated transit, resulting in heavy financial penalties and client rejections.

To resolve this, this project deploys an AI-powered visual inspection pipeline:
* **Pre-trained Backbone:** Leverages **ResNet50** transfer learning initialized with ImageNet weights to reduce GPU training costs and accelerate convergence (achieving target accuracy in just 6 epochs).
* **High-Accuracy Classification:** Accurately classifies 16 distinct fruit categories (8 fruits $\times$ Fresh vs. Spoiled).
* **Interactive Operations Interface:** A sleek **Streamlit** web app equipped with drag-and-drop image upload, real-time probability distributions, and automated warehouse dispatch decisions (Route to Packaging Conveyor A vs. Reject to Diverter B).

---

## 🏆 Key Performance Metrics

| Evaluation Metric | Baseline CNN (Scratch) | ResNet50 Transfer Learning | Client KPI Target |
| :--- | :---: | :---: | :---: |
| **Training Epochs** | 10 Epochs | **6 Epochs (-40% Time)** | $\le$ 10 Epochs |
| **16-Class Test Accuracy** | 95.27% | **98.68%** | $\ge$ 90.00% |
| **Holdout Test Loss** | 0.1770 | **0.0455 (-74% Loss)** | Lowest Possible |
| **Binary Freshness KPI** | 97.45% | **99.05%** | $\ge$ 95.00% |
| **Inference Latency** | ~35 ms / image | **~18 ms / image** | Real-time (<50 ms) |

---

## 🍓 Supported Produce Classes (16 Classes)

The system inspects 8 core fruit types across two condition states:

| Fruit Type | Fresh Class Code | Spoiled Class Code | Quality Decision |
| :--- | :---: | :---: | :--- |
| **Banana** | `F_Banana` | `S_Banana` | Fresh $\rightarrow$ Conveyor A / Spoiled $\rightarrow$ Diverter B |
| **Lemon** | `F_Lemon` | `S_Lemon` | Fresh $\rightarrow$ Conveyor A / Spoiled $\rightarrow$ Diverter B |
| **Lulo (Naranjilla)** | `F_Lulo` | `S_Lulo` | Fresh $\rightarrow$ Conveyor A / Spoiled $\rightarrow$ Diverter B |
| **Mango** | `F_Mango` | `S_Mango` | Fresh $\rightarrow$ Conveyor A / Spoiled $\rightarrow$ Diverter B |
| **Orange** | `F_Orange` | `S_Orange` | Fresh $\rightarrow$ Conveyor A / Spoiled $\rightarrow$ Diverter B |
| **Strawberry** | `F_Strawberry` | `S_Strawberry` | Fresh $\rightarrow$ Conveyor A / Spoiled $\rightarrow$ Diverter B |
| **Tamarillo** | `F_Tamarillo` | `S_Tamarillo` | Fresh $\rightarrow$ Conveyor A / Spoiled $\rightarrow$ Diverter B |
| **Tomato** | `F_Tomato` | `S_Tomato` | Fresh $\rightarrow$ Conveyor A / Spoiled $\rightarrow$ Diverter B |

---

## 🏗️ Model Architecture

The transfer learning architecture connects a frozen feature extractor with a custom classification head:

```
Input Image (128x128x3)
       │
       ▼
Data Augmentation (Flip, Rotation, Zoom, Contrast)
       │
       ▼
ResNet50 Preprocessing (Zero-centered ImageNet BGR)
       │
       ▼
ResNet50 Backbone (Weights: ImageNet, Trainable: False)
       │
       ▼
GlobalAveragePooling2D
       │
       ▼
BatchNormalization
       │
       ▼
Dense (256, ReLU) + Dropout (0.35)
       │
       ▼
Dense (16, Softmax) -> 16-Class Probabilities
```

---

## 🚀 Quickstart & Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ridwanenam/FreshHarvest-Fruit-Inspection-AI.git
cd FreshHarvest-Fruit-Inspection-AI
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Ensure Model File is Present
Place the trained model file `resnet50_freshharvest_transfer_model.keras` inside the `models/` directory:
```
FreshHarvest-Fruit-Inspection-AI/
└── models/
    └── resnet50_freshharvest_transfer_model.keras
```
*(Note: Because model weights are ~101 MB, exceeding standard GitHub 100 MB limits, files are tracked via Git LFS or direct release download).*

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ Streamlit App Features

* **Drag-and-Drop Image Uploader:** Supports PNG, JPG, JPEG, and WEBP formats directly from conveyor camera streams or local folders.
* **Warehouse Sample Selector:** Includes pre-loaded sample images (`fresh_banana.jpg`, `spoiled_orange.jpg`, etc.) for zero-setup instant testing.
* **Executive Quality Decision:** Displays high-visibility color-coded badges (🟢 Fresh Grade A vs. 🔴 Spoiled Bio-hazard) and routing recommendations.
* **Top-5 Probability Distribution:** Interactive Plotly horizontal bar chart highlighting confidence distribution across top candidate classes.
* **Hardware Telemetry:** Real-time display of inference latency in milliseconds and model baseline metrics.

---

## 📁 Repository Structure

```
FreshHarvest-Fruit-Inspection-AI/
├── .streamlit/
│   └── config.toml          # Custom theme settings
├── assets/
│   └── samples/             # Demo fruit images for quick testing
│       ├── fresh_banana.jpg
│       ├── fresh_orange.jpg
│       ├── fresh_strawberry.jpg
│       ├── spoiled_banana.jpg
│       ├── spoiled_orange.jpg
│       └── spoiled_strawberry.jpg
├── models/
│   └── resnet50_freshharvest_transfer_model.keras  # Trained weights
├── .gitignore               # Ignored files (caches, large weights)
├── app.py                   # Streamlit web application
├── requirements.txt         # Project dependencies
└── README.md                # Comprehensive documentation
```

---

## 👥 Contributors & Contact

* **Developer:** Ridwan Triputra ([@ridwanenam](https://github.com/ridwanenam))
* **Organization:** FreshHarvest Logistics / Codebasics Virtual Bootcamp
* **Role:** Machine Learning Engineer
