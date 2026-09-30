# FreshHarvest - Fruit Quality Inspection System

This project is an image classification application built for FreshHarvest Logistics to classify fruit freshness (Fresh vs. Spoiled) across 16 classes. It uses Transfer Learning with a pre-trained ResNet50 model and a Streamlit web application.

---

## Project Overview

In fresh produce logistics, checking fruit quality manually is slow and can lead to errors. This project provides an automated image classification tool:
* Model: Pre-trained ResNet50 with frozen weights and a custom classification head.
* Training: Trained for 6 epochs, achieving fast convergence and reducing computing costs.
* Web App: A Streamlit application where users can drag and drop images to see the predicted class and confidence score.

---

## Model Performance

| Metric | Baseline CNN (10 Epochs) | ResNet50 Transfer Learning (6 Epochs) |
| :--- | :---: | :---: |
| Training Epochs | 10 Epochs | 6 Epochs |
| 16-Class Test Accuracy | 95.27% | 98.68% |
| Test Loss | 0.1770 | 0.0455 |
| Fresh vs. Spoiled Accuracy | 97.45% | 99.05% |
| Latency | ~35 ms | ~18 ms |

---

## Supported Classes (16 Classes)

The model classifies 8 types of fruit into Fresh (F_) or Spoiled (S_):

| Fruit | Fresh Class | Spoiled Class |
| :--- | :---: | :---: |
| Banana | F_Banana | S_Banana |
| Lemon | F_Lemon | S_Lemon |
| Lulo (Naranjilla) | F_Lulo | S_Lulo |
| Mango | F_Mango | S_Mango |
| Orange | F_Orange | S_Orange |
| Strawberry | F_Strawberry | S_Strawberry |
| Tamarillo | F_Tamarillo | S_Tamarillo |
| Tomato | F_Tomato | S_Tomato |

---

## Model Architecture

The transfer learning model consists of:
1. Input Layer: (128, 128, 3)
2. Data Augmentation: Random Flip, Rotation, Zoom, Contrast
3. ResNet50 Preprocessing: Zero-centered ImageNet mean
4. Pre-trained ResNet50 Backbone (weights='imagenet', trainable=False)
5. GlobalAveragePooling2D
6. BatchNormalization
7. Dense (256, ReLU) + Dropout (0.35)
8. Dense (16, Softmax)

---

## How to Run Locally

### 1. Clone Repository
```bash
git clone https://github.com/ridwanenam/FreshHarvest-Fruit-Inspection.git
cd FreshHarvest-Fruit-Inspection
```

### 2. Create Virtual Environment
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

### 4. Model File
Place the trained model file `resnet50_freshharvest_transfer_model.keras` inside the `models/` directory:
```
FreshHarvest-Fruit-Inspection/
└── models/
    └── resnet50_freshharvest_transfer_model.keras
```
Note: If running on Streamlit Cloud, the application will automatically download the model from GitHub Release v1.0.0 if not present locally.

### 5. Run Streamlit App
```bash
streamlit run app.py
```

---

## Repository Structure

```
FreshHarvest-Fruit-Inspection/
├── .streamlit/
│   └── config.toml
├── assets/
│   └── samples/
│       ├── fresh_banana.jpg
│       ├── fresh_orange.jpg
│       ├── fresh_strawberry.jpg
│       ├── spoiled_banana.jpg
│       ├── spoiled_orange.jpg
│       └── spoiled_strawberry.jpg
├── models/
│   └── resnet50_freshharvest_transfer_model.keras
├── notebook/
│   └── freshharvest-logistics (2).ipynb
├── .gitignore
├── app.py
├── requirements.txt
└── README.md
```

---

## Author

* Ridwan Triputra (GitHub: @ridwanenam)
* Codebasics Virtual Bootcamp - FreshHarvest Logistics
