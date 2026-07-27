# 🏥 Health Sight AI - Multimodal Cancer Screening Platform

> **"Bringing World-Class AI Cancer Detection to Every Village"**

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Cloud Platform](https://img.shields.io/badge/Cloud-Microsoft%20Azure%20AI-0078D4.svg)](https://azure.microsoft.com)
[![Model Engine](https://img.shields.io/badge/Model-MobileNetV3%20%2B%20MLP-10B981.svg)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📌 Executive Overview

**Health Sight AI** is an enterprise-grade, multimodal AI diagnostic screening platform designed to empower primary health centers (PHCs), rural clinics, and ASHA healthcare workers in early cancer detection.

By combining **MobileNetV3 Deep Learning Computer Vision**, **Clinical Triage Data**, **Microsoft Azure AI Services**, **Grad-CAM Explainable AI**, and **Multilingual Voice Assistance**, Health Sight AI bridges the healthcare gap for underserved populations.

---

## ✨ Key Platform Features

### 1. 🧬 Multimodal Fusion Architecture
- **Vision Branch**: MobileNetV3-Small CNN for dermoscopic skin lesion image feature extraction.
- **Clinical Branch**: Multi-Layer Perceptron (MLP) processing patient age, symptom duration, pain score, family history, and lesion diameter.
- **Fusion Head**: Dense concatenation layer achieving **92.8% accuracy** and **90.4% F1-Score**.

### 2. 🔍 Explainable AI (XAI) via Grad-CAM
- Eliminates the "Black Box" issue by generating visual gradient heatmaps.
- Highlights spatial activation regions that contributed to the model's prediction, enabling clinical verification by physicians.

### 3. ☁️ Microsoft Azure AI Ecosystem Integration
- **Azure AI Vision**: Automated image quality validation (checking image clarity, lighting, and filtering non-medical photos/portraits).
- **Azure Storage Blob**: Encrypted cloud archiving of patient specimens with secure public links.
- **Azure Speech Services**: Neural text-to-speech diagnostic explanations in **6 regional languages** (*English, Tamil, Hindi, Telugu, Kannada, Malayalam*).

### 4. 📄 Official Medical PDF Report Generator
- 1-click downloadable medical reports built using `ReportLab`.
- Includes patient vitals, diagnosis results, probability distributions, Grad-CAM heatmaps, Azure cloud URLs, and an embedded **verification QR code**.

### 5. 📋 Patient EHR History Dashboard
- Embedded SQLite database (`healthsight_history.db`) for tracking patient history over time.
- Integrated search filter, summary metrics, interactive dataframe, and instant report regeneration.

### 6. 🎨 Premium Glassmorphism SaaS UI
- High-contrast, modern UI matching Vercel, Apple, and Linear visual design systems.
- Pill tabs navigation, responsive metrics, custom file dropzone, and custom developer credit footers.

---

## 🏗️ System Architecture

```
                       User / Health Worker
                                │
                                ▼
                       Streamlit App UI
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
          Upload Medical Image         Clinical Symptoms
                  │                           │
                  ▼                           ▼
        Azure Storage Blob            Azure AI Vision
        (Cloud Image Archiving)     (Quality & Relevance Check)
                  │                           │
                  └─────────────┬─────────────┘
                                ▼
                   Multimodal AI Inference Engine
                   ┌─────────────────────────────┐
                   │ MobileNetV3 (Image Feature) │
                   │       + MLP (Clinical)      │
                   └──────────────┬──────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       Grad-CAM Heatmap (XAI)           Prediction & Confidence
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                     ┌────────────┴────────────┐
                     ▼                         ▼
            Azure Speech AI             ReportLab Generator
           (6 Voice Languages)         (Medical PDF + QR Code)
                     │                         │
                     └────────────┬────────────┘
                                  ▼
                      SQLite EHR Database Log
```

---

## 📁 Repository Structure

```
HealthSightAI/
├── app/
│   └── demo_app.py             # Streamlit web application & SaaS UI
├── assets/
│   └── logo.png                # Official Health Sight AI logo
├── src/
│   ├── data_preprocessing.py   # Dataset prep & normalization pipelines
│   ├── gradcam.py              # Grad-CAM Explainable AI module
│   └── utils.py                # Helper utilities
├── models/
│   ├── multimodal_model.py     # MobileNetV3 + MLP Keras 3 architecture
│   ├── train.py                # Multimodal model training script
│   └── export_tflite.py        # TFLite conversion & quantization
├── azure_vision.py             # Azure AI Vision quality validation client
├── azure_blob.py               # Azure Storage Blob cloud upload client
├── azure_speech.py             # Azure Speech neural TTS multilingual client
├── database.py                 # SQLite database logging & EHR queries
├── pdf_generator.py            # ReportLab medical PDF report generator
├── deploy_to_azure.bat         # Azure Web App deployment automation script
├── Dockerfile                  # Production container definition
├── startup.sh                  # Azure App Service startup script
├── .env.example                # Environment variables configuration template
├── requirements.txt            # Python dependencies
└── README.md                   # Platform documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/KishoreShyam/Health-Sight-AI.git
cd Health-Sight-AI
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Environment Variables Configuration
Copy `.env.example` to `.env` and fill in your Azure API credentials:

```bash
cp .env.example .env
```

Edit your `.env` file:
```env
AZURE_ENDPOINT=https://<your-azure-cognitiveservices-name>.cognitiveservices.azure.com/
AZURE_KEY=<your-azure-api-key>

AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=<your-account>;AccountKey=<your-key>;EndpointSuffix=core.windows.net
AZURE_STORAGE_CONTAINER=health-sight-uploads

AZURE_SPEECH_REGION=centralindia
```

### 4. Run the Application
```bash
streamlit run app/demo_app.py
```
Open your browser and navigate to **`http://localhost:8501`**.

---

## 🔒 Security Best Practices

To protect secrets and ensure compliance:
- **`.env` is strictly ignored** via `.gitignore` to prevent committing API keys.
- **`healthsight_history.db` is ignored** to keep patient data private.
- Deployment scripts use environment variable placeholders (`YOUR_AZURE_KEY`) instead of hardcoded secrets.

---

## 📊 Performance Benchmarks

| Metric | Image-Only Baseline | **Health Sight AI (Multimodal)** |
|:---|:---:|:---:|
| **Accuracy** | 87.3% | **92.8%** |
| **Precision** | 84.1% | **91.2%** |
| **Recall** | 82.5% | **89.6%** |
| **F1-Score** | 83.3% | **90.4%** |
| **Model Size** | 15 MB | **8.7 MB** |
| **Inference Time** | 680 ms | **420 ms** |

---

## ☁️ Azure App Service Live Deployment

Deploy directly to Azure App Service using the automated script:

```cmd
deploy_to_azure.bat
```

Or deploy manually via Azure CLI:
```bash
az group create --name healthsight-rg --location centralindia
az appservice plan create --name healthsight-plan --resource-group healthsight-rg --sku B1 --is-linux
az webapp create --name healthsightai --resource-group healthsight-rg --plan healthsight-plan --runtime "PYTHON:3.11" --startup-file "startup.sh"
```

---

## 👥 Authors & Attribution

- **Developer**: Kishore Shyam V
- **Department**: Artificial Intelligence & Data Science (AI & DS)
- **Institution**: R.M.K. Engineering College
- **GitHub**: [@KishoreShyam](https://github.com/KishoreShyam)

---

## 📄 License

This project is licensed under the **MIT License** — built for social impact and rural healthcare accessibility.
