# 💧 YOLO Water Meter Reader

A FastAPI-powered service that uses a YOLO model to detect and extract numeric readings from water meter images. Ideal for automating utility data collection in factories, smart homes, or IoT systems.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![YOLO](https://img.shields.io/badge/YOLOv8-Model-red)

---

## 🚀 Features

- 🔍 Detects digits from water meter images using YOLO
- 🧠 Trained on custom YOLOv8 model (best v3(100).pt)
- 📷 Upload images via REST API and get back the numeric reading
- ✅ Confidence filtering and left-to-right digit sorting
- ⚙ Production-ready FastAPI setup

---

## 📦 Requirements

- Python 3.10+
- FastAPI
- Uvicorn
- ultralytics (YOLOv8)
- numpy
- Pillow

Install dependencies:

```bash
pip install -r requirements.txt
