# 👤 Face Verification API with FastAPI & InsightFace

A simple API that verifies if two face images belong to the same person using **FastAPI**, **InsightFace**, and **cosine similarity**.

---

## 🚀 Features
- Upload two images (`img1` and `img2`)
- Detect faces using **InsightFace (buffalo_s model)**
- Compute embeddings and compare them
- Return **similarity score** and decision (`same` or `different`)

---

## 🛠️ Requirements

### Install dependencies
```bash
pip install fastapi uvicorn opencv-python-headless numpy insightface

