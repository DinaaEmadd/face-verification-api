from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
from numpy.linalg import norm
import insightface

app = FastAPI()
face_model = insightface.app.FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
face_model.prepare(ctx_id=0)

def read_image(file: UploadFile):
    image = np.frombuffer(file.file.read(), np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

@app.post("/verify/")
async def verify_faces(img1: UploadFile = File(...), img2: UploadFile = File(...), threshold: float = 0.6):
    image1 = read_image(img1)
    image2 = read_image(img2)
    
    faces1 = face_model.get(image1)
    faces2 = face_model.get(image2)

    if len(faces1) == 0 or len(faces2) == 0:
        return {"error": "وجه واحد على الأقل لم يُكتشف"}

    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding
    similarity = cosine_similarity(emb1, emb2)

    return {
        "similarity": round(float(similarity), 4),
        "result": "same" if similarity > threshold else "diffrent"
    }
