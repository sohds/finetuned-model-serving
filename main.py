import io
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import logging
import os
from torchvision import models

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 디바이스 설정
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available, using CPU")
    return device

# 모델 클래스 정의
class PneumoniaModelModified(nn.Module):
    def __init__(self, num_classes=4, input_features=1536):
        super(PneumoniaModelModified, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

# 전역 변수
device = None
model = None
feature_extractor = None
class_names = ['COVID-19', 'Bacterial Pneumonia', 'Viral Pneumonia', 'Normal']

def load_models():
    global model, feature_extractor, device
    
    try:
        # 디바이스 설정
        device = get_device()
        
        # EfficientNet 특징 추출기 로드 (수정된 부분)
        logger.info("Loading feature extractor...")
        feature_extractor = models.efficientnet_b3(pretrained=True)
        # 마지막 분류층 제거
        feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
        feature_extractor.eval()
        feature_extractor.to(device)
        logger.info("Feature extractor loaded successfully")
        
        # 분류 모델 로드
        logger.info("Loading classification model...")
        model = PneumoniaModelModified(num_classes=4, input_features=1536)
        # 체크포인트 존재 여부 확인
        if os.path.exists('./checkpoint/best_model.pth'):
            model.load_state_dict(torch.load('./checkpoint/best_model.pth', map_location=device))
            logger.info("Checkpoint loaded successfully")
        else:
            logger.warning("No checkpoint found, using initialized weights")
        model.eval()
        model.to(device)
        logger.info("Classification model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    load_models()

def process_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    return image

def get_prediction(image_tensor):
    try:
        with torch.no_grad():
            # 특징 추출
            features = feature_extractor(image_tensor)
            features = features.view(features.size(0), -1)
            
            # 분류
            outputs = model(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
        return {
            "predicted_class": class_names[predicted_class],
            "confidence": float(confidence),
            "probabilities": {
                class_name: float(prob)
                for class_name, prob in zip(class_names, probabilities[0].tolist())
            }
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 이미지 읽기
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # 이미지 전처리
        image_tensor = process_image(image)
        
        # 예측
        prediction = get_prediction(image_tensor)
        return prediction
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}, 500

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "models_loaded": model is not None and feature_extractor is not None
    }

@app.get("/system-info")
async def system_info():
    gpu_info = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": "GPU" if torch.cuda.is_available() else "CPU"
    }
    return gpu_info