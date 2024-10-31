# finetuned-model-serving
2024-2 '딥러닝기반데이터분석' 강의 과제 2 'CNN Fine-tuning과 Serving'
## Chest X-Ray로 폐 질환 분류하기
![image](https://github.com/user-attachments/assets/6a46b487-ea83-47e8-8ef7-240a87269a2d)
![image](https://github.com/user-attachments/assets/8d89bf49-3c68-4dc1-946f-ce1f4baf2ac5)]
![image](https://github.com/user-attachments/assets/86a217da-9b48-4769-b2d5-2d7fa27eb8dd)

## Model Serve
### Pipeline
```mermaid
---
config:
  theme: neutral
  look: neo
  layout: dagre
---
graph LR
subgraph Frontend[Streamlit Frontend]
A[X-ray 이미지 업로드] --> B[이미지 보여주기]
B --> C[Backend로 보내기]
H[결과 받기] --> I[예측 보여주기]
I --> J[Probability Plot 보여주기]
end
subgraph Backend[FastAPI Backend]
    D[이미지 받기] --> E[이미지 전처리]
    E --> F[Feature Extraction<br/>EfficientNet-B3]
    F --> G[Custom Classifier]
    G --> K[Return Prediction]
end
C -->|HTTP POST| D
K -->|JSON Response| H
```

### Demo
![](https://github.com/sohds/finetuned-model-serving/blob/main/readme/4%EB%B0%B0%EC%86%8D_%EA%B5%AC%ED%98%84%EC%98%81%EC%83%81.gif)
