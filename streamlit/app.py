import streamlit as st
import requests
from PIL import Image
import io
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="Pneumonia Classification",
    page_icon="🫁",
    layout="wide"
)

# 제목
st.title("Pneumonia Classification System")
st.markdown("Upload a chest X-ray image to classify the type of pneumonia")

# 파일 업로드
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 표시
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    # 예측 요청
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://localhost:8000/predict", files=files)
    
    if response.status_code == 200:
        prediction = response.json()
        
        with col2:
            st.subheader("Prediction Results")
            
            # 예측 클래스와 신뢰도 표시
            st.markdown(f"### Predicted Class: {prediction['predicted_class']}")
            st.markdown(f"### Confidence: {prediction['confidence']*100:.2f}%")
            
            # 확률 분포 시각화
            probabilities = prediction['probabilities']
            
            # Plotly를 사용한 막대 그래프
            fig = go.Figure(data=[
                go.Bar(
                    x=list(probabilities.keys()),
                    y=list(probabilities.values()),
                    text=[f'{prob*100:.1f}%' for prob in probabilities.values()],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Probability Distribution",
                xaxis_title="Class",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Error occurred during prediction. Please try again.")

# 추가 정보
st.markdown("""
---
### About the Model
This system uses a deep learning model based on EfficientNet-B3 to classify chest X-ray images into four categories:
- COVID-19
- Bacterial Pneumonia
- Viral Pneumonia
- Normal

### How to Use
1. Upload a chest X-ray image using the file uploader above
2. The system will automatically process the image and display the results
3. Results include the predicted class, confidence score, and probability distribution
""")