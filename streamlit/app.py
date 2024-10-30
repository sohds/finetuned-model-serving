import streamlit as st
import requests
from PIL import Image
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="Lung Disease Classification",
    page_icon="🫁",
    layout="wide"
)

st.image('streamlit/dataset-cover.jpeg', use_column_width=True)
# 제목
st.title("👩‍⚕️ Pneumonia & COVID-19 Classification System")
st.markdown("흉부 X-ray 이미지를 업로드 해서, 폐렴인지 코로나인지 확인해 보세요!")

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
            st.subheader("예측 결과")
            
            # 예측 클래스 표시
            st.markdown(f"##### 흉부 X-Ray 예측 클래스: {prediction['predicted_class']}")
            
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
            
            st.markdown('')
            st.markdown('- 2024-2 딥러닝기반데이터분석 개인 과제: CNN Fine-tuning')
            st.markdown('- 해당 진단 결과는 의료 전문가의 판단을 보조하는 용도로만 사용되어야 합니다.')
            st.markdown('- 제작자: 오서연 (sohtks@gmail.com), 오류가 발견되거나 궁금한 사항 있으시면 언제든 연락주세요!')
            st.markdown('- 데이터셋 출처: [Kaggle](https://www.kaggle.com/datasets/gibi13/pneumonia-covid19-image-dataset)')
    else:
        st.error("예측 중에 오류가 발생했어요. 다시 시도해 주세요.")

# 추가 정보
st.markdown("""
---
### 🤖 About the Model
이 시스템은 EfficientNet-B3 딥러닝 모델을 fine-tuning해 사용하여 흉부 X-ray 이미지를 네 가지 범주로 분류합니다:
- COVID-19 (코로나)
- Bacterial Pneumonia (세균성 폐렴)
- Viral Pneumonia (바이러스성 폐렴)
- Normal (정상)

### 🧐 How to Use
1. 파일 업로더를 사용하여 흉부 X-ray 이미지를 업로드합니다.
2. 시스템은 FastPAI로 모델을 서빙해, 이미지를 처리하고 진단 결과를 표시합니다.
3. 결과에는 예측된 클래스, 신뢰도 점수 및 확률 분포가 포함됩니다.
""")