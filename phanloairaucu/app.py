import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from PIL import Image

# Tiêu đề giao diện
st.title("🍎 Ứng dụng Phân Loại Nông Sản (Táo)")
st.write("Tải lên một bức ảnh nông sản để mô hình dự đoán chất lượng/loại.")

# Tải mô hình và nhãn đã train
@st.cache_resource
def load_model():
    model = joblib.load('agricultural_model.pkl')
    class_names = joblib.load('class_names.pkl')
    return model, class_names

try:
    model, class_names = load_model()
except Exception as e:
    st.error(f"❌ Không tìm thấy file mô hình (.pkl). Hãy chắc chắn bạn đã chạy file train.py trước! Lỗi: {e}")
    st.stop()

# Khung tải ảnh lên từ máy tính
uploaded_file = st.file_uploader("Chọn một tệp ảnh (jpg, png, webp)...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Hiển thị ảnh người dùng vừa chọn lên giao diện
    image_pil = Image.open(uploaded_file).convert('RGB')
    st.image(image_pil, caption='📷 Ảnh bạn đã tải lên', use_column_width=True)
    
    if st.button('🚀 Bắt đầu Phân loại'):
        with st.spinner('Đang phân tích đặc trưng màu sắc và hình dáng...'):
            # Chuyển ảnh PIL sang định dạng OpenCV (BGR)
            image = np.array(image_pil)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Xử lý ảnh giống hệt lúc train
            image_resized = cv2.resize(image, (128, 128))
            
            # 1. Color Histogram
            hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            cv2.normalize(hist_h, hist_h)
            cv2.normalize(hist_s, hist_s)
            cv2.normalize(hist_v, hist_v)
            color_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
            
            # 2. HOG Features
            gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), block_norm='L2-Hys')
            
            # Kết hợp đặc trưng
            features = np.concatenate([color_features, hog_features]).reshape(1, -1)
            
            # Dự đoán kết quả
            pred_idx = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = probabilities[pred_idx] * 100
            
            predicted_label = class_names[pred_idx]
            
            # In kết quả nổi bật trên giao diện
            st.success("✨ Phân loại hoàn tất!")
            st.markdown(f"### 🏷️ Kết quả dự đoán: **{predicted_label}**")
            st.info(f"📊 Độ tin cậy của mô hình: **{confidence:.2f}%**")