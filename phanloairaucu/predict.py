import os
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from sklearn.metrics import accuracy_score
from PIL import Image

# 1. Tải mô hình và nhãn đã huấn luyện
model = joblib.load('agricultural_model.pkl')
class_names = joblib.load('class_names.pkl')

def evaluate_validation_set(val_dir):
    if not os.path.exists(val_dir):
        print(f"❌ Không tìm thấy thư mục: {val_dir}")
        return

    true_labels = []
    pred_labels = []
    
    print(f"📁 Đang quét thư mục kiểm thử: {val_dir}\n" + "-"*40)
    
    # Duyệt qua các thư mục nhãn con trong val
    for label_idx, class_name in enumerate(sorted(os.listdir(val_dir))):
        class_folder = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_folder):
            continue
            
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            
            try:
                with Image.open(img_path) as pil_img:
                    image = np.array(pil_img.convert('RGB'))
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception:
                continue
                
            # Tiền xử lý & trích xuất đặc trưng
            image_resized = cv2.resize(image, (128, 128))
            
            hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            cv2.normalize(hist_h, hist_h)
            cv2.normalize(hist_s, hist_s)
            cv2.normalize(hist_v, hist_v)
            color_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
            
            gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), block_norm='L2-Hys')
            
            features = np.concatenate([color_features, hog_features]).reshape(1, -1)
            
            # Dự đoán
            pred_idx = model.predict(features)[0]
            
            true_labels.append(label_idx)
            pred_labels.append(pred_idx)
            
            print(f"📷 Ảnh: {img_name} | Thực tế: {class_name} --> Dự đoán: {class_names[pred_idx]}")

    if len(true_labels) > 0:
        acc = accuracy_score(true_labels, pred_labels)
        print("-" * 40)
        print(f"🎯 Tổng kết độ chính xác trên tập Val: {acc * 100:.2f}%")
    else:
        print("⚠️ Không tìm thấy ảnh hợp lệ nào trong thư mục val để đánh giá.")

if __name__ == "__main__":
    evaluate_validation_set('dataset/val')