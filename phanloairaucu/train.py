import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Hàm trích xuất đặc trưng
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Cảnh báo: Không thể đọc được file ảnh: {image_path}")
        return None
    
    image = cv2.resize(image, (128, 128))
    
    # Color Histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    color_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    
    # HOG Features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys')
    
    combined_features = np.concatenate([color_features, hog_features])
    return combined_features

# 2. Hàm tải dataset
def load_dataset(data_dir):
    features = []
    labels = []
    if not os.path.exists(data_dir):
        print(f"❌ Lỗi: Không tìm thấy thư mục {data_dir}")
        return np.array(features), np.array(labels), []
        
    class_names = sorted(os.listdir(data_dir))
    print(f"📁 Đang quét thư mục {data_dir}, phát hiện các lớp: {class_names}")
    
    for label_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_folder):
            continue
        img_list = os.listdir(class_folder)
        print(f"   - Lớp '{class_name}' có {len(img_list)} file.")
        for img_name in img_list:
            img_path = os.path.join(class_folder, img_name)
            feat = extract_features(img_path)
            if feat is not None:
                features.append(feat)
                labels.append(label_idx)
                
    return np.array(features), np.array(labels), class_names

if __name__ == "__main__":
    train_dir = './dataset/train'
    val_dir = './dataset/val'
    
    X_train, y_train, class_names = load_dataset(train_dir)
    
    if len(X_train) == 0:
        print("❌ Lỗi: Không tìm thấy ảnh nào trong tập train! Vui lòng kiểm tra lại đường dẫn và thư mục ảnh.")
    else:
        X_val, y_val, _ = load_dataset(val_dir)
        
        print(f"🚀 Bắt đầu huấn luyện với {len(X_train)} mẫu...")
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train)
        
        if len(X_val) > 0:
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            print(f"🎯 Độ chính xác trên tập Val: {acc * 100:.2f}%")
            
        joblib.dump(model, 'agricultural_model.pkl')
        joblib.dump(class_names, 'class_names.pkl')
        print("✅ Đã lưu mô hình thành công vào file 'agricultural_model.pkl'!")