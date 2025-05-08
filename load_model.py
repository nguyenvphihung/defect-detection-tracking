#file này Chứa hàm để tải mô hình YOLOv5 (đơn giản là để không phải xử lý lại từ đầu mỗi lần chạy file detect)
import os
import torch
from pathlib import Path

# Đường dẫn đến thư mục dự án
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

def load_yolov5_model(conf_threshold=0.25, only_person=True):
    """Tải mô hình YOLOv5 đã huấn luyện"""
    print("🔄 Đang tải mô hình YOLOv5 đã huấn luyện...")
    
    # Đường dẫn đến file weights
    weights_path = PROJECT_ROOT / "runs" / "mot16_train" / "weights" / "best.pt"
    
    if not weights_path.exists():
        print(f"⚠️ Không tìm thấy file weights tại {weights_path}")
        print("⚠️ Sử dụng mô hình YOLOv5 mặc định...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    else:
        print(f"✅ Sử dụng mô hình đã huấn luyện: {weights_path}")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path))
    
    # Cấu hình model
    model.conf = conf_threshold  # Ngưỡng tin cậy
    if only_person:
        model.classes = [0]  # Chỉ phát hiện người (class 0)
    
    return model

if __name__ == "__main__":
    # Ví dụ sử dụng
    model = load_yolov5_model()
    print("✅ Đã tải mô hình thành công!")
    print(f"📋 Thông tin mô hình: {model.names}")