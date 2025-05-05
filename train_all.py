import os
from pathlib import Path

# Đường dẫn đến thư mục dự án
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

def train_yolov5():
    """Huấn luyện YOLOv5 với toàn bộ dataset MOT16"""
    print("🚀 Bắt đầu huấn luyện YOLOv5...")
    
    # Clone YOLOv5 repository nếu chưa có
    yolov5_path = PROJECT_ROOT / "yolov5"
    if not os.path.exists(yolov5_path):
        print("📥 Đang clone YOLOv5 repository...")
        os.system(f'git clone https://github.com/ultralytics/yolov5 "{yolov5_path}"')
        os.system(f'pip install -r "{yolov5_path}/requirements.txt"')
    
    # Đường dẫn đến file data.yaml
    data_yaml_path = PROJECT_ROOT / "datasets" / "mot16_yolo" / "data.yaml"
    
    # Lệnh huấn luyện YOLOv5 với cấu hình phù hợp cho toàn bộ dataset
    train_cmd = f'python "{yolov5_path}/train.py" --img 320 --batch 2 --epochs 50 --data "{data_yaml_path}" --weights yolov5n.pt --project "{PROJECT_ROOT}/runs" --name mot16_train --cache'
    
    print(f"🔄 Đang chạy lệnh: {train_cmd}")
    os.system(train_cmd)
    
    print("✅ Huấn luyện hoàn tất!")
    print(f"📁 Kết quả được lưu tại: {PROJECT_ROOT}/runs/mot16_train")

if __name__ == "__main__":
    train_yolov5()