#file này File chính để phát hiện người trong dataset MOT16-02 sử dụng YOLOv5
# đầu ra hắn là folder outputs 
import os
import cv2
import configparser
from pathlib import Path
from load_model import load_yolov5_model  # Import hàm tải mô hình

# Đường dẫn đến thư mục dự án
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn đến thư mục chứa ảnh MOT16-02
IMAGES_PATH = Path("d:/Defect - Detection - Tracking/defect-detection-tracking/datasets/mot16_yolo/images/train/MOT16-02")

# Đường dẫn đến thư mục lưu kết quả
OUTPUT_PATH = Path("d:/Defect - Detection - Tracking/defect-detection-tracking/outputs/MOT16-02_detected")
os.makedirs(OUTPUT_PATH, exist_ok=True)

def detect_persons():
    """Phát hiện người trong dataset MOT16-02 sử dụng YOLOv5"""
    print("🚀 Bắt đầu phát hiện người trong MOT16-02...")
    
    # Tải mô hình YOLOv5 đã huấn luyện
    model = load_yolov5_model(conf_threshold=0.25, only_person=True)
    
    # Lấy danh sách ảnh
    image_files = sorted([f for f in os.listdir(IMAGES_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    if not image_files:
        print(f"⚠️ Không tìm thấy ảnh trong thư mục {IMAGES_PATH}")
        return
    
    print(f"📷 Tìm thấy {len(image_files)} ảnh để xử lý")
    
    # Đọc tốc độ khung hình từ seqinfo.ini nếu có
    seq_path = Path("data/MOT16/train/MOT16-02")
    seqinfo_path = seq_path / "seqinfo.ini"
    frame_rate = 30  # mặc định
    
    if seqinfo_path.exists():
        config = configparser.ConfigParser()
        config.read(seqinfo_path)
        try:
            frame_rate = int(config["Sequence"]["frameRate"])
            print(f"📄 FPS lấy từ seqinfo.ini: {frame_rate}")
        except:
            print("⚠️ Không đọc được frameRate, dùng mặc định 30")
    
    delay = int(1000 / frame_rate)
    
    # Xử lý từng ảnh
    for i, img_file in enumerate(image_files):
        img_path = IMAGES_PATH / img_file
        
        # Đọc ảnh
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Không thể đọc ảnh: {img_path}")
            continue
        
        # Phát hiện đối tượng
        results = model(img)
        
        # Lấy thông tin về các đối tượng phát hiện được
        detections = results.xyxy[0].cpu().numpy()
        num_detections = len(detections)
        
        # Vẽ bounding box
        img_with_boxes = results.render()[0].copy()
        
        # Thêm đoạn resize ảnh chỉ để hiển thị
        scale_percent = 50  # Tỷ lệ phần trăm của kích thước mới
        display_width = int(img_with_boxes.shape[1] * scale_percent / 100)
        display_height = int(img_with_boxes.shape[0] * scale_percent / 100)
        display_img = cv2.resize(img_with_boxes, (display_width, display_height))
        
        # Hiển thị số lượng đối tượng phát hiện được
        cv2.putText(display_img, f"Số người: {num_detections}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Lưu ảnh kết quả với kích thước gốc
        output_path = OUTPUT_PATH / img_file
        cv2.imwrite(str(output_path), img_with_boxes)
        
        # Hiển thị ảnh đã resize (chỉ hiển thị một cửa sổ)
        cv2.imshow("YOLOv5 Detection", display_img)
        key = cv2.waitKey(delay)
        if key == 27:  # Nhấn ESC để thoát
            break
        
        # Hiển thị tiến độ
        if (i + 1) % 10 == 0 or i == len(image_files) - 1:
            print(f"✅ Đã xử lý {i + 1}/{len(image_files)} ảnh")
    
    cv2.destroyAllWindows()
    print(f"✅ Hoàn tất! Kết quả được lưu tại: {OUTPUT_PATH}")

if __name__ == "__main__":
    detect_persons()