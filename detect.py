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
        scale_percent = 100  # Tăng từ 80% lên 100% kích thước gốc
        display_width = int(img_with_boxes.shape[1] * scale_percent / 100)
        display_height = int(img_with_boxes.shape[0] * scale_percent / 100)
        display_img = cv2.resize(img_with_boxes, (display_width, display_height))
        
        # Điều chỉnh kích thước chữ cho phù hợp với khung hình lớn hơn
        cv2.putText(display_img, f"Số người: {num_detections}", (25, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Tạo cửa sổ có thể điều chỉnh kích thước và đặt kích thước ban đầu
        cv2.namedWindow("YOLOv5 Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv5 Detection", display_width, display_height)
        
        # Hiển thị ảnh đã resize
        cv2.imshow("YOLOv5 Detection", display_img)
        
        # Xóa dòng hiển thị số người trùng lặp
        
        # Lưu ảnh kết quả với kích thước gốc và thêm prefix
        output_filename = f"detected_{img_file}"
        output_path = OUTPUT_PATH / output_filename
        cv2.imwrite(str(output_path), img_with_boxes)
        
        key = cv2.waitKey(delay)
        if key == 27:  # Nhấn ESC để thoát
            break
        elif key == ord('f'):  # Nhấn 'f' để chuyển đổi chế độ toàn màn hình
            is_fullscreen = not is_fullscreen  # Đảo trạng thái toàn màn hình
            if is_fullscreen:
                cv2.setWindowProperty("YOLOv5 Detection", cv2.WND_PROP_FULLSCREEN, 
                                    cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty("YOLOv5 Detection", cv2.WND_PROP_FULLSCREEN, 
                                    cv2.WINDOW_NORMAL)
                cv2.resizeWindow("YOLOv5 Detection", display_width, display_height)

        # Hiển thị tiến độ
        if (i + 1) % 10 == 0 or i == len(image_files) - 1:
            print(f"✅ Đã xử lý {i + 1}/{len(image_files)} ảnh")
    
    cv2.destroyAllWindows()
    print(f"✅ Hoàn tất! Kết quả được lưu tại: {OUTPUT_PATH}")

if __name__ == "__main__":
    detect_persons()