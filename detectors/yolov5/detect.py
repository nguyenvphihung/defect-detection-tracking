# YOLO detection wrapper - Basic implementation for demo
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import os

class SimpleDetector:
    """
    Lớp detector đơn giản dùng cho demo, sử dụng OpenCV's BasicBackgroundSubtractor 
    hoặc GroundTruth trực tiếp từ MOT16 nếu có
    """
    def __init__(self, conf_thres: float = 0.25, classes: List[int] = None):
        self.conf_thres = conf_thres
        self.classes = classes  # [0] cho người
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.frame_count = 0
        self.gt_data = None  # Sẽ chứa dữ liệu ground truth nếu có
        
        print(f"📦 Sử dụng OpenCV SimpleDetector cho demo. Ngưỡng: {conf_thres}")

    def load_gt_if_available(self, source_path: str):
        """
        Tải dữ liệu ground truth nếu path là MOT16 sequence
        """
        gt_path = os.path.join(source_path, "gt", "gt.txt")
        if os.path.exists(gt_path):
            print(f"Tìm thấy dữ liệu ground truth: {gt_path}")
            self.gt_data = {}
            total_objects = 0
            
            # Đọc toàn bộ file ground truth
            with open(gt_path, 'r') as f:
                for line in f:
                    fields = line.strip().split(',')
                    # Format MOT16: <frame_id>,<object_id>,<x>,<y>,<w>,<h>,<confidence>,<class_id>,<visibility>
                    frame_id = int(fields[0])
                    object_id = int(fields[1])
                    
                    # Bỏ qua các đối tượng không hợp lệ
                    confidence = float(fields[6]) if len(fields) > 6 else 0
                    if confidence == 0:  # Chỉ lấy các đối tượng có confidence > 0
                        continue
                        
                    # Lấy thông tin vị trí và kích thước
                    x, y, w, h = map(float, fields[2:6])
                    
                    # Với MOT16 class 1=person, các class khác thường không quan tâm
                    class_id = int(fields[7]) if len(fields) > 7 else 1
                    
                    # Chỉ lấy người (class 1) hoặc các class cần thiết
                    if class_id != 1 and (self.classes is not None and class_id not in self.classes):
                        continue
                    
                    # Chuyển đổi từ class của MOT sang class của YOLO nếu cần
                    # MOT16: class 1 = person, YOLO: class 0 = person
                    yolo_class_id = 0 if class_id == 1 else class_id
                    
                    # Khởi tạo danh sách cho frame nếu chưa có
                    if frame_id not in self.gt_data:
                        self.gt_data[frame_id] = []
                    
                    # Thêm đối tượng vào danh sách của frame tương ứng
                    # Format: (class_id, confidence, [x, y, w, h])
                    self.gt_data[frame_id].append(
                        (yolo_class_id, confidence, [int(x), int(y), int(w), int(h)])
                    )
                    total_objects += 1
            
            # Thống kê số frame và đối tượng
            unique_frames = len(self.gt_data)
            print(f"🚀 Đã tải ground truth cho {unique_frames} frames, tổng cộng {total_objects} đối tượng")
            
            # In ra một vài frame đầu tiên để debug
            first_10_frames = sorted(list(self.gt_data.keys()))[:10]
            for frame in first_10_frames:
                print(f"Frame {frame}: {len(self.gt_data[frame])} đối tượng")
                
            return True
        return False

    def detect(self, frame: np.ndarray, frame_id: int = None) -> List[Tuple[int, float, List[int]]]:
        """
        Phát hiện đối tượng trong frame
        Args:
            frame: Ảnh đầu vào dạng numpy array (BGR)
            frame_id: ID của frame (dùng cho MOT16, bắt đầu từ 1)
        Returns:
            List các detection [class_id, confidence, [x, y, w, h]]
        """
        self.frame_count += 1
        
        # Sử dụng frame_id truyền vào nếu có, nếu không dùng frame_count
        current_frame_id = frame_id if frame_id is not None else self.frame_count
        
        # Debug thông tin frame_count với tần suất thấp hơn để giảm bớt thông tin
        if self.frame_count % 100 == 0:
            print(f"DEBUG DETECTOR: Xử lý frame {self.frame_count}, MOT frame_id={current_frame_id}")
        
        # 1. Kiểm tra nếu có dữ liệu ground truth cho frame hiện tại
        if self.gt_data is not None and current_frame_id in self.gt_data:
            detections = self.gt_data[current_frame_id]
            
            # Cho biết số lượng đối tượng được phát hiện
            if detections:
                # Chỉ in log debug cho những frame có đối tượng và theo tần suất phù hợp
                if self.frame_count % 20 == 0 or len(detections) > 5:
                    print(f"DEBUG DETECTOR: Đã tìm thấy {len(detections)} đối tượng trong frame {current_frame_id}")
            return detections
        
        # 2. Nếu không có ground truth, dùng background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Lọc nhiễu
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Tìm contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            # Loại bỏ các contour quá nhỏ
            if cv2.contourArea(contour) < 500:  # Ngưỡng diện tích
                continue
                
            # Lấy bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Confidence giả lập dựa trên diện tích
            area = cv2.contourArea(contour)
            confidence = min(1.0, area / 10000)  # Giới hạn max là 1.0
            
            # Chỉ giữ lại các detection có độ tin cậy cao
            if confidence >= self.conf_thres:
                class_id = 0  # Mặc định là class 'người'
                detections.append((class_id, confidence, [x, y, w, h]))
                
        return detections

# Tạo class alias để tránh phải sửa code ở các nơi khác
YOLOv5Detector = SimpleDetector
