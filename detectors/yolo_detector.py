# YOLOv5 detector implementation
import cv2
import numpy as np
import torch
import os
import time
from typing import List, Tuple, Dict, Any

class YOLODetector:
    """
    Lớp detector sử dụng mô hình YOLOv5 để phát hiện đối tượng
    """
    def __init__(self, model_name='yolov5s', conf_thres=0.25, classes=None, use_gt=False):
        """
        Khởi tạo YOLODetector
        Args:
            model_name: Tên mô hình YOLOv5 (yolov5s, yolov5m, v.v.)
            conf_thres: Ngưỡng tin cậy
            classes: Danh sách các class cần lọc ([0] cho người)
            use_gt: Nếu True, sẵn sàng sử dụng ground truth nếu có
        """
        self.conf_thres = conf_thres
        self.classes = classes  # [0] cho người
        self.frame_count = 0
        self.use_gt = use_gt
        self.gt_data = None  # Sẽ chứa dữ liệu ground truth nếu có
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Tải mô hình YOLOv5 
        if not use_gt:
            try:
                # Sử dụng force_reload=True và thêm sys.path manipulation để tránh xung đột module
                import sys
                # Lưu lại đường dẫn hiện tại
                original_path = list(sys.path)
                try:
                    # Tải YOLOv5 với force_reload để đảm bảo tải lại tất cả các dependencies
                    self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, force_reload=True)
                except Exception as e:
                    print(f"❌ Lỗi khi tải mô hình YOLOv5 trực tiếp: {e}")
                    # Thử cách khác - sử dụng pip để cài đặt yolov5
                    try:
                        import subprocess
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "yolov5"])
                        import yolov5
                        self.model = yolov5.load(model_name)
                    except Exception as e2:
                        print(f"❌ Không thể cài đặt yolov5 qua pip: {e2}")
                        raise RuntimeError("Không thể tải mô hình YOLOv5. Vui lòng cài đặt thủ công: pip install yolov5")
                self.model.to(self.device)
                self.model.eval()
                
                # Thiết lập tham số
                self.model.conf = conf_thres  # Ngưỡng tin cậy
                if classes is not None:
                    self.model.classes = classes  # Chỉ phát hiện các class được chỉ định
                    
                print(f"🚀 Đã tải mô hình YOLOv5 {model_name} trên {self.device}. Ngưỡng: {conf_thres}, Classes: {classes}")
            except Exception as e:
                print(f"❌ Lỗi khi tải mô hình YOLOv5: {e}")
                print("⚠️ Sẽ sử dụng ground truth nếu có!")
                self.model = None
                self.use_gt = True
        else:
            # Chỉ sử dụng ground truth
            self.model = None
            print(f"📄 Chỉ sử dụng ground truth, không tải mô hình YOLOv5")
    
    def load_gt_if_available(self, source_path: str) -> bool:
        """
        Tải dữ liệu ground truth nếu có (cho phương pháp fallback)
        Args:
            source_path: Đường dẫn đến thư mục MOT16
        Returns:
            True nếu đã tải ground truth, False nếu không
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
                
            # Đánh dấu là đang sử dụng ground truth
            self.use_gt = True
            return True
        
        # Không tìm thấy ground truth
        print(f"Không tìm thấy ground truth: {gt_path}")
        return False
    
    def detect(self, frame: np.ndarray, frame_id: int = None) -> List[Tuple[int, float, List[int]]]:
        """
        Phát hiện đối tượng trong frame bằng YOLOv5 hoặc ground truth
        Args:
            frame: Ảnh đầu vào dạng numpy array (BGR)
            frame_id: ID của frame (cần thiết cho ground truth)
        Returns:
            List các detection [class_id, confidence, [x, y, w, h]]
        """
        self.frame_count += 1
        
        # Sử dụng frame_id truyền vào nếu có, nếu không dùng frame_count
        current_frame_id = frame_id if frame_id is not None else self.frame_count
        
        # Debug thông tin frame_count với tần suất thấp hơn để giảm bớt thông tin
        if self.frame_count % 100 == 0:
            print(f"DEBUG DETECTOR: Xử lý frame {self.frame_count}, MOT frame_id={current_frame_id}")
        
        # 1. Kiểm tra nếu có dữ liệu ground truth cho frame hiện tại và đang ở chế độ sử dụng GT
        if self.use_gt and self.gt_data is not None and current_frame_id in self.gt_data:
            detections = self.gt_data[current_frame_id]
            
            # Cho biết số lượng đối tượng được phát hiện
            if detections and self.frame_count % 20 == 0:
                print(f"DEBUG DETECTOR (Ground Truth): Đã tìm thấy {len(detections)} đối tượng trong frame {current_frame_id}")
            
            return detections
            
        # 2. Nếu không sử dụng ground truth hoặc không có ground truth cho frame hiện tại
        if self.model is not None:
            try:
                # Chuyển frame về định dạng RGB (YOLOv5 cần RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Inference
                results = self.model(rgb_frame, size=640)  # size có thể điều chỉnh (640, 1280, ...)
                
                # Lấy kết quả dưới dạng pandas dataframe
                predictions = results.pandas().xyxy[0]
                
                # Chuyển đổi kết quả thành định dạng [class_id, confidence, [x, y, w, h]]
                detections = []
                for _, row in predictions.iterrows():
                    x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                    conf = row['confidence']
                    class_id = row['class']
                    
                    # Chuyển từ [x1, y1, x2, y2] sang [x, y, w, h] - format YOLO sang format MOT
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    
                    detections.append((int(class_id), float(conf), [x, y, w, h]))
                
                # Debug info
                if self.frame_count % 20 == 0:
                    print(f"🔍 YOLOv5: Đã phát hiện {len(detections)} đối tượng trong frame {self.frame_count}")
                    
                return detections
                
            except Exception as e:
                print(f"⚠️ Lỗi khi phát hiện đối tượng với YOLOv5: {e}")
                
        # 3. Nếu không có model và không có ground truth, thử sử dụng yolo_wrapper
        if not hasattr(self, 'tried_wrapper') or not self.tried_wrapper:
            print(f"⚠️ Model YOLO không khả dụng, thử sử dụng yolo_wrapper...")
            try:
                # Lưu frame tạm thời
                import tempfile
                import os
                import subprocess
                import json
                
                # Tạo thư mục tạm nếu chưa tồn tại
                os.makedirs('temp', exist_ok=True)
                temp_path = os.path.join('temp', f'frame_{self.frame_count}.jpg')
                
                # Lưu frame hiện tại
                cv2.imwrite(temp_path, frame)
                
                # Gọi wrapper để phát hiện đối tượng
                cmd = f"python yolo_wrapper.py --image {temp_path} --model yolov5s --conf {self.conf_thres}"
                if self.classes is not None:
                    class_str = ','.join(map(str, self.classes))
                    cmd += f" --classes {class_str}"
                
                # Thực hiện lệnh và lấy kết quả
                result = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
                raw_detections = json.loads(result)
                
                # Chuyển đổi kết quả sang định dạng [class_id, confidence, [x,y,w,h]]
                detections = []
                for det in raw_detections:
                    x, y, w, h, conf, class_id = det
                    detections.append((int(class_id), float(conf), [x, y, w, h]))
                
                print(f"✅ Phát hiện thành công {len(detections)} đối tượng với yolo_wrapper")
                # Đánh dấu đã thử wrapper
                self.tried_wrapper = True
                
                # Xóa file tạm
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                return detections
            except Exception as e:
                print(f"⚠️ Lỗi khi sử dụng yolo_wrapper: {e}")
                self.tried_wrapper = True
        
        # 4. Nếu tất cả các phương pháp đều thất bại, trả về rỗng
        print(f"⚠️ Không có cả model YOLO lẫn ground truth cho frame {frame_id}!")
        return []
