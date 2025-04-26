# DeepSORT tracking logic
import cv2
import numpy as np
import torch
import yaml
import os
from typing import List, Tuple, Dict, Optional

# Tải thư viện DeepSORT - giả định đã cài đặt
# (Tạo thư viện giả lập vì chưa có thư viện thật trong dự án, đây chỉ là mẫu)
class DeepFeatureExtractor:
    """Class giả lập cho encoder DeepSORT"""
    def __init__(self, model_type='osnet_x0_25'):
        self.model_type = model_type
        print(f"📦 Sử dụng mô hình embedding: {model_type}")
        
    def __call__(self, crops: List[np.ndarray]):
        """Trích xuất features từ các crop hình ảnh"""
        if not crops:
            return np.array([])
        # Giả lập trích xuất feature
        # Cài đặt thực tế sẽ sử dụng mạng CNN/ResNet/...
        return np.random.rand(len(crops), 512) # 512 dimensions

class KalmanTracker:
    """Class giả lập cho Kalman filter tracker của DeepSORT"""
    def __init__(self, bbox, track_id):
        self.track_id = track_id
        self.bbox = bbox
        self.time_since_update = 0
        self.hits = 1
        self.age = 1
    
    def predict(self):
        # Giả lập dự đoán Kalman
        pass
    
    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
    
    def get_state(self):
        return self.bbox

class DeepSORT:
    def __init__(self, config_path: str):
        """
        Khởi tạo DeepSORT tracker
        Args:
            config_path: Đường dẫn đến file cấu hình DeepSORT YAML
        """
        self.trackers = []
        self.next_id = 1
        
        # Đọc cấu hình
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            self.config = cfg['DEEPSORTC']
        
        # Tải các tham số cấu hình
        self.max_dist = self.config['MAX_DIST']
        self.min_confidence = self.config['MIN_CONFIDENCE']
        self.nms_max_overlap = self.config['NMS_MAX_OVERLAP']
        self.max_iou_distance = self.config['MAX_IOU_DISTANCE']
        self.max_age = self.config['MAX_AGE']
        self.n_init = self.config['N_INIT']
        self.nn_budget = self.config['NN_BUDGET']
        
        # Khởi tạo feature extractor
        model_type = self.config['MODEL_TYPE']
        self.extractor = DeepFeatureExtractor(model_type)
        
        print(f"📌 Đã khởi tạo DeepSORT: max_dist={self.max_dist}, max_age={self.max_age}")
    
    def _get_features(self, frame: np.ndarray, detections: List[Tuple]):
        """Lấy feature từ các cạnh cắt (crops) của detections"""
        crops = []
        for _, conf, bbox in detections:
            if conf < self.min_confidence:
                continue
            x, y, w, h = bbox
            crop = frame[y:y+h, x:x+w]
            crops.append(crop)
        
        # Trích xuất features
        features = self.extractor(crops)
        return features
    
    def _match_detections_to_tracks(self, detections, features):
        """Ghép cặp detections với các tracks hiện tại"""
        # Đây là phần giả lập đơn giản, DeepSORT thực tế phức tạp hơn nhiều
        assignments = []
        unmatched_tracks = list(range(len(self.trackers)))
        unmatched_detects = list(range(len(detections)))
        
        # Xử lý giả lập
        for t in unmatched_tracks[:]: 
            for d in unmatched_detects[:]:
                # Giả lập ghép cặp dựa trên IoU và cosine distance
                # Thực tế DeepSORT sử dụng Hungarian algorithm + Mahalanobis + cosine distance
                tracker_bbox = self.trackers[t].get_state()
                detect_bbox = detections[d][2]
                
                # Mô phỏng tính IoU và kiểm tra ghép cặp
                iou = self._calc_iou(tracker_bbox, detect_bbox)
                if iou > 0.5: # ngưỡng IoU giả lập
                    assignments.append((t, d))
                    unmatched_tracks.remove(t)
                    unmatched_detects.remove(d)
                    break
        
        return assignments, unmatched_tracks, unmatched_detects
    
    def _calc_iou(self, bbox1, bbox2):
        """Tính IoU giữa hai bounding box"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Tính tọa độ phần chồng nhau
        xx1 = max(x1, x2)
        yy1 = max(y1, y2)
        xx2 = min(x1 + w1, x2 + w2)
        yy2 = min(y1 + h1, y2 + h2)
        
        # Tính diện tích
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        overlap = w * h
        
        # Tính IoU
        area1 = w1 * h1
        area2 = w2 * h2
        iou = overlap / (area1 + area2 - overlap + 1e-6)
        
        return iou
    
    def update(self, frame: np.ndarray, detections: List[Tuple]) -> List[Tuple]:
        """
        Cập nhật tracker với danh sách phát hiện mới
        Args:
            frame: Frame hiện tại
            detections: Danh sách phát hiện [(class_id, confidence, [x,y,w,h]),...]
        Returns:
            Danh sách theo dõi [(track_id, class_id, [x,y,w,h]),...]  
        """
        # 1. Lọc các detection có độ tin cậy cao
        dets = [det for det in detections if det[1] >= self.min_confidence]
        
        # 2. Trích xuất features
        features = self._get_features(frame, dets)
        
        # 3. Dự đoán vị trí của các track hiện tại
        for tracker in self.trackers:
            tracker.predict()
        
        # 4. Ghép cặp detections với các tracks hiện tại
        assignments, unmatched_tracks, unmatched_detects = \
            self._match_detections_to_tracks(dets, features)
        
        # 5. Cập nhật các tracks đã ghép cặp
        for track_idx, det_idx in assignments:
            _, _, bbox = dets[det_idx]
            self.trackers[track_idx].update(bbox)
        
        # 6. Xử lý các track không ghép cặp được
        for track_idx in unmatched_tracks:
            tracker = self.trackers[track_idx]
            tracker.time_since_update += 1
        
        # 7. Xóa các track quá cũ
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        # 8. Tạo track mới cho các detection chưa ghép cặp
        for det_idx in unmatched_detects:
            class_id, _, bbox = dets[det_idx]
            # Tạo tracker mới
            new_track = KalmanTracker(bbox, self.next_id)
            self.trackers.append(new_track)
            self.next_id += 1
        
        # 9. Trả về kết quả tracking
        results = []
        for tracker in self.trackers:
            # Chỉ trả về track đã được xác nhận (confirmed)
            if tracker.time_since_update == 0 and tracker.hits >= self.n_init:
                # Giả định class_id là 0 (person)
                class_id = 0
                track_id = tracker.track_id
                bbox = tracker.get_state()
                results.append((track_id, class_id, bbox))
        
        return results
