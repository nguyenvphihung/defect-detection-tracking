# Hàm vẽ bbox, ID, mũi tên,...
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

# Constant
COLOR_PALETTE = {
    0: (0, 255, 0),    # màu xanh lá cho người
    1: (255, 0, 0),    # màu đỏ cho xe
    2: (0, 0, 255),    # màu xanh dương cho túi 
}

def visualize_detections(frame: np.ndarray, detections: List[Tuple], min_conf_threshold: float = 0.5) -> np.ndarray:
    """
    Vẽ các bounding box cho phát hiện (detection)
    Args:
        frame: Frame hiện tại
        detections: Danh sách phát hiện [(class_id, confidence, [x,y,w,h]),...]
        min_conf_threshold: Ngưỡng tin cậy tối thiểu để hiển thị
    Returns:
        Frame với các bbox đã được vẽ
    """
    vis_frame = frame.copy()
    for det in detections:
        class_id, confidence, bbox = det
        
        # Bỏ qua các phát hiện có độ tin cậy thấp
        if confidence < min_conf_threshold:
            continue
            
        # Lấy thông tin bounding box
        x, y, w, h = [int(c) for c in bbox]
        
        # Xác định màu sắc
        color = COLOR_PALETTE.get(class_id, (255, 255, 255))  # mặc định trắng nếu không có class
        
        # Vẽ bounding box
        cv2.rectangle(vis_frame, (x, y), (x+w, y+h), color, 2)
        
        # Hiển thị thông tin class và confidence
        label = f"Class: {class_id}, Conf: {confidence:.2f}"
        cv2.putText(vis_frame, label, (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return vis_frame

def visualize_tracks(frame: np.ndarray, tracks: List[Tuple]) -> np.ndarray:
    """
    Vẽ các bounding box và ID cho các đối tượng được theo dõi (track)
    Args:
        frame: Frame hiện tại
        tracks: Danh sách theo dõi [(track_id, class_id, [x,y,w,h]),...]
    Returns:
        Frame với các track bbox và ID đã được vẽ
    """
    vis_frame = frame.copy()
    
    for track in tracks:
        track_id, class_id, bbox = track
        
        # Lấy thông tin bounding box
        x, y, w, h = [int(c) for c in bbox]
        
        # Xác định màu sắc dựa trên class và ID
        # Sử dụng màu cơ bản theo class_id
        base_color = COLOR_PALETTE.get(class_id, (255, 255, 255))
        
        # Vẽ bounding box
        cv2.rectangle(vis_frame, (x, y), (x+w, y+h), base_color, 2)
        
        # Vẽ ID ở góc trên bên trái
        id_label = f"ID: {track_id}"
        cv2.putText(vis_frame, id_label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, base_color, 2)
        
        # Vẽ class những phía dưới
        class_label = f"Class: {class_id}"
        cv2.putText(vis_frame, class_label, (x, y+h+15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, base_color, 1)
    
    # Hiển thị tổng số đối tượng theo dõi
    cv2.putText(vis_frame, f"Total: {len(tracks)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return vis_frame
