# đọc ghi file log, nhãn
# Xử lý file, log tracking,...
import os
import csv
import json
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any

def create_output_dirs(output_dir: str):
    """
    Tạo các thư mục đầu ra cần thiết
    Args:
        output_dir: Thư mục đầu ra chính
    """
    # Tạo thư mục đầu ra chính nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Tạo các thư mục con
    video_output_dir = os.path.join(output_dir, "videos")
    log_output_dir = os.path.join(output_dir, "logs")
    
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
    
    if not os.path.exists(log_output_dir):
        os.makedirs(log_output_dir)
    
    return video_output_dir, log_output_dir

def create_tracking_log(log_dir: str, sequence_name: str):
    """
    Tạo file log CSV cho tracking
    Args:
        log_dir: Thư mục chứa file log
        sequence_name: Tên của sequence video
    Returns:
        Đường dẫn đến file log
    """
    # Tạo tên file với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{sequence_name}_{timestamp}.csv")
    
    # Tạo file log với header
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "track_id", "class_id", "x", "y", "w", "h", "timestamp"])
    
    print(f"🔍 Đã tạo file log: {log_path}")
    return log_path

def log_tracking_results(log_path: str, frame_id: int, tracks: List[Tuple]):
    """
    Ghi log kết quả tracking vào file CSV
    Args:
        log_path: Đường dẫn đến file log
        frame_id: ID của frame hiện tại
        tracks: Danh sách theo dõi [(track_id, class_id, [x,y,w,h]),...]
    """
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        for track in tracks:
            track_id, class_id, bbox = track
            x, y, w, h = bbox
            
            # Ghi thông tin vào file log
            writer.writerow([frame_id, track_id, class_id, x, y, w, h, timestamp])
