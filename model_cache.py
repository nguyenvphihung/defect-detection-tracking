import os
from pathlib import Path

# Biến toàn cục để lưu trữ mô hình đã tải
_cached_model = None

def get_cached_model(conf_threshold=0.25, only_person=True, force_reload=False):
    """Lấy mô hình từ cache hoặc tải mới nếu cần"""
    global _cached_model
    
    if _cached_model is None or force_reload:
        # Import ở đây để tránh import vòng tròn
        from load_model import load_yolov5_model
        _cached_model = load_yolov5_model(conf_threshold, only_person)
        print("✅ Đã tải mô hình mới vào cache")
    else:
        print("✅ Sử dụng mô hình từ cache")
        # Cập nhật cấu hình nếu cần
        _cached_model.conf = conf_threshold
        if only_person:
            _cached_model.classes = [0]
    
    return _cached_model