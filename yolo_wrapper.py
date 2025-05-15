#!/usr/bin/env python
# Tệp này hoạt động như một wrapper độc lập để chạy YOLOv5
import sys
import os
import argparse
import torch
import json
import numpy as np

def load_yolo_model(model_name='yolov5s', conf_thres=0.25, classes=None):
    """
    Tải mô hình YOLOv5 trong một môi trường độc lập
    """
    try:
        # Tải mô hình YOLOv5 từ gói cài đặt (pip install yolov5)
        import yolov5
        model = yolov5.load(model_name)
        
        # Thiết lập tham số
        model.conf = conf_thres
        if classes is not None:
            model.classes = classes
            
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình YOLOv5: {e}")
        return None

def detect_objects(model, img_path):
    """
    Phát hiện đối tượng trong ảnh và trả về kết quả dưới dạng JSON
    """
    if model is None:
        return []
    
    try:
        # Thực hiện phát hiện
        results = model(img_path)
        
        # Lấy kết quả dưới dạng pandas dataframe
        predictions = results.pandas().xyxy[0]
        
        # Chuyển đổi kết quả thành định dạng [x, y, w, h, confidence, class_id]
        detections = []
        for _, row in predictions.iterrows():
            x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            w = x2 - x1
            h = y2 - y1
            confidence = row['confidence']
            class_id = row['class']
            
            # Thêm vào danh sách kết quả
            detections.append([int(x1), int(y1), int(w), int(h), float(confidence), int(class_id)])
            
        return detections
    except Exception as e:
        print(f"Lỗi khi phát hiện đối tượng: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='YOLOv5 detection wrapper')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, default='yolov5s', help='YOLOv5 model name')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--classes', type=str, default=None, help='Classes to detect, comma separated')
    
    args = parser.parse_args()
    
    # Convert classes string to list if provided
    class_list = None
    if args.classes:
        class_list = [int(c.strip()) for c in args.classes.split(',')]
    
    # Load model
    model = load_yolo_model(args.model, args.conf, class_list)
    
    # Detect objects
    detections = detect_objects(model, args.image)
    
    # Print result as JSON
    print(json.dumps(detections))

if __name__ == "__main__":
    main()
