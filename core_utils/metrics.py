# Mô đun tính precision, recall, mô đun đánh giá MOT
import numpy as np
from typing import List, Tuple, Dict, Any

def calculate_iou(bbox1, bbox2):
    """
    Tính IoU (Intersection over Union) giữa hai bounding box
    Args:
        bbox1: Bounding box 1 [x, y, w, h]
        bbox2: Bounding box 2 [x, y, w, h]
    Returns:
        Giá trị IoU
    """
    # Chuyển đổi sang [x1, y1, x2, y2]
    x1, y1, w1, h1 = bbox1
    x1_2, y1_2 = x1 + w1, y1 + h1
    
    x2, y2, w2, h2 = bbox2
    x2_2, y2_2 = x2 + w2, y2 + h2
    
    # Tính điểm chéo phần giao nhau
    xx1 = max(x1, x2)
    yy1 = max(y1, y2)
    xx2 = min(x1_2, x2_2)
    yy2 = min(y1_2, y2_2)
    
    # Kiểm tra nếu có giao nhau không
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    intersection = w * h
    
    # Tính diện tích của hai bbox
    area1 = w1 * h1
    area2 = w2 * h2
    
    # Tính IoU
    union = area1 + area2 - intersection
    iou = intersection / max(union, 1e-6)  # Tránh chia cho 0
    
    return iou

def evaluate_tracking_frame(gt_tracks: List[Tuple], pred_tracks: List[Tuple], 
                          iou_threshold: float = 0.5, class_threshold: bool = True):
    """
    Đánh giá kết quả tracking cho một frame
    Args:
        gt_tracks: Danh sách ground truth [(gt_id, class_id, [x,y,w,h])]
        pred_tracks: Danh sách dự đoán [(track_id, class_id, [x,y,w,h])]
        iou_threshold: Ngưỡng IoU để xác định trùng lập
        class_threshold: Nếu True, class_id phải giống nhau
    Returns:
        Dict kết quả đánh giá
    """
    # Kết quả đánh giá
    results = {
        "TP": 0,  # True positive
        "FP": 0,  # False positive
        "FN": 0,  # False negative
        "IDSW": 0,  # ID switches
        "gt_tracks": len(gt_tracks),
        "pred_tracks": len(pred_tracks),
        "matched_tracks": []  # (gt_id, track_id, iou)
    }
    
    # Danh sách các ground truth đã được ghép cặp
    matched_gt_ids = []
    
    # Kiểm tra từng đối tượng dự đoán
    for pred in pred_tracks:
        track_id, pred_class, pred_bbox = pred
        best_iou = iou_threshold
        best_gt_id = -1
        best_gt_idx = -1
        
        # So sánh với từng ground truth
        for i, gt in enumerate(gt_tracks):
            gt_id, gt_class, gt_bbox = gt
            
            # Nếu đã ghép cặp rồi thì bỏ qua
            if gt_id in matched_gt_ids:
                continue
            
            # Nếu cần so khớp class và không giống nhau, bỏ qua
            if class_threshold and pred_class != gt_class:
                continue
            
            # Tính IoU
            iou = calculate_iou(pred_bbox, gt_bbox)
            
            # Tìm ground truth có IoU cao nhất
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
                best_gt_idx = i
        
        # Xử lý kết quả ghép cặp
        if best_gt_id != -1:
            # Đã tìm được ground truth khớp với dự đoán
            results["TP"] += 1
            matched_gt_ids.append(best_gt_id)
            results["matched_tracks"].append((best_gt_id, track_id, best_iou))
        else:
            # Không tìm được ground truth khớp => false positive
            results["FP"] += 1
    
    # Số false negative là số ground truth chưa được ghép cặp
    results["FN"] = len(gt_tracks) - len(matched_gt_ids)
    
    # Tính precision và recall
    if results["TP"] + results["FP"] > 0:
        results["precision"] = results["TP"] / (results["TP"] + results["FP"])
    else:
        results["precision"] = 0
        
    if results["TP"] + results["FN"] > 0:
        results["recall"] = results["TP"] / (results["TP"] + results["FN"])
    else:
        results["recall"] = 0
    
    # Tính F1-score
    if results["precision"] + results["recall"] > 0:
        results["f1_score"] = 2 * (results["precision"] * results["recall"]) / (results["precision"] + results["recall"])
    else:
        results["f1_score"] = 0
    
    return results
