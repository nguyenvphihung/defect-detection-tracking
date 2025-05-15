#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import time
import sys

# Thêm thư mục gốc vào path để có thể import các module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import các module tự viết
from detectors.yolo_detector import YOLODetector  # Sử dụng YOLODetector mới thay thế cho YOLOv5Detector cũ
from trackers.deep_sort.deep_sort import DeepSORT
from core_utils.drawing import visualize_detections, visualize_tracks
from core_utils.file_io import create_output_dirs, create_tracking_log, log_tracking_results
from core_utils.counter import PersonCounter

def parse_args():
    """Xử lý các tham số dòng lệnh"""
    parser = argparse.ArgumentParser(description='YOLOv5 + DeepSORT Tracking Demo')
    
    # Tham số đầu vào
    parser.add_argument('--source', type=str, default='data/MOT16/train/MOT16-02',
                        help='Thư mục chuyển động MOT16 hoặc đường dẫn video/camera')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Thư mục đầu ra')
    
    # Tham số đếm người
    parser.add_argument('--counter-direction', type=str, default='horizontal',
                        choices=['horizontal', 'vertical'],
                        help='Hướng của đường đếm (horizontal/vertical)')
    parser.add_argument('--counter-line', type=float, default=0.5,
                        help='Vị trí của đường đếm (0.0-1.0, tỷ lệ phần trăm của chiều cao/rộng)')
    
    # Tham số YOLO
    parser.add_argument('--yolo-model', type=str, default='yolov5s',
                        help='Tên mô hình YOLOv5 (yolov5s, yolov5m, v.v.)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Ngưỡng tin cậy YOLO')
    parser.add_argument('--classes', nargs='+', type=int, default=[0],
                        help='Các class cần lọc (0: người)')
    parser.add_argument('--use-gt', action='store_true',
                        help='Sử dụng dữ liệu ground truth nếu có')
    parser.add_argument('--use-yolo', action='store_true',
                        help='Ưu tiên sử dụng YOLO ngay cả khi có ground truth')
    parser.add_argument('--detection-method', type=str, default='auto',
                        choices=['auto', 'yolo', 'ground-truth', 'both'],
                        help='Phương pháp phát hiện: auto (tự động), yolo (chỉ dùng YOLO), ground-truth (chỉ dùng GT), both (kết hợp)')
    
    # Tham số DeepSORT
    parser.add_argument('--deepsort-config', type=str, default='configs/deepsort.yaml',
                        help='File cấu hình DeepSORT')
    
    # Tham số hiển thị
    parser.add_argument('--display', action='store_true', help='Hiển thị khung hình kết quả')
    parser.add_argument('--save-vid', action='store_true', help='Lưu video kết quả')
    
    args = parser.parse_args()
    return args

def process_mot_sequence(seq_path, args):
    """Xử lý sequence MOT16"""
    # Đường dẫn ảnh và cấu hình
    img_dir = os.path.join(seq_path, "img1")
    seq_name = os.path.basename(seq_path)
    
    # Tạo thư mục đầu ra
    video_output_dir, log_output_dir = create_output_dirs(args.output_dir)
    
    # Tạo file log tracking
    log_path = create_tracking_log(log_output_dir, seq_name)
    
    # Khởi tạo bộ đếm người
    counter_position = args.counter_line
    counter_direction = args.counter_direction
    
    # Danh sách file ảnh
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    # Lấy kích thước frame
    first_img = cv2.imread(os.path.join(img_dir, img_files[0]))
    frame_height, frame_width = first_img.shape[:2]
    
    # Khởi tạo bộ đếm người với kích thước frame
    person_counter = PersonCounter(frame_height, frame_width, 
                               line_position=counter_position,
                               line_direction=counter_direction)
    
    # Khởi tạo video writer nếu cần
    video_writer = None
    if args.save_vid:
        video_path = os.path.join(video_output_dir, f"{seq_name}_tracking.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30, (frame_width, frame_height))
        print(f"🎥 Video output: {video_path}")
    
    # Bắt đầu tracking
    frame_id = 0  # Frame ID trong mã nguồn
    mot_frame_id = 1  # Frame ID dùng trong MOT16, bắt đầu từ 1
    total_frames = len(img_files)
    processing_times = []
    
    for img_file in tqdm(img_files, desc=f"Processing {seq_name}"):
        start_time = time.time()
        frame_id += 1
        mot_frame_id = frame_id  # MOT16 frame_id bắt đầu từ 1
        
        # Đọc frame
        img_path = os.path.join(img_dir, img_file)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Warning: Cannot read frame: {img_path}")
            continue
            
        # 1. Phát hiện đối tượng bằng YOLO
        # Truyền frame_id theo định dạng MOT16
        detections = detector.detect(frame, frame_id=mot_frame_id)
        
        # Debug: In thông tin về detections
        if frame_id % 20 == 0:
            print(f"DEBUG: Frame {frame_id} (MOT frame_id={mot_frame_id}) có {len(detections)} detections")
        
        # 2. Cập nhật tracker
        tracks = tracker.update(frame, detections)
        
        # Debug: In thông tin về tracks
        if frame_id % 20 == 0 and tracks:
            print(f"DEBUG: Frame {frame_id} có {len(tracks)} tracks")
        
        # 3. Cập nhật bộ đếm người
        person_counter.update(tracks)
        
        # 4. Ghi log và hiển thị
        log_tracking_results(log_path, frame_id, tracks)
        
        # 5. Vẽ kết quả
        if args.display or args.save_vid:
            # Vẽ các vị trí phát hiện (màu xanh lá nhạt)
            vis_img = visualize_detections(frame, detections, args.conf_thres)
            # Vẽ các đối tượng đang track (ID + bbox)
            vis_img = visualize_tracks(vis_img, tracks)
            # Vẽ đường đếm và số liệu thống kê
            vis_img = person_counter.draw_counter(vis_img)
            # Vẽ thông tin phụ
            cv2.putText(vis_img, f"Frame: {frame_id}/{total_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            if args.display:
                cv2.imshow(f"YOLOv5 + DeepSORT - {seq_name}", vis_img)
                if cv2.waitKey(1) == 27:  # Esc key
                    break
            
            if args.save_vid and video_writer:
                video_writer.write(vis_img)
        
        # Tính thời gian xử lý
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
    
    # Kết thúc
    if video_writer:
        video_writer.release()
    
    if args.display:
        cv2.destroyAllWindows()
    
    # Thống kê
    avg_time = sum(processing_times) / len(processing_times)
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"\n=== Thống kê ===\n")
    print(f"Tổng số frame: {total_frames}")
    print(f"Thời gian trung bình mỗi frame: {avg_time:.4f}s")
    print(f"FPS trung bình: {avg_fps:.2f}")
    print(f"Kết quả tracking đã được lưu vào: {log_path}")
    
    # Thống kê đếm người
    print(f"\n=== Thống kê đếm người ===\n")
    direction_text = "lên/xuống" if counter_direction == "horizontal" else "trái/phải"
    print(f"Tổng số người đi qua đường đếm: {person_counter.total_count}")
    print(f"Số người đi {direction_text.split('/')[0]}: {person_counter.count_up}")
    print(f"Số người đi {direction_text.split('/')[1]}: {person_counter.count_down}")

def process_video(video_path, args):
    """Xử lý video thông thường"""
    # Tạo thư mục đầu ra
    video_output_dir, log_output_dir = create_output_dirs(args.output_dir)
    
    # Lấy tên video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Tạo file log tracking
    log_path = create_tracking_log(log_output_dir, video_name)
    
    # Khởi tạo tham số đếm người
    counter_position = args.counter_line
    counter_direction = args.counter_direction
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Không thể mở video {video_path}")
        return
    
    # Lấy thông tin video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Khởi tạo bộ đếm người
    person_counter = PersonCounter(frame_height, frame_width, 
                               line_position=counter_position,
                               line_direction=counter_direction)
    
    # Khởi tạo video writer nếu cần
    video_writer = None
    if args.save_vid:
        video_path = os.path.join(video_output_dir, f"{video_name}_tracking.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
        print(f"🎥 Video output: {video_path}")
    
    # Bắt đầu tracking
    frame_id = 0
    processing_times = []
    
    with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                break
                
            frame_id += 1
            
            # 1. Phát hiện đối tượng bằng YOLO
            detections = detector.detect(frame)
            
            # 2. Cập nhật tracker
            tracks = tracker.update(frame, detections)
            
            # 3. Ghi log và hiển thị
            log_tracking_results(log_path, frame_id, tracks)
            
            # 3.1 Cập nhật bộ đếm người
            person_counter.update(tracks)
            
            # 4. Vẽ kết quả
            if args.display or args.save_vid:
                # Vẽ các vị trí phát hiện (màu xanh lá nhạt)
                vis_img = visualize_detections(frame, detections, args.conf_thres)
                # Vẽ các đối tượng đang track (ID + bbox)
                vis_img = visualize_tracks(vis_img, tracks)
                # Vẽ đường đếm và thông tin đếm người
                vis_img = person_counter.draw_counter(vis_img)
                # Vẽ thông tin phụ
                cv2.putText(vis_img, f"Frame: {frame_id}/{total_frames}", (200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                if args.display:
                    cv2.imshow(f"YOLOv5 + DeepSORT - {video_name}", vis_img)
                    if cv2.waitKey(1) == 27:  # Esc key
                        break
                
                if args.save_vid and video_writer:
                    video_writer.write(vis_img)
            
            # Tính thời gian xử lý
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Cập nhật thanh tiến trình
            pbar.update(1)
    
    # Kết thúc
    cap.release()
    if video_writer:
        video_writer.release()
    
    if args.display:
        cv2.destroyAllWindows()
    
    # Thống kê
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"\n=== Thống kê ===\n")
    print(f"Tổng số frame: {frame_id}")
    print(f"Thời gian trung bình mỗi frame: {avg_time:.4f}s")
    print(f"FPS trung bình: {avg_fps:.2f}")
    print(f"Kết quả tracking đã được lưu vào: {log_path}")
    
    # Thống kê đếm người
    print(f"\n=== Thống kê đếm người ===\n")
    direction_text = "lên/xuống" if counter_direction == "horizontal" else "trái/phải"
    print(f"Tổng số người đi qua đường đếm: {person_counter.total_count}")
    print(f"Số người đi {direction_text.split('/')[0]}: {person_counter.count_up}")
    print(f"Số người đi {direction_text.split('/')[1]}: {person_counter.count_down}")


if __name__ == "__main__":
    # Xử lý tham số dòng lệnh
    args = parse_args()
    
    # Khởi tạo detector
    # Xác định phương pháp phát hiện theo các tham số dòng lệnh
    gt_available = os.path.isdir(args.source) and os.path.exists(os.path.join(args.source, "gt"))
    
    # Xử lý tham số detection-method
    if args.detection_method == 'auto':
        # Tự động chọn phương pháp tốt nhất:
        # - Dùng ground truth nếu có và --use-yolo không được chỉ định
        # - Dùng YOLO trong các trường hợp khác
        use_gt = gt_available and not args.use_yolo
    elif args.detection_method == 'yolo':
        # Chỉ dùng YOLO
        use_gt = False
    elif args.detection_method == 'ground-truth':
        # Chỉ dùng ground truth nếu có
        use_gt = gt_available
        if not gt_available:
            print(f"\n⚠️ Cảnh báo: Không tìm thấy dữ liệu ground truth trong {args.source}/gt")
            print("Sẽ sử dụng YOLOv5 thay thế.")
    elif args.detection_method == 'both':
        # Sử dụng cả hai (hiện tại chưa hỗ trợ đầy đủ, sẽ sử dụng GT nếu có)
        use_gt = gt_available
        print("\nℹ️ Chế độ 'both' hiện tại sẽ ưu tiên dùng ground truth nếu có, ngược lại sẽ dùng YOLO.")
        
    # Trường hợp đặc biệt: Nếu chỉ định --use-gt, ưu tiên dùng GT nếu có
    if args.use_gt and gt_available:
        use_gt = True
        
    # Trường hợp đặc biệt: Nếu chỉ định --use-yolo, ưu tiên dùng YOLO
    if args.use_yolo:
        use_gt = False
        
    # Hiển thị phương pháp được chọn
    if use_gt:
        print(f"\nℹ️ Sử dụng dữ liệu GROUND TRUTH từ {args.source}/gt")
    else:
        print(f"\nℹ️ Sử dụng mô hình YOLOv5 {args.yolo_model}")

    
    print("\n=== Khởi tạo YOLO Detector ===\n")
    # Khởi tạo YOLODetector mới với đầy đủ tham số
    detector = YOLODetector(
        model_name=args.yolo_model,
        conf_thres=args.conf_thres,
        classes=args.classes,
        use_gt=use_gt
    )
    
    # Nếu nguồn dữ liệu là MOT16, thử tải ground truth
    if use_gt:
        detector.load_gt_if_available(args.source)
    
    # Khởi tạo tracker
    print("\n=== Khởi tạo DeepSORT Tracker ===\n")
    tracker = DeepSORT(config_path=args.deepsort_config)
    
    # Kiểm tra loại đầu vào
    source = args.source
    
    if os.path.isdir(source) and os.path.exists(os.path.join(source, "img1")):
        # Xử lý sequence MOT16
        print(f"\n=== Xử lý sequence MOT16: {source} ===\n")
        process_mot_sequence(source, args)
    elif os.path.isfile(source) and source.endswith((".mp4", ".avi", ".mov")):
        # Xử lý video thông thường
        print(f"\n=== Xử lý video: {source} ===\n")
        process_video(source, args)
    elif source.isdigit():
        # Xử lý webcam
        print(f"\n=== Bắt đầu webcam ID: {source} ===\n") 
        process_video(int(source), args)
    else:
        print(f"Error: Không hỗ trợ loại đầu vào: {source}")
