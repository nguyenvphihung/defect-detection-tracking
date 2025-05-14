# 🎯 Dự án Theo Dõi Đối Tượng Trên Video - MOTChallenge16

Đây là dự án sử dụng mô hình YOLO và thuật toán DeepSORT để thực hiện bài toán **phát hiện và theo dõi nhiều đối tượng (Multi-Object Tracking)** trên video, sử dụng tập dữ liệu chuẩn **MOTChallenge 2016**.

---

## 📌 Mục tiêu dự án

- Phát hiện người hoặc đối tượng trong video bằng YOLOv5/YOLOv8
- Gán ID và theo dõi chuyển động của từng đối tượng bằng DeepSORT
- Ghi log quá trình tracking (ID, tọa độ, thời gian)
- Hiển thị kết quả trực quan (bounding box + ID)
- Thực hiện theo plan 5 tuần, bám sát hướng dẫn của repo tham khảo `boxmot`

---

## 📁 Cấu trúc thư mục

```
defect-detection-tracking/
├── configs/                  # Cấu hình DeepSORT
├── data/                     # Nơi chứa dataset 
├── detectors/yolov5/         # Mã nguồn YOLOv5 (wrapper)
├── trackers/deep_sort/       # Thuật toán DeepSORT
├── utils/                    # Hàm hỗ trợ (vẽ box, log,...)
├── outputs/                  # Kết quả (video, CSV)
├── main.py                   # Pipeline chính
├── visualize_gt.py           # Kiểm tra dữ liệu ground truth
├── requirements.txt          # Thư viện cần cài
└── README.md                 # File này
```

> 📦 Dữ liệu `MOT16` KHÔNG đưa vào GitHub → Tự tải về từ (https://motchallenge.net/data/MOT16/)

---

## 🚀 Cách chạy dự án

1. **Tải dữ liệu MOT16** và đặt vào thư mục:
```
data/MOT16/test/MOT16-01...
data/MOT16/train/MOT16-02...
```

2. **Cài thư viện:**
```bash
pip install -r requirements.txt
```

3. **Chạy thử YOLO detect hoặc visualization:**
```bash
python visualize_gt.py
python main.py
```
4. **Chạy phát hiện đối tượng với Yolov5:**
```bash
python detect.py
```
- Mô hình Yolov5 sẽ được tải tự động và lưu vào thư mục `detectors/yolov5/runs/detect/`
- Kết quả phát sẽ hiển thị trực tiếp trên màn hình với tỷ lệ 50% kích thước gốc
- Kết quả sẽ được lưu vào thư mục `outputs/`
- Nhấn phim ESC để dừng quá trình xử lý.

---

## Giải thích về các file trong dự án

- detect.py: File này chứa mã nguồn để phát hiện đối tượng trên video sử dụng mô hình YOLOv5.
- visualize_gt.py: File này chứa mã nguồn để hiển thị các bounding box và ID của các đối tượng trong video.
- load_model.py: File này chứa mã nguồn để tải mô hình YOLOv5 đã được huấn luyện.
- train_all.py: File này chứa mã nguồn để huấn luyện mô hình YOLOv5.
