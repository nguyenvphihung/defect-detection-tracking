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

---