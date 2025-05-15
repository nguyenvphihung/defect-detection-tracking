# 🎯 Dự án Theo Dõi Đối Tượng Trên Video - MOTChallenge16

Đây là dự án sử dụng mô hình YOLO và thuật toán DeepSORT để thực hiện bài toán **phát hiện và theo dõi nhiều đối tượng (Multi-Object Tracking)** trên video, sử dụng tập dữ liệu chuẩn **MOTChallenge 2016**. Hệ thống cũng tích hợp tính năng đếm người đi qua đường ảo với khả năng theo dõi hướng di chuyển.

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
# Hiển thị ground truth từ MOT16
python visualize_gt.py

# Chạy với detector mặc định (ưu tiên ground truth nếu có)
python main.py --source data/MOT16/train/MOT16-02 --display

# Chạy với đường đếm ngang (mặc định)
python main.py --source data/MOT16/train/MOT16-02 --display --counter-direction horizontal --counter-line 0.5

# Chạy với đường đếm dọc
python main.py --source data/MOT16/train/MOT16-02 --display --counter-direction vertical --counter-line 0.1
```

---

## 🔍 Quan trọng: Về việc sử dụng YOLOv5 vs Ground Truth

Hệ thống được thiết kế để **ưu tiên sử dụng dữ liệu ground truth** khi có sẵn, thay vì chạy YOLOv5 detector. Điều này giúp:

1. **Tối ưu hóa quá trình debug**: Đảm bảo kết quả nhất quán và chính xác
2. **Tăng tốc độ xử lý**: Ground truth không cần tính toán so với inference của YOLO

### 🔄 Cách dữ liệu ground truth được ưu tiên:

- Khi chạy với dữ liệu MOT16, hệ thống tự động kiểm tra thư mục `gt` và file `gt.txt`
- Nếu tìm thấy, `use_gt` được set thành `True` và YOLODetector sẽ tải ground truth thay vì mô hình YOLOv5
- Dòng log `"📄 Chỉ sử dụng ground truth, không tải mô hình YOLOv5"` sẽ xuất hiện

### 🔧 Để buộc hệ thống sử dụng YOLOv5 thay vì ground truth:

- Sử dụng nguồn dữ liệu không có ground truth (như video thông thường)
- Hoặc sửa code trong `main.py` để vô hiệu hóa việc tự động đặt `use_gt=True`

```python
# Sửa đoạn code này trong main.py
use_gt = False
# Thêm điều kiện hoặc tham số để kiểm soát việc sử dụng ground truth
```

---

Lệnh chạy với data gt.txt : python3 main.py --source data/MOT16/train/MOT16-02 --display --detection-method ground-truth
Lệnh chạy với yolo : python3 main.py --source data/MOT16/train/MOT16-02 --display --counter-direction vertical --counter-line 0.1 --detection-method yolo