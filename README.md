
# 🛠️ Hệ thống Phát hiện và Theo dõi Sản phẩm Lỗi Thời gian Thực

Dự án nhằm xây dựng một hệ thống **phát hiện và theo dõi sản phẩm lỗi** trong dây chuyền sản xuất công nghiệp bằng cách sử dụng các kỹ thuật học sâu (deep learning), đảm bảo hoạt động **theo thời gian thực**.

## 🎯 Mục tiêu dự án

- Phát hiện sản phẩm lỗi bằng mô hình **YOLO**.
- Theo dõi sản phẩm lỗi qua dây chuyền bằng **DeepSORT**.
- Phân tích xu hướng lỗi và hiển thị kết quả trực quan.
- Hệ thống đủ nhẹ để triển khai thực tế trong môi trường công nghiệp.

## 🧱 Cấu trúc thư mục dự án (gợi ý)

```
defect-detection-tracking/
│
├── data/                  # Dữ liệu (ảnh, nhãn, video)
├── yolov5/                # Mã nguồn YOLO (detection)
├── tracking/              # Mã nguồn DeepSORT
├── utils/                 # Các hàm phụ trợ
├── main.py                # File chạy chính
├── requirements.txt       # Danh sách thư viện cần cài
└── README.md              # Tài liệu này
```

## 🚀 Cách bắt đầu

1. **Clone repo về máy:**
   ```bash
   git clone https://github.com/tai-khoan/ten-repo.git
   cd ten-repo
   ```

2. **Cài đặt thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Chạy thử chương trình đọc video:**
   ```bash
   python main.py
   ```

## 🧠 Công nghệ sử dụng

- YOLOv5 (Phát hiện vật thể)
- DeepSORT (Theo dõi đối tượng)
- OpenCV (Xử lý ảnh và video)
- Matplotlib, Pandas (Phân tích & trực quan hóa)
- Python 3.8+

## 📌 Trạng thái

🚧 **Đang triển khai** — Tuần 1: Tìm hiểu, đọc dữ liệu và hiển thị ảnh thử nghiệm.