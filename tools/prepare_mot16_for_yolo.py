import os
import shutil
import cv2
import yaml

# === Cấu hình ===
mot16_train_path = "data/MOT16/train"
target_base = "datasets/mot16_yolo"
images_target_base = os.path.join(target_base, "images/train")
labels_target_base = os.path.join(target_base, "labels/train")

# === Tạo thư mục gốc nếu chưa có ===
os.makedirs(images_target_base, exist_ok=True)
os.makedirs(labels_target_base, exist_ok=True)

# === Duyệt từng sequence trong data/MOT16/train/ ===
sequences = sorted(os.listdir(mot16_train_path))

for seq_name in sequences:
    seq_path = os.path.join(mot16_train_path, seq_name)
    img_dir = os.path.join(seq_path, "img1")
    gt_path = os.path.join(seq_path, "gt", "gt.txt")

    if not os.path.exists(gt_path) or not os.path.exists(img_dir):
        print(f"⚠️ Bỏ qua {seq_name} vì thiếu gt.txt hoặc img1/")
        continue

    # Tạo thư mục images và labels cho sequence này
    images_target = os.path.join(images_target_base, seq_name)
    labels_target = os.path.join(labels_target_base, seq_name)
    os.makedirs(images_target, exist_ok=True)
    os.makedirs(labels_target, exist_ok=True)

    # Copy ảnh
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    for img_file in img_files:
        src = os.path.join(img_dir, img_file)
        dst = os.path.join(images_target, img_file)
        shutil.copyfile(src, dst)

    print(f"✅ Copy {len(img_files)} ảnh từ {seq_name}")

    # Lấy kích thước ảnh
    sample_img = cv2.imread(os.path.join(img_dir, img_files[0]))
    img_h, img_w = sample_img.shape[:2]

    # Chuyển ground truth
    with open(gt_path, 'r') as f:
        for line in f:
            fields = line.strip().split(',')
            frame_id = int(fields[0])
            obj_id = int(fields[1])
            x, y, w, h = map(float, fields[2:6])

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            class_id = 0  # tất cả là người

            label_filename = os.path.join(labels_target, f"{frame_id:06d}.txt")
            with open(label_filename, "a") as out_f:
                out_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    print(f"✅ Convert ground truth cho {seq_name}")

# === Sinh file data.yaml ===
yaml_content = {
    'train': os.path.join(target_base, 'images/train'),
    'val': os.path.join(target_base, 'images/train'),  # tạm thời dùng tập train làm validation
    'nc': 1,
    'names': ['person']
}

yaml_path = os.path.join(target_base, "data.yaml")
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_content, f)

print(f"📄 Đã tạo file data.yaml tại {yaml_path}")
print("🎯 Hoàn tất chuyển toàn bộ MOT16/train sang format YOLO!")
