import os
import cv2
import configparser

# === Đường dẫn đến sequence ===
seq_path = "data/MOT16/train/MOT16-02"
img_dir = os.path.join(seq_path, "img1")
gt_path = os.path.join(seq_path, "gt", "gt.txt")
seqinfo_path = os.path.join(seq_path, "seqinfo.ini")

# === Đọc frameRate từ seqinfo.ini ===
frame_rate = 30  # mặc định
if os.path.exists(seqinfo_path):
    config = configparser.ConfigParser()
    config.read(seqinfo_path)
    try:
        frame_rate = int(config["Sequence"]["frameRate"])
        print(f"📄 FPS lấy từ seqinfo.ini: {frame_rate}")
    except:
        print("⚠️ Không đọc được frameRate, dùng mặc định 30")

delay = int(1000 / frame_rate)

# === Đọc ground truth từ gt.txt ===
# Format: frame, id, x, y, w, h, conf, class, visibility
gt_dict = {}

with open(gt_path, 'r') as f:
    for line in f:
        fields = line.strip().split(',')
        frame_id = int(fields[0])
        obj_id = int(fields[1])
        x, y, w, h = map(float, fields[2:6])
        bbox = (int(x), int(y), int(w), int(h))

        if frame_id not in gt_dict:
            gt_dict[frame_id] = []
        gt_dict[frame_id].append((obj_id, bbox))

# === Duyệt từng ảnh và vẽ bounding box ===
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

for i, img_file in enumerate(img_files):
    frame_id = i + 1
    img_path = os.path.join(img_dir, img_file)
    frame = cv2.imread(img_path)
    if frame is None:
        continue

    if frame_id in gt_dict:
        for obj_id, (x, y, w, h) in gt_dict[frame_id]:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {obj_id}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("MOT16 Ground Truth Viewer", frame)
    if cv2.waitKey(delay) == 27:
        break

cv2.destroyAllWindows()
