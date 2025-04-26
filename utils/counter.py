# Mô đun đếm đối tượng đi qua đường ảo
import cv2
import numpy as np
from typing import List, Tuple, Dict, Set

class PersonCounter:
    """
    Lớp đếm số người đi qua một đường ảo trong khung hình
    """
    def __init__(self, frame_height: int, frame_width: int, 
                line_position: float = 0.5, line_direction: str = 'horizontal'):
        """
        Khởi tạo bộ đếm người
        Args:
            frame_height: Chiều cao của khung hình
            frame_width: Chiều rộng của khung hình
            line_position: Vị trí của đường đếm (0.0-1.0, phần trăm của chiều cao hoặc rộng)
            line_direction: Hướng của đường đếm ('horizontal' hoặc 'vertical')
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.line_position = line_position
        self.line_direction = line_direction
        
        # Xác định tọa độ đường đếm
        if line_direction == 'horizontal':
            self.line_y = int(frame_height * line_position)
            self.line_start = (0, self.line_y)
            self.line_end = (frame_width, self.line_y)
        else:  # vertical
            self.line_x = int(frame_width * line_position)
            self.line_start = (self.line_x, 0)
            self.line_end = (self.line_x, frame_height)
        
        # Các bộ đếm
        self.count_up = 0      # Đếm người đi lên/trái
        self.count_down = 0    # Đếm người đi xuống/phải
        self.total_count = 0   # Tổng số người đi qua
        
        # Lưu vị trí trước đó của từng ID
        self.prev_positions = {}
        
        # Các ID đã được đếm (để tránh đếm nhiều lần)
        self.counted_ids = set()
        
        # Biến đếm frame để xuất debug info định kỳ
        self.frame_count = 0
        
        print(f"📊 Đã khởi tạo bộ đếm người. Đường {line_direction} tại vị trí {line_position:.2f}")
        if line_direction == 'horizontal':
            print(f"Tọa độ đường ngang: y={self.line_y}")
        else:
            print(f"Tọa độ đường dọc: x={self.line_x}")
    
    def _is_crossing_line(self, 
                        track_id: int,
                        prev_pos: Tuple[int, int], 
                        curr_pos: Tuple[int, int]) -> Tuple[bool, str]:
        """
        Kiểm tra xem đối tượng có đi qua đường đếm không và theo hướng nào
        Args:
            track_id: ID của track để debug
            prev_pos: Vị trí trước đó (center_x, center_y)
            curr_pos: Vị trí hiện tại (center_x, center_y)
        Returns:
            (đã đi qua, hướng)
        """
        prev_x, prev_y = prev_pos
        curr_x, curr_y = curr_pos
        
        # Debug info
        debug_cross = False
        
        # Tính khoảng cách di chuyển
        distance_moved = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
        
        if self.line_direction == 'horizontal':
            # Nếu vị trí trước và sau ở hai bên của đường ngang
            # Kiểm tra cả trường hợp đi ngang qua đúng đường và cả trường hợp di chuyển nhanh vượt qua đường
            
            # Kiểm tra chuẩn - đi từ trên xuống dưới hoặc từ dưới lên trên qua đường
            if (prev_y <= self.line_y and curr_y > self.line_y) or \
               (prev_y >= self.line_y and curr_y < self.line_y):
                # Xác định hướng
                direction = 'down' if prev_y < self.line_y else 'up'
                debug_cross = True
                print(f"✅ CROSSING: ID {track_id} vượt đường ngang (y={self.line_y}) từ ({prev_x},{prev_y}) đến ({curr_x},{curr_y}), hướng: {direction}")
                return True, direction
                
            # Kiểm tra thêm trường hợp di chuyển quá nhanh hoặc có frame bỏ qua
            if distance_moved > 30:  # Di chuyển đáng kể
                # Tính xem đường thẳng nối prev_pos và curr_pos có cắt qua đường đếm không
                if (prev_y < self.line_y and curr_y > self.line_y + 30) or \
                   (prev_y > self.line_y + 30 and curr_y < self.line_y):
                    direction = 'down' if prev_y < self.line_y else 'up'
                    print(f"✅ FAST CROSSING: ID {track_id} vượt nhanh qua đường ngang (y={self.line_y}) từ ({prev_x},{prev_y}) đến ({curr_x},{curr_y}), hướng: {direction}")
                    return True, direction
            
        else:  # vertical
            # Kiểm tra chuẩn - đi từ trái sang phải hoặc từ phải sang trái qua đường dọc
            if (prev_x <= self.line_x and curr_x > self.line_x) or \
               (prev_x >= self.line_x and curr_x < self.line_x):
                # Xác định hướng
                direction = 'right' if prev_x < self.line_x else 'left'
                debug_cross = True
                print(f"✅ CROSSING: ID {track_id} vượt đường dọc (x={self.line_x}) từ ({prev_x},{prev_y}) đến ({curr_x},{curr_y}), hướng: {direction}")
                return True, direction
                
            # Kiểm tra thêm trường hợp di chuyển quá nhanh hoặc có frame bỏ qua
            if distance_moved > 30:  # Di chuyển đáng kể
                if (prev_x < self.line_x and curr_x > self.line_x + 30) or \
                   (prev_x > self.line_x + 30 and curr_x < self.line_x):
                    direction = 'right' if prev_x < self.line_x else 'left'
                    print(f"✅ FAST CROSSING: ID {track_id} vượt nhanh qua đường dọc (x={self.line_x}) từ ({prev_x},{prev_y}) đến ({curr_x},{curr_y}), hướng: {direction}")
                    return True, direction
        
        # Debug info khi có đối tượng đi gần đường
        if self.line_direction == 'horizontal':
            distance_to_line = abs(curr_y - self.line_y)
            if distance_to_line < 30:
                print(f"👁 NEAR LINE: ID {track_id} gần đường ngang (y={self.line_y}), khoảng cách={distance_to_line}px, di chuyển từ ({prev_x},{prev_y}) đến ({curr_x},{curr_y})")
        else:
            distance_to_line = abs(curr_x - self.line_x)
            if distance_to_line < 30:
                print(f"👁 NEAR LINE: ID {track_id} gần đường dọc (x={self.line_x}), khoảng cách={distance_to_line}px, di chuyển từ ({prev_x},{prev_y}) đến ({curr_x},{curr_y})")
        
        return False, None
    
    def update(self, tracks: List[Tuple]) -> None:
        """
        Cập nhật bộ đếm với vị trí mới của các đối tượng
        Args:
            tracks: Danh sách theo dõi [(track_id, class_id, [x,y,w,h]),...]
        """
        # In thông tin debug về số lượng track được cập nhật
        self.frame_count += 1
        if self.frame_count % 20 == 0 or len(tracks) > 0:  # In mỗi 20 frame hoặc khi có tracks
            print(f"📊 FRAME {self.frame_count}: Đang cập nhật {len(tracks)} đối tượng cho bộ đếm")
            print(f"📊 Tổng số ID đã đếm: {len(self.counted_ids)}, ID được theo dõi: {len(self.prev_positions)}")
        
        # Lưu danh sách track_id để làm sạch danh sách prev_positions
        active_track_ids = set()
        
        for track in tracks:
            track_id, class_id, bbox = track
            
            # Thêm ID vào danh sách active
            active_track_ids.add(track_id)
            
            # Tính tọa độ trung tâm
            x, y, w, h = bbox
            center_x = x + w // 2
            center_y = y + h // 2
            current_pos = (center_x, center_y)
            
            # Bỏ qua nếu ID này đã được đếm
            if track_id in self.counted_ids:
                if self.frame_count % 50 == 0:  # Log định kỳ
                    print(f"🔄 Bỏ qua ID {track_id} vì đã được đếm trước đó")
                continue
            
            # Debug thông tin track cơ bản
            if self.frame_count % 50 == 0 or track_id not in self.prev_positions:  # Log định kỳ hoặc track mới
                line_pos = self.line_y if self.line_direction == 'horizontal' else self.line_x
                print(f"👤 Track ID: {track_id}, Vị trí: ({center_x}, {center_y}), Kích thước: {w}x{h}, Đường {self.line_direction}: {line_pos}")
            
            # Kiểm tra nếu có vị trí trước đó
            if track_id in self.prev_positions:
                prev_pos = self.prev_positions[track_id]
                
                # Kiểm tra xem có đi qua đường không
                crossed, direction = self._is_crossing_line(track_id, prev_pos, current_pos)
                
                if crossed:
                    # Cập nhật bộ đếm theo hướng
                    if direction in ['up', 'left']:
                        self.count_up += 1
                    else:  # down, right
                        self.count_down += 1
                    
                    self.total_count += 1
                    self.counted_ids.add(track_id)
                    
                    print(f"🚶 ID {track_id} đi qua đường theo hướng {direction}. Tổng: {self.total_count} (↑:{self.count_up}/↓:{self.count_down})")
            else:
                if self.frame_count % 50 == 0:  # Log định kỳ
                    print(f"🆕 Track mới xuất hiện: ID {track_id} tại ({center_x}, {center_y})")
            
            # Cập nhật vị trí cho lần sau
            self.prev_positions[track_id] = current_pos
        
        # Xóa các ID không còn hoạt động khỏi prev_positions
        ids_to_remove = set(self.prev_positions.keys()) - active_track_ids
        if ids_to_remove:
            for id_to_remove in ids_to_remove:
                del self.prev_positions[id_to_remove]
            print(f"🧹 Đã xóa {len(ids_to_remove)} ID không còn hoạt động khỏi bộ nhớ: {ids_to_remove}")
    
    def draw_counter(self, frame: np.ndarray) -> np.ndarray:
        """
        Vẽ đường đếm và số liệu thống kê lên khung hình
        Args:
            frame: Khung hình cần vẽ
        Returns:
            Khung hình đã vẽ
        """
        # Vẽ đường đếm với độ đậm cao hơn và màu nổi bật
        cv2.line(frame, self.line_start, self.line_end, (0, 255, 255), 3)  # Màu vàng đậm hơn
        
        # Thêm chữ tại đường đếm
        if self.line_direction == 'horizontal':
            cv2.putText(frame, "Duong dem", (self.line_start[0] + 10, self.line_start[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Duong dem", (self.line_start[0] + 10, self.line_start[1] + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Vẽ hướng dẫn
        if self.line_direction == 'horizontal':
            # Vẽ mũi tên hướng lên
            arrow_up_start = (20, self.line_y + 50)
            arrow_up_end = (20, self.line_y - 50)
            cv2.arrowedLine(frame, arrow_up_start, arrow_up_end, (0, 0, 255), 3, tipLength=0.3)
            cv2.putText(frame, "Len", (25, self.line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Vẽ mũi tên hướng xuống
            arrow_down_start = (80, self.line_y - 50)
            arrow_down_end = (80, self.line_y + 50)
            cv2.arrowedLine(frame, arrow_down_start, arrow_down_end, (255, 0, 0), 3, tipLength=0.3)
            cv2.putText(frame, "Xuong", (85, self.line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:  # vertical
            # Vẽ mũi tên hướng trái
            arrow_left_start = (self.line_x + 50, 30)
            arrow_left_end = (self.line_x - 50, 30)
            cv2.arrowedLine(frame, arrow_left_start, arrow_left_end, (0, 0, 255), 3, tipLength=0.3)
            cv2.putText(frame, "Trai", (self.line_x - 10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Vẽ mũi tên hướng phải
            arrow_right_start = (self.line_x - 50, 90)
            arrow_right_end = (self.line_x + 50, 90)
            cv2.arrowedLine(frame, arrow_right_start, arrow_right_end, (255, 0, 0), 3, tipLength=0.3)
            cv2.putText(frame, "Phai", (self.line_x - 10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Vẽ số liệu thống kê với nền đen mờ và chữ lớn hơn
        text_bg_color = (0, 0, 0, 0.7)  # Màu nền đen với độ trong suốt
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        
        # Tạo một bảng thống kê
        stats_x = frame.shape[1] - 220
        stats_y = 30
        stats_w = 210
        stats_h = 120
        
        # Vẽ nền cho bảng thống kê
        overlay = frame.copy()
        cv2.rectangle(overlay, (stats_x, stats_y), (stats_x + stats_w, stats_y + stats_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)  # Tạo hiệu ứng trong suốt
        
        # Tiêu đề bảng
        title_text = "THONG KE"
        cv2.putText(frame, title_text, (stats_x + 10, stats_y + 25), font, font_scale, (0, 255, 255), thickness)
        
        # Tổng số người
        total_text = f"Tong so: {self.total_count}"
        cv2.putText(frame, total_text, (stats_x + 10, stats_y + 55), font, font_scale, text_color, thickness)
        
        # Số người đi lên/trái
        if self.line_direction == 'horizontal':
            up_text = f"Di len: {self.count_up}"
        else:
            up_text = f"Di trai: {self.count_up}"
        cv2.putText(frame, up_text, (stats_x + 10, stats_y + 85), font, font_scale, text_color, thickness)
        
        # Số người đi xuống/phải
        if self.line_direction == 'horizontal':
            down_text = f"Di xuong: {self.count_down}"
        else:
            down_text = f"Di phai: {self.count_down}"
        cv2.putText(frame, down_text, (stats_x + 10, stats_y + 115), font, font_scale, text_color, thickness)
        
        return frame
