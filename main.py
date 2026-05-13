import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

canvas = np.zeros((720, 1280, 3), np.uint8)

xp, yp = 0, 0
draw_color = (255, 0, 255)
brush_thickness = 15

colors = [
    (255, 0, 255),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 0, 0)
]

print("Đang khởi động camera...")
print("-----------------------------------------------------")
print("HƯỚNG DẪN SỬ DỤNG:")
print("1. Đưa bàn tay vào trước camera.")
print("2. Giơ 1 NGÓN TRỎ để BẮT ĐẦU VẼ.")
print("3. Giơ 2 NGÓN (Trỏ + Giữa) để TẠM DỪNG hoặc CHỌN MÀU ở phía trên cùng màn hình.")
print("4. Giơ CẢ BÀN TAY (tất cả các ngón) để XÓA TOÀN BỘ bức tranh.")
print("5. Nhấn phím 'q' trên bàn phím (hoặc đóng cửa sổ) để thoát.")
print("6. Nhấn phím 'c' để xóa toàn bộ bức tranh.")
print("-----------------------------------------------------")

start_time = time.time()
fist_start_time = 0

while True:
    success, img = cap.read()
    if not success:
        print("Không thể đọc từ camera. Vui lòng kiểm tra lại thiết bị!")
        break
        
    img = cv2.flip(img, 1)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    timestamp_ms = int((time.time() - start_time) * 1000)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    
    lmList = []
    if detection_result.hand_landmarks:
        handLms = detection_result.hand_landmarks[0]
        h, w, c = img.shape
        
        for id, lm in enumerate(handLms):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
            
        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            lm_start = handLms[start_idx]
            lm_end = handLms[end_idx]
            cx1, cy1 = int(lm_start.x * w), int(lm_start.y * h)
            cx2, cy2 = int(lm_end.x * w), int(lm_end.y * h)
            cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
        for lm in handLms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
            
    if len(lmList) != 0:
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[12][1], lmList[12][2]
        
        fingers = []
        tipIds = [8, 12, 16, 20]
        
        for id in tipIds:
            dist_tip = np.hypot(lmList[id][1] - lmList[0][1], lmList[id][2] - lmList[0][2])
            dist_pip = np.hypot(lmList[id - 2][1] - lmList[0][1], lmList[id - 2][2] - lmList[0][2])
            
            if dist_tip > dist_pip:
                fingers.append(1)
            else:
                fingers.append(0)
                
        if sum(fingers) >= 4:
            fist_start_time = 0
            canvas = np.zeros((720, 1280, 3), np.uint8)
            cv2.putText(img, "Da xoa toan bo tranh!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            xp, yp = 0, 0
            
        elif len(fingers) >= 2 and fingers[0] == 1 and fingers[1] == 1 and sum(fingers) == 2:
            fist_start_time = 0
            xp, yp = 0, 0
            cv2.circle(img, (x1, y1), 15, (0, 255, 255), cv2.FILLED)
            cv2.putText(img, "Chon mau / Di chuyen", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if y1 < 100:
                if 0 < x1 < 250:
                    draw_color = colors[0]
                    brush_thickness = 15
                elif 250 < x1 < 500:
                    draw_color = colors[1]
                    brush_thickness = 15
                elif 500 < x1 < 750:
                    draw_color = colors[2]
                    brush_thickness = 15
                elif 750 < x1 < 1000:
                    draw_color = colors[3]
                    brush_thickness = 15
                elif 1000 < x1 < 1280:
                    draw_color = colors[4]
                    brush_thickness = 50
            
        elif len(fingers) >= 1 and fingers[0] == 1 and sum(fingers) == 1:
            fist_start_time = 0
            cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
            
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
                
            cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
            xp, yp = x1, y1
            cv2.putText(img, "Dang ve...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        elif sum(fingers) == 0:
            if fist_start_time == 0:
                fist_start_time = time.time()
                
            elapsed_time = time.time() - fist_start_time
            time_left = max(0, 3 - int(elapsed_time))
            
            cv2.putText(img, f"Thoat sau: {time_left}s", (450, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            if elapsed_time >= 3:
                print("Đã nắm tay 3 giây! Đang thoát ứng dụng...")
                break
        else:
            xp, yp = 0, 0
            fist_start_time = 0
    else:
        fist_start_time = 0

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)
    
    cv2.rectangle(img, (0, 0), (250, 100), colors[0], cv2.FILLED)
    cv2.rectangle(img, (250, 0), (500, 100), colors[1], cv2.FILLED)
    cv2.rectangle(img, (500, 0), (750, 100), colors[2], cv2.FILLED)
    cv2.rectangle(img, (750, 0), (1000, 100), colors[3], cv2.FILLED)
    cv2.rectangle(img, (1000, 0), (1280, 100), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, "Tay (Eraser)", (1050, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.imshow("Nhan dien tay ve hinh", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((720, 1280, 3), np.uint8)

cap.release()
cv2.destroyAllWindows()
