import cv2
import mediapipe as mp
import numpy as np
import time
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

save_dir = "hasil_foto"
os.makedirs(save_dir, exist_ok=True)

# Shared state for async callback
latest_result = None
latest_timestamp = 0

# Hand connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]

def result_callback(result, output_image, timestamp_ms):
    global latest_result, latest_timestamp
    latest_result = result
    latest_timestamp = timestamp_ms

def is_peace_sign(hand_landmarks):
    """Deteksi spesifik untuk pose 2 jari (Peace Sign ✌️)."""
    # Telunjuk & Tengah harus tegak
    index_up = hand_landmarks[8].y < hand_landmarks[6].y
    middle_up = hand_landmarks[12].y < hand_landmarks[10].y
    # Manis & Kelingking harus menekuk
    ring_down = hand_landmarks[16].y > hand_landmarks[14].y
    pinky_down = hand_landmarks[20].y > hand_landmarks[18].y
    
    return index_up and middle_up and ring_down and pinky_down

def count_fingers(hand_landmarks):
    """Hitung jumlah jari menggunakan jarak absolut."""
    tips_ids = [8, 12, 16, 20]
    count = 0

    # Logika Jempol: Cek jarak ujung jempol ke pangkal kelingking
    thumb_tip_dist = abs(hand_landmarks[4].x - hand_landmarks[17].x)
    thumb_ip_dist = abs(hand_landmarks[3].x - hand_landmarks[17].x)
    
    if thumb_tip_dist > thumb_ip_dist:
        count += 1

    # Cek 4 jari lainnya
    for tip in tips_ids:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            count += 1

    return count

def draw_landmarks_on_frame(frame, detection_result):
    """Draw hand landmarks on the frame."""
    if detection_result is None or not detection_result.hand_landmarks:
        return frame

    annotated = frame.copy()
    h, w, _ = annotated.shape

    for hand_landmarks in detection_result.hand_landmarks:
        # Draw each landmark as a circle
        for landmark in hand_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)

        # Draw connections between landmarks
        for connection in HAND_CONNECTIONS:
            start = hand_landmarks[connection[0]]
            end = hand_landmarks[connection[1]]
            sx, sy = int(start.x * w), int(start.y * h)
            ex, ey = int(end.x * w), int(end.y * h)
            cv2.line(annotated, (sx, sy), (ex, ey), (255, 255, 255), 2)

    return annotated

# Configure HandLandmarker for live stream
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=4, 
    result_callback=result_callback
)

camera_index = 0
cap = cv2.VideoCapture(camera_index)

cv2.namedWindow("Gesture Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Camera", 1280, 720)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Convert to MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Send frame for async detection
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        # Inisialisasi variabel deteksi
        total_finger_count = 0
        trigger_blur = False

        if latest_result is not None and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                # Hitung kumulatif jari untuk indikator teks
                total_finger_count += count_fingers(hand_landmarks)
                
                # Cek per tangan: Apakah ada satu tangan yang memunculkan pose Peace / 2 jari?
                if is_peace_sign(hand_landmarks) or count_fingers(hand_landmarks) == 2:
                    trigger_blur = True

        # Buat canvas tampilan
        display_frame = frame.copy()
        
        # 1. Gambar tracking tangan duluan
        if latest_result is not None and latest_result.hand_landmarks:
            display_frame = draw_landmarks_on_frame(display_frame, latest_result)

        # 2. Tampilkan teks indikator jumlah jari (Tetap Hijau untuk info sistem)
        cv2.putText(display_frame, f"Jari Terdeteksi: {total_finger_count}", (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 3. KONDISI SAKLAR: Efek blur diturunkan intensitasnya ke (25, 25)
        if total_finger_count == 2 or trigger_blur:
            display_frame = cv2.GaussianBlur(display_frame, (25, 25), 0)
            # Teks tetap warna pink cerah (BGR: 180, 105, 255)
            cv2.putText(display_frame, "foto kita blur..", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 105, 180), 2)

        # 4. Crop ke aspect ratio 16:9 biar tidak gepeng, lalu resize
        h, w = display_frame.shape[:2]
        target_aspect = 16 / 9
        current_aspect = w / h
        
        if current_aspect < target_aspect:
            new_h = int(w / target_aspect)
            y_offset = (h - new_h) // 2
            display_frame = display_frame[y_offset:y_offset+new_h, :]
        elif current_aspect > target_aspect:
            new_w = int(h * target_aspect)
            x_offset = (w - new_w) // 2
            display_frame = display_frame[:, x_offset:x_offset+new_w]

        # Tampilkan ke layar monitor (Windowed)
        display_frame = cv2.resize(display_frame, (1280, 720))
        cv2.imshow("Gesture Camera", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            cap.release()
            found = False
            for i in range(1, 6):
                next_index = (camera_index + i) % 6
                cap = cv2.VideoCapture(next_index)
                if cap.isOpened() and cap.read()[0]:
                    camera_index = next_index
                    found = True
                    break
                else:
                    cap.release()
            
            if not found:
                cap = cv2.VideoCapture(camera_index)

cap.release()
cv2.destroyAllWindows()