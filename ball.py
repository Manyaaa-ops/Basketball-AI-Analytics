
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# 1. Setup
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture('test_video.mp4')
pts = deque(maxlen=20)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_demo.mp4', fourcc, 30.0, (960, 540))

# 2. SELECT THE HOOP MANUALLY (The 'Golden' Fix)
ret, first_frame = cap.read()
if not ret:
    print("Video error")
    exit()

first_frame = cv2.resize(first_frame, (960, 540))
print("INSTRUCTIONS: Draw a box over the HOOP using your mouse, then press ENTER.")
# This opens a window for you to draw the 'Target Zone'
roi = cv2.selectROI("Select Hoop Zone", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Hoop Zone")

# hoop_rect format: (x, y, w, h)
hoop_x, hoop_y, hoop_w, hoop_h = roi
hoop_rect = (hoop_x, hoop_y, hoop_x + hoop_w, hoop_y + hoop_h)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (960, 540))

    # 3. BALL DETECTION
    results = model.predict(frame, conf=0.3, classes=[32], verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            pts.appendleft((int((x1 + x2) / 2), int((y1 + y2) / 2)))

    # 4. DRAW HOOP & PREDICTION
    cv2.rectangle(frame, (hoop_rect[0], hoop_rect[1]), (hoop_rect[2], hoop_rect[3]), (255, 255, 0), 2)
    
    prob = 0
    if len(pts) > 8:
        x_coords = np.array([p[0] for p in pts])
        y_coords = np.array([p[1] for p in pts])
        
        # Draw path tail (Yellow)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 2)

        try:
            # Quadratic fit for gravity curve
            poly = np.polyfit(x_coords, y_coords, 2)
            predict = np.poly1d(poly)
            
            # Draw and Check Prediction (Green dots)
            for future_x in range(int(x_coords[0]), int(x_coords[0] + 500), 10):
                future_y = int(predict(future_x))
                cv2.circle(frame, (future_x, future_y), 2, (0, 255, 0), -1)
                
                # Check if path enters the box you drew
                if hoop_rect[0] < future_x < hoop_rect[2] and \
                   hoop_rect[1] < future_y < hoop_rect[3]:
                    prob = 98
                    break
        except: pass

    # 5. DRAW PROBABILITY BAR
    cv2.rectangle(frame, (50, 150), (80, 400), (50, 50, 50), -1)
    bar_color = (0, 255, 0) if prob > 50 else (0, 0, 255)
    bar_height = int(np.interp(prob, [0, 100], [400, 150]))
    cv2.rectangle(frame, (50, bar_height), (80, 400), bar_color, -1)
    cv2.putText(frame, f"SHOT PROB: {prob}%", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)
    out.write(frame) # Save frame to video
    cv2.imshow('Manya-Ball Analytics', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
