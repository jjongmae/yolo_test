from collections import defaultdict
import cv2, numpy as np
from ultralytics import YOLO

# ── 설정 ────────────────────────────────────────────────
model       = YOLO("rtdetr-l.pt")                       # 모델
video_path  = r"output_pedestrain.mp4"
cap         = cv2.VideoCapture(video_path)
palette     = np.random.randint(0, 255, size=(1000, 3))  # 클래스 ID → 색상

# 차량 관련 클래스 ID들 (COCO 데이터셋 기준)
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

def cls_color(cls_id: int):
    return tuple(int(c) for c in palette[cls_id % len(palette)])

frame_idx = 0
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1

    # ── 검출만 수행 ──────────────────────────────────────
    result = model.predict(frame, conf=0.15, iou=0.8, verbose=False)[0]

    if result.boxes:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        cls_ids     = result.boxes.cls.int().cpu().numpy()  # 클래스 ID들

        for (x1, y1, x2, y2), cls_id in zip(boxes_xyxy, cls_ids):
            # 차량 클래스만 필터링
            if cls_id in VEHICLE_CLASSES:
                color = cls_color(cls_id)
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )

    cv2.imshow("rtdetr Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
