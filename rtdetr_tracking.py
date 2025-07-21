from collections import defaultdict
import cv2, numpy as np
from ultralytics import YOLO

# ── 설정 ────────────────────────────────────────────────
model       = YOLO("rtdetr-l.pt")                       # 모델
video_path  = r"output_pedestrain.mp4"
cap         = cv2.VideoCapture(video_path)
track_hist  = defaultdict(list)                        # {id: [(x, y), ...]}
max_hist    = 30                                       # 궤적 길이
palette     = np.random.randint(0, 255, size=(1000, 3))  # ID→색 팔레트 (최대 1,000개)

def id_color(tid: int):
    """트랙 ID별 고정 색상(BGR) 반환"""
    return tuple(int(c) for c in palette[tid % len(palette)])

frame_idx = 0
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1

    # ── 추적 ────────────────────────────────────────────
    res = model.track(frame, persist=True)[0]          # 첫 번째 결과 객체
    if res.boxes and res.boxes.is_track:
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()      # (N,4) [x1,y1,x2,y2]
        tids       = res.boxes.id.int().cpu().tolist() # 트랙 ID들

        for (x1, y1, x2, y2), tid in zip(boxes_xyxy, tids):
            color = id_color(tid)

            # ── 박스만 그리기 (두께 2) ────────────────────
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # ── (선택) 궤적 저장 & 그리기 ───────────────
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hist = track_hist[tid]
            hist.append((int(cx), int(cy)))
            if len(hist) > max_hist:
                hist.pop(0)
            if len(hist) > 1:
                pts = np.int32(hist).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)

    cv2.imshow("YOLO11 Tracking (box-only)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
