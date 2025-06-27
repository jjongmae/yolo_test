# -*- coding: utf-8 -*-
# 1) imports ────────────────────────────────────────────────────────────────
from collections import defaultdict
import cv2, numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker  # ★ NEW
from types import SimpleNamespace  

# 2) 모델 & 트래커 초기화 ────────────────────────────────────────────────────
model = YOLO("yolo11n.pt")

bt_cfg = {
    "track_thresh": 0.25,
    "track_high_thresh": 0.5,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.6,        # 신규 트랙 생성을 위한 최소 score
    "match_thresh": 0.8,
    "track_buffer": 30,
    "momentum": 0.2,
    "min_box_area": 10,
    "min_box_score": 0.1,
    "max_age": 30,
    "nms_thresh": 0.7,
    "fuse_score": False,            # ← 지금 에러 난 부분
    "mot20": False                  # MOT20 평가 방식 사용 여부
}

tracker = BYTETracker(SimpleNamespace(**bt_cfg), frame_rate=30)  # ★ NEW

video_path = r"D:\data\여주시험도로_20250610\카메라1_202506101340.mp4"
cap = cv2.VideoCapture(video_path)

# 시각화용 유틸 -------------------------------------------------------------
palette = np.random.randint(0, 255, (1000, 3))
def id_color(t): return tuple(int(c) for c in palette[int(t) % len(palette)])

track_hist = defaultdict(list)
max_hist   = 30

# 3) 루프 ───────────────────────────────────────────────────────────────────
frame_idx = 0
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1

    # ── 3-1 YOLO 검출 (track X) ───────────────────────────────────────────
    det = model.predict(frame, conf=0.25, iou=0.5, verbose=False)[0]

    boxes  = det.boxes.xyxy.cpu().numpy()        # (N,4)
    scores = det.boxes.conf.cpu().numpy()        # (N,)
    clss   = det.boxes.cls.cpu().numpy()         # (N,)

    # ByteTrack 입력 형식 [x1,y1,x2,y2,score,class]  (없으면 빈 배열)
    if len(boxes):
        dets = np.hstack((boxes, scores[:, None], clss[:, None])).astype(np.float32)
    else:
        dets = np.empty((0, 6), dtype=np.float32)

    # ── 3-2 ByteTrack 업데이트 ────────────────────────────────────────────
    online_targets = tracker.update(det.boxes.cpu(), frame.shape[:2], frame_idx)

    # ── 3-3 시각화 (박스 + 궤적) ──────────────────────────────────────────
    for t in online_targets:
        # track 객체는 tlbr(x1,y1,x2,y2)·track_id·cls 속성 보유
        # 방법 1: 필요한 값만 슬라이싱
        x1, y1, x2, y2, tid = map(int, t[:5])
        cls = int(t[6]) if len(t) > 6 else -1  # 없으면 -1로 처리
        color          = id_color(tid)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        # cv2.putText(frame, str(tid), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # 궤적(선택)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        hist   = track_hist[tid]
        hist.append((cx, cy))
        if len(hist) > max_hist:
            hist.pop(0)
        if len(hist) > 1:
            pts = np.int32(hist).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)

    cv2.imshow("YOLO11 + SAHI + ByteTrack", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
