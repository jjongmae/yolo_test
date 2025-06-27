import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ── SAHI 모델 설정 ─────────────────────────────────────
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo11n.pt",
    confidence_threshold=0.25,
    device="cuda:0"  # 또는 "cpu"
)

video_path = r"D:\data\여주시험도로_20250610\카메라1_202506101340.mp4"
cap = cv2.VideoCapture(video_path)

palette = np.random.randint(0, 255, size=(1000, 3))
def cls_color(cls_id): return tuple(map(int, palette[cls_id % len(palette)]))

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── SAHI 슬라이스 기반 검출 수행 ───────────────────
    result = get_sliced_prediction(
        image=frame_rgb,
        detection_model=detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    for obj in result.object_prediction_list:
        x1, y1, x2, y2 = map(int, obj.bbox.to_voc_bbox())
        cls_id = int(obj.category.id)
        color = cls_color(cls_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"cls:{cls_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    cv2.imshow("SAHI Detection Only", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
