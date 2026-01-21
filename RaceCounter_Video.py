__name__ = "Racecount"
__license__ = "GNUv3"
__author__ = "Nikola Cajic, Theo Hubinger"
__repository__ = "https://github.com/htl3r-2136/HTL3R_Racecount_4AX"

import cv2
import numpy as np
import supervision as sv
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

# ================== CONFIG ==================
ROBOFLOW_API_KEY = "c3P356etbTH7VIVpbpCk"
MODEL_ID = "my-first-project-rj433/7"
VIDEO_PATH = r"C:\Users\nikol\OneDrive\Desktop\Racecount_POR.mp4"  # <-- VIDEO DATEI
# ============================================

# ================== STATE ==================
line_points = []
line_ready = False
line_zone = None

video_paused = True

mercedes_laps = 0
porsche_laps = 0
# ============================================

tracker = sv.ByteTrack()

box_annotator = sv.RoundBoxAnnotator()
label_annotator = sv.LabelAnnotator()
line_annotator = sv.LineZoneAnnotator(
    display_in_count=False,
    display_out_count=False,
    display_text_box=False
)

# ================== MOUSE ==================
def mouse_callback(event, x, y, flags, param):
    global line_points, line_ready, video_paused

    if event == cv2.EVENT_LBUTTONDOWN and not line_ready:
        line_points.append((x, y))
        print("Linienpunkt:", (x, y))

        if len(line_points) == 2:
            line_ready = True
            video_paused = False
            print("Ziellinie gesetzt → VIDEO START")
# ============================================

# ================== PIPELINE CALLBACK ==================
def on_prediction(predictions, video_frame: VideoFrame):
    global line_zone, mercedes_laps, porsche_laps, video_paused

    frame = video_frame.image

    # ---------- KEY HANDLING ----------
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        video_paused = not video_paused
        print("PAUSE" if video_paused else "PLAY")
    if key == ord("q"):
        exit(0)

    # ---------- PAUSE ----------
    if video_paused:
        preview = frame.copy()

        if len(line_points) >= 1:
            cv2.circle(preview, line_points[0], 5, (0, 255, 0), -1)
        if len(line_points) == 2:
            cv2.circle(preview, line_points[1], 5, (0, 0, 255), -1)
            cv2.line(preview, line_points[0], line_points[1], (255, 0, 0), 2)

        if line_zone is not None:
            preview = line_annotator.annotate(preview, line_zone)

        cv2.putText(
            preview,
            "PAUSED (SPACE)",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        cv2.imshow("RaceCount", preview)
        return

    # ---------- LINE INIT ----------
    if line_zone is None and line_ready:
        start = sv.Point(*line_points[0])
        end = sv.Point(*line_points[1])
        line_zone = sv.LineZone(start=start, end=end)

    # ---------- DETECTIONS ----------
    det_list = predictions.get("predictions", []) if predictions else []

    if not det_list:
        detections = sv.Detections.empty()
    else:
        boxes, confs, cids = [], [], []

        for det in det_list:
            x, y, w, h = det["x"], det["y"], det["width"], det["height"]
            boxes.append([x - w/2, y - h/2, x + w/2, y + h/2])
            confs.append(det["confidence"])

            cls = det.get("class", "").lower()
            if cls == "mercedes":
                cids.append(0)
            elif cls == "porsche":
                cids.append(1)
            else:
                cids.append(2)

        detections = sv.Detections(
            xyxy=np.array(boxes, float),
            confidence=np.array(confs, float),
            class_id=np.array(cids, int)
        )

    # ---------- TRACKING ----------
    detections = tracker.update_with_detections(detections)

    # ---------- LINE CROSS ----------
    crossed_in, crossed_out = line_zone.trigger(detections)

    for i in range(len(detections)):
        if crossed_in[i] or crossed_out[i]:
            if detections.class_id[i] == 0:
                mercedes_laps += 1
            elif detections.class_id[i] == 1:
                porsche_laps += 1

    # ---------- LABELS ----------
    labels = []
    for cid in detections.class_id:
        labels.append("MERCEDES" if cid == 0 else "PORSCHE" if cid == 1 else "CAR")

    # ---------- DRAW ----------
    annotated = frame.copy()
    annotated = line_annotator.annotate(annotated, line_zone)
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels)

    cv2.putText(
        annotated,
        f"Mercedes: {mercedes_laps}",
        (10, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        1.5,
        (0, 255, 0),
        1
    )
    cv2.putText(
        annotated,
        f"Porsche: {porsche_laps}",
        (10, 60),
        cv2.FONT_HERSHEY_DUPLEX,
        1.5,
        (0, 255, 0),
        1
    )

    cv2.imshow("RaceCount", annotated)
# =======================================================

# ================== WINDOW ==================
cv2.namedWindow("RaceCount", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RaceCount", 1280, 720)
cv2.setMouseCallback("RaceCount", mouse_callback)

print("Pipeline startet mit Video:", VIDEO_PATH)

pipeline = InferencePipeline.init(
    model_id=MODEL_ID,
    api_key=ROBOFLOW_API_KEY,
    video_reference=VIDEO_PATH,
    on_prediction=on_prediction,
    confidence=0.5,
    max_fps=None
)

pipeline.start()
pipeline.join()
cv2.destroyAllWindows()
