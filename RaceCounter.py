__name__ = "Racecount"
__license__ = "GNUv3"
__author__ = "Nikola Cajic, Theo Hubinger"
__repository__ = "https://github.com/htl3r-2136/HTL3R_Racecount_4AX"

import cv2
import numpy as np
import supervision as sv
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

ROBOFLOW_API_KEY = "c3P356etbTH7VIVpbpCk"
MODEL_ID = "my-first-project-rj433/6"
VIDEO_REFERENCE = 1  # 0 oder 1 je nach Webcam

# === Globale Variablen für Linie & Runden ===
line_points = []        # [(x1, y1), (x2, y2)]
line_ready = False
line_zone = None

mercedes_laps = 0
porsche_laps = 0

tracker = sv.ByteTrack()   # Tracking bleibt, IDs werden danach überschrieben

box_annotator = sv.RoundBoxAnnotator()
label_annotator = sv.LabelAnnotator()
line_annotator = sv.LineZoneAnnotator(
    display_in_count=False,
    display_out_count=False,
    display_text_box=False
)

# === Maus-Callback zum Setzen der Ziellinie ===
def mouse_callback(event, x, y, flags, param):
    global line_points, line_ready

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            line_points.append((x, y))
            print("Linienpunkt gesetzt:", (x, y))
        if len(line_points) == 2:
            line_ready = True
            print("Ziellinie fertig:", line_points[0], "->", line_points[1])


# === on_prediction: Hauptlogik ===
def on_prediction(predictions, video_frame: VideoFrame):
    global line_zone, mercedes_laps, porsche_laps

    frame = video_frame.image

    # Kamera horizontal invertieren (Spiegelung links/rechts)
    frame = cv2.flip(frame, 1)

    # 1) Falls Linie noch nicht gesetzt ist: nur Vorschau + Rückgabe
    if not line_ready:
        preview = frame.copy()
        if len(line_points) >= 1:
            cv2.circle(preview, line_points[0], 5, (0, 255, 0), -1)
        if len(line_points) == 2:
            cv2.circle(preview, line_points[1], 5, (0, 0, 255), -1)
            cv2.line(preview, line_points[0], line_points[1], (255, 0, 0), 2)

        cv2.imshow("RaceCount", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit(0)
        return

    # 2) Linie ist gesetzt -> ggf. LineZone erzeugen
    if line_zone is None:
        start = sv.Point(*line_points[0])
        end = sv.Point(*line_points[1])
        line_zone = sv.LineZone(start=start, end=end)

    # 3) Predictions -> leeres Detections, falls nichts erkannt
    det_list = predictions.get("predictions", []) if predictions else []

    if len(det_list) == 0:
        detections = sv.Detections(
            xyxy=np.empty((0, 4), dtype=float),
            confidence=np.empty((0,), dtype=float),
            class_id=np.empty((0,), dtype=int),
        )
    else:
        boxes = []
        confidences = []
        class_ids = []

        for det in det_list:
            # Roboflow: x,y = Mittelpunkt, width/height = Breite/Höhe
            x, y = det["x"], det["y"]
            w, h = det["width"], det["height"]

            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes.append([x1, y1, x2, y2])
            confidences.append(det["confidence"])

            # Klassenname aus Roboflow holen und auf 0/1 mappen
            class_name = det.get("class", "")
            if class_name.lower() == "mercedes":
                cid = 0
            elif class_name.lower() == "porsche":
                cid = 1
            else:
                cid = 2  # unbekannt/sonstiges
            class_ids.append(cid)

        detections = sv.Detections(
            xyxy=np.array(boxes, dtype=float),
            confidence=np.array(confidences, dtype=float),
            class_id=np.array(class_ids, dtype=int),
        )

    # 4) Tracking
    detections = tracker.update_with_detections(detections)

    # 4b) tracker_id fest nach Klasse setzen (nur 0 und 1)
    if detections.tracker_id is None:
        detections.tracker_id = np.zeros(len(detections), dtype=int)

    for i in range(len(detections)):
        cid = detections.class_id[i]
        if cid == 0:      # Mercedes
            detections.tracker_id[i] = 0
        elif cid == 1:    # Porsche
            detections.tracker_id[i] = 1
        else:
            detections.tracker_id[i] = -1  # andere Klassen (falls vorhanden)

    # 5) LineZone triggern
    crossed_in, crossed_out = line_zone.trigger(detections)

    # 6) Runden pro Klasse zählen (über tracker_id 0/1)
    for i in range(len(detections)):
        if crossed_in[i] or crossed_out[i]:
            tid = detections.tracker_id[i]
            if tid == 0:
                mercedes_laps += 1
            elif tid == 1:
                porsche_laps += 1

    # 7) Labels pro Auto (Tracker-ID = 0/1, plus Klassenkürzel)
    labels = []
    for i in range(len(detections)):
        tid = detections.tracker_id[i]
        if tid == 0:
            labels.append("MERC #0")
        elif tid == 1:
            labels.append("POR #1")
        else:
            labels.append("CAR")

    # 8) Zeichnen
    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )
    annotated = line_annotator.annotate(
        annotated,
        line_zone
    )

    # Text mit nur 2 IDs (Mercedes/Porsche)
    cv2.putText(
        annotated,
        f"Mercedes (id 0): {mercedes_laps}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,  # kleinere Schrift
        (0, 0, 0),  # schwarz (B, G, R)
        1,  # dünner
        cv2.LINE_AA
    )

    cv2.putText(
        annotated,
        f"Porsche (id 1): {porsche_laps}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    # Keine Größenänderung -> kein Verzerren, Fenster ist nur „Container“
    cv2.imshow("RaceCount", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        exit(0)


# === Pipeline starten ===
cv2.namedWindow("RaceCount", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RaceCount", 1280, 720)  # Startgröße, Bild bleibt unverzerrt
cv2.setMouseCallback("RaceCount", mouse_callback)

pipeline = InferencePipeline.init(
    model_id=MODEL_ID,
    api_key=ROBOFLOW_API_KEY,
    video_reference=VIDEO_REFERENCE,
    on_prediction=on_prediction,
)

pipeline.start()
pipeline.join()