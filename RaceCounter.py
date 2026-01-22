"""
Racecount - Automatisierte Rundenzählung für Slotcars.

Dieses Skript nutzt das Roboflow Inference SDK und die Supervision-Library, um
bestimmte Fahrzeugmarken (Mercedes, Porsche) auf einer Rennstrecke zu erkennen,
zu tracken und Runden zu zählen, sobald sie eine manuell definierte Ziellinie überqueren.
"""

__name__ = "Racecount"
__license__ = "GNUv3"
__author__ = "Nikola Cajic, Theo Hubinger"
__repository__ = "https://github.com/htl3r-2136/HTL3R_Racecount_4AX"

import cv2
import numpy as np
import supervision as sv
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.sinks import render_boxes

ROBOFLOW_API_KEY = "c3P356etbTH7VIVpbpCk"
MODEL_ID = "my-first-project-rj433/8"
VIDEO_REFERENCE = 0  # 0, 1 oder 2 je nach Webcam

# === Globale Variablen für Linie & Runden ===
line_points = []  # [(x1, y1), (x2, y2)]
line_ready = False
line_zone = None

mercedes_laps = 0
porsche_laps = 0
# ============================================
# Initialisierung der Supervision-Werkzeuge
tracker = sv.ByteTrack()

box_annotator = sv.RoundBoxAnnotator()
label_annotator = sv.LabelAnnotator()
line_annotator = sv.LineZoneAnnotator(
    display_in_count=False,
    display_out_count=False,
    display_text_box=False
)

def mouse_callback(event, x, y, flags, param):
    """
        Verarbeitet Maus-Events zum Zeichnen der Ziellinie im Vorschaufenster.

        Args:
            event (int): Der Typ des OpenCV-Mausereignisses.
            x (int): X-Koordinate der Mausposition.
            y (int): Y-Koordinate der Mausposition.
            flags (int): Event-spezifische Flags.
            param: Optionale Parameter (nicht verwendet).
    """
    global line_points, line_ready

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            line_points.append((x, y))
            print("Linienpunkt gesetzt:", (x, y))
        if len(line_points) == 2:
            line_ready = True
            print("Ziellinie fertig:", line_points[0], "->", line_points[1])


def on_prediction(predictions, video_frame: VideoFrame):
    """
        Callback-Funktion für die Inference-Pipeline. Verarbeitet Detektionen,
        aktualisiert das Tracking und zählt die Runden bei Linienüberquerung.

        Args:
            predictions (dict): Die Vorhersagen des Roboflow-Modells.
            video_frame (VideoFrame): Das aktuelle Frame-Objekt inklusive Bilddaten.
    """
    global line_zone, mercedes_laps, porsche_laps

    frame = video_frame.image

    # 1) Initialisierungsphase: Ziellinie setzen
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

    # 2) Initialisierung der LineZone nach Definition der Punkte
    if line_zone is None:
        start = sv.Point(*line_points[0])
        end = sv.Point(*line_points[1])
        line_zone = sv.LineZone(start=start, end=end)

    # 3) Konvertierung der Predictions in Supervision-Detections
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
            # Umrechnung von Roboflow (Center X, Y) zu XYXY (Corner)
            x, y = det["x"], det["y"]
            w, h = det["width"], det["height"]

            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes.append([x1, y1, x2, y2])
            confidences.append(det["confidence"])

            # Mapping der Klassennamen auf IDs
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

    # 4) Objekt-Tracking (Zuweisung von IDs über Frames hinweg)
    detections = tracker.update_with_detections(detections)

    # Manuelle Korrektur der Tracker-ID basierend auf der Klasse (für die Zähllogik)
    if detections.tracker_id is None:
        detections.tracker_id = np.zeros(len(detections), dtype=int)

    for i in range(len(detections)):
        cid = detections.class_id[i]
        if cid == 0:  # Mercedes
            detections.tracker_id[i] = 0
        elif cid == 1:  # Porsche
            detections.tracker_id[i] = 1
        else:
            detections.tracker_id[i] = -1 # andere Klassen (falls vorhanden)

    # 5) Linienüberquerung prüfen
    crossed_in, crossed_out = line_zone.trigger(detections)

    # 6) Rundenzähler inkrementieren
    for i in range(len(detections)):
        if crossed_in[i] or crossed_out[i]:
            tid = detections.tracker_id[i]
            if tid == 0:
                mercedes_laps += 1
            elif tid == 1:
                porsche_laps += 1

    # 7) Visualisierung vorbereiten (Labels)
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
    annotated = frame
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )
    annotated = line_annotator.annotate(
        annotated,
        line_zone
    )

    # UI Overlays (Text-Anzeige)
    cv2.putText(
        annotated,
        f"Mercedes (id 0): {mercedes_laps}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        annotated,
        f"Porsche (id 1): {porsche_laps}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

    cv2.imshow("RaceCount", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        exit(0)


# --- Kamera-Setup & Pipeline-Start ---
print("Stelle Kamera-Qualität ein...")
cap = cv2.VideoCapture(VIDEO_REFERENCE)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"Auflösung: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
    cap.release()
else:
    print(f"WARNUNG: Kamera {VIDEO_REFERENCE} konnte nicht geöffnet werden!")

# === Pipeline starten ===
cv2.namedWindow("RaceCount", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RaceCount", 1280, 720)
cv2.setMouseCallback("RaceCount", mouse_callback)

print("Initialisiere Pipeline...")
print(f"Model: {MODEL_ID}")
print(f"Kamera: {VIDEO_REFERENCE}")

try:
    pipeline = InferencePipeline.init(
        model_id=MODEL_ID,
        api_key=ROBOFLOW_API_KEY,
        video_reference=VIDEO_REFERENCE,
        on_prediction=on_prediction,
        max_fps=30,  # Erhöht für bessere Performance
        confidence=0.5,  # Nur sichere Detections
    )

    print("Pipeline initialisiert! Starte Stream...")
    print("Klicke 2 Punkte im Fenster um die Ziellinie zu setzen.")
    print("Drücke 'q' zum Beenden.")

    pipeline.start()
    pipeline.join()

except KeyboardInterrupt:
    print("\nProgramm durch Benutzer beendet")
    cv2.destroyAllWindows()

except Exception as e:
    print(f"FEHLER: {e}")
    import traceback

    traceback.print_exc()
    cv2.destroyAllWindows()