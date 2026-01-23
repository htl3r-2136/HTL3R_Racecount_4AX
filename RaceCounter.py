"""
Racecount - Automatisierte Rundenzählung für Slotcars.

Dieses Programm verwendet:
- Roboflow Inference SDK für Objekterkennung
- Supervision für Tracking, Linien-Logik und Visualisierung
- OpenCV für Videoanzeige, Maus- und Tastatureingaben

Ziel:
Slotcars (Mercedes & Porsche) werden erkannt, verfolgt (Tracking)
und gezählt, sobald sie eine vom Benutzer gesetzte Ziellinie überqueren.
"""

# ================== METADATEN ==================
__name__ = "Racecount"
__license__ = "GNUv3"
__author__ = "Nikola Cajic, Theo Hubinger"
__repository__ = "https://github.com/htl3r-2136/HTL3R_Racecount_4AX"
# ===============================================


# ================== IMPORTS =====================
import cv2                  # Video, Fenster, Maus- & Tastatur-Handling
import numpy as np           # Effiziente numerische Datenstrukturen
import supervision as sv     # Tracking & Visualisierung (ByteTrack, LineZone)
from inference import InferencePipeline   # Roboflow Inference Pipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.sinks import render_boxes
# ===============================================


# ================== KONFIGURATION ==================
# API-Key für den Zugriff auf das Roboflow-Modell
ROBOFLOW_API_KEY = "c3P356etbTH7VIVpbpCk"

# Modell-ID im Format: projektname/version
MODEL_ID = "my-first-project-rj433/8"

# Kameraquelle:
# 0 = Standard-Webcam
# 1 / 2 = externe Kameras (falls vorhanden)
VIDEO_REFERENCE = 0
# ===================================================


# ================== GLOBALE ZUSTÄNDE ==================
# Zwei Punkte definieren die Ziellinie (per Mausklick)
line_points = []          # [(x1, y1), (x2, y2)]
line_ready = False        # True, sobald beide Punkte gesetzt wurden
line_zone = None          # Supervision LineZone (wird später erzeugt)

# Rundenzähler für die beiden Fahrzeugklassen
mercedes_laps = 0
porsche_laps = 0
# =====================================================


# ================== TRACKING & ANNOTATION ==================
# ByteTrack verfolgt Objekte über mehrere Frames hinweg
# → verhindert doppelte Zählungen
tracker = sv.ByteTrack()

# Zeichnet abgerundete Bounding Boxes
box_annotator = sv.RoundBoxAnnotator()

# Zeichnet Textlabels (z.B. MER / POR)
label_annotator = sv.LabelAnnotator()

# Zeichnet die Ziellinie (ohne Standard-Zähltexte)
line_annotator = sv.LineZoneAnnotator(
    display_in_count=False,
    display_out_count=False,
    display_text_box=False
)
# ===========================================================


# ================== MAUS CALLBACK ==================
def mouse_callback(event, x, y, flags, param):
    """
    Ermöglicht das interaktive Setzen der Ziellinie.

    Ablauf:
    - 1. Linksklick → Startpunkt der Linie
    - 2. Linksklick → Endpunkt der Linie
    - Danach wird die Linie fixiert
    """
    global line_points, line_ready

    # Reagiere nur auf Linksklicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            line_points.append((x, y))
            print("Linienpunkt gesetzt:", (x, y))

        # Sobald zwei Punkte existieren → Linie fertig
        if len(line_points) == 2:
            line_ready = True
            print("Ziellinie fertig:", line_points[0], "->", line_points[1])
# ==================================================


# ================== PIPELINE CALLBACK ==================
def on_prediction(predictions, video_frame: VideoFrame):
    """
    Zentrale Callback-Funktion der Inference Pipeline.

    Diese Funktion wird für jedes einzelne Video-Frame aufgerufen
    und übernimmt:
    - Linien-Setup
    - Verarbeitung der Modellvorhersagen
    - Objekt-Tracking
    - Rundenzählung
    - Darstellung im UI
    """
    global line_zone, mercedes_laps, porsche_laps

    # Aktuelles Bild aus dem VideoFrame extrahieren
    frame = video_frame.image


    # ---------- 1) ZIELLINIEN-SETUP ----------
    # Solange keine Linie definiert ist, läuft das System im Setup-Modus
    if not line_ready:
        preview = frame.copy()

        # Visualisierung der bereits gesetzten Punkte
        if len(line_points) >= 1:
            cv2.circle(preview, line_points[0], 5, (0, 255, 0), -1)
        if len(line_points) == 2:
            cv2.circle(preview, line_points[1], 5, (0, 0, 255), -1)
            cv2.line(preview, line_points[0], line_points[1], (255, 0, 0), 2)

        cv2.imshow("RaceCount", preview)

        # Abbruch per Taste 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit(0)
        return  # Verarbeitung dieses Frames abbrechen


    # ---------- 2) LINEZONE INITIALISIERUNG ----------
    # Wird genau einmal erstellt, sobald zwei Punkte vorhanden sind
    if line_zone is None:
        start = sv.Point(*line_points[0])
        end = sv.Point(*line_points[1])
        line_zone = sv.LineZone(start=start, end=end)


    # ---------- 3) DETEKTIONSVERARBEITUNG ----------
    # Rohdaten vom Roboflow-Modell
    det_list = predictions.get("predictions", []) if predictions else []

    if len(det_list) == 0:
        # Leere Detections (kein Objekt erkannt)
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
            # Roboflow liefert Center-Koordinaten → Umrechnung zu Eckpunkten
            x, y = det["x"], det["y"]
            w, h = det["width"], det["height"]

            boxes.append([
                x - w / 2, y - h / 2,
                x + w / 2, y + h / 2
            ])

            confidences.append(det["confidence"])

            # Klassenname → numerische ID
            class_name = det.get("class", "").lower()
            if class_name == "mercedes":
                class_ids.append(0)
            elif class_name == "porsche":
                class_ids.append(1)
            else:
                class_ids.append(2)  # unbekannt

        detections = sv.Detections(
            xyxy=np.array(boxes, dtype=float),
            confidence=np.array(confidences, dtype=float),
            class_id=np.array(class_ids, dtype=int),
        )


    # ---------- 4) TRACKING ----------
    # Verknüpft Detektionen über mehrere Frames hinweg
    detections = tracker.update_with_detections(detections)

    # Falls keine Tracker-IDs existieren → Initialisieren
    if detections.tracker_id is None:
        detections.tracker_id = np.zeros(len(detections), dtype=int)

    # Vereinfachte ID-Zuweisung für die Zähllogik
    for i in range(len(detections)):
        if detections.class_id[i] == 0:
            detections.tracker_id[i] = 0
        elif detections.class_id[i] == 1:
            detections.tracker_id[i] = 1
        else:
            detections.tracker_id[i] = -1


    # ---------- 5) LINIENÜBERQUERUNG ----------
    crossed_in, crossed_out = line_zone.trigger(detections)


    # ---------- 6) RUNDENZÄHLUNG ----------
    for i in range(len(detections)):
        if crossed_in[i] or crossed_out[i]:
            if detections.tracker_id[i] == 0:
                mercedes_laps += 1
            elif detections.tracker_id[i] == 1:
                porsche_laps += 1


    # ---------- 7) LABELS ----------
    labels = []
    for tid in detections.tracker_id:
        if tid == 0:
            labels.append("MERCEDES")
        elif tid == 1:
            labels.append("PORSCHE")
        else:
            labels.append("CAR")


    # ---------- 8) DARSTELLUNG ----------
    annotated = frame
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels)
    annotated = line_annotator.annotate(annotated, line_zone)

    # UI-Text (Rundenzähler)
    cv2.putText(annotated, f"Mercedes: {mercedes_laps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(annotated, f"Porsche: {porsche_laps}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow("RaceCount", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        exit(0)
# ==================================================


# ================== PIPELINE START ==================
cv2.namedWindow("RaceCount", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RaceCount", 1280, 720)
cv2.setMouseCallback("RaceCount", mouse_callback)

print("Initialisiere Pipeline...")
print("Modell:", MODEL_ID)
print("Kamera:", VIDEO_REFERENCE)

pipeline = InferencePipeline.init(
    model_id=MODEL_ID,
    api_key=ROBOFLOW_API_KEY,
    video_reference=VIDEO_REFERENCE,
    on_prediction=on_prediction,
    max_fps=30,
    confidence=0.5
)

print("Pipeline gestartet")
print("→ 2 Klicks setzen die Ziellinie")
print("→ Taste 'q' beendet das Programm")

pipeline.start()
pipeline.join()
cv2.destroyAllWindows()
# ===================================================
