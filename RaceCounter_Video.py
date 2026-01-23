"""
Racecount - Video-basierte Rundenzählung für Slotcars.

Dieses Skript analysiert ein Rennvideo, erkennt Slotcars (Mercedes & Porsche)
mithilfe eines Roboflow-Objekterkennungsmodells und zählt automatisch die Runden.

Eine Runde wird gezählt, sobald ein Fahrzeug eine manuell gesetzte Ziellinie
überquert. Tracking (ByteTrack) verhindert Mehrfachzählungen desselben Fahrzeugs.
"""

# ================== METADATEN ==================
__name__ = "Racecount"
__license__ = "GNUv3"
__author__ = "Nikola Cajic, Theo Hubinger"
__repository__ = "https://github.com/htl3r-2136/HTL3R_Racecount_4AX"
# ===============================================

# ================== IMPORTS =====================
import cv2                      # OpenCV → Video, Fenster, Maus & Tastatur
import numpy as np               # Numerische Arrays für Bounding Boxes
import supervision as sv         # Tracking & Visualisierung (ByteTrack, Annotatoren)
from inference import InferencePipeline   # Roboflow Inference Pipeline
from inference.core.interfaces.camera.entities import VideoFrame
# ===============================================


# ================== KONFIGURATION ==================
# Roboflow API-Key (Zugriff auf das trainierte Modell)
ROBOFLOW_API_KEY = "c3P356etbTH7VIVpbpCk"

# Modell-ID im Format: projektname/version
MODEL_ID = "my-first-project-rj433/8"

# Pfad zur Video-Datei (lokal)
VIDEO_PATH = r"C:\Users\nikol\OneDrive\Desktop\Racecount_MER.mp4"
# ===================================================


# ================== GLOBALE ZUSTÄNDE ==================
# Punkte zur Definition der Ziellinie (per Mausklick)
line_points = []

# True, sobald zwei Punkte gesetzt wurden
line_ready = False

# LineZone-Objekt von Supervision (für Überquerungslogik)
line_zone = None

# Startzustand: Video pausiert (warten auf Linien-Set)
video_paused = True

# Rundenzähler pro Fahrzeugtyp
mercedes_laps = 0
porsche_laps = 0
# =====================================================


# ================== TRACKING & ANNOTATION ==================
# ByteTrack sorgt dafür, dass Fahrzeuge über mehrere Frames
# hinweg eine eindeutige ID behalten
tracker = sv.ByteTrack()

# Zeichnet abgerundete Bounding Boxes
box_annotator = sv.RoundBoxAnnotator()

# Zeichnet Textlabels über den Boxen
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
    Erlaubt dem Benutzer, die Ziellinie per Maus festzulegen.

    - Erster Linksklick: Startpunkt
    - Zweiter Linksklick: Endpunkt
    - Danach startet das Video automatisch
    """
    global line_points, line_ready, video_paused

    # Reagiere nur auf Linksklicks, solange die Linie nicht fertig ist
    if event == cv2.EVENT_LBUTTONDOWN and not line_ready:
        line_points.append((x, y))
        print("Linienpunkt gesetzt:", (x, y))

        # Sobald zwei Punkte existieren → Linie fertig
        if len(line_points) == 2:
            line_ready = True
            video_paused = False
            print("Ziellinie gesetzt → VIDEO START")
# ==================================================


# ================== PIPELINE CALLBACK ==================
def on_prediction(predictions, video_frame: VideoFrame):
    """
    Zentrale Callback-Funktion der Inference Pipeline.

    Wird für jedes Frame aufgerufen und übernimmt:
    - Tasteneingaben (Pause / Quit)
    - Detektionsverarbeitung
    - Tracking
    - Linienüberquerung
    - Rundenzählung
    - Darstellung (UI)
    """
    global line_zone, mercedes_laps, porsche_laps, video_paused

    # Aktuelles Frame aus dem Video extrahieren
    frame = video_frame.image

    # ---------- TASTENABFRAGE ----------
    key = cv2.waitKey(1) & 0xFF

    # Leertaste → Pause / Play
    if key == ord(" "):
        video_paused = not video_paused
        print("PAUSE" if video_paused else "PLAY")

    # Q → Programm sofort beenden
    if key == ord("q"):
        exit(0)

    # ---------- PAUSENMODUS ----------
    if video_paused:
        preview = frame.copy()

        # Visualisierung der gesetzten Linienpunkte
        if len(line_points) >= 1:
            cv2.circle(preview, line_points[0], 5, (0, 255, 0), -1)
        if len(line_points) == 2:
            cv2.circle(preview, line_points[1], 5, (0, 0, 255), -1)
            cv2.line(preview, line_points[0], line_points[1], (255, 0, 0), 2)

        # Falls LineZone existiert → anzeigen
        if line_zone is not None:
            preview = line_annotator.annotate(preview, line_zone)

        # Pause-Text einblenden
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
        return  # Verarbeitung für dieses Frame abbrechen

    # ---------- ZIELLINIE INITIALISIEREN ----------
    # Wird nur einmal erstellt, sobald zwei Punkte gesetzt sind
    if line_zone is None and line_ready:
        start = sv.Point(*line_points[0])
        end = sv.Point(*line_points[1])
        line_zone = sv.LineZone(start=start, end=end)

    # ---------- DETEKTIONSVERARBEITUNG ----------
    # Rohdaten von Roboflow
    det_list = predictions.get("predictions", []) if predictions else []

    if not det_list:
        detections = sv.Detections.empty()
    else:
        boxes, confs, cids = [], [], []

        for det in det_list:
            # Mittelpunkt + Breite/Höhe → xyxy Bounding Box
            x, y, w, h = det["x"], det["y"], det["width"], det["height"]
            boxes.append([x - w/2, y - h/2, x + w/2, y + h/2])
            confs.append(det["confidence"])

            # Klassenzuordnung
            cls = det.get("class", "").lower()
            if cls == "mercedes":
                cids.append(0)
            elif cls == "porsche":
                cids.append(1)
            else:
                cids.append(2)

        # Erstellung des Supervision-Detections-Objekts
        detections = sv.Detections(
            xyxy=np.array(boxes, float),
            confidence=np.array(confs, float),
            class_id=np.array(cids, int)
        )

    # ---------- TRACKING ----------
    # Verbindet aktuelle Detektionen mit bestehenden Tracks
    detections = tracker.update_with_detections(detections)

    # ---------- LINIENÜBERQUERUNG ----------
    crossed_in, crossed_out = line_zone.trigger(detections)

    # Rundenzählung pro Fahrzeugtyp
    for i in range(len(detections)):
        if crossed_in[i] or crossed_out[i]:
            if detections.class_id[i] == 0:
                mercedes_laps += 1
            elif detections.class_id[i] == 1:
                porsche_laps += 1

    # ---------- LABELS ----------
    labels = [
        "MERCEDES" if cid == 0 else "PORSCHE" if cid == 1 else "CAR"
        for cid in detections.class_id
    ]

    # ---------- DARSTELLUNG ----------
    annotated = frame.copy()
    annotated = line_annotator.annotate(annotated, line_zone)
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels)

    # Rundenzähler einblenden
    cv2.putText(annotated, f"Mercedes: {mercedes_laps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"Porsche: {porsche_laps}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("RaceCount", annotated)
# =======================================================


# ================== FENSTER & PIPELINE ==================
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
# =======================================================
