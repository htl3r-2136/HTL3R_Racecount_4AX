# Importiert die InferencePipeline-Klasse aus dem inference-Modul
# Diese Klasse wird verwendet, um ein Echtzeit-Inferenz-Pipeline-System zu erstellen
from inference import InferencePipeline

# Importiert die Funktion render_boxes
# Diese Funktion zeichnet die erkannten Bounding Boxes direkt ins Videobild
from inference.core.interfaces.stream.sinks import render_boxes

# Dein persönlicher Roboflow API-Key
# Wird benötigt, um Zugriff auf dein Modell in der Cloud zu bekommen
ROBOFLOW_API_KEY = "c3P356etbTH7VIVpbpCk"

# ID deines trainierten Roboflow-Modells
# Format: "projektname/version"
MODEL_ID = "my-first-project-rj433/6"

# Videoquelle:
# 0 oder 1 = Webcam (je nach System)
# Alternativ könnte hier auch ein Pfad zu einer Videodatei stehen
VIDEO_REFERENCE = 1

# Initialisiert die InferencePipeline
# - model_id: gibt an, welches Modell verwendet wird
# - api_key: authentifiziert dich bei Roboflow
# - video_reference: bestimmt die Videoquelle
# - on_prediction: Funktion, die bei jeder Vorhersage aufgerufen wird
pipeline = InferencePipeline.init(
    model_id=MODEL_ID,
    api_key=ROBOFLOW_API_KEY,
    video_reference=VIDEO_REFERENCE,
    on_prediction=render_boxes
)

# Startet die Pipeline
# Ab jetzt wird das Video gelesen und das Modell macht Vorhersagen
pipeline.start()

# Wartet, bis die Pipeline beendet wird
# Verhindert, dass das Programm sofort endet
pipeline.join()
