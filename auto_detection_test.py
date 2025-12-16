from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

ROBOFLOW_API_KEY = "c3P356etbTH7VIVpbpCk"
MODEL_ID = "my-first-project-rj433/6"
VIDEO_REFERENCE = 1

pipeline = InferencePipeline.init(
    model_id=MODEL_ID,
    api_key=ROBOFLOW_API_KEY,
    video_reference=VIDEO_REFERENCE,
    on_prediction=render_boxes)

pipeline.start()
pipeline.join()
