import io
import json
from PIL import Image
from fastapi import File, FastAPI, UploadFile
import torch


model = torch.hub.load('./yolov5', 'custom', path='./static/best.pt', source='local' ,force_reload=True) 

app = FastAPI()

@app.post("/objectdetection/")
async def objectdetection(file: UploadFile = File(...)):
    contents = await file.read() # read the contents of the UploadFile object
    input_image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = model(input_image)
    results_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    return {"result": results_json}
