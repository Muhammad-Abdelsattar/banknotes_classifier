import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from inference import InferencePipeline

app = FastAPI()

pipeline = InferencePipeline("model.onnx")

async def log_request(image):
    logs_dir = "/teamspace/studios/this_studio/images_requests"
    idx = len(os.listdir(logs_dir))
    cv2.imwrite(os.path.join(logs_dir,str(idx) + '.jpg'),image)

@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)) :
    image = await file.read()
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if image.shape[0] > image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    background_tasks.add_task(log_request, image)
    output = pipeline(image)
    return JSONResponse(content={"prediction": output})