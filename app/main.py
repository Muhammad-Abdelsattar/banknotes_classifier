from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from inference import InferencePipeline


app = FastAPI()

pipeline = InferencePipeline("model.onnx")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    output = pipeline(image)
    return JSONResponse(content={"prediction": output})