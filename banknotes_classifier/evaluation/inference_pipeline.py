import os
import onnxruntime as ort
import cv2
import numpy as np

class InferencePipeline:
    def __init__(self, model_path):
        self.model = ort.InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

    def __call__(self, image):
        image = self._prepare_input(image)
        output = self.predict(image)
        return output.max().item(), output.argmax().item()

    def _prepare_input(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image):
        output = self.model.run([self.output_name], {self.input_name: image})[0]
        return output
