import numpy as np
import cv2
import onnxruntime as ort 


class InferencePipeline:
    def __init__(self, model_path):
        self.model = ort.InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.mapping = {
            0: "1 pound face",
            1: "1 pound back",
            2: "5 pounds face",
            3: "5 pounds back",
            4: "10 pounds face",
            5: "10 pounds back",
            6: "20 pounds face",
            7: "20 pounds back",
            8: "50 pounds face",
            9: "50 pounds back",
            10: "100 pounds face",
            11: "100 pounds back",
            12: "200 pounds face",
            13: "200 pounds back",
        }

    def __call__(self, image):
        image = self._prepare_input(image)
        output = self.predict(image)
        predicted_class = self.mapping[output.argmax().item()]
        predicted_confidence = output[0][output.argmax().item()] * 100
        return f"The model is {predicted_confidence:.2f}% confident that the image is {predicted_class}"

    def _prepare_input(self, image):
        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image):
        output = self.model.run([self.output_name], {self.input_name: image})[0]
        return output