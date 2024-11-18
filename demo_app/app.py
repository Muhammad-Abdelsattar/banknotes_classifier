import gradio as gr
import cv2
from inference import InferencePipeline

pipeline = InferencePipeline("artifacts/model.onnx")

def predict(image):
    if image.shape[0] > image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    return pipeline(image)


with gr.Blocks() as demo:
    gr.Markdown("# Banknotes Classifier",)
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Input Image")
            predict_button = gr.Button("Predict")
        with gr.Column():
            output = gr.Textbox(label="Prediction")
    predict_button.click(predict, inputs=image_input, outputs=output)

if __name__ == "__main__":
    demo.launch()