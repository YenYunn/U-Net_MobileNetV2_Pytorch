import os
import cv2
import numpy as np
import onnxruntime

INPUT_IMAGE_WIDTH = 320
INPUT_IMAGE_HEIGHT = 320


def load_model(model_path):
    providers_config = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    return onnxruntime.InferenceSession(model_path, providers=providers_config)


def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


def predict_mask(image, model):
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    image = preprocess_image(image)
    pr_mask = model.run([output_name], {input_name: image})[0]

    return np.squeeze(pr_mask).round()


def visualize_results(image, mask):
    pred_mask_show = (mask * 255).astype(np.uint8)
    mask_rgb = cv2.merge([pred_mask_show, np.zeros_like(pred_mask_show), np.zeros_like(pred_mask_show)])

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)

    cv2.imshow('Prediction Mask', pred_mask_show)
    cv2.imshow('Overlay Result', blended)
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)


def main():
    model_path = 'model/onnx/guidewire_track_simplified.onnx'
    image_path = 'data/images/0000.jpg'

    model = load_model(model_path)

    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    image = cv2.resize(image, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))

    mask = predict_mask(image, model)
    visualize_results(image, mask)


if __name__ == '__main__':
    main()
