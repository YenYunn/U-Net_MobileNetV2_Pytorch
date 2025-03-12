import os
import cv2
import json
import torch
import base64
import numpy as np
from torchvision import transforms

INPUT_IMAGE_WIDTH = 320
INPUT_IMAGE_HEIGHT = 320
DEFAULT_X_START = 640
DEFAULT_CROP_WIDTH = 640


def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def extract_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_points = []
    for contour in contours:
        points = [[float(point[0][0]), float(point[0][1])] for point in contour]
        if points:
            all_points.append(points)
    return all_points


def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image):
    _, width = image.shape[:2]
    if width >= DEFAULT_X_START + DEFAULT_CROP_WIDTH:
        crop_image_original = image[:, DEFAULT_X_START:DEFAULT_X_START + DEFAULT_CROP_WIDTH]
    else:
        crop_image_original = image

    img = cv2.cvtColor(crop_image_original, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
    transform = transforms.Compose([transforms.ToTensor()])
    input_img = transform(img_resized).unsqueeze(0)

    return crop_image_original, input_img


def predict_mask(image, model, device):
    height, width = image.shape[:2]
    crop_image_original, input_img = preprocess_image(image)
    input_img = input_img.to(device)

    with torch.no_grad():
        pr_mask = model.predict(input_img)

    pr_mask_img = ((pr_mask.squeeze().cpu().numpy() >= 0.1) * 255).astype(np.uint8)
    pr_mask_img = cv2.resize(pr_mask_img, (crop_image_original.shape[1], crop_image_original.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    full_mask = np.zeros((height, width), dtype=np.uint8)

    x_start = DEFAULT_X_START
    crop_width = crop_image_original.shape[1]

    if width >= x_start + crop_width:
        full_mask[:, x_start:x_start + crop_width] = pr_mask_img
    else:
        full_mask[:, -crop_width:] = pr_mask_img

    return image, full_mask


def generate_label_json(image, mask, image_path):
    height, width = image.shape[:2]
    base64_str = encode_image_to_base64(image)
    contours = extract_contours(mask)

    shapes = []
    for points in contours:
        shapes.append({
            'label': 'guidewire',
            'points': points,
            'group_id': None,
            'description': '',
            'shape_type': 'polygon',
            'flags': {},
            'mask': None
        })

    label_data = {
        'version': '5.5.0',
        'flags': {},
        'shapes': shapes,
        'imagePath': os.path.basename(image_path),
        'imageData': base64_str,
        'imageHeight': height,
        'imageWidth': width
    }
    return label_data


def save_label_json(label_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(label_data, f, indent=4)


def process_folder(image_folder, model, device, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff'))]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        if image is None:
            print(f'Error loading image: {image_path}')
            continue

        full_image, full_mask = predict_mask(image, model, device)
        label_data = generate_label_json(full_image, full_mask, image_path)

        output_json_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.json')
        save_label_json(label_data, output_json_path)
        print(f'Label saved: {output_json_path}')


if __name__ == '__main__':
    image_folder = '../data/rawframes'
    model_path = '../runs/train/weights/best.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path, device)
    process_folder(image_folder, model, device, image_folder)
