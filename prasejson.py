import os
import json
import cv2
import numpy as np


def parse_json(json_path, labels):
    with open(json_path, encoding='gb18030', errors='ignore') as file:
        json_data = json.load(file)

    image_path = next(
        (p for p in [json_path.replace('.json', ext) for ext in ['.jpg', '.png']] if os.path.exists(p)),
        None
    )
    if image_path is None:
        return None, None

    height, width = json_data['imageHeight'], json_data['imageWidth']
    mask = np.zeros((height, width), dtype=np.uint8)

    num_labels = len(labels)
    step = 255 // (num_labels + 1)

    label_map = {label: (idx + 1) * step for idx, label in enumerate(labels)}

    for shape in json_data['shapes']:
        label = shape['label']
        if label not in label_map:
            continue

        points = np.array(shape['points'], np.int32)
        cv2.fillPoly(mask, [points], label_map[label])

    return mask, image_path


def process_json_files(image_folder, labels):
    json_files = [f for f in os.listdir(image_folder) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(image_folder, json_file)
        mask, image_path = parse_json(json_path, labels)

        if mask is not None:
            mask_path = json_path.replace('.json', '_mask.png')
            cv2.imencode('.png', mask)[1].tofile(mask_path)

            print(f'Generated mask: {mask_path}')


def clean_unlabeled_images(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

    for image_file in image_files:
        json_path = os.path.join(image_folder, image_file.rsplit('.', 1)[0] + '.json')
        if not os.path.exists(json_path):
            os.remove(os.path.join(image_folder, image_file))
            print(f'Removed unlabeled image: {image_file}')


if __name__ == '__main__':
    image_folder = 'data/test'
    labels = ['guidewire']

    clean_unlabeled_images(image_folder)
    process_json_files(image_folder, labels)
