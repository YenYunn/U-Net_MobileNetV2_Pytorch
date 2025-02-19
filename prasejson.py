# import os
# import json
# import cv2
# import numpy as np
#
#
# def parse_json(path, labels):
#     fp = open(path, encoding='gb18030', errors='ignore')
#     impath = path[:-5] + '.jpg'
#     if os.path.exists(impath):
#         im = cv2_imread(impath)
#     else:
#         impath = path[:-5] + '.png'
#         im = cv2_imread(impath)
#
#     json_data = json.load(fp)
#     fp.close()
#     num = len(labels)
#     step = 255 // (num + 1)
#     shapes = json_data['shapes']
#     h = json_data['imageHeight']
#     w = json_data['imageWidth']
#     mask = np.zeros((h, w), dtype=np.uint8)
#     output = []
#     for shape in shapes:
#         print(shape)
#         point = shape['points']
#
#         box = np.array(point, np.int32)
#
#         idx = 0
#         value = (idx + 1) * step
#
#         output.append([shape['label'], value])
#         cv2.fillPoly(mask, [box], (idx + 1))
#
#     return mask, output
#
#
# def getLabel(path, jsonlist):
#     labels = []
#     for name in jsonlist:
#         jsonpath = path + name
#         fp = open(jsonpath)
#         json_data = json.load(fp)
#         fp.close()
#         shapes = json_data['shapes']
#         for shape in shapes:
#             if shape['label'] not in labels:
#                 labels.append(shape['label'])
#     return labels
#
#
# def cv2_imread(file_path, color=1):
#     if color == 0:
#         cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
#
#     else:
#         cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
#     return cv_img
#
#
# def cv2_imwrite(file_path, img):
#     cv2.imencode('.png', img)[1].tofile(file_path)
#
#
# if __name__ == '__main__':
#     path = 'data/images/'
#     for file in os.listdir(path):
#         if file[-4:] == '.jpg' or file[-4:] == '.png':
#             if not os.path.exists(path + '/' + file[:-4] + '.json'):
#                 os.remove(path + '/' + file)
#     outfolder = path
#     jsonlist = []
#     for file in os.listdir(path):
#         print(file)
#         if file[-4:] == 'json':
#             jsonlist.append(file)
#
#     # labels = getLabel(path, jsonlist)
#     labels = ['guidewire', 'condyle-R', 'cza', 'brake_e', 'break_f', 'break_h', 'brake_g']
#     # labels = ['brake_g']
#     for name in jsonlist:
#         jsonpath = path + name
#         print(jsonpath)
#         # try:
#         if parse_json(jsonpath, labels):
#             mask, output = parse_json(jsonpath, labels)
#             outpath = outfolder + name[:-5] + '_mask.png'
#             print(outpath)
#             outim = outfolder + name.split('.')[0] + '.jpg'
#
#             cv2_imwrite(outpath, mask)


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
