import os
import cv2
import numpy as np
from torch.utils.data import Dataset


def generate_one_hot_mask(mask, label_info):
    height, width = mask.shape
    num_classes = len(label_info)
    one_hot_mask = np.zeros((height, width, num_classes), dtype=np.float64)

    for class_idx in range(num_classes):
        label = label_info[class_idx][0].split(' ', 1)
        one_hot_mask[:, :, class_idx][mask == int(label[1])] = 1

    return one_hot_mask


class SegDataset(Dataset):
    def __init__(self, file_list_path, data_dir, transform=None):
        with open(file_list_path, 'r', encoding='utf-8') as f:
            self.file_list = f.read().splitlines()

        self.data_dir = data_dir
        self.transform = transform

        mean = (61, 61, 61)
        std = (36, 36, 36)
        self.mean = np.array(mean, dtype=np.float32) / 255.0
        self.std = np.array(std) / 255.0

        self.label_info = []
        label_file_path = os.path.join(data_dir, 'label_names.txt')
        with open(label_file_path, 'r') as label_file:
            for line in label_file.readlines():
                self.label_info.append(list(line.strip('\n').split(',')))

    def __getitem__(self, index):
        file_name = self.file_list[index]

        image_path = os.path.join(self.data_dir, file_name + '.jpg')
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        if image is None:
            print(f'Error loading image: {image_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.data_dir, file_name + '_mask.png')
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        one_hot_mask = generate_one_hot_mask(mask, self.label_info)

        if self.transform:
            augmented = self.transform(image=image, mask=one_hot_mask)
            image = augmented['image']
            one_hot_mask = augmented['mask'][0, :, :, :].permute(2, 0, 1)

        return image, one_hot_mask

    def __len__(self):
        return len(self.file_list)
