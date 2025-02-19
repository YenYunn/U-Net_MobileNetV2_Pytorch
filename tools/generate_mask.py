# _*_ encoding    : utf-8 _*_
# @Author         : Karon
# @Author EMail   : zhjr2020@qq.com
# @File           : generate_mask.py
# @Created Date   : 2021年04月20日 13:01:46
# @Change Activity: 2021年04月20日
# @Modified By    : Karon
# @Software       : PyCharm
# @version        : Python 3.9、PyTorch 1.8.1、TensorFlow 2.5.0
"""
===========================================================================================================
关键:
内容: 从原图中，根据边框txt文件截取闸片，截取出来的闸片与json文件一起生成mask
目标:
总结:
===========================================================================================================
"""
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def xywh2xyxy(xywh, img_w, img_h):
    """

    :param xywh: xywh = [0.541234, 0.585679, 0.161307, 0.447848]
    :param img_w:
    :param img_h:
    :return:
    """
    box_w = xywh[2] * img_w
    box_h = xywh[3] * img_h

    x_min = xywh[0] * img_w - box_w / 2
    y_min = xywh[1] * img_h - box_h / 2

    x_max = x_min + box_w
    y_max = y_min + box_h

    return round(x_min), round(y_min), round(x_max), round(y_max)


def polygon2box(polygon):
    """

    :param polygon: [[596, 575],
                     [597, 642],
                     [628, 641],
                     [625, 576]]
    :return:
    """
    xs = []
    ys = []

    for point in polygon:
        xs.append(point[0])
        ys.append(point[1])

    return min(xs), min(ys), max(xs), max(ys)


def get_roi_mask(root_path, brake_labels, output_path):
    """首先，根据yolo标签txt文件，从原图中截取闸片，然后再根据json文件生成闸片mask
    root_path
        |- 1.jpg
        |- 1.txt
        |- 1.json

    :param output_path:
    :param brake_labels:
    :param root_path:
    :return:
    """
    for root, dirs, files in os.walk(root_path):
        with tqdm(total=len(files)) as pbar:
            for file in files:
                if ('.txt' in file) and (file != 'classes.txt'):  # 一个txt文件
                    # print(file)

                    brake_img = cv2.imread(os.path.join(root, file[:-4] + '.jpg'))  # 一张图片
                    img_h, img_w, channels = brake_img.shape

                    with open(os.path.join(root, file), mode="r") as txt_file:
                        for cxywh in txt_file.readlines():
                            list_cxywh = cxywh.strip().split(" ")  # cxywh 对应一个框
                            list_cxywh[0] = list(map(int, cxywh.strip().split(" ")[0]))[0]
                            list_cxywh[1:] = list(map(float, cxywh.strip().split(" ")[1:]))
                            # print(list_cxywh)
                            x1_min, y1_min, x1_max, y1_max = xywh2xyxy(list_cxywh[1:], img_w, img_h)
                            # print(x1_min, y1_min, x1_max, y1_max)

                            roi_brake = brake_img[y1_min:y1_max, x1_min:x1_max]  # 一个闸片ROI

                            with open(os.path.join(root, file[:-4] + '.json')) as json_file:  # 对应图片的json文件
                                json_content = json.load(json_file)
                                brake_shapes = json_content['shapes']
                                # print(brake_shapes)
                                index = 0
                                for brake_shape in brake_shapes:
                                    brake_shape_points = brake_shape['points']
                                    # labelme标注的polygon转box
                                    x2_min, y2_min, x2_max, y2_max = polygon2box(brake_shape_points)

                                    # box: x1_min, y1_min, x1_max, y1_max
                                    # seg: x2_min, y2_min, x2_max, y2_max
                                    if (x1_min < x2_min) and (x1_max > x2_max) and (y1_min < y2_min) and (
                                            y1_max > y2_max):
                                        # cv2.imshow('brake', roi_brake)
                                        # cv2.waitKey()
                                        # cv2.destroyAllWindows()

                                        roi_brake_mask = np.zeros(shape=(roi_brake.shape[0], roi_brake.shape[1]),
                                                                  dtype=np.uint8)

                                        single_channel = np.zeros_like(roi_brake_mask, dtype=np.uint8)

                                        # cv2.imshow('brake_mask', roi_brake_mask)
                                        # cv2.waitKey()
                                        # cv2.destroyAllWindows()

                                        brake_mask = np.array(brake_shape_points, dtype=np.int32) - np.array(
                                            [x1_min, y1_min])
                                        # print(brake_mask)
                                        brake_label_index = brake_labels.index(brake_shape['label'])
                                        cv2.fillPoly(roi_brake_mask, pts=[brake_mask], color=brake_label_index + 1)

                                        roi_brake_mask1 = np.zeros_like(roi_brake_mask, dtype=np.uint8)
                                        cv2.fillPoly(roi_brake_mask1, pts=[brake_mask], color=brake_label_index + 210)

                                        # cv2.imshow('brake_mask', roi_brake_mask)
                                        # cv2.waitKey()
                                        # cv2.destroyAllWindows()

                                        roi_brake_mask_rgb = cv2.merge(
                                            [roi_brake_mask1, single_channel, single_channel])
                                        roi_brake_mask_val = cv2.addWeighted(roi_brake, 0.5, roi_brake_mask_rgb, 0.5, 0)

                                        cv2.imwrite(filename=os.path.join(output_path, f"{file[:-4]}_{index}" + '.jpg'),
                                                    img=roi_brake)
                                        cv2.imwrite(
                                            filename=os.path.join(output_path, f"{file[:-4]}_{index}" + '_mask.png'),
                                            img=roi_brake_mask, params=[int(cv2.IMWRITE_JPEG_QUALITY), 100])
                                        cv2.imwrite(
                                            filename=os.path.join(output_path, f"{file[:-4]}_{index}" + '_val.jpg'),
                                            img=roi_brake_mask_val)

                                        index += 1

                pbar.update(1)
                pbar.set_description(desc=f"{file}")


if __name__ == '__main__':
    get_roi_mask(root_path=r"D:\zhjr\dataset\20210323\lvz_brake_zhapian\segment_UNet\batch2\all_filter_brakes",
                 brake_labels=['brake_a', 'brake_b', 'brake_c', 'brake_d', 'brake_e', 'brake_f'],
                 output_path=r"D:\zhjr\dataset\20210323\lvz_brake_zhapian\segment_UNet\brake_train_seg"
                 )
