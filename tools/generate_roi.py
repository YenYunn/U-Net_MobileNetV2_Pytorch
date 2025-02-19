# _*_ encoding    : utf-8 _*_
# @Author         : Karon
# @Author EMail   : zhjr2020@qq.com
# @File           : generate_roi.py
# @Created Date   : 2021年04月20日 13:51:37
# @Change Activity: 2021年04月20日
# @Modified By    : Karon
# @Software       : PyCharm
# @version        : Python 3.9, PyTorch 1.8.1, TensorFlow 2.5.0
"""
===========================================================================================================
关键:
内容: 根据txt边框坐标提取边框区域
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

    return list(map(round, [x_min, y_min, x_max, y_max]))


def get_roi(root_path: str, output_path: str) -> None:
    """
    :param output_path:
    :param root_path:
    :return:
    """
    for root, dirs, files in os.walk(root_path):
        with tqdm(total=len(files)) as pbar:
            for file in files:
                if '.txt' in file:  # 一个txt文件
                    # print(file)

                    brake_img = cv2.imread(os.path.join(root, file[:-4] + '.jpg'))  # 一张图片
                    img_h, img_w, channels = brake_img.shape

                    index = 0
                    with open(os.path.join(root, file), mode="r") as txt_file:
                        for cxywh in txt_file.readlines():
                            list_cxywh = cxywh.strip().split(' ')  # cxywh 对应一个框
                            list_cxywh[0] = list(map(int, cxywh.strip().split(" ")[0]))[0]
                            list_cxywh[1:] = list(map(float, cxywh.strip().split(" ")[1:]))

                            x1_min, y1_min, x1_max, y1_max = xywh2xyxy(list_cxywh[1:], img_w, img_h)

                            roi_brake = brake_img[y1_min:y1_max, x1_min:x1_max]

                            # cv2.imshow('brake', roi_brake)
                            # cv2.waitKey()
                            # cv2.destroyAllWindows()

                            cv2.imwrite(filename=os.path.join(output_path, f"{file[:-4]}_{index}" + '.jpg'), img=roi_brake)

                            index += 1

                pbar.update(1)
                pbar.set_description(desc=f'{file}')


if __name__ == '__main__':
    root_path = 'xxx'
    output_path = 'xxx'

    get_roi(root_path, output_path)
