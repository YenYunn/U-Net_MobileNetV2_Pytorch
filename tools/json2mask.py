# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:28:14 2020

@author: Zhao Chunlin
解析json文件。
"""
import os
import json
import cv2
import numpy as np
import copy


def parseJson(path, labels):
    fp = open(path)
    impath = path.split('.')[0] + '.jpg'
    im = cv2.imread(impath)
    jsondata = json.load(fp)
    fp.close()
    num = len(labels)
    step = 255 // (num + 1)
    shapes = jsondata['shapes']
    h = jsondata['imageHeight']
    w = jsondata['imageWidth']
    mask = np.zeros((h, w), dtype=np.uint8)
    output = []
    for shape in shapes:
        point = shape['points']
        box = np.array(point, np.int32)
        idx = labels.index(shape['label'])
        value = (idx + 1) * step
        print("label: %s, label value:%d" % (shape['label'], value))
        output.append([shape['label'], value])
        cv2.fillPoly(mask, [box], idx + 1)

        # if shape['label'] == 'sandtube':
        #     cv2.fillPoly(mask, [box], 50)    ###根据自己的需求修改数值。
        # elif shape['label'] == 'rail':
        #     cv2.fillPoly(mask, [box], 100)
        # else:
        #     cv2.fillPoly(mask, [box], 150)

    return mask, output, im


def getLabel(path, jsonlist):
    labels = []
    for name in jsonlist:
        jsonpath = path + name
        fp = open(jsonpath)
        jsondata = json.load(fp)
        fp.close()
        shapes = jsondata['shapes']
        for shape in shapes:
            if shape['label'] not in labels:
                labels.append(shape['label'])
    return labels


if __name__ == "__main__":
    path = r"D:\zhengrui\jpg(1)"
    outfolder = r"D:\zhengrui\jpg(1)"
    jsonlist = []
    for file in os.listdir(path):
        if file[-4:] == 'json':
            jsonlist.append(file)

    # labels = getLabel(path, jsonlist)
    labels = ['brake_a', 'brake_b', 'brake_c', 'brake_d', 'brake_e', 'brake_f']
    print(labels)
    for name in jsonlist:
        jsonpath = os.path.join(path, name)
        print(jsonpath)
        mask, output, im = parseJson(jsonpath, labels)
        outpath = os.path.join(outfolder, name.split('.')[0] + '_mask.png')
        outim = os.path.join(outfolder, name.split('.')[0] + '.jpg')
        cv2.imwrite(outpath, mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(outim, im)

        zero_mask = np.zeros(shape=mask.shape, dtype=np.uint8)
        mask_copy = copy.deepcopy(mask)
        mask_rgb = cv2.merge([mask_copy * 255, zero_mask, zero_mask])
        mask_val = cv2.addWeighted(im, 0.7, mask_rgb, 0.3, 0)

        outmask = os.path.join(outfolder, name.split('.')[0] + '_val.jpg')
        cv2.imwrite(outmask, mask_val)
