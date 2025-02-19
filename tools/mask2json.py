import os, cv2, json
import base64
from tqdm import tqdm


def polygon2quadrangle(points: list) -> list:
    """

    :param points: [[1,2],
                    [3, 8],
                    [5, 6],[2, 9], ]
    :return: [[1,2],
              [5, 8]]
    """
    xs = []
    ys = []
    for point in points:
        xs.append(point[0])
        ys.append(point[1])
    xs.sort()
    ys.sort()
    try:
        x1, x2, x3, x4 = xs[0], xs[1], xs[-2], xs[-1]
        y1, y2, y3, y4 = ys[0], ys[1], ys[-2], ys[-1]

        return [[x1, y1], [x3, y2], [x4, y4], [x2, y3]]
    except:
        return points


def polygon2polygon(points: list) -> list:
    """

    :param points:
    :return:
    """
    new_points = []
    i = 0
    while i < len(points):
        new_points.append(points[min(i, len(points))])
        i += 3

    return new_points


def mask2json(root_path: str) -> None:
    shape = {"label": None,
             "points": None,
             "group_id": None,  # null
             "shape_type": "polygon",
             "flags": {}
             }
    mask_dict = {"version": "4.5.6",
                 "flags": {},
                 "shapes": [],
                 "imagePath": None,
                 "imageData": None,
                 "imageHeight": None,
                 "imageWidth": None
                 }

    with open(os.path.join(root_path, "image2label.json"), mode="r", encoding='UTF-8') as json_file:
        image2label = json.loads(json_file.read())

    original_image_info_dict = {}  # 图片名称：imageData
    for root, dir, files in os.walk(root_path):
        with tqdm(total=len(files)) as pbar:
            for file in files:
                if '.jpg' in file:
                    with open(os.path.join(root_path, file), mode='rb') as img:
                        original_image_info_dict[file[:-4]] = base64.b64encode(img.read()).decode()

                pbar.update(1)
                pbar.set_description(desc=f"图片转二进制:{file:<24}")

    for root, dir, files in os.walk(root_path):
        with tqdm(total=len(files)) as pbar:
            for file in files:
                if '.png' in file:
                    mask = cv2.imread(os.path.join(root, file), flags=cv2.CV_8UC1)
                    _, mask = cv2.threshold(mask, thresh=0.1, maxval=1, type=cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_KCOS)

                    mask_dict["imagePath"] = file[:-4] + ".jpg"
                    mask_dict["imageData"] = original_image_info_dict[file[:-4]]
                    mask_dict["imageHeight"] = mask.shape[0]
                    mask_dict["imageWidth"] = mask.shape[1]

                    # cv2.drawContours(mask, contours, contourIdx=-1, color=(255, 0, 255), thickness=1)
                    # cv2.imshow("mask", mask)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()

                    for index in range(len(contours)):
                        contours_list = [i[0] for i in contours[index].tolist()]
                        shape["label"] = image2label[file[:-4]]
                        if shape["label"] == "brake_b":
                            shape["points"] = polygon2quadrangle(contours_list)
                        else:
                            # shape["points"] = polygon2polygon(contours_list)
                            shape["points"] = contours_list

                        mask_dict["shapes"] = []
                        mask_dict["shapes"].append(shape)

                    with open(os.path.join(root, file[:-4] + ".json"), mode="w", encoding="utf-8") as json_file:
                        json_file.truncate(0)
                        json_file.write(json.dumps(mask_dict, indent=2))

                pbar.update(1)
                pbar.set_description(desc=f"写入json文件:{file:<24}")


if __name__ == '__main__':
    mask2json(root_path=r"D:\zhjr\dataset\20210323\lvz_brake_zhapian\segment_UNet\Temporary\test_brakes_ori_mask")
