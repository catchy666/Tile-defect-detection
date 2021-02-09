import json
import os
import cv2
import time
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from pycocotools.coco import COCO
from argparse import ArgumentParser

"""
1: "边异常",
2: "角异常",
3: "白色点瑕疵",
4: "浅色块瑕疵",
5: "深色点块瑕疵",
6: "光圈瑕疵",
7: "记号笔",
8: "划伤"
"""

RESULT_JSON = json.load(open('../../prediction_result/result.json', 'r'))


def bbox_result(img_name, res):
    anns = list()
    image = cv2.imread(img_name)
    for _ in res:
        if _['name'] == os.path.basename(img_name):
            anns.append(_)
    for ann in anns:
        x0, y0, x1, y1 = ann['bbox']
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255))
        cv2.putText(image,
                    str(ann['category']),
                    (x0, y0 - 2),
                    cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=2, color=(0, 0, 255))
    cv2.imwrite('_truth.jpg', image)


def main():
    parser = ArgumentParser(description='Draw test result for single img.')
    parser.add_argument('--img',
                        help='image file path',
                        default=os.path.join('../../tcdata/tile_round2xxxxxxxxx',
                                             'xxxxxxxxx.jpg')
                        )
    args = parser.parse_args()
    bbox_result(img_name=args.img, res=RESULT_JSON)


if __name__ == '__main__':
    main()
