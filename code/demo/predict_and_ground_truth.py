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

CONFIG = '../config/cascade_rcnn_r50_fpn_70e_coco.py'
CHECKPOINTS = '../../user_data/model_data/cascade_rcnn_r50_fpn_70e/latest.pth'
ANN_FILE = '../../tcdata/tile_round2_train/annotations/instances_train.json'


def predict(img_name):
    model = init_detector(CONFIG, CHECKPOINTS)
    result = inference_detector(model, img_name)
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img_name, result, score_thr=0.3, show=False, out_file='./_predict.jpg')


def draw_gt(img_name):
    coco = COCO(ANN_FILE)
    ann_ids = coco.getAnnIds(imgIds=get_img_id(ANN_FILE, os.path.basename(img_name)), iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    image = cv2.imread(img_name)
    for i in range(len(anns)):
        x, y, w, h = anns[i]['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255))
        cv2.putText(image,
                    str(anns[i]['category_id']),
                    (x, y - 2),
                    cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=2, color=(0, 0, 255))
    cv2.imwrite('_truth.jpg', image)


def get_img_id(ann, name):
    with open(ann, 'r') as f:
        res = json.load(f)
    for img in res["images"]:
        if name == img["file_name"]:
            return img["id"]


def main():
    parser = ArgumentParser(description='Predicting & Drawing Ground Truth')
    parser.add_argument('--img', help='image file path',
                        default=os.path.join('../../tcdata/tile_round2_train/train_imgs',
                                             '271_180_t20201215152001656_CAM1_1.jpg')
                        )
    args = parser.parse_args()
    # model predict result
    predict(img_name=args.img)
    # bbox ground truth
    draw_gt(img_name=args.img)


if __name__ == '__main__':
    main()
