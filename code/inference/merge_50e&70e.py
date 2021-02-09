import numpy as np
import os
import json
import glob

# Need to set: Line 7
TEST_IMGS = '../../tcdata/tile_round1_testB_20210128/testB_imgs/*'
BEFORE_RES1_PATH = '../../user_data/tmp_data/70e/result0.json'  # before 'result.json'
BEFORE_RES2_PATH = '../../user_data/tmp_data/result0.json'  # before 'result.json'
RESULT_PATH = '../../prediction_result/B_50e+70e/result.json'  # 最终提交结果文件
CLASSES = 6


# nms
def nms(dets, thresh=0.3):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def merger_json():
    th_nms = 0.5
    vote = False
    th_vote = 0.5

    result = []

    with open(BEFORE_RES1_PATH, 'r') as load_f:
        json_data_1 = json.load(load_f)
    with open(BEFORE_RES2_PATH, 'r') as load_f:
        json_data_3 = json.load(load_f)

    img_list = glob.glob(TEST_IMGS)

    for _ in img_list:
        name = os.path.basename(_)
        boxes = []

        for box in json_data_1:
            img_name = box['name']
            if name != img_name:
                continue
            else:
                category = box['category']
                bbox = box['bbox']
                score = box['score']
                boxes.append(bbox + [score] + [category])
        for box in json_data_3:
            img_name = box['name']
            if name != img_name:
                continue
            else:
                category = box['category']
                bbox = box['bbox']
                score = box['score']
                boxes.append(bbox + [score] + [category])
        if boxes:
            boxes = np.array(boxes)
            for kind in range(1, CLASSES + 1, 1):
                boxes_kind = boxes[boxes[:, -1] == kind][:, :-1]
                order_nms = nms(boxes_kind, thresh=th_nms)
                boxes_kind = boxes_kind[order_nms]

                if vote:
                    boxes_kind_raw = boxes[boxes[:, -1] == kind][:, :-1]
                    boxes_kind = box_voting(boxes_kind, boxes_kind_raw, th_vote)

                if len(boxes_kind) <= 0:
                    continue
                for box in boxes_kind:
                    l = {'name': name,
                         'category': kind,
                         'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                         'score': float(box[4])}
                    result.append(l)

    with open(RESULT_PATH, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))


def box_iou_vote(b1, b2):
    b1 = np.expand_dims(b1, -2)
    b2 = np.expand_dims(b2, 0)

    b1_mins = b1[..., :2]
    b1_maxes = b1[..., 2:4]
    b1_wh = b1[..., 2:4] - b1[..., 0:2]

    b2_mins = b2[..., :2]
    b2_maxes = b2[..., 2:4]
    b2_wh = b1[..., 2:4] - b1[..., 0:2]

    intersect_mins = np.maximum(b1_mins, b2_mins)
    intersect_maxes = np.minimum(b1_maxes, b2_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def box_voting(boxes, boxes_all, th_vote):
    mask_iou = box_iou_vote(boxes, boxes_all)
    mask_iou = mask_iou >= th_vote

    for i, box in enumerate(boxes):
        boxes_sample = boxes_all[mask_iou[i]]
        boxes_sample = boxes_sample[:, :4] * boxes_sample[:, -1:] / np.sum(boxes_sample[:, -1:])
        boxes[i, :4] = np.sum(boxes_sample[:, :4], axis=0)

    return boxes


if __name__ == '__main__':
    merger_json()
