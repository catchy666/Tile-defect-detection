import json
from tqdm import tqdm

"""
二次过滤：根据单个图片bbox最高score来对图像进行二次过滤, 判断该图像是否是正常样本
"""
# Need to set: Line 8, Line 11
test_json_raw = json.load(
    open("../../tcdata/tile_round1_testB_20210128/annotations/instances_test_crop_2000.json", "r"))
test_json = json.load(open('../../user_data/tmp_data/tile_result.bbox.json', "r"))
CROP_RES_PATH = '../../user_data/tmp_data/crop_result.json'  # 切块测试集的测试结果 --> json2submit

images_ids = {}
for img in test_json_raw["images"]:
    images_ids[img["id"]] = img["file_name"]  # image_ids['20210000001']='197_103_t20201119094043570_CAM1.jpg'

img_scores = dict()
for ann in tqdm(test_json):
    img_id = ann["image_id"]
    score = ann["score"]
    if img_id not in img_scores.keys():
        img_scores[img_id] = score
    else:
        if score > img_scores[img_id]:
            img_scores[img_id] = score

results = []
for ann in tqdm(test_json):
    img_id = ann["image_id"]
    if img_scores[img_id] > 0.1:
        label = ann["category_id"]
        bbox = ann["bbox"]
        filename = images_ids[img_id].split('/')[-1]
        w, h = bbox[2], bbox[3]
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + w
        ymax = bbox[1] + h
        score = ann["score"]
        results.append(
            {'name': filename, 'category': int(label), 'bbox': [xmin, ymin, xmax, ymax], 'score': float(score)})

with open(CROP_RES_PATH, 'w') as fp:
    json.dump(results, fp, indent=4, separators=(',', ': '))
