import json
from tqdm import tqdm
"""
切片bbox坐标转换为对应原图坐标
"""
# Need to set: Line 7, Line 8
CROP_RES_PATH = '../../user_data/tmp_data/crop_result.json'  # 切块测试集的测试结果 --> json2submit --> here
BEFORE_RES_PATH = '../../user_data/tmp_data/result0.json'  # before 'result.json'


def update(crop_result_data, step):
    result_info = []
    for info in tqdm(crop_result_data):
        to_map = {}
        img_name = info["name"]

        x_i = img_name.split('.jpg')[1].split('_')[1]
        y_i = img_name.split('.jpg')[1].split('_')[2]
        x_i, y_i = int(x_i), int(y_i)

        new_bbox_x_min = y_i * step + info["bbox"][0]
        new_bbox_x_max = y_i * step + info["bbox"][2]
        new_bbox_y_min = x_i * step + info["bbox"][1]
        new_bbox_y_max = x_i * step + info["bbox"][3]

        to_map["name"] = img_name.split('.')[0] + '.jpg'
        to_map["category"] = info["category"]
        to_map["bbox"] = [new_bbox_x_min, new_bbox_y_min, new_bbox_x_max, new_bbox_y_max]
        to_map["score"] = info["score"]
        result_info.append(to_map)

    with open(BEFORE_RES_PATH, 'w') as fp:
        json.dump(result_info, fp, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    with open(CROP_RES_PATH, 'r') as f:
        crop_res = json.load(f)
    update(crop_res, 2000 * 0.8)  # arg1: 切块测试文件信息 arg2: 测试集图片切块尺寸
