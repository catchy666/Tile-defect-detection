import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

tile_label2name = {1: "边异常",
                   2: "角异常",
                   3: "白色点瑕疵",
                   4: "浅色块瑕疵",
                   5: "深色点块瑕疵",
                   6: "光圈瑕疵",
                   7: "记号笔",
                   8: "划伤"}

DATASET_PATH = '../../tcdata/tile_round2_train/'  # Dataset path
ANNOTATION_FILE = os.path.join(DATASET_PATH, 'merged_train_annos.json')  # Dataset annotations json file
IMG_PREFIX = os.path.join(DATASET_PATH, 'train_imgs')  # Dataset images path
OUTPUT = os.path.join(DATASET_PATH, 'annotations/instances_{}.json'.format('train'))  # coco save path


class Tile2COCO:
    def __init__(self):
        self.annotations = []
        self.images = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def to_coco(self, ann_file, out_file, image_prefix):
        self._init_categories()
        data_infos = pd.read_json(open(ann_file, 'r'))
        name_list = data_infos['name'].unique()
        for img_name in tqdm(name_list):
            img_ann = data_infos[data_infos['name'] == img_name]
            if len(img_ann) > 100:
                print(img_name)
                continue

            bboxes = img_ann['bbox'].tolist()
            defect_categories = img_ann['category'].tolist()
            assert img_ann['name'].unique()[0] == img_name

            img_path = os.path.join(image_prefix, img_name)
            h, w = int(img_ann['image_height'].values[0]), int(img_ann['image_width'].values[0])
            self.images.append(self._image(img_path, h, w))

            for bbox, defect_category in zip(bboxes, defect_categories):
                if bbox[1] >= h:
                    continue
                if bbox[0] >= w:
                    continue
                label = defect_category
                annotation = self._annotation(label, bbox, h, w)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'tile defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        with open(out_file, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))

    def _init_categories(self):
        for k, v in tile_label2name.items():
            category = {}
            category['id'] = k
            category['name'] = v
            # category['supercategory'] = 'category'
            self.categories.append(category)

    def _image(self, path, h, w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    def _annotation(self, label, bbox, h, w):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area <= 0:
            print(bbox)
            input()
        points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points, h, w)
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    def _get_box(self, points, img_h, img_w):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        '''coco,[x,y,w,h]'''
        w = max_x - min_x
        h = max_y - min_y
        if w > img_w:
            w = img_w
        if h > img_h:
            h = img_h
        return [min_x, min_y, w, h]


if __name__ == '__main__':
    tile = Tile2COCO()
    tile.to_coco(ann_file=ANNOTATION_FILE,
                 out_file=OUTPUT,
                 image_prefix=IMG_PREFIX)
