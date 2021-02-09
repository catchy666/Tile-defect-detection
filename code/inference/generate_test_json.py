import json
import os
from glob import glob
from tqdm import tqdm
from PIL import Image

# Need to set: Line 8, Line 9
TEST_IMG_PATH = '../../tcdata/tile_round1_testB_20210128/test_imgs_crop_2000'
TEST_ANN_PATH = '../../tcdata/tile_round1_testB_20210128/annotations/instances_test_crop_2000.json'


def save(images, annotations, name=''):
    ann = {}
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations
    category = [
        {'id': 1, 'name': '1'},
        {'id': 2, 'name': '2'},
        {'id': 3, 'name': '3'},
        {'id': 4, 'name': '4'},
        {'id': 5, 'name': '5'},
        {'id': 6, 'name': '6'},
    ]
    ann['categories'] = category
    json.dump(ann, open(TEST_ANN_PATH, 'w'), indent=4, separators=(',', ': '))


def test_dataset(img_dir):
    img_list = glob(img_dir + '/*.jpg')

    image_id, idx = 20210000000, 1
    images, annotations = [], []

    for img_path in tqdm(img_list):
        img = Image.open(img_path)
        w, h = img.size
        image_id += 1
        image = {
            'file_name': os.path.split(img_path)[-1],
            'width': w,
            'height': h,
            'id': image_id
        }
        images.append(image)
        labels = [[10, 10, 20, 20]]
        for label in labels:
            bbox = [label[0], label[1], label[2] - label[0], label[3] - label[1]]
            seg = []
            ann = {'segmentation': [seg], 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'image_id': image_id,
                   'bbox': bbox, 'category_id': 1, 'id': idx, 'ignore': 0}
            idx += 1
            annotations.append(ann)
    save(images, annotations)


if __name__ == '__main__':
    print('generate test json label file.')
    test_dataset(TEST_IMG_PATH)
