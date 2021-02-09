import os
import cv2 as cv
from tqdm import tqdm
"""
Crop Test Dataset Images
"""
# Need to set: Line 8, Line 9
TEST_IMG_PATH = '../../tcdata/tile_round1_testB_20210128/testB_imgs'  # test dataset to crop
SAVE_PATH = '../../tcdata/tile_round1_testB_20210128/test_imgs_crop_2000/'  # save path


def crop_test(image_paths, step):
    for image_path in tqdm(image_paths):
        image_dir_path = os.path.join(TEST_IMG_PATH, image_path)
        image = cv.imread(image_dir_path)
        image_size = image.shape
        x_i = 0
        y_i = 0
        while y_i * step * 0.8 + step < image_size[1]:
            while x_i * step * 0.8 + step < image_size[0]:
                image_crop = image[int(x_i * step * 0.8):int(x_i * step * 0.8 + step),
                             int(y_i * step * 0.8):int(y_i * step * 0.8 + step)]
                new_image_name = image_path + '_' + str(x_i) + '_' + str(y_i) + '.jpg'
                new_image_path = os.path.join(SAVE_PATH, new_image_name)
                cv.imwrite(new_image_path, image_crop)
                x_i += 1
            image_crop = image[int(x_i * step * 0.8):image_size[0], int(y_i * step * 0.8):int(y_i * step * 0.8 + step)]
            new_image_name = image_path + '_' + str(x_i) + '_' + str(y_i) + '.jpg'
            new_image_path = os.path.join(SAVE_PATH, new_image_name)
            cv.imwrite(new_image_path, image_crop)
            y_i += 1
            x_i = 0
        while x_i * step * 0.8 + step < image_size[0]:
            image_crop = image[int(x_i * step * 0.8):int(x_i * step * 0.8 + step), int(y_i * step * 0.8):image_size[1]]
            new_image_name = image_path + '_' + str(x_i) + '_' + str(y_i) + '.jpg'
            new_image_path = os.path.join(SAVE_PATH, new_image_name)
            cv.imwrite(new_image_path, image_crop)
            x_i += 1
        image_crop = image[int(x_i * step * 0.8):image_size[0], int(y_i * step * 0.8):image_size[1]]
        new_image_name = image_path + '_' + str(x_i) + '_' + str(y_i) + '.jpg'
        new_image_path = os.path.join(SAVE_PATH, new_image_name)
        cv.imwrite(new_image_path, image_crop)


if __name__ == '__main__':
    crop_test(os.listdir(TEST_IMG_PATH), 2000)  # arg1: 测试集图片列表 arg2: 测试集图片切块尺寸
