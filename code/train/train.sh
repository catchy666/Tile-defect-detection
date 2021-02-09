#!/bin/bash
mv ../../tcdata/tile_round2_train_20210204 ../../tcdata/tile_round2_train
mkdir -p ../../tcdata/tile_round2_train/annotations
cp -r ../../tcdata/tile_round2_train_20210208/train_imgs/* ../../tcdata/tile_round2_train/train_imgs
cp -r ../../tcdata/tile_round2_train_20210208/train_template_imgs/* ../../tcdata/tile_round2_train/train_template_imgs
cp ../../tcdata/tile_round2_train_20210208/train_annos.json ../../tcdata/tile_round2_train/train_annos_20210208.json
wait
rm -rf ../../tcdata/tile_round2_train_20210208
rm ../../tcdata/tile_round2_train/train_annos.json ../../tcdata/tile_round2_train/train_annos_20210208.json ../../tcdata/tile_round2_train/Readme.md
mkdir -p ../pretrained
python merge_data_json.py
python tile2coco.py
wget http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth -O ../pretrained/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth
python tile_convert_model.py
CUDA_VISIBLE_DEVICES=1,0 ./dist_train.sh ../config/cascade_rcnn_r50_fpn_70e_coco.py 2