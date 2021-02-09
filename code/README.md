# TianChi-2021-Tile-Defect-Detection


### 比赛地址：[2021广东工业智造创新大赛—智能算法赛：瓷砖表面瑕疵质检](https://tianchi.aliyun.com/competition/entrance/531846/introduction)
-----


## 解决方案及算法

 - 数据预处理
    - 由于数据集单张图片尺寸过大且所持算力有限，考虑使用2000x2000尺寸，步长2000*0.8进行图像切块
    - 将bbox原坐标转换为对应切块后图像的坐标
 - 模型设计
    - 基本框架：Cascade R-CNN
    - backbone：ResNet50
    - Cascade阈值调整为0.4 0.5 0.6，以适应0.1 0.3 0.5的mAP评价标准
    - RCNN正负样例放松overlap要求 放松为 0.6 0.2
    - 根据建议区域采用ROI Align 的方式提取特征
 - 后处理
    - Soft-NMS
    - 最大score二类后处理, 根据单个图片bbox最高score来对图像进行二次过滤, 判断该图像是否是正常样本
    
## Requirements
 - OS: Ubuntu18.04
 - GPU: 2080Ti * 2
 - Python: 3.7.9
 - GCC: GCC 5+
 - CUDA: 10.2
 - CuDNN: 7.6.5
 - PyTorch: 1.6.0

## Install

 1. Create a conda virtual environment and activate it.
```
 conda create -n tile python=3.7 -y
 conda activate tile
```
 2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/) , e.g.,
 ```
 # CUDA 10.2
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
 ```
 3. Install mmcv-full.
 ```
 pip install mmcv-full==1.1.5 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html
 ```
 4. Install build requirements and then install MMDetection.
 ```bash
 cd code/
 pip install -r requirements/build.txt
 pip install -v -e .  # or "python setup.py develop"
 ```

## Prepare Data

**<u>（以下准备操作已写入train的shell脚本中，训练时可一次性执行）</u>**


 - 建立所需文件夹
   
     - 在tcdata/训练集目录下创建文件夹：
     
        annotations
        
        train_crop_2000
        
     - 在tcdata/测试集目录下创建文件夹：
     
        annotations
        
        test_crop_2000
     
 - 图像切块
    - 切块后训练集&测试集图像存储于train_imgs_crop_2000/ & test_imgs_crop_2000/目录下；
    - 根据原始标注文件(train_annos.json)生成切块后的新标注文件(train_annos_crop_2000.json)；
    
 - 标注文件格式转换
   
    - 将新标注文件转换为mmdetection框架所需的COCO格式：instances_train_crop_2000.json


 ## Train & Test

  - **训练**
    
    1. 训练脚本
    ```
    cd tools/
    ./train.sh
    ```
    2. 训练过程的模型权重、训练日志保存在user_data目录中
    
 - **测试**

    1. 运行test.sh

    ```
    /test.sh
    ```

    2. 测试结果保存在prediction_result/result.json


