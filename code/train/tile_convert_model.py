# for cascade rcnn
import torch

model_name = "../pretrained/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth"
model = torch.load(model_name)

# weight
num_classes = 9
model["state_dict"]["roi_head.bbox_head.0.fc_cls.weight"] = model["state_dict"]["roi_head.bbox_head.0.fc_cls.weight"][
                                                            :num_classes, :]
model["state_dict"]["roi_head.bbox_head.1.fc_cls.weight"] = model["state_dict"]["roi_head.bbox_head.1.fc_cls.weight"][
                                                            :num_classes, :]
model["state_dict"]["roi_head.bbox_head.2.fc_cls.weight"] = model["state_dict"]["roi_head.bbox_head.2.fc_cls.weight"][
                                                            :num_classes, :]
# bias
model["state_dict"]["roi_head.bbox_head.0.fc_cls.bias"] = model["state_dict"]["roi_head.bbox_head.0.fc_cls.bias"][
                                                          :num_classes]
model["state_dict"]["roi_head.bbox_head.1.fc_cls.bias"] = model["state_dict"]["roi_head.bbox_head.1.fc_cls.bias"][
                                                          :num_classes]
model["state_dict"]["roi_head.bbox_head.2.fc_cls.bias"] = model["state_dict"]["roi_head.bbox_head.2.fc_cls.bias"][
                                                          :num_classes]

# change input channels



# save new model
torch.save(model, "../pretrained/modified_cascade_rcnn_r50_fpn_20e_coco.pth")
print("The pretrained model is converted. \n Note: (The number of classes is {} currently.)".format(num_classes))
