import torch, torchvision
import matplotlib.pyplot as plt
import json
from PIL import Image
import cv2
import numpy as np
from os import walk
from copy import deepcopy
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.image_list import ImageList
from detectron2.data import transforms as T
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures.boxes import Boxes
from detectron2.layers import nms
from detectron2 import model_zoo
from detectron2.config import get_cfg
from tqdm.auto import tqdm
import pickle 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

def load_config_and_model_weights(cfg_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

    # ROI HEADS SCORE THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Comment the next line if you're using 'cuda'
    # cfg['MODEL']['DEVICE']='cpu'

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

    return cfg

cfg = load_config_and_model_weights(cfg_path)

def get_model(cfg):
    # build model
    model = build_model(cfg)

    # load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # eval mode
    model.eval()
    return model

model = get_model(cfg)

def prepare_image_inputs(cfg, img_list):
    # Resizing the image according to the configuration
    transform_gen = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )
    img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

    # Convert to C,H,W format
    convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1))

    batched_inputs = [{"image":convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in img_list]

    # Normalizing the image
    num_channels = len(cfg.MODEL.PIXEL_MEAN)
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    images = [normalizer(x["image"]) for x in batched_inputs]

    # Convert to ImageList
    images =  ImageList.from_tensors(images,model.backbone.size_divisibility)
    
    return images, batched_inputs

# images, batched_inputs = prepare_image_inputs(cfg, [img_bgr1])

def get_features(model, images):
    features = model.backbone(images.tensor.to(device))
    return features

# features = get_features(model, images)
# print(features.keys())

# plt.imshow(cv2.resize(img2, (images.tensor.shape[-2:][::-1])))
# plt.show()
# for key in features.keys():
#     print(features[key].shape)
#     plt.imshow(features[key][0,0,:,:].squeeze().detach().cpu().numpy(), cmap='jet')
#     plt.show()

def get_proposals(model, images, features):
    proposals, _ = model.proposal_generator(images, features)
    return proposals

# proposals = get_proposals(model, images, features)

def get_box_features(model, features, proposals):
    features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
    box_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    box_features = model.roi_heads.box_head.flatten(box_features)
    box_features = model.roi_heads.box_head.fc1(box_features)
    box_features = model.roi_heads.box_head.fc_relu1(box_features)
    box_features = model.roi_heads.box_head.fc2(box_features)

    box_features = box_features.reshape(1, box_features.shape[0], box_features.shape[1]) # depends on your config and batch size
    return box_features, features_list

# box_features, features_list = get_box_features(model, features, proposals)  
# print(box_features.shape)


def get_prediction_logits(model, features_list, proposals):
    cls_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    cls_features = model.roi_heads.box_head(cls_features)
    pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(cls_features)
    return pred_class_logits, pred_proposal_deltas

# pred_class_logits, pred_proposal_deltas = get_prediction_logits(model, features_list, proposals)

def get_box_scores(cfg, pred_class_logits, pred_proposal_deltas):
    box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
    smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

    outputs = FastRCNNOutputs(
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta,
    )

    boxes = outputs.predict_boxes()
    scores = outputs.predict_probs()
    image_shapes = outputs.image_shapes

    return boxes, scores, image_shapes

# boxes, scores, image_shapes = get_box_scores(cfg, pred_class_logits, pred_proposal_deltas)
# print(boxes)

def get_output_boxes(boxes, batched_inputs, image_size):
    proposal_boxes = boxes.reshape(-1, 4).clone()
    scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
    output_boxes = Boxes(proposal_boxes)

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(image_size)

    return output_boxes

# output_boxes = [get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]

def select_boxes(cfg, output_boxes, scores):
    test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cls_prob = scores.detach().cpu()
    cls_boxes = output_boxes.tensor.detach().reshape(output_boxes.tensor.shape[0]//80,80,4).cpu()
    max_conf = torch.zeros((cls_boxes.shape[0]))
    for cls_ind in range(0, cls_prob.shape[1]-1):
        cls_scores = cls_prob[:, cls_ind+1]
        det_boxes = cls_boxes[:,cls_ind,:]
        keep = np.array(nms(det_boxes, cls_scores, test_nms_thresh))
        max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
    keep_boxes = torch.where(max_conf >= test_score_thresh)[0]
    return keep_boxes, max_conf

# temp = [select_boxes(cfg, output_boxes[i], scores[i]) for i in range(len(scores))]
# keep_boxes, max_conf = [],[]
# for keep_box, mx_conf in temp:
#     keep_boxes.append(keep_box)
#     max_conf.append(mx_conf)

MIN_BOXES=10
MAX_BOXES=100
def filter_boxes(keep_boxes, max_conf, min_boxes, max_boxes):
    if len(keep_boxes) < min_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:min_boxes]
    elif len(keep_boxes) > max_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:max_boxes]
    return keep_boxes

# keep_boxes = [filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in zip(keep_boxes, max_conf)]

def get_visual_embeds(box_features, keep_boxes):
    return box_features[keep_boxes.copy()]

# visual_embeds = [torch.tensor(get_visual_embeds(box_feature, keep_box)).to(device) for box_feature, keep_box in zip(box_features, keep_boxes)]
# visual_embeds

# Convert Image to Model Input
def walker_arr(root_path):
    output = [] 
    for (dirpath, dirnames, filenames) in walk(root_path):
            for j in filenames:
                output.append(dirpath + '/' + j)
    return output

images_path = walker_arr('/bigdata/amirhossein/LLaVA/SemEval/dev/dev_images')

visual_embeds = []
for image_path in tqdm(images_path):
    img = plt.imread(image_path, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert Image to Model Input
    image, batched_inputs = prepare_image_inputs(cfg, [img])

    # Get ResNet+FPN features
    features = get_features(model, image)

    # Get region proposals from RPN
    proposals = get_proposals(model, image, features)

    # Get Box Features for the proposals
    box_features, features_list = get_box_features(model, features, proposals)

    # Get prediction logits and boxes
    pred_class_logits, pred_proposal_deltas = get_prediction_logits(model, features_list, proposals)

    # Get FastRCNN scores and boxes
    boxes, scores, image_shapes = get_box_scores(cfg, pred_class_logits, pred_proposal_deltas)

    # Rescale the boxes to original image size
    output_boxes = [get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]

    # Select the Boxes using NMS
    temp = [select_boxes(cfg, output_boxes[i], scores[i]) for i in range(len(scores))]
    keep_boxes, max_conf = [], []
    for keep_box, mx_conf in temp:
        keep_boxes.append(keep_box)
        max_conf.append(mx_conf)

    # Limit the total number of boxes
    keep_boxes = [filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in zip(keep_boxes, max_conf)]

    # Get the visual embedding
    visual_embed = [torch.tensor(get_visual_embeds(box_feature, keep_box)).to(device) for box_feature, keep_box in zip(box_features, keep_boxes)]
    visual_embeds.append({"path": image_path, "embed": visual_embed[0]})


with open('/bigdata/amirhossein/visualbert/visual_embeds_dev.pkl', 'wb') as f:
    pickle.dump(visual_embeds, f)