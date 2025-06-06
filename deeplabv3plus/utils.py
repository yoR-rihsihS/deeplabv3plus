import json
import numpy as np

import torch

def compute_batch_metrics(preds, targets, num_classes):
    """
    Compute metrics for a batch of predictions and targets to later compute metrics for the whole dataset.
    Args:
        - preds (torch.Tensor): The predictions of shape (N, H, W)
        - targets (torch.Tensor): The targets of shape (N, H, W)
        - num_classes (int): The number of classes
    Returns:
        - intersection (torch.Tensor): The intersection of the predictions and targets of shape (num_classes,)
        - union (torch.Tensor): The union of the predictions and targets of shape (num_classes,)
        - pred_cardinality (torch.Tensor): The cardinality of the predictions of shape (num_classes,)
        - target_cardinality (torch.Tensor): The cardinality of the targets of shape (num_classes,)
    """
    intersection = torch.zeros(num_classes, device=preds.device, dtype=torch.float)
    union = torch.zeros(num_classes, device=preds.device, dtype=torch.float)
    pred_cardinality = torch.zeros(num_classes, device=preds.device, dtype=torch.float)
    target_cardinality = torch.zeros(num_classes, device=preds.device, dtype=torch.float)

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)

        inter = (pred_mask & target_mask).sum() # the intersection is same as true positive or class correct
        uni = (pred_mask | target_mask).sum()

        intersection[cls] = inter
        union[cls] = uni
        pred_cardinality[cls] = pred_mask.sum()
        target_cardinality[cls] = target_mask.sum() # the target cardinality is same as class total

    return intersection, union, pred_cardinality, target_cardinality


def convert_trainid_mask(trainid_mask, to, name_to_trainId_path, name_to_labelId_path, name_to_color_path):
    """
    Convert a mask from train id to label id or color.
    Args:
        - trainid_mask (np.array): The output of the model of shape H x W
        - to (str): The type of mask to convert to either 'labelid' or 'color'
    Returns:
        - np.array: The converted mask of shape H x W or H x W x 3
    """
    assert to in ["labelid", "color"], f"to {to} is not supported. Choose from 'labelid' or 'color'"
    name_to_trainid = json.load(open(name_to_trainId_path, 'r'))
    name_to_labelid = json.load(open(name_to_labelId_path, 'r'))
    name_to_color = json.load(open(name_to_color_path, 'r'))
    for name, train_id in name_to_trainid.items():
        if train_id == 255:
            name_to_trainid[name] = 19

    trainid_to_labelid = {}
    for name, train_id in name_to_trainid.items():
        label_id = name_to_labelid[name]
        trainid_to_labelid[train_id] = label_id
    trainid_to_labelid[19] = 0 # train id 19 is background, label id 0 is unlabeled

    # Creating lookup table for mapping
    max_trainid = max(trainid_to_labelid.keys())
    labelid_lut = np.zeros((max_trainid + 1), dtype=np.uint8)
    for train_id in range(max_trainid + 1):
        labelid_lut[train_id] = trainid_to_labelid[train_id]

    # Applying label_lut to convert mask
    labelid_mask = labelid_lut[trainid_mask]
    if to == 'labelid':
        return labelid_mask
    
    labelid_to_color = {}
    for name, color in name_to_color.items():
        label_id = name_to_labelid[name]
        labelid_to_color[label_id] = color

    # Creating lookup table for mapping
    max_labelid = max(labelid_to_color.keys())
    color_lut = np.zeros((max_labelid + 1, 3), dtype=np.uint8)
    for label_id in range(max_labelid + 1):
        color_lut[label_id] = labelid_to_color[label_id]

    # Applying color_lut to convert mask
    color_mask = color_lut[labelid_mask]  # shape: (H, W, 3)
    return color_mask