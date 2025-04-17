import numpy as np
import torch
import torchvision.transforms.functional as F

from .labels import name_to_labelid, name_to_trainid, name_to_color

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
    intersection = torch.zeros(num_classes).to(preds.device)
    union = torch.zeros(num_classes).to(preds.device)
    pred_cardinality = torch.zeros(num_classes).to(preds.device)
    target_cardinality = torch.zeros(num_classes).to(preds.device)

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


def convert_trainid_mask(trainid_mask, to="labelid"):
    """
    Convert a mask from train id to label id.
    Args:
        - trainid_mask (np.array): The output of the model of shape H x W
        - to (str): The type of mask to convert to either 'labelid' or 'color'
    Returns:
        - np.array: The converted mask of shape H x W or H x W x 3
    """
    assert to in ["labelid", "color"], "to must be either 'labelid' or 'color'"

    trainid_to_labelid = {}
    for name, train_id in name_to_trainid.items():
        label_id = name_to_labelid[name]
        trainid_to_labelid[train_id] = label_id
    trainid_to_labelid[19] = 0 # train id 19 is background, label id 0 is unlabeled

    # Creating lookup table for mapping
    max_trainid = max(trainid_to_labelid.keys())
    labelid_lut = np.zeros((max_trainid + 1), dtype=np.uint8)
    for train_id, label_id in trainid_to_labelid.items():
        labelid_lut[train_id] = label_id

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
    for label_id, color in labelid_to_color.items():
        color_lut[label_id] = color

    # Applying color_lut to convert mask
    color_mask = color_lut[labelid_mask]  # shape: (H, W, 3)
    return color_mask


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    _mean = -mean / std
    _std = 1 / std
    
    return F.normalize(tensor, _mean.tolist(), _std.tolist())