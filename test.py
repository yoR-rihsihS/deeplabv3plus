import os
from time import time
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import v2

from codebase import DeepLabV3Plus, convert_trainid_mask

DEVICE = 'cuda:0'
torch.cuda.empty_cache()

def inference(model, data_path, save_path, transform):
    time_taken = []
    model.eval()
    for city in os.listdir(data_path):
        img_dir = os.path.join(data_path, city)
        tgt_dir = os.path.join(save_path, city)
        os.makedirs(tgt_dir, exist_ok=True)
        for file_name in os.listdir(img_dir):
            if file_name.endswith('_leftImg8bit.png'):
                img_path = os.path.join(img_dir, file_name)
                tgt_name = file_name.replace('_leftImg8bit.png', '_trainId_preds.png')
                tgt_path = os.path.join(tgt_dir, tgt_name)

                start = time()

                image = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEVICE)
                pred_logits = model(image)
                pred_mask = torch.argmax(pred_logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                pred_mask_color = convert_trainid_mask(pred_mask, to="labelid").astype(np.uint8)
            
                end = time()

                img = Image.fromarray(pred_mask_color, mode='L')
                img.save(tgt_path)
                time_taken.append(end - start)

    print(f"Model inference on images from {data_path} has been recorded in {save_path}")
    print(f"Average time to run model on one image : {(sum(time_taken) / len(time_taken)):.4f}")


def main(config):
    transform_val_test = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=config['normalize_mean'], std=config['normalize_std']),
    ])

    model = DeepLabV3Plus(
        backbone=config['backbone'],
        num_classes=config['num_classes'],
        aspp_dilate=config['aspp_dilate'],
        output_features_name=config['output_features_name'],
        low_level_features_name=config['low_level_features_name'],
        intermediacte_channels=config['intermediacte_channels']
    )

    model_state_dict = torch.load(config['model_weights_path'], map_location='cpu', weights_only=True)
    if config['multi_gpu_trained']:
        new_model_state_dict = {}
        for k, v in model_state_dict.items():
            new_key = k.replace("module.", "")  # strip off the prefix
            new_model_state_dict[new_key] = v
        model.load_state_dict(new_model_state_dict, strict=True)
    else:
        model.load_state_dict(model_state_dict, strict=True)
    model.to(DEVICE)

    inference(model, data_path='./data/leftImg8bit/test/', save_path='./saved/', transform=transform_val_test)



if __name__=="__main__":
    config = {
        'num_classes': 20,
        'ignore_class': 19,
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],
        'backbone': 'resnet50',
        'aspp_dilate': [6, 12, 18],
        'output_features_name': 'layer_4',
        'low_level_features_name': 'layer_2',
        'intermediacte_channels': 256,
        'model_weights_path': './deeplabv3plus_epoch_20.pth',
        'multi_gpu_trained': True,
    }
    main(config)