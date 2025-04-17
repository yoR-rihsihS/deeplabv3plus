import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchinfo import summary

from torch.amp import GradScaler

from codebase import DeepLabV3Plus, FocalLoss, CityScapes, compute_batch_metrics, convert_trainid_mask, denormalize

import wandb

DEVICE = 'cuda'
USING_MULTI_GPU = False
torch.cuda.empty_cache()

def predict_mask_one_mini_batch(model, data_loader, mean, std):
    results_img, results_gt_mask, results_pred_mask = [], [], []
    model.eval()
    with torch.no_grad():
        for image, mask in data_loader:
            image = image.to(DEVICE)
            mask = mask.squeeze(1)

            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                pred_logits = model(image)
            pred_mask = torch.argmax(pred_logits, dim=1).squeeze(1)
            break

    pred_mask = pred_mask.cpu()
    mask = mask.cpu()
    image = image.cpu()
    for i in range(image.shape[0]):
        img = denormalize(image[i], mean, std).clamp(0, 1)
        img = (img * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        msk = mask[i].numpy().astype(np.uint8)
        pred_msk = pred_mask[i].numpy().astype(np.uint8)
        pred_mask_color = convert_trainid_mask(pred_msk, to="color").astype(np.uint8)
        mask_color = convert_trainid_mask(msk, to="color").astype(np.uint8)

        results_img.append(img)
        results_gt_mask.append(mask_color)
        results_pred_mask.append(pred_mask_color)

    return results_img, results_gt_mask, results_pred_mask


def evaluate(model, data_loader, loss_fn, num_classes):
    running_loss = 0
    total_samples = 0
    total_intersection = torch.zeros(num_classes).to(DEVICE)
    total_union = torch.zeros(num_classes).to(DEVICE)
    total_pred_cardinality = torch.zeros(num_classes).to(DEVICE)
    total_target_cardinality = torch.zeros(num_classes).to(DEVICE)
    model.eval()
    with torch.no_grad():
        for image, mask in data_loader:
            image = image.to(DEVICE)
            mask = mask.to(DEVICE).squeeze(1)

            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                pred_logits = model(image)
                loss = loss_fn(pred_logits, mask)
            running_loss += loss.item() * mask.shape[0]

            pred_mask = torch.argmax(pred_logits, dim=1).squeeze(1)
            total_samples += mask.shape[0]
            intersection, union, pred_cardinality, target_cardinality = compute_batch_metrics(pred_mask, mask, num_classes)

            total_intersection += intersection
            total_union += union
            total_pred_cardinality += pred_cardinality
            total_target_cardinality += target_cardinality

    iou_per_class = total_intersection / (total_union + 1e-6)
    dice_per_class = 2 * total_intersection / (total_pred_cardinality + total_target_cardinality + 1e-6)
    pixel_accuracy_per_class = total_intersection / (total_target_cardinality + 1e-6)
    return running_loss / total_samples, iou_per_class, dice_per_class, pixel_accuracy_per_class

def train_one_epoch(model, data_loader, loss_fn, optimizer, num_classes, scaler):
    running_loss = 0
    total_samples = 0
    total_intersection = torch.zeros(num_classes).to(DEVICE)
    total_union = torch.zeros(num_classes).to(DEVICE)
    total_pred_cardinality = torch.zeros(num_classes).to(DEVICE)
    total_target_cardinality = torch.zeros(num_classes).to(DEVICE)
    model.train()
    for image, mask in data_loader:
        image = image.to(DEVICE)
        mask = mask.to(DEVICE).squeeze(1)

        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            pred_logits = model(image)
            loss = loss_fn(pred_logits, mask)
        running_loss += loss.item() * mask.shape[0]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred_mask = torch.argmax(pred_logits, dim=1).squeeze(1)
        total_samples += mask.shape[0]
        intersection, union, pred_cardinality, target_cardinality = compute_batch_metrics(pred_mask, mask, num_classes)

        total_intersection += intersection
        total_union += union
        total_pred_cardinality += pred_cardinality
        total_target_cardinality += target_cardinality

    iou_per_class = total_intersection / (total_union + 1e-6)
    dice_per_class = 2 * total_intersection / (total_pred_cardinality + total_target_cardinality + 1e-6)
    pixel_accuracy_per_class = total_intersection / (total_target_cardinality + 1e-6)
    return running_loss / total_samples, iou_per_class, dice_per_class, pixel_accuracy_per_class

def make_per_class_metrics_table(metrics_per_class):
    metrics_per_class = [float(m) for m in metrics_per_class]
    table = wandb.Table(columns=[f"class_{i}" for i in range(len(metrics_per_class))])
    table.add_data(*metrics_per_class)
    return table

def main(config):
    with wandb.init(project="vjt_assignment", config=config, name="run") as run:
        transform_train = v2.Compose([
            v2.ToImage(),
            v2.RandomCrop(size=run.config.train_crop_size),
            v2.RandomHorizontalFlip(p=run.config.random_transform_p),
            v2.RandomPhotometricDistort(brightness=run.config.brightness, contrast=run.config.contrast, saturation=run.config.saturation, hue=run.config.hue, p=run.config.random_transform_p),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=run.config.normalize_mean, std=run.config.normalize_std),
        ])

        transform_val_test = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=run.config.normalize_mean, std=run.config.normalize_std),
        ])

        train_set = CityScapes(root_dir='./data/', split='train', transform=transform_train)
        val_set = CityScapes(root_dir='./data/', split='val', transform=transform_val_test)
        test_set = CityScapes(root_dir='./data/', split='test', transform=transform_val_test)

        train_loader = DataLoader(train_set, batch_size=run.config.batch_size, shuffle=True, num_workers=5, persistent_workers=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=run.config.batch_size, shuffle=False, num_workers=2, persistent_workers=True)
        test_loader = DataLoader(test_set, batch_size=run.config.batch_size, shuffle=False, num_workers=3, persistent_workers=True)

        model = DeepLabV3Plus(
            backbone=run.config['backbone'],
            num_classes=run.config['num_classes'],
            aspp_dilate=run.config['aspp_dilate'],
            output_features_name=run.config['output_features_name'],
            low_level_features_name=run.config['low_level_features_name'],
            intermediacte_channels=run.config['intermediacte_channels']
        )
        model.to(DEVICE)

        print("Model summary:")
        summary(model, input_size=(1, 3, 1024, 2048), device=DEVICE)

        num_parameters = sum(p.numel() for p in model.parameters())
        print("Number of parameters =", num_parameters)

        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of trainable parameters =", num_parameters)

        if run.config.multi_gpu:
            print("Using", torch.cuda.device_count(), "GPUs")
            devices = list(range(torch.cuda.device_count()))
            model = nn.DataParallel(model, device_ids=devices, output_device=devices[0])
            USING_MULTI_GPU = True

        if USING_MULTI_GPU:
            resnet_params = list(model.module.resnet.parameters())
            other_params = [p for n, p in model.module.named_parameters() if not n.startswith('resnet')]
        else:
            resnet_params = list(model.resnet.parameters())
            other_params = [p for n, p in model.named_parameters() if not n.startswith('resnet')]

        optimizer = optim.SGD([
            {'params': resnet_params, 'lr': run.config.learning_rate * 0.1},
            {'params': other_params, 'lr': run.config.learning_rate}
        ], momentum=run.config.momentum, weight_decay=run.config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=run.config.epochs, eta_min=run.config.eta_min)

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, size_average=True, ignore_index=run.config.ignore_class)

        scaler = GradScaler()

        results_table = wandb.Table(columns=["image", "pred_mask", "gt_mask"])
        train_iou_per_class_table = wandb.Table(columns=[f"class_{i}" for i in range(run.config.num_classes)])
        val_iou_per_class_table = wandb.Table(columns=[f"class_{i}" for i in range(run.config.num_classes)])
        train_dice_per_class_table = wandb.Table(columns=[f"class_{i}" for i in range(run.config.num_classes)])
        val_dice_per_class_table = wandb.Table(columns=[f"class_{i}" for i in range(run.config.num_classes)])
        train_pixel_accuracy_per_class_table = wandb.Table(columns=[f"class_{i}" for i in range(run.config.num_classes)])
        val_pixel_accuracy_per_class_table = wandb.Table(columns=[f"class_{i}" for i in range(run.config.num_classes)])

        for epoch in range(run.config.epochs):
            train_loss, train_iou_per_class, train_dice_per_class, train_pixel_accuracy_per_class = train_one_epoch(model, train_loader, loss_fn, optimizer, run.config.num_classes, scaler)

            val_loss, val_iou_per_class, val_dice_per_class, val_pixel_accuracy_per_class = evaluate(model, val_loader, loss_fn, run.config.num_classes)
            ignore_index_mask = torch.arange(val_iou_per_class.shape[0]).to(DEVICE) != run.config.ignore_class

            print(f"Epoch {epoch + 1}/{run.config.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            print(f"Train IOU: {train_iou_per_class[ignore_index_mask].mean().cpu().item():.4f} - Val IOU: {val_iou_per_class[ignore_index_mask].mean().cpu().item():.4f}")
            print(f"Train Dice: {train_dice_per_class[ignore_index_mask].mean().cpu().item():.4f} - Val Dice: {val_dice_per_class[ignore_index_mask].mean().cpu().item():.4f}")
            print(f"Train Pixel Accuracy: {train_pixel_accuracy_per_class[ignore_index_mask].mean().cpu().item():.4f} - Val Pixel Accuracy: {val_pixel_accuracy_per_class[ignore_index_mask].mean().cpu().item():.4f}")
            print()
            scheduler.step()

            results_img, results_gt_mask, results_pred_mask = predict_mask_one_mini_batch(model, val_loader, run.config.normalize_mean, run.config.normalize_std)

            for i in range(len(results_img)):
                results_table.add_data(
                    wandb.Image(results_img[i], caption=f"image_{epoch}_{i}"),
                    wandb.Image(results_pred_mask[i], caption=f"pred_mask_{epoch}_{i}"),
                    wandb.Image(results_gt_mask[i], caption=f"gt_mask_{epoch}_{i}")
                )

            train_dice_per_class_table.add_data(*train_dice_per_class.cpu().tolist())
            train_iou_per_class_table.add_data(*train_iou_per_class.cpu().tolist())
            train_pixel_accuracy_per_class_table.add_data(*train_pixel_accuracy_per_class.cpu().tolist())
            val_dice_per_class_table.add_data(*val_dice_per_class.cpu().tolist())
            val_iou_per_class_table.add_data(*val_iou_per_class.cpu().tolist())
            val_pixel_accuracy_per_class_table.add_data(*val_pixel_accuracy_per_class.cpu().tolist())

            wandb.log({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_iou': train_iou_per_class[ignore_index_mask].mean().cpu().item(),
                'train_dice': train_dice_per_class[ignore_index_mask].mean().cpu().item(),
                'train_pixel_accuracy': train_pixel_accuracy_per_class[ignore_index_mask].mean().cpu().item(),
                'val_iou': val_iou_per_class[ignore_index_mask].mean().cpu().item(),
                'val_dice': val_dice_per_class[ignore_index_mask].mean().cpu().item(),
                'val_pixel_accuracy': val_pixel_accuracy_per_class[ignore_index_mask].mean().cpu().item(),
            })

        test_loss, test_iou_per_class, test_dice_per_class, test_pixel_accuracy_per_class = evaluate(model, test_loader, loss_fn, run.config.num_classes)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test IOU: {test_iou_per_class[ignore_index_mask].mean().cpu().item():.4f}")
        print(f"Test Dice: {test_dice_per_class[ignore_index_mask].mean().cpu().item():.4f}")
        print(f"Test Pixel Accuracy: {test_pixel_accuracy_per_class[ignore_index_mask].mean().cpu().item():.4f}")
        print()

        results_img, results_gt_mask, results_pred_mask = predict_mask_one_mini_batch(model, test_loader, run.config.normalize_mean, run.config.normalize_std)

        for i in range(len(results_img)):
            results_table.add_data(
                wandb.Image(results_img[i], caption=f"test_image_{epoch}_{i}"),
                wandb.Image(results_pred_mask[i], caption=f"test_pred_mask_{epoch}_{i}"),
                wandb.Image(results_gt_mask[i], caption=f"test_gt_mask_{epoch}_{i}")
            )

        wandb.log({
            'test_loss': test_loss,
            'test_iou': test_iou_per_class[ignore_index_mask].mean().cpu().item(),
            'test_dice': test_dice_per_class[ignore_index_mask].mean().cpu().item(),
            'test_pixel_accuracy': test_pixel_accuracy_per_class[ignore_index_mask].mean().cpu().item(),
            'results_table': results_table,
            'train_iou_per_class_table': train_iou_per_class_table,
            'train_dice_per_class_table': train_dice_per_class_table,
            'train_pixel_accuracy_per_class_table': train_pixel_accuracy_per_class_table,
            'val_iou_per_class_table': val_iou_per_class_table,
            'val_dice_per_class_table': val_dice_per_class_table,
            'val_pixel_accuracy_per_class_table': val_pixel_accuracy_per_class_table,
            'test_iou_per_class_table': make_per_class_metrics_table(test_iou_per_class.cpu().tolist()),
            'test_dice_per_class_table': make_per_class_metrics_table(test_dice_per_class.cpu().tolist()),
            'test_pixel_accuracy_per_class_table': make_per_class_metrics_table(test_pixel_accuracy_per_class.cpu().tolist()),
        })

        torch.save(model.state_dict(), f"deeplabv3plus_epoch_{epoch+1}.pth")
        wandb.save(f"deeplabv3plus_epoch_{epoch+1}.pth")

if __name__=="__main__":
    config = {
        'batch_size': 12,
        'multi_gpu': True,
        'num_classes': 20,
        'ignore_class': 19,
        'train_crop_size': [1024, 1024],
        'random_transform_p': 0.5,
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],
        'brightness': [0.75, 1.25],
        'contrast': [0.75, 1.25],
        'saturation': [0.75, 1.25],
        'hue': [-0.08, 0.08],
        'backbone': 'resnet50',
        'num_classes': 20,
        'aspp_dilate': [6, 12, 18],
        'output_features_name': 'layer_4',
        'low_level_features_name': 'layer_2',
        'intermediacte_channels': 256,
        'epochs': 20,
        'learning_rate': 0.05,
        'weight_decay': 0.001,
        'momentum': 0.9,
        'eta_min': 0.0001,
    }
    wandb.login(key="xxxxx") # This is removed intentionally
    main(config)
