import torch
import torch.nn.functional as F
import numpy as np

def mIoU(pred_mask, label_mask, smooth=1e-10, n_classes=23):
    '''
    pred_mask = label_mask are the grayscale mask that class id = pixel value
    '''
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        label_mask = label_mask.contiguous().view(-1)

        iou_per_class = []
        for pixel_value in range(0, n_classes): # class_id = pixel value
            true_class = pred_mask == pixel_value
            true_label = label_mask == pixel_value

            if true_label.long().sum().item() == 0: # no exist label -> set nan
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

def dice_score(pred_mask, label_mask, smooth=1e-10, n_classes=23):
    '''
    pred_mask and label_mask are the grayscale mask with class id as pixel value
    '''
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        label_mask = label_mask.contiguous().view(-1)

        dice_per_class = []
        for pixel_value in range(0, n_classes): # class_id = pixel value
            true_class = pred_mask == pixel_value
            true_label = label_mask == pixel_value

            if true_label.long().sum().item() == 0: # no exist label -> set nan
                dice_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                dice = (2 * intersect + smooth) / (true_class.sum().float().item() + true_label.sum().float().item() + smooth)
                dice_per_class.append(dice)
        return np.nanmean(dice_per_class)