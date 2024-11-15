import glob
import os
import albumentations as A
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2

class CustomDataset(Dataset):
    '''
        dataset_folder/
            - Images
                - Training
                    - 0.jpg
                    - 1.jpg
                    ...
                - Validation
                    - 0.jpg
                    - 1.jpg
                    ...
            - Annotations
                - Training
                    - 0.jpg
                    - 1.jpg
                    ...
                - Validation
                    - 0.jpg
                    - 1.jpg
                    ...
    '''
    def __init__(self, image_dir, label_dir, image_size_HW=(640, 480), ignore_classids=[], is_train=False, save_dir="output"):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.label_paths = sorted(os.listdir(label_dir))
        self.height_size = image_size_HW[0]
        self.width_size = image_size_HW[1]
        self.save_dir = save_dir
        self.ignore_classids = ignore_classids

        os.makedirs(self.save_dir, exist_ok=True)

        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Resize(height=self.height_size, width=self.width_size),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=self.height_size, width=self.width_size),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        label_path = os.path.join(self.label_dir, self.label_paths[idx])

        original_image = cv2.imread(image_path)
        label_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # 모든 class 다 있음[0~255 사이](pixel value = class id)
        original_label_mask = label_mask.copy()

        transformed = self.transform(image=original_image, mask=label_mask)
        image = transformed['image']
        label_mask = transformed['mask']

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # (H, W, C) -> (C, H, W)
        label_mask = torch.from_numpy(label_mask).long()

        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        return image, label_mask


class CustomCOCODataset(Dataset):
    '''
        dataset_folder/
            - Images
                - 0.jpg
                - 1.jpg
                ...
            - Annotations
                - 0.jpg
                - 1.jpg
                ...
    '''
    def __init__(self, image_dir, label_dir, is_train=False, save_dir="output"):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.label_paths = sorted(os.listdir(label_dir))
        self.save_dir = save_dir

        # 저장 디렉토리가 없으면 생성
        os.makedirs(self.save_dir, exist_ok=True)

        ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
        ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

        if is_train:
            self.transform = A.Compose([
                A.CenterCrop(height=1280, width=960),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=ADE_MEAN, std=ADE_STD),
            ])
        else:
            self.transform = A.Compose([
                A.CenterCrop(height=1280, width=960),
                A.Normalize(mean=ADE_MEAN, std=ADE_STD),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        label_path = os.path.join(self.label_dir, self.label_paths[idx])

        original_image = cv2.imread(image_path)
        label_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        transformed = self.transform(image=original_image, mask=label_mask)
        image = transformed['image']
        label_mask = transformed['mask']

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # (H, W, C) -> (C, H, W)
        label_mask = (torch.from_numpy(label_mask) > 0).long()  # Convert to binary
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        return {"pixel_values": image, "mask_labels": label_mask, "original_image": original_image_rgb}
