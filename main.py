import os
import torch
from my_datasets.dataset import CustomDataset
from torch.utils.data import DataLoader
from models.unet.unet import UNet
from utils.loss import DiceLoss, FocalLoss
from trainer import Trainer

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device :', device)

    BATCH_SIZE = 4
    TOTAL_NUM_CLASSES = 3  # 배경(0), class1(1), class2(2)
    IGNORE_CLASS_IDS = [0,1] # 배경과 class1은 제외
    NUM_EPOCHS = 5000
    LEARNING_RATE = 5e-5
    load_model = "./checkpoints/blackhead_unet/checkpoint_27.pth"
    model_output_dir = "./checkpoints/blackhead_unet"
    os.makedirs(model_output_dir, exist_ok=True)
    output_dir = "output"

    root_dir = "../../dataset/Acne/bwhead/cropped"
    train_image_dir = os.path.join(root_dir, "images/training")
    train_label_dir = os.path.join(root_dir, "annotations/training")
    val_image_dir = os.path.join(root_dir, "images/validation")
    val_label_dir = os.path.join(root_dir, "annotations/validation")

    train_dataset = CustomDataset(train_image_dir, train_label_dir, image_size_HW=(1280, 960), ignore_classids= IGNORE_CLASS_IDS, is_train=True)
    val_dataset = CustomDataset(val_image_dir, val_label_dir, image_size_HW=(1280, 960), ignore_classids= IGNORE_CLASS_IDS, is_train=False)

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(3, num_classes=(TOTAL_NUM_CLASSES)).to(device)

    criterion = DiceLoss(ignore_index=IGNORE_CLASS_IDS)
    # criterion = FocalLoss(ignore_index=IGNORE_CLASS_IDS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.000001)

    trainer = Trainer(trainloader=trainloader, validloader=validloader, model=model, batch_size=BATCH_SIZE,
                      criterion=criterion, optimizer=optimizer, scheduler=scheduler, output_dir=output_dir,
                      model_output_dir = model_output_dir, ignore_index=IGNORE_CLASS_IDS, device=device)

    if load_model !=None:
        trainer.load_checkpoint(load_model)

    trainer.train(epochs=NUM_EPOCHS)

