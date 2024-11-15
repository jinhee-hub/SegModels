import os
import torch
from tqdm import tqdm
from utils.visaulization import save_samples
from utils.metrics import mIoU, dice_score

class Trainer():
    def __init__(self,
                 trainloader=None,
                 validloader=None,
                 model=None,
                 batch_size = 4,
                 num_classes= 2,
                 output_dir="./output",
                 model_output_dir = "./checkpoints",
                 ignore_index=[0],
                 criterion=None,
                 optimizer = None,
                 scheduler = None,
                 metrics = None,
                 device = "cuda",
                 ):
        self.trainloader = trainloader
        self.validloader = validloader
        self.model = model
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self.output_dir = output_dir
        self.model_output_dir = model_output_dir
        self.ignore_index = ignore_index
        self.best_losses = []  # Top 3
        self.checkpoints = []
        self.device = device

    def train(self, epochs=10):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()
            running_loss = 0.0
            for i, batch in enumerate(tqdm(self.trainloader,  desc="Training")):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_train_loss = running_loss / len(self.trainloader)
            print(f"Average Training Loss: {avg_train_loss:.4f}")


            self.model.eval()
            val_loss = 0.0
            total_miou = 0.0
            total_dsc = 0.0
            max_dsc = 0.0
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(self.validloader, desc="Validation")):
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(images)

                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    batch_miou = mIoU(outputs, labels)
                    batch_dsc = dice_score(outputs, labels)
                    total_miou += batch_miou
                    total_dsc += batch_dsc

                    ########### TODO: Implemnetation - save images
                    if batch_dsc > max_dsc:
                        print(f"Best DSC: {batch_dsc:.4f} -- Evaluation samples are saved in {self.output_dir}")
                        max_dsc = batch_dsc
                        save_samples(images=images, labels=labels, outputs=outputs,batch_index=idx * self.batch_size + epoch,
                                     output_dir=self.output_dir, ignore_classids=self.ignore_index)


            avg_val_loss = val_loss / len(self.validloader)
            avg_miou = total_miou / len(self.validloader)
            avg_dsc = total_dsc / len(self.validloader)
            print(f"Average Validation Loss: {avg_val_loss:.4f}")
            print(f"Mean IoU: {avg_miou:.4f}, Mean DSC: {avg_dsc:.4f}")

            if self.scheduler != None:
                self.scheduler.step(val_loss)

            self.save_checkpoint(epoch, val_loss)

    def evaluate(self):
        self.model.eval()

        val_loss = 0.0
        total_miou = 0.0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.validloader, desc="Validating")):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.7).long()



    def save_checkpoint(self, epoch, val_loss):
        if len(self.best_losses) < 3:
            self.best_losses.append(val_loss)
            checkpoint_path = os.path.join(self.model_output_dir, f'checkpoint_{epoch}.pth')
            self.checkpoints.append(checkpoint_path)
            self._save_state(checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch} with validation loss {val_loss:.4f}")
        else:
            max_loss = max(self.best_losses)
            if val_loss < max_loss:
                max_index = self.best_losses.index(max_loss)
                os.remove(self.checkpoints[max_index])
                self.best_losses[max_index] = val_loss
                checkpoint_path = os.path.join(self.model_output_dir, f'checkpoint_{epoch}.pth')
                self.checkpoints[max_index] = checkpoint_path
                self._save_state(checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch} with validation loss {val_loss:.4f}")

    def _save_state(self, checkpoint_path):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, filepath, epoch):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # start_epoch = checkpoint['epoch']
        start_epoch = epoch
        print(f"Loaded checkpoint from epoch {start_epoch}")
        return start_epoch
