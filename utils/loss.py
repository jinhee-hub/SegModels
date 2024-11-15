import torch
from torch import nn
from torch import Tensor

class MSELoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets, **kwargs):
        loss = self.loss(inputs, targets)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=[]):
        super().__init__()
        self.ignore_index = ignore_index
    def dice_coeff(self, pred: Tensor, label: Tensor, epsilon: float = 1e-6):
        '''
            pred: [batch * num_classes, height, width],  label: [batch * num_classes, height, width]
        '''
        assert pred.size() == label.size()

        sum_dim = (-1, -2)

        inter = 2 * (pred * label).sum(dim=sum_dim)
        sets_sum = pred.sum(dim=sum_dim) + label.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice.mean()


    def multiclass_dice_coeff(self, pred: Tensor, label: Tensor):
        '''
        Input -> pred: [batch, num_classes, height, width],  label: [batch, height, width]
        '''

        # class별로 One-hot encoding되며, class가 존재할 경우 1, 없으면 0.
        # label: [batch, height, width] -> label_one_hot: [batch, num_classes, height, width]
        label_one_hot = nn.functional.one_hot(label, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()

        num_classes = pred.size(1)
        dice_scores = {}
        for class_index in range(num_classes):
            if class_index in self.ignore_index:
                dice_scores[class_index] = None
            else:
                # pred[:, class_index] = label_one_hot[:, class_index] = [batch, height, width]
                class_pred = pred[:, class_index, :, :]
                class_label = label_one_hot[:, class_index, :, :]
                dice_score = self.dice_coeff(class_pred, class_label)
                dice_scores[class_index] = dice_score

        valid_dice_scores = [score for score in dice_scores.values() if score is not None]
        return torch.mean(torch.stack(valid_dice_scores))


    def forward(self, pred: Tensor, label: Tensor):
        '''
        Input ->  pred: [batch, num_classes, height, width],  label: [batch, height, width]
        '''
        pred = torch.softmax(pred, dim=1)
        dice = self.multiclass_dice_coeff(pred, label)  # even if it's binary, able to use this.

        dice_loss = 1 - dice

        return dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, ignore_index=[]):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def focal_coeff(self, pred: Tensor, label: Tensor, epsilon: float = 1e-6):
        """
        pred: [batch * num_classes, height, width], label: [batch * num_classes, height, width]
        """
        assert pred.size() == label.size()

        pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
        log_pred = torch.log(pred)

        focal_weight = (1 - pred) ** self.gamma
        ce_loss = -(label * log_pred)
        loss = self.alpha * focal_weight * ce_loss

        return loss.mean()

    def multiclass_focal_coeff(self, pred: Tensor, label: Tensor):
        """
        Input -> pred: [batch, num_classes, height, width], label: [batch, height, width]
        """
        # One-hot encode label: [batch, height, width] -> [batch, num_classes, height, width]
        label_one_hot = nn.functional.one_hot(label, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()

        num_classes = pred.size(1)
        focal_losses = {}
        for class_index in range(num_classes):
            if class_index in self.ignore_index:
                focal_losses[class_index] = None
            else:
                # pred[:, class_index] = label_one_hot[:, class_index] = [batch, height, width]
                class_pred = pred[:, class_index, :, :]
                class_label = label_one_hot[:, class_index, :, :]
                focal_loss = self.focal_coeff(class_pred, class_label)
                focal_losses[class_index] = focal_loss

        valid_focal_losses = [loss for loss in focal_losses.values() if loss is not None]
        return torch.mean(torch.stack(valid_focal_losses))

    def forward(self, pred: Tensor, label: Tensor):
        """
        Input ->  pred: [batch, num_classes, height, width], label: [batch, height, width]
        """
        pred = torch.softmax(pred, dim=1)
        focal_loss = self.multiclass_focal_coeff(pred, label)

        return focal_loss