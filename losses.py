import torch.nn
import torch.nn as nn
import segmentation_models_pytorch as smp


class FocalLoss(nn.Module):
    def __init__(self, gamma, sigmoid=False):
        super().__init__()
        self.gamma = gamma
        self.bce = torch.nn.BCELoss()
        self.sigmoid = sigmoid

    def forward(self, logit, label):
        if self.sigmoid:
            logit = torch.sigmoid(logit)

        negative_log_p = self.bce(logit, label)
        p = torch.exp(-negative_log_p).item()
        focal_loss = (1 - p) ** self.gamma * negative_log_p

        return focal_loss


class combinedLoss(smp.utils.base.Loss):
    def __init__(self, weights=None, name='combinedLoss'):
        super().__init__()
        if weights is None:
            weights = [1, 1]
        self.name = name
        self.weights = weights
        self.bce_loss = smp.utils.losses.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(gamma=10)
        self.dice_loss = smp.utils.losses.DiceLoss()

    def forward(self, y_pr, y_gt):
        # loss = self.weights[0] * self.bce_loss(y_pr, y_gt) + self.weights[1] * self.dice_loss(y_pr, y_gt)
        loss = self.weights[0] * self.focal_loss(y_pr, y_gt) + self.weights[1] * self.dice_loss(y_pr, y_gt)+ self.bce_loss(y_pr, y_gt)
        return loss


cLoss = combinedLoss([1, 0.8])
