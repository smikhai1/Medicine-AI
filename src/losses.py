import torch.nn as nn
import torch.nn.functional as F
import torch


def dice_loss(input, target, num_classes=3):
    # for binary case!!!
    # need to be extended
    EPS = 1e-10
    dice_target = F.one_hot(target, num_classes=3).to(torch.float32)[:, :, :, 1:]
    dice_input = F.softmax(input, dim=1)[:, 1:, :, :]
    dice_input = torch.transpose(dice_input, 1, 3)



    intersection = (dice_target * dice_input).sum() + EPS
    union = dice_target.sum() + dice_input.sum() + EPS
    return 2.0 * intersection / union


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, num_classes=3):
        return dice_loss(input, target, num_classes)


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight, reduction=reduction)

    def forward(self, logits, targets):
        return self.loss(logits, targets)

class CrossEntropyDiceLoss(nn.Module):

    def __init__(self, dice_weight=1.0):
        super(CrossEntropyDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_loss = CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.ce_loss(logits, targets) + self.dice_weight * (1 - dice_loss(logits, targets))

class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        loss = 0
        for i in range(2):
            probs_flat = probs[:, i].contiguous().view(-1)
            targets_flat = (targets==i+1).float().contiguous().view(-1)
            loss += self.bce_loss(probs_flat, targets_flat)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.bce_with_logits((1 - torch.sigmoid(input)) ** self.gamma * F.logsigmoid(input), target)


class LossBinaryDice(nn.Module):
    def __init__(self, dice_weight=1):
        super(LossBinaryDice, self).__init__()        
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def forward(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        if self.dice_weight:
            smooth = 1e-500
            target = (targets > 0.0).float()
            prediction = F.sigmoid(outputs)
#             prediction = (prediction>.5).float()
            dice_part = 1 - (2*torch.sum(prediction * target) + smooth) / \
                            (torch.sum(prediction) + torch.sum(target) + smooth)


            loss += self.dice_weight * dice_part
        return loss
