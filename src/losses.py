import torch.nn as nn
import torch.nn.functional as F
import torch


def dice_score_combined(input, target, num_classes=3):

    EPS = 1e-10

    dice_target = F.one_hot(target, num_classes=3).to(torch.float32)[:, :, :, 1:]
    dice_input = F.softmax(input, dim=1)[:, 1:, :, :]
    dice_input = torch.transpose(dice_input, 1, 3)

    intersection = (dice_target * dice_input).sum()
    union = dice_target.sum() + dice_input.sum() + EPS

    return 2.0 * intersection / union

def dice_score(input, target, channel, num_classes=3):
    """
    Dice score computed separately for a body part and the tumor in it

    :param input:
    :param target:
    :param num_classes:
    :return:
    """
    EPS = 1e-10

    one_hot_mask = F.one_hot(target, num_classes=3).to(torch.float32)
    predict = F.softmax(input, dim=1)

    mask = one_hot_mask[:, :, :, channel]
    predict = torch.transpose(predict, 1, 3)[:, :, :, channel]

    intersection = torch.sum(mask * predict, dim=(1, 2))
    union = torch.sum(mask, dim=(1, 2)) + torch.sum(predict, dim=(1, 2))

    dice_score = 2 * torch.mean((intersection + EPS) / (union + EPS))

    return dice_score

class DiceScore(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.channel = channel

    def forward(self, input, target, num_classes=3):
        return dice_score(input, target, self.channel, num_classes)

class DiceScoreComb(nn.Module):
    # for metrics
    def __init__(self):
        super().__init__()

    def forward(self, input, target, num_classes=3):
        return dice_score_combined(input, target, num_classes=3)

class BodyDiceScore(DiceScore):
    # for metrics
    def __init__(self, channel=1):
        super().__init__(channel)
        self.channel = channel


class TumorDiceScore(DiceScore):
    # for metrics
    def __init__(self, channel=2):
        super().__init__(channel)
        self.channel = channel

class CrossEntropyDiceLoss(nn.Module):

    def __init__(self, body_weight=1.0, tumor_weight=1.0):
        super(CrossEntropyDiceLoss, self).__init__()
        self.body_weight = body_weight
        self.tumor_weight = tumor_weight
        self.class_weights = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float32, device='cuda')
        self.ce_loss = CrossEntropyLoss(weight=self.class_weights)
        self.body_dice_score = DiceScore(channel=1)
        self.tumor_dice_score = DiceScore(channel=2)

    def forward(self, logits, targets):
        return self.ce_loss(logits, targets) \
               + self.body_weight * (1 - self.body_dice_score(logits, targets)) \
               + self.tumor_weight * (1 - self.tumor_dice_score(logits, targets))


class CrossEntropyDiceLossComb(nn.Module):

    def __init__(self, dice_weight=1.0):
        super(CrossEntropyDiceLossComb, self).__init__()
        self.dice_weight = dice_weight
        self.ce_loss = CrossEntropyLoss()


    def forward(self, logits, targets):
        return self.ce_loss(logits, targets) \
               + self.dice_weight * (1 - dice_score_combined(logits, targets))



class GeneralizedDiceLoss2D(nn.Module):

    def __init__(self):
        """
        Implementation of Generalized Dice Loss from https://arxiv.org/pdf/1707.03237.pdf
        """
        super(GeneralizedDiceLoss2D, self).__init__()

    def forward(self, logits, masks):
        """
        Computes Generalized Dice Loss

        :param logits: tensor of shape batch_size x c x d1 x d2 ... x dk , where c is the number of classes, k is the
                       number of image dimensions
        :param masks: tensor of shape batch_size x d1 x d2 ... x dk consists of class labels (integers) of each element
        :return:
        """

        EPS = 1e-10

        one_hot_masks = F.one_hot(masks, num_classes=3).to(torch.float32)
        probas = F.softmax(logits, dim=1)
        probas = torch.transpose(probas, 1, 3)

        weights = (torch.sum(one_hot_masks, dim=(1, 2)))/ (torch.sum(one_hot_masks, dim=(1, 2, 3), keepdim=True) + EPS)


        intersection = torch.sum(torch.sum(probas * one_hot_masks, dim=(1, 2)) * weights)
        union = torch.sum(torch.sum(probas + one_hot_masks, dim=(1, 2)) * weights)

        dice_loss = 1 - 2 * (intersection + EPS) / (union + EPS)

        return dice_loss

######################################


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight, reduction=reduction)

    def forward(self, logits, targets):
        return self.loss(logits, targets)

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
