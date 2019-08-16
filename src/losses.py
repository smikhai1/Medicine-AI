import torch.nn as nn
import torch.nn.functional as F
import torch

def dice_score(input, target, channel, num_classes=3):
    """
    Dice score computed separately for a specified channel

    :param input:
    :param target:
    :param num_classes:
    :return:
    """
    EPS = 1e-10

    one_hot_mask = F.one_hot(target, num_classes=3).to(torch.float32)
    predict = F.softmax(input, dim=1)

    mask = one_hot_mask[:, :, :, channel]
    predict = predict[:, channel, :, :]

    intersection = torch.sum(mask * predict)
    union = torch.sum(mask) + torch.sum(predict)

    dice_score = 2 * ((intersection + EPS) / (union + EPS))

    return dice_score

def dice_score3d(input, target, channel, num_classes=3):
    """
    Dice score computed separately for a specified channel

    :param input:
    :param target:
    :param num_classes:
    :return:
    """
    EPS = 1e-10

    one_hot_mask = F.one_hot(target, num_classes=3).to(torch.float32)
    predict = F.softmax(input, dim=1)

    mask = one_hot_mask[:, :, :, :, channel]
    predict = predict[:, channel, :, :, :]

    intersection = torch.sum(mask * predict)
    union = torch.sum(mask) + torch.sum(predict)

    dice_score = 2 * ((intersection + EPS) / (union + EPS))

    return dice_score

class DiceScore(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.channel = channel

    def forward(self, input, target, num_classes=3):
        return dice_score3d(input, target, self.channel, num_classes)


class BackgroundDiceScore(DiceScore):
    # for metrics
    def __init__(self, channel=0):
        super().__init__(channel)
        self.channel = channel


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

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        if weight:
            self.weight = torch.tensor(weight, device='cuda')
        else:
            self.weight = None

        self.loss = nn.CrossEntropyLoss(self.weight, reduction=reduction)

    def forward(self, logits, targets):
        return self.loss(logits, targets)

class CrossEntropyDiceLoss(nn.Module):

    def __init__(self, body_weight=1.0, tumor_weight=1.0):
        super(CrossEntropyDiceLoss, self).__init__()
        self.body_weight = body_weight
        self.tumor_weight = tumor_weight
        self.ce_loss = CrossEntropyLoss()
        self.body_dice_score = DiceScore(channel=1)
        self.tumor_dice_score = DiceScore(channel=2)

    def forward(self, logits, targets):
        return self.ce_loss(logits, targets) \
               + self.body_weight * (1 - self.body_dice_score(logits, targets)) \
               + self.tumor_weight * (1 - self.tumor_dice_score(logits, targets))


class GeneralizedDiceLoss2D(nn.Module):

    def __init__(self):
        """
        Implementation of Generalized Dice Loss from https://arxiv.org/pdf/1707.03237.pdf

        Doesn't work!
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

        EPS = 1e-8

        one_hot_masks = F.one_hot(masks, num_classes=3).to(torch.float32)
        probas = F.softmax(logits, dim=1)
        probas = probas.permute(0, 2, 3, 1)

        weights = 1.0 / (torch.pow(torch.sum(one_hot_masks, dim=(1, 2)), 1) + EPS)

        intersection = torch.sum(torch.sum(probas * one_hot_masks, dim=(1, 2)) * weights)
        union = torch.sum(torch.sum(probas + one_hot_masks, dim=(1, 2)) * weights)

        dice_loss =  1 - 2 * (intersection + EPS) / (union + EPS)

        return dice_loss


class MeanDiceLoss2D(nn.Module):

    def __init__(self):
        """
        Implementation of Mean Dice Loss from https://arxiv.org/pdf/1707.01992.pdf
        """
        super(MeanDiceLoss2D, self).__init__()

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
        probas = probas.permute(0, 2, 3, 1)

        intersection = torch.sum(probas * one_hot_masks, dim=(1, 2))
        union = torch.sum(torch.pow(probas, 2), dim=(1, 2)) + torch.sum(torch.pow(one_hot_masks, 2), dim=(1, 2))
        dice_loss =  torch.mean(2 * (intersection + EPS) / (union + EPS), dim=1)

        return -torch.mean(dice_loss)

class CrossEntropyDiceLoss2D(nn.Module):

    def __init__(self, dice_weight=0.7):
        super(CrossEntropyDiceLoss2D, self).__init__()
        self.dice_weight = dice_weight
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = MeanDiceLoss2D()

    def forward(self, logits, targets):

        return self.ce_loss(logits, targets) \
               + self.dice_weight * self.dice_loss(logits, targets)

class MeanDiceLoss3D(nn.Module):

    def __init__(self):
        """
        Implementation of Mean Dice Loss from https://arxiv.org/pdf/1707.01992.pdf
        """
        super(MeanDiceLoss3D, self).__init__()

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
        probas = probas.permute(0, 2, 3, 4, 1)

        intersection = torch.sum(probas * one_hot_masks, dim=(1, 2, 3))
        union = torch.sum(torch.pow(probas, 2), dim=(1, 2, 3)) + torch.sum(torch.pow(one_hot_masks, 2), dim=(1, 2, 3))
        dice_loss =  torch.mean(2 * (intersection + EPS) / (union + EPS), dim=1)

        return -torch.mean(dice_loss)

class CrossEntropyDiceLoss3D(nn.Module):

    def __init__(self, dice_weight=0.7):
        super(CrossEntropyDiceLoss3D, self).__init__()
        self.dice_weight = dice_weight
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = MeanDiceLoss3D()

    def forward(self, logits, targets):

        return self.ce_loss(logits, targets) \
               + self.dice_weight * self.dice_loss(logits, targets)


class TverskyLoss2D(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5):
        super(TverskyLoss2D, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, masks):

        one_hot_masks = F.one_hot(masks, num_classes=3).to(torch.float32)
        probas = F.softmax(logits, dim=1)
        probas = probas.permute(0, 2, 3, 1)

        ones = torch.ones_like(one_hot_masks)
        p0 = probas
        p1 = ones - probas
        g0 = one_hot_masks
        g1 = ones - one_hot_masks

        num = torch.sum(p0 * g0, dim=(0, 1, 2))
        denum = num + self.alpha * torch.sum(p0 * g1, dim=(0, 1, 2)) + self.beta * torch.sum(p1 * g0, dim=(0, 1, 2))

        T = torch.sum(num / denum)

        return 3 - T


######################################
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
