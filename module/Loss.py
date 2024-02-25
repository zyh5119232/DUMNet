from __future__ import print_function, division
from torch.autograd import Variable
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class MultiClassBCE(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        loss = []
        for i in range(inputs.shape[1]):
            yp = inputs[:, i]
            yt = targets[:, i]
            BCE = F.binary_cross_entropy(yp, yt, reduction='mean')

            if i == 0:
                loss = BCE
            else:
                loss += BCE

        return loss


class IEL(nn.Module):
    def __init__(self):
        super(IEL, self).__init__()
        self.eps = 1e-6
        self.bceloss = nn.BCELoss(reduction='mean')

    def edgeLoss(self, pred, mask):
        edge = mask - F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)
        edge[edge != 0] = 1
        edge_tp = (edge * (pred - mask).abs_()).sum([2, 3])
        edgeNum = edge.sum([2, 3]) + self.eps
        loss = (edge_tp / edgeNum).mean()

        return loss

    def regionLoss(self, pred, mask):
        tp = (pred * mask).sum([2, 3])
        tp_fp_fn = (pred + mask - pred * mask).sum([2, 3]) + self.eps
        loss = (1 - tp / tp_fp_fn).mean()

        return loss

    def forward(self, pred, mask):
        bce = self.bceloss(pred, mask)
        edge = self.edgeLoss(pred, mask)
        region = self.regionLoss(pred, mask)
        total = bce + edge + region
        return total


class sc_IEL(nn.Module):
    def __init__(self):
        super(sc_IEL, self).__init__()
        self.eps = 1e-6
        self.bceloss = nn.BCELoss(reduction='mean')

    def edgeLoss(self, pred, mask):
        edge = mask - F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)
        edge[edge != 0] = 1
        edge_tp = (edge * (pred - mask).abs_()).sum([2, 3])
        edgeNum = edge.sum([2, 3]) + self.eps
        loss = (edge_tp / edgeNum).mean()

        return loss

    def regionLoss(self, pred, mask):
        tp = (pred * mask).sum([2, 3])
        tp_fp_fn = (pred + mask - pred * mask).sum([2, 3]) + self.eps
        loss = (1 - tp / tp_fp_fn).mean()

        return loss

    def forward(self, pred, mask):
        bce = self.bceloss(pred, mask)
        # edge = self.edgeLoss(pred, mask)
        # region = self.regionLoss(pred, mask)
        total = bce
        return total


# ceshi
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class Polyp(nn.Module):
    def __init__(self):
        super(Polyp, self).__init__()
        self.eps = 1e-6
        self.bceloss = nn.BCEWithLogitsLoss()
        # self.lovasz = lovasz_hinge(logits, labels)

    def structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()

    def edgeLoss(self, pred, mask):

        edge = mask - F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)
        edge[edge != 0] = 1
        edge_tp = (edge * (pred - mask).abs_()).sum([2, 3])
        edgeNum = edge.sum([2, 3]) + self.eps
        loss = (edge_tp / edgeNum).mean()

        return loss

    def regionLoss(self, pred, mask):

        tp = (pred * mask).sum([2, 3])
        tp_fp_fn = (pred + mask - pred * mask).sum([2, 3]) + self.eps
        loss = (1 - tp / tp_fp_fn).mean()

        return loss

    def diceloss(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

    ## 不确定性损失
    def ual_loss(self, pred, mask: torch.Tensor, iter_percentage: float = 0, method="cos"):
        if method == "linear":
            milestones = (0.3, 0.7)
            coef_range = (0, 1)
            min_point, max_point = min(milestones), max(milestones)
            min_coef, max_coef = min(coef_range), max(coef_range)
            if iter_percentage < min_point:
                ual_coef = min_coef
            elif iter_percentage > max_point:
                ual_coef = max_coef
            else:
                ratio = (max_coef - min_coef) / (max_point - min_point)
                ual_coef = ratio * (iter_percentage - min_point)
        elif method == "cos":
            coef_range = (0, 1)
            min_coef, max_coef = min(coef_range), max(coef_range)
            normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
            ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
        else:
            ual_coef = 1.0

        assert pred.shape == mask.shape, (pred.shape, mask.shape)
        sigmoid_x = pred.sigmoid()
        loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
        loss = loss_map.mean()
        ual_loss = loss
        ual_loss *= ual_coef
        return ual_loss
    ## 不确定性损失

    # # Gdice loss
    # def Gdiceloss(self, pred, mask,smooth = 1):
    #     # y_true,y_pred shape=[num_label,H,W,C]
    #     num_label = pred.shape[0]
    #     w = K.zeros(shape=(num_label,))
    #     w = K.sum(mask, axis=(1, 2, 3))
    #     w = 1 / (w ** 2 + 0.000001)
    #     # Compute gen dice coef:
    #     intersection_w = w * K.sum(mask * pred, axis=[1, 2, 3])
    #     union_w = w * K.sum(mask + pred, axis=[1, 2, 3])
    #     loss =  K.mean((2. * intersection_w + smooth) / (union_w + smooth), axis=0)
    #     return 1 - loss

    def iou_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return wiou.mean()

    def seg_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

    def Kappa_loss(self, pred, mask, N=384 * 384):  # 更考虑背景像素 相当于dice的升级
        Gi = torch.flatten(mask)
        Pi = torch.flatten(pred)
        # Gi = K.flatten(mask)
        # Pi = K.flatten(pred)
        numerator = 2 * torch.sum(Pi * Gi) - torch.sum(Pi) * torch.sum(Gi) / N
        denominator = torch.sum(Pi * Pi) + torch.sum(Gi * Gi) - 2 * torch.sum(Pi * Gi) / N
        Kappa_loss = 1 - numerator / denominator
        return Kappa_loss

    def tversky_loss(inputs, targets, beta=0.7, weights=None):  # cunyi
        batch_size = targets.size(0)
        loss = 0.0

        for i in range(batch_size):
            prob = inputs[i]
            ref = targets[i]

            alpha = 1.0 - beta

            tp = (ref * prob).sum()
            fp = ((1 - ref) * prob).sum()
            fn = (ref * (1 - prob)).sum()
            tversky = tp / (tp + alpha * fp + beta * fn)
            loss = loss + (1 - tversky)
        return loss / batch_size

    # class WeightedFocalLoss(nn.Module):
    #     "Non weighted version of Focal Loss"

    def WeightedFocalLoss(self, pred, mask, alpha=.25, gamma=2):
        alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        gamma = gamma
        BCE_loss = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        mask = mask.type(torch.long)
        at = alpha.gather(0, mask.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** gamma * BCE_loss
        return F_loss.mean()

    def forward(self, pred, mask):
        # print (pred.shape)
        # print (mask.shape)
        structure_loss = self.structure_loss(pred, mask)
        bce = self.bceloss(pred, mask)
        edge = self.edgeLoss(pred, mask)
        region = self.regionLoss(pred, mask)
        dice = self.diceloss(pred, mask)
        iou = self.iou_loss(pred, mask)
        seg_loss = self.seg_loss(pred, mask)
        # lovasz_loss = lovasz_hinge(pred, mask)
        # ual = self.ual_loss(pred, mask)
        # total = bce + edge + region + dice
        total = seg_loss * 2 + region
        # return structure_loss
        return bce + iou
        # return bce+dice
        # return bce+dice+lovasz_loss*0.1


class GuideLoss(nn.Module):
    def __init__(self):
        super(GuideLoss, self).__init__()
        self.eps = 1e-6
        self.bceloss = nn.BCELoss(reduction='mean')

    def cross_entropy2d(self, input, target, temperature=1):
        T = temperature
        loss = self.bceloss(input / T, target)
        return loss

    def KD_KLDivLoss(self, Stu_output, Tea_output, temperature):
        T = temperature
        KD_loss = nn.KLDivLoss()(F.log_softmax(Stu_output / T, dim=1), F.softmax(Tea_output / T, dim=1))
        KD_loss = KD_loss * T * T
        return KD_loss

    def edgeLoss(self, pred, mask):
        edge = mask - F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)
        edge[edge != 0] = 1
        edge_tp = (edge * (pred - mask).abs_()).sum([2, 3])
        edgeNum = edge.sum([2, 3]) + self.eps
        loss = (edge_tp / edgeNum).mean()

        return loss

    def regionLoss(self, pred, mask):
        tp = (pred * mask).sum([2, 3])
        tp_fp_fn = (pred + mask - pred * mask).sum([2, 3]) + self.eps
        loss = (1 - tp / tp_fp_fn).mean()

        return loss

    def forward(self, pred, mask, teacher):
        loss_depth = self.cross_entropy2d(teacher.detach(), mask, temperature=20)
        loss_gt = self.cross_entropy2d(pred, mask, temperature=1)
        LIPU_loss = self.KD_KLDivLoss(pred, teacher.detach(), temperature=20)
        alpha = math.exp(-70 * loss_depth)
        loss_adptative = (1 - alpha) * loss_gt + alpha * LIPU_loss

        # bceloss = self.bceloss(pred, mask)
        edge = self.edgeLoss(pred, mask)
        region = self.regionLoss(pred, mask)
        total = loss_adptative + edge + region
        return total


class MSL(nn.Module):
    def __init__(self):
        super(MSL, self).__init__()
        self.bceloss = nn.BCELoss(reduction='mean')

    def forward(self, pred, mask):
        total = 0
        for p in pred:
            m = F.interpolate(mask, size=p.size()[2:], mode="bilinear", align_corners=True)
            total += self.bceloss(p, m)
        return total

# Lovasz loss
"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""
try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


# 函数输入是一个排过序的 标签组  越靠近前面的标签 表示这个像素点与真值的误差越大
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    # print("p = ", p)
    # print("gt_sorted = ", gt_sorted)
    gts = gt_sorted.sum()  # 求个和
    # gt_sorted.float().cumsum(0) 1维的 计算的是累加和 例如 【1 2 3 4 5】 做完后就是【1 3 6 10 15】
    # 这个intersection是用累加和的值按维度减 累加数组的值，目的是做啥呢  看字面是取交集
    intersection = gts - gt_sorted.float().cumsum(0)  # 对应论文Algorithm 1的第3行
    union = gts + (1 - gt_sorted).float().cumsum(0)  # 对应论文Algorithm 1的第4行
    jaccard = 1. - intersection / union  # 对应论文Algorithm 1的第5行
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]  # 对应论文Algorithm 1的第7行
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)  # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]  # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    print("probas.shape = ", probas.shape)
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        # lovasz_softmax_flat的输入就是probas 【262144 2】  labels【262144】
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


# 这个函数是计算损失函数的部位
def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    # 预测像素点个数，一张512*512的图
    if probas.numel() == 0:  # 返回数组中元素的个数
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)  # 获得通道数呗  就是预测几类
    losses = []
    # class_to_sum = [0 1]  类的种类总数 用list存储
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c  如果语义标注数据与符合第c类，fg中存储1.0样数据
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]  # 取出第c类预测值 是介于 0~1之间的float数
        # errors 是预测结果与标签结果差的绝对值
        errors = (Variable(fg) - class_pred).abs()
        # 对误差排序 从大到小排   perm是下标值 errors_sorted 是排序后的预测值
        errors_sorted, perm = torch.sort(errors, 0, descending=True)

        perm = perm.data
        # 排序后的标签值
        fg_sorted = fg[perm]

        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    # 在这维度为probas 【1 2 512 512】  labels维度为【1 1 512 512】
    if probas.dim() == 3:  # dim()数组维度
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()  # 数组维度
    # 维度交换并变形   将probas.permute(0, 2, 3, 1)变换后的前3维合并成1维，通道不变
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    #
    labels = labels.view(-1)
    # 我的代码是用默认值 直接返回了 probas  labels  两个压缩完事的东西
    # 在这维度为probas 【262144 2】  labels维度为【262144】

    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

