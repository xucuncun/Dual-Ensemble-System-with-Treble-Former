import torch
from torch import Tensor
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt



class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray, device='cuda') -> np.ndarray:

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.percentile(distances[indexes], 95))

    def compute(self, pred: torch.Tensor, target: torch.Tensor, device='cuda') -> torch.Tensor:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
            ), "Only binary channel supported"

        pred = (pred > 0.5).byte()
        target = (target > 0.5).byte()
        if torch.sum(pred) == 0:
            pred[0][0][0][0] = 1
        if torch.sum(target) == 0:
            target[0][0][0][0] = 1
        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
            ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
            ).float()

        return torch.max(right_hd, left_hd).to(device=device)

hd_distance_95 = HausdorffDistance()

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)

        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

def iou_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)

        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (inter + epsilon) / (sets_sum - inter + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += iou_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def comment_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6, device='cuda'):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        W, H = target.shape
        tem = torch.ones(W, H).to(device=device)
        tp = torch.dot(input.reshape(-1), target.reshape(-1))
        fp = torch.dot(input.reshape(-1), ((target * -1) + tem).reshape(-1))
        fn = torch.dot(((input * -1) + tem).reshape(-1), target.reshape(-1))
        tn = torch.dot(((input * -1) + tem).reshape(-1), ((target * -1) + tem).reshape(-1))

        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        # in facet use 'tp' replace 'inter', use 'fp + fn + tp + tp' replace 'sets_sum' is completely practicable,
        # But I'm worried that the editor and reader won't believe this simple calculation method,
        # because some SCI Zone 1 papers use 'inter' and 'sets_sum' to calculate dice and iou
        # dice0 = (2 * tp + epsilon) / (fp + fn + tp + tp + epsilon)
        # iou0 = (tp + epsilon) / (fp + fn + tp + epsilon)
        # dice1 = (2 * inter + epsilon) / (sets_sum + epsilon)
        # iou1 = (inter + epsilon) / (sets_sum - inter + epsilon)
        # print('dice0=', dice0)
        # print('dice1=', dice1)
        # print('iou0=', iou0)
        # print('iou1=', iou1)

        dice = (2 * inter + epsilon) / (sets_sum + epsilon)
        iou = (inter + epsilon) / (sets_sum - inter + epsilon)
        Pre = (tp + epsilon) / (tp + fp + epsilon)
        Rec = (tp + epsilon) / (tp + fn + epsilon)
        Spe = (tn + epsilon) / (tn + fp + epsilon)
        Acc = (tp + tn + epsilon) / (tp + tn + fp + fn + epsilon)
        F2 = (5 * Pre * Rec) / (4 * (Pre + Rec))
        return dice, iou, Pre, Rec, Spe, Acc, F2
    else:
        # compute and average metric for each batch element
        Dice, IoU, PP, RR, SS, AA, FF = 0, 0, 0, 0, 0, 0, 0
        for i in range(input.shape[0]):
            dice, iou, pre, rec, spe, acc, f2 = comment_coeff(input[i, ...], target[i, ...], device=device)
            Dice += dice
            IoU += iou
            PP += pre
            RR += rec
            SS += spe
            AA += acc
            FF += f2
        Dice = Dice / input.shape[0]
        IoU = IoU / input.shape[0]
        PP = PP / input.shape[0]
        RR = RR / input.shape[0]
        SS = SS / input.shape[0]
        AA = AA / input.shape[0]
        FF = FF / input.shape[0]
        return Dice, IoU, PP, RR, SS, AA, FF





def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

def multiclass_iou_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += iou_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def multiclass_comment_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, device='cuda', epsilon=1e-6):
    # Average of miou coefficient for all classes
    assert input.size() == target.size()
    mDice, mIoU, mPP, mRR, mSS, mAA, mFF = 0, 0, 0, 0, 0, 0, 0
    for channel in range(input.shape[1]):
        dice, iou, pre, rec, spe, acc, f2 = comment_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon, device=device)
        mDice += dice
        mIoU += iou
        mPP += pre
        mRR += rec
        mSS += spe
        mAA += acc
        mFF += f2
    mDice = mDice / input.shape[1]
    mIoU = mIoU / input.shape[1]
    mPP = mPP / input.shape[1]
    mRR = mRR / input.shape[1]
    mSS = mSS / input.shape[1]
    mAA = mAA / input.shape[1]
    mFF = mFF / input.shape[1]

    return mDice, mIoU, mPP, mRR, mSS, mAA, mFF


def multiclass_comment_coeff_onepic(input: Tensor, target: Tensor, reduce_batch_first: bool = False, device='cuda', epsilon=1e-6):

    assert input.size() == target.size()

    # mDice_onepic, mIoU_onepic, mPP_onepic, mRR_onepic, mSS_onepic, mAA_onepic, mFF_onepic = 0, 0, 0, 0, 0, 0, 0
    mDice_list, mIoU_list, mPP_list, mRR_list, mSS_list, mAA_list, mFF_list, mHD95_list = [], [], [], [], [], [], [], []
    Dice_list, IoU_list, PP_list, RR_list, SS_list, AA_list, FF_list, HD95_list = [], [], [], [], [], [], [], []

    for N in range(input.shape[0]):
        mDice_onepic, mIoU_onepic, mPP_onepic, mRR_onepic, mSS_onepic, mAA_onepic, mFF_onepic, mHD95_onepic = 0, 0, 0, 0, 0, 0, 0, 0
        for channel in range(input.shape[1]):
            dice, iou, pre, rec, spe, acc, f2 = comment_coeff(input[N, channel, ...], target[N, channel, ...], reduce_batch_first, epsilon, device=device)
            HD95 = hd_distance_95.compute(input[N, channel, ...].unsqueeze(0).unsqueeze(0), target[N, channel, ...].unsqueeze(0).unsqueeze(0), device=device)
            if channel == 1:
                Dice_list.append(dice)
                IoU_list.append(iou)
                PP_list.append(pre)
                RR_list.append(rec)
                SS_list.append(spe)
                AA_list.append(acc)
                FF_list.append(f2)
                HD95_list.append(HD95)

            mDice_onepic += dice
            mIoU_onepic += iou
            mPP_onepic += pre
            mRR_onepic += rec
            mSS_onepic += spe
            mAA_onepic += acc
            mFF_onepic += f2
            mHD95_onepic += HD95

        mDice_onepic = mDice_onepic / input.shape[1]
        mIoU_onepic = mIoU_onepic / input.shape[1]
        mPP_onepic = mPP_onepic / input.shape[1]
        mRR_onepic = mRR_onepic / input.shape[1]
        mSS_onepic = mSS_onepic / input.shape[1]
        mAA_onepic = mAA_onepic / input.shape[1]
        mFF_onepic = mFF_onepic / input.shape[1]
        mHD95_onepic = mHD95_onepic / input.shape[1]

        mDice_list.append(mDice_onepic)
        mIoU_list.append(mIoU_onepic)
        mPP_list.append(mPP_onepic)
        mRR_list.append(mRR_onepic)
        mSS_list.append(mSS_onepic)
        mAA_list.append(mAA_onepic)
        mFF_list.append(mFF_onepic)
        mHD95_list.append(mHD95_onepic)

    return Dice_list, IoU_list, PP_list, RR_list, SS_list, AA_list, FF_list, HD95_list, mDice_list, mIoU_list, mPP_list, mRR_list, mSS_list, mAA_list, mFF_list, mHD95_list




def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)
    return acc


def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
        # TP : True Positive
        # FN : False Negative
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2
    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SP = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
        # TN : True Negative
        # FP : False Positive
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    return SP

def get_precision(SR,GT,threshold=0.5):
    PC = 0
    SR = SR > threshold
    GT = GT== torch.max(GT)
        # TP : True Positive
        # FP : False Positive
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)
    return PC
