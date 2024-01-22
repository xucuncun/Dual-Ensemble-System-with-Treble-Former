import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import get_sensitivity, get_precision ,get_specificity ,get_accuracy
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.dice_score import multiclass_comment_coeff, comment_coeff, multiclass_comment_coeff_onepic
from utils.dice_score import multiclass_iou_coeff


def evaluate_test(net, dataloader, device, w_ustm, w_ust, w_m, w_us, w_t, ornot):
    net.eval()
    num_val_batches = len(dataloader)
    mask_preds = [None] * num_val_batches
    epoch_eval = int(-1)
    Dice, IoU, Pre, Rre, Spe, Acc, F2 = 0, 0, 0, 0, 0, 0, 0
    mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2 = 0, 0, 0, 0, 0, 0, 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Test round', unit='batch', leave=False):
        epoch_eval += 1
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        # print('mask_true.shape=', mask_true.shape)
        mask_true = F.one_hot(mask_true.squeeze_(1), net.n_classes).permute(0, 3, 1, 2).float()


        with torch.no_grad():
            # predict the mask
            if ornot:
                deep_out = net(image)
                mask_pred = deep_out[0] * w_ustm + deep_out[1] * w_ust + deep_out[2] * w_m + deep_out[3] * w_us + deep_out[4] * w_t
            else:
                mask_pred = net(image)
            mask_preds[epoch_eval] = mask_pred.cpu()

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice, iou, pre, rec, spe, acc, f2 = comment_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                mdice, miou, mpre, mrec, mspe, macc, mf2 = multiclass_comment_coeff(mask_pred[:, 0:, ...],
                                                                               mask_true[:, 0:, ...],
                                                                               reduce_batch_first=False)
                dice, iou, pre, rec, spe, acc, f2 = multiclass_comment_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                   reduce_batch_first=False)
            Dice += dice
            IoU += iou
            Pre += pre
            Rre += rec
            Spe += spe
            Acc += acc
            F2 += f2

            mDice += mdice
            mIoU += miou
            mPre += mpre
            mRre += mrec
            mSpe += mspe
            mAcc += macc
            mF2 += mf2




    net.train()

    now_pred = [None] * (epoch_eval + 1)
    for i in range(epoch_eval + 1):
        now_pred[i] = mask_preds[i]

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        Dice = Dice
        IoU = IoU
        Pre = Pre
        Rre = Rre
        Spe = Spe
        Acc = Acc
        F2 = F2

        mDice = mDice
        mIoU = mIoU
        mPre = mPre
        mRre = mRre
        mSpe = mSpe
        mAcc = mAcc
        mF2 = mF2
    else:
        Dice = Dice / num_val_batches
        IoU = IoU / num_val_batches
        Pre = Pre / num_val_batches
        Rre = Rre / num_val_batches
        Spe = Spe / num_val_batches
        Acc = Acc / num_val_batches
        F2 = F2 / num_val_batches

        mDice = mDice / num_val_batches
        mIoU = mIoU / num_val_batches
        mPre = mPre / num_val_batches
        mRre = mRre / num_val_batches
        mSpe = mSpe / num_val_batches
        mAcc = mAcc / num_val_batches
        mF2 = mF2 / num_val_batches


    return Dice, IoU, Pre, Rre, Spe, Acc, F2, mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2, mask_preds, now_pred

def evaluate_val(net, dataloader, device, w_ustm, w_ust, w_m, w_us, w_t, popfits_masks, ornot):
    net.eval()
    num_val_batches = len(dataloader)
    mask_preds = [None] * num_val_batches
    epoch_eval = int(-1)
    dice_score = 0
    dice1 = 0
    dice2 = 0
    dice3 = 0
    dice4 = 0
    dice5 = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

        epoch_eval += 1
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            if ornot:
                deep_out = net(image)
                combine_out = deep_out[0] * w_ustm + deep_out[1] * w_ust + deep_out[2] * w_m + deep_out[3] * w_us + deep_out[4] * w_t
            else:
                combine_out = net(image)
            mask_preds[epoch_eval] = combine_out.cpu()


            # convert to one-hot format
            if net.n_classes == 1:
                combine_out = (F.sigmoid(combine_out) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(combine_out, mask_true, reduce_batch_first=False)
            else:
                combine_out = F.one_hot(combine_out.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(combine_out[:, 0:, ...], mask_true[:, 0:, ...],
                                                    reduce_batch_first=False)

            if ornot:
                deep_out1 = F.one_hot(deep_out[0].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice1 += multiclass_dice_coeff(deep_out1[:, 0:, ...], mask_true[:, 0:, ...],
                                                    reduce_batch_first=False)
                deep_out2 = F.one_hot(deep_out[1].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice2 += multiclass_dice_coeff(deep_out2[:, 0:, ...], mask_true[:, 0:, ...],
                                                    reduce_batch_first=False)
                deep_out3 = F.one_hot(deep_out[2].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice3 += multiclass_dice_coeff(deep_out3[:, 0:, ...], mask_true[:, 0:, ...],
                                                    reduce_batch_first=False)
                deep_out4 = F.one_hot(deep_out[3].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice4 += multiclass_dice_coeff(deep_out4[:, 0:, ...], mask_true[:, 0:, ...],
                                                    reduce_batch_first=False)
                deep_out5 = F.one_hot(deep_out[4].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice5 += multiclass_dice_coeff(deep_out5[:, 0:, ...], mask_true[:, 0:, ...],
                                                reduce_batch_first=False)

    net.train()

    popfits_masks[0] = popfits_masks[1]
    popfits_masks[1] = popfits_masks[2]
    popfits_masks[2] = popfits_masks[3]
    popfits_masks[3] = popfits_masks[4]
    popfits_masks[4] = popfits_masks[5]
    popfits_masks[5] = popfits_masks[6]
    popfits_masks[6] = popfits_masks[7]
    popfits_masks[7] = popfits_masks[8]
    popfits_masks[8] = popfits_masks[9]
    popfits_masks[9] = mask_preds

    # popfits_masks[0] = popfits_masks[6]
    # popfits_masks[1] = popfits_masks[6]
    # popfits_masks[2] = popfits_masks[6]
    # popfits_masks[3] = popfits_masks[6]
    # popfits_masks[4] = popfits_masks[6]
    # popfits_masks[5] = mask_preds
    # popfits_masks[6] = mask_preds
    # popfits_masks[7] = mask_preds
    # popfits_masks[8] = mask_preds
    # popfits_masks[9] = mask_preds

    now_pred = [None] * (epoch_eval + 1)
    for i in range(epoch_eval + 1):
        now_pred[i] = mask_preds[i]



    # Fixes a potential division by zero error
    if num_val_batches == 0:
        dice = dice_score
    else:
        dice = dice_score / num_val_batches
        dice1 = dice1 / num_val_batches
        dice2 = dice2 / num_val_batches
        dice3 = dice3 / num_val_batches
        dice4 = dice4 / num_val_batches
        dice5 = dice5 / num_val_batches


    return dice, dice1, dice2, dice3, dice4, dice5, popfits_masks, now_pred


def evaluate_ensemble_val(ensemble_pre, dataloader, net_n_classes, device):

    num_val_batches = len(dataloader)
    epoch_eval = int(-1)
    dice_score = 0


    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

        epoch_eval += 1
        _, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        mask_pred = ensemble_pre[epoch_eval]
        mask_pred = mask_pred.to(device=device)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()

        if net_n_classes == 1:
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            # compute the Dice score
            dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
        else:
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:, 0:, ...], mask_true[:, 0:, ...],
                                                    reduce_batch_first=False)

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        dice = dice_score
    else:
        dice = dice_score / num_val_batches

    return dice


def evaluate_ensemble_label(preloader, dataloader, pre_val, time, number_net, label, threshold, bestpop, net_n_classes, device):

    num_val_batches = len(dataloader)
    val_preds = [None] * num_val_batches
    test_preds = [None] * num_val_batches
    bestpop = torch.from_numpy(bestpop)
    bestpop = bestpop.to(device=device)
    epoch_eval = int(-1)
    Dice, IoU, Pre, Rre, Spe, Acc, F2 = 0, 0, 0, 0, 0, 0, 0
    mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2 = 0, 0, 0, 0, 0, 0, 0

    for batch_mask_true in tqdm(dataloader, total=num_val_batches, desc='Ensemble Validation round', unit='batch', leave=False):
        epoch_eval += 1
        mask_true = batch_mask_true['mask']
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre_test = torch.zeros(size=mask_true.shape)
        ensemable_mask_pre_test = ensemable_mask_pre_test.to(device=device)
        ensemable_mask_pre_val = torch.zeros(size=mask_true.shape)
        ensemable_mask_pre_val = ensemable_mask_pre_val.to(device=device)
        with torch.no_grad():
            label_time = int(0)
            for epoch_net in range(number_net):
                if label[epoch_net] >= threshold:
                    label_time += 1
                else:
                    ensemable_mask_pre_test += bestpop[epoch_net - label_time] * preloader[epoch_net][epoch_eval]
                    ensemable_mask_pre_val += bestpop[epoch_net - label_time] * pre_val[epoch_net][time][epoch_eval]
                    val_preds[epoch_eval] = ensemable_mask_pre_val.cpu()
                    test_preds[epoch_eval] = ensemable_mask_pre_test.cpu()
                # IndexError: index 2 is out of bounds for dimension 0 with size 2

            # convert to one-hot format
            if net_n_classes == 1:
                ensemable_mask_pre_test = (F.sigmoid(ensemable_mask_pre_test) > 0.5).float()
                # compute the Dice score
                dice, iou, pre, rec, spe, acc, f2 = comment_coeff(ensemable_mask_pre_test, mask_true, reduce_batch_first=False, device=device)
            else:
                ensemable_mask_pre_test = F.one_hot(ensemable_mask_pre_test.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                mdice, miou, mpre, mrec, mspe, macc, mf2 = multiclass_comment_coeff(ensemable_mask_pre_test[:, 0:, ...],
                                                                               mask_true[:, 0:, ...],
                                                                               reduce_batch_first=False, device=device)
                dice, iou, pre, rec, spe, acc, f2 = multiclass_comment_coeff(ensemable_mask_pre_test[:, 1:, ...], mask_true[:, 1:, ...],
                                             reduce_batch_first=False, device=device)
            Dice += dice
            IoU += iou
            Pre += pre
            Rre += rec
            Spe += spe
            Acc += acc
            F2 += f2

            mDice += mdice
            mIoU += miou
            mPre += mpre
            mRre += mrec
            mSpe += mspe
            mAcc += macc
            mF2 += mf2

    now_pred_val = [None] * (epoch_eval + 1)
    now_pred_test = [None] * (epoch_eval + 1)
    for i in range(epoch_eval + 1):
        now_pred_val[i] = val_preds[i]
        now_pred_test[i] = test_preds[i]
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        Dice = Dice
        IoU = IoU
        Pre = Pre
        Rre = Rre
        Spe = Spe
        Acc = Acc
        F2 = F2

        mDice = mDice
        mIoU = mIoU
        mPre = mPre
        mRre = mRre
        mSpe = mSpe
        mAcc = mAcc
        mF2 = mF2
    else:
        Dice = Dice / num_val_batches
        IoU = IoU / num_val_batches
        Pre = Pre / num_val_batches
        Rre = Rre / num_val_batches
        Spe = Spe / num_val_batches
        Acc = Acc / num_val_batches
        F2 = F2 / num_val_batches

        mDice = mDice / num_val_batches
        mIoU = mIoU / num_val_batches
        mPre = mPre / num_val_batches
        mRre = mRre / num_val_batches
        mSpe = mSpe / num_val_batches
        mAcc = mAcc / num_val_batches
        mF2 = mF2 / num_val_batches
    return Dice, IoU, Pre, Rre, Spe, Acc, F2, mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2, now_pred_val, now_pred_test

# def evaluate_ensemble_pl(preloader, dataloader, number_net, bestpop, net_n_classes, device):
#
#     num_val_batches = len(dataloader)
#     bestpop = torch.from_numpy(bestpop)
#     bestpop = bestpop.to(device=device)
#     epoch_eval = int(-1)
#     Dice, mDice, mIoU, Pre, Rre, Spe, Acc, F2 = 0, 0, 0, 0, 0, 0, 0, 0
#
#     for batch_mask_true in tqdm(dataloader, total=num_val_batches, desc='Ensemble Validation round', unit='batch', leave=False):
#         epoch_eval += 1
#         mask_true = batch_mask_true['mask']
#         mask_true = mask_true.to(device=device, dtype=torch.long)
#         mask_true = F.one_hot(mask_true.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
#         ensemable_mask_pre = torch.zeros(size=mask_true.shape)
#         ensemable_mask_pre = ensemable_mask_pre.to(device=device)
#         with torch.no_grad():
#             for epoch_net in range(number_net):
#                 ensemable_mask_pre += bestpop[epoch_net] * preloader[epoch_net][epoch_eval]
#                 # IndexError: index 2 is out of bounds for dimension 0 with size 2
#
#             # convert to one-hot format
#             if net_n_classes == 1:
#                 ensemable_mask_pre = (F.sigmoid(ensemable_mask_pre) > 0.5).float()
#                 # compute the Dice score
#                 _, _, pre, rec, spe, acc, f2 = comment_coeff(ensemable_mask_pre, mask_true, reduce_batch_first=False)
#                 dice = dice_coeff(ensemable_mask_pre, mask_true, reduce_batch_first=False)
#             else:
#                 ensemable_mask_pre = F.one_hot(ensemable_mask_pre.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
#                 # compute the Dice score, ignoring background
#                 mdice, miou, pre, rec, spe, acc, f2 = multiclass_comment_coeff(ensemable_mask_pre[:, 0:, ...],
#                                                                                mask_true[:, 0:, ...],
#                                                                                reduce_batch_first=False, device=device)
#                 dice = multiclass_dice_coeff(ensemable_mask_pre[:, 1:, ...], mask_true[:, 1:, ...],
#                                              reduce_batch_first=False)
#             Dice += dice
#             mDice += mdice
#             mIoU += miou
#             Pre += pre
#             Rre += rec
#             Spe += spe
#             Acc += acc
#             F2 += f2
#
#     # Fixes a potential division by zero error
#     if num_val_batches == 0:
#         Dice = Dice
#         mDice = mDice
#         mIoU = mIoU
#         Pre = Pre
#         Rre = Rre
#         Spe = Spe
#         Acc = Acc
#         F2 = F2
#     else:
#         Dice = Dice / num_val_batches
#         mDice = mDice / num_val_batches
#         mIoU = mIoU / num_val_batches
#         Pre = Pre / num_val_batches
#         Rre = Rre / num_val_batches
#         Spe = Spe / num_val_batches
#         Acc = Acc / num_val_batches
#         F2 = F2 / num_val_batches
#     return Dice, mDice, mIoU, Pre, Rre, Spe, Acc, F2

def evaluate_ensemble(preloader_test, dataloader_test, preloader_val, time, number_net, bestpop, net_n_classes, device):

    num_val_batches = len(dataloader_test)
    val_preds = [None] * num_val_batches
    test_preds = [None] * num_val_batches
    bestpop = torch.from_numpy(bestpop)
    bestpop = bestpop.to(device=device)
    epoch_eval = int(-1)
    bestdic = 0
    Dice, IoU, Pre, Rre, Spe, Acc, F2 = 0, 0, 0, 0, 0, 0, 0
    mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2 = 0, 0, 0, 0, 0, 0, 0

    for batch_mask_true_test in tqdm(dataloader_test, total=num_val_batches, desc='Ensemble Test round', unit='batch', leave=False):
        epoch_eval += 1
        mask_true_test = batch_mask_true_test['mask']
        mask_true_test = mask_true_test.to(device=device, dtype=torch.long)
        mask_true_test = F.one_hot(mask_true_test.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre_test = torch.zeros(size=mask_true_test.shape)
        ensemable_mask_pre_test = ensemable_mask_pre_test.to(device=device)
        ensemable_mask_pre_val = torch.zeros(size=mask_true_test.shape)
        ensemable_mask_pre_val = ensemable_mask_pre_val.to(device=device)
        with torch.no_grad():
            for epoch_net in range(number_net):
                ensemable_mask_pre_test += bestpop[epoch_net] * preloader_test[epoch_net][epoch_eval]
                ensemable_mask_pre_val += bestpop[epoch_net] * preloader_val[epoch_net][time][epoch_eval]
                val_preds[epoch_eval] = ensemable_mask_pre_val.cpu()
                test_preds[epoch_eval] = ensemable_mask_pre_test.cpu()
                # IndexError: index 2 is out of bounds for dimension 0 with size 2


            # convert to one-hot format
            if net_n_classes == 1:
                ensemable_mask_pre_test = (F.sigmoid(ensemable_mask_pre_test) > 0.5).float()
                # compute the Dice score
                dice, iou, pre, rec, spe, acc, f2 = comment_coeff(ensemable_mask_pre_test, mask_true_test, reduce_batch_first=False, device=device)
            else:
                ensemable_mask_pre_test = F.one_hot(ensemable_mask_pre_test.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                mdice, miou, mpre, mrec, mspe, macc, mf2 = multiclass_comment_coeff(ensemable_mask_pre_test[:, 0:, ...],
                                                                               mask_true_test[:, 0:, ...],
                                                                               reduce_batch_first=False, device=device)
                dice, iou, pre, rec, spe, acc, f2 = multiclass_comment_coeff(ensemable_mask_pre_test[:, 1:, ...], mask_true_test[:, 1:, ...],
                                             reduce_batch_first=False, device=device)
            Dice += dice
            IoU += iou
            Pre += pre
            Rre += rec
            Spe += spe
            Acc += acc
            F2 += f2

            mDice += mdice
            mIoU += miou
            mPre += mpre
            mRre += mrec
            mSpe += mspe
            mAcc += macc
            mF2 += mf2

    now_pred_val = [None] * (epoch_eval + 1)
    now_pred_test = [None] * (epoch_eval + 1)
    for i in range(epoch_eval + 1):
        now_pred_val[i] = val_preds[i]
        now_pred_test[i] = test_preds[i]

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        Dice = Dice
        IoU = IoU
        Pre = Pre
        Rre = Rre
        Spe = Spe
        Acc = Acc
        F2 = F2

        mDice = mDice
        mIoU = mIoU
        mPre = mPre
        mRre = mRre
        mSpe = mSpe
        mAcc = mAcc
        mF2 = mF2
    else:
        Dice = Dice / num_val_batches
        IoU = IoU / num_val_batches
        Pre = Pre / num_val_batches
        Rre = Rre / num_val_batches
        Spe = Spe / num_val_batches
        Acc = Acc / num_val_batches
        F2 = F2 / num_val_batches

        mDice = mDice / num_val_batches
        mIoU = mIoU / num_val_batches
        mPre = mPre / num_val_batches
        mRre = mRre / num_val_batches
        mSpe = mSpe / num_val_batches
        mAcc = mAcc / num_val_batches
        mF2 = mF2 / num_val_batches
    return Dice, IoU, Pre, Rre, Spe, Acc, F2, mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2, now_pred_val, now_pred_test

def evaluate_ensembleA_test(preloader_test, dataloader_test, number_net,  bestpop, net_n_classes, device):

    num_val_batches = len(dataloader_test)
    test_preds = [None] * num_val_batches
    bestpop = torch.from_numpy(bestpop)
    bestpop = bestpop.to(device=device)
    epoch_eval = int(-1)
    bestdic = 0
    Dice, IoU, Pre, Rre, Spe, Acc, F2 = 0, 0, 0, 0, 0, 0, 0
    mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2 = 0, 0, 0, 0, 0, 0, 0

    for batch_mask_true_test in tqdm(dataloader_test, total=num_val_batches, desc='Ensemble Test round', unit='batch', leave=False):
        epoch_eval += 1
        mask_true_test = batch_mask_true_test['mask']
        mask_true_test = mask_true_test.to(device=device, dtype=torch.long)
        mask_true_test = F.one_hot(mask_true_test.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre_test = torch.zeros(size=mask_true_test.shape)
        ensemable_mask_pre_test = ensemable_mask_pre_test.to(device=device)
        with torch.no_grad():
            for epoch_net in range(number_net):
                ensemable_mask_pre_test += bestpop[epoch_net] * preloader_test[epoch_net][epoch_eval]
                test_preds[epoch_eval] = ensemable_mask_pre_test.cpu()
                # IndexError: index 2 is out of bounds for dimension 0 with size 2

            # convert to one-hot format
            if net_n_classes == 1:
                ensemable_mask_pre_test = (F.sigmoid(ensemable_mask_pre_test) > 0.5).float()
                # compute the Dice score
                dice, iou, pre, rec, spe, acc, f2 = comment_coeff(ensemable_mask_pre_test, mask_true_test, reduce_batch_first=False, device=device)
            else:
                ensemable_mask_pre_test = F.one_hot(ensemable_mask_pre_test.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                mdice, miou, mpre, mrec, mspe, macc, mf2 = multiclass_comment_coeff(ensemable_mask_pre_test[:, 0:, ...],
                                                                               mask_true_test[:, 0:, ...],
                                                                               reduce_batch_first=False, device=device)
                dice, iou, pre, rec, spe, acc, f2 = multiclass_comment_coeff(ensemable_mask_pre_test[:, 1:, ...], mask_true_test[:, 1:, ...],
                                             reduce_batch_first=False, device=device)
            Dice += dice
            IoU += iou
            Pre += pre
            Rre += rec
            Spe += spe
            Acc += acc
            F2 += f2

            mDice += mdice
            mIoU += miou
            mPre += mpre
            mRre += mrec
            mSpe += mspe
            mAcc += macc
            mF2 += mf2

    now_pred_test = [None] * (epoch_eval + 1)
    for i in range(epoch_eval + 1):
        now_pred_test[i] = test_preds[i]

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        Dice = Dice
        IoU = IoU
        Pre = Pre
        Rre = Rre
        Spe = Spe
        Acc = Acc
        F2 = F2

        mDice = mDice
        mIoU = mIoU
        mPre = mPre
        mRre = mRre
        mSpe = mSpe
        mAcc = mAcc
        mF2 = mF2
    else:
        Dice = Dice / num_val_batches
        IoU = IoU / num_val_batches
        Pre = Pre / num_val_batches
        Rre = Rre / num_val_batches
        Spe = Spe / num_val_batches
        Acc = Acc / num_val_batches
        F2 = F2 / num_val_batches

        mDice = mDice / num_val_batches
        mIoU = mIoU / num_val_batches
        mPre = mPre / num_val_batches
        mRre = mRre / num_val_batches
        mSpe = mSpe / num_val_batches
        mAcc = mAcc / num_val_batches
        mF2 = mF2 / num_val_batches
    return Dice, IoU, Pre, Rre, Spe, Acc, F2, mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2, now_pred_test

def evaluate_ensembleB_test(preloader_test, dataloader_test, number_net, label, threshold,  bestpop, net_n_classes, device):

    num_val_batches = len(dataloader_test)
    test_preds = [None] * num_val_batches
    bestpop = torch.from_numpy(bestpop)
    bestpop = bestpop.to(device=device)
    epoch_eval = int(-1)
    bestdic = 0
    Dice, IoU, Pre, Rre, Spe, Acc, F2 = 0, 0, 0, 0, 0, 0, 0
    mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2 = 0, 0, 0, 0, 0, 0, 0

    for batch_mask_true_test in tqdm(dataloader_test, total=num_val_batches, desc='Ensemble Test round', unit='batch', leave=False):
        epoch_eval += 1
        mask_true_test = batch_mask_true_test['mask']
        mask_true_test = mask_true_test.to(device=device, dtype=torch.long)
        mask_true_test = F.one_hot(mask_true_test.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre_test = torch.zeros(size=mask_true_test.shape)
        ensemable_mask_pre_test = ensemable_mask_pre_test.to(device=device)
        with torch.no_grad():
            label_time = int(0)
            for epoch_net in range(number_net):
                if label[epoch_net] >= threshold:
                    label_time += 1
                else:
                    ensemable_mask_pre_test += bestpop[epoch_net - label_time] * preloader_test[epoch_net][epoch_eval]
                    test_preds[epoch_eval] = ensemable_mask_pre_test.cpu()
                # IndexError: index 2 is out of bounds for dimension 0 with size 2

            # convert to one-hot format
            if net_n_classes == 1:
                ensemable_mask_pre_test = (F.sigmoid(ensemable_mask_pre_test) > 0.5).float()
                # compute the Dice score
                dice, iou, pre, rec, spe, acc, f2 = comment_coeff(ensemable_mask_pre_test, mask_true_test, reduce_batch_first=False, device=device)
            else:
                ensemable_mask_pre_test = F.one_hot(ensemable_mask_pre_test.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                mdice, miou, mpre, mrec, mspe, macc, mf2 = multiclass_comment_coeff(ensemable_mask_pre_test[:, 0:, ...],
                                                                               mask_true_test[:, 0:, ...],
                                                                               reduce_batch_first=False, device=device)
                dice, iou, pre, rec, spe, acc, f2 = multiclass_comment_coeff(ensemable_mask_pre_test[:, 1:, ...], mask_true_test[:, 1:, ...],
                                             reduce_batch_first=False, device=device)
            Dice += dice
            IoU += iou
            Pre += pre
            Rre += rec
            Spe += spe
            Acc += acc
            F2 += f2

            mDice += mdice
            mIoU += miou
            mPre += mpre
            mRre += mrec
            mSpe += mspe
            mAcc += macc
            mF2 += mf2

    now_pred_test = [None] * (epoch_eval + 1)
    for i in range(epoch_eval + 1):
        now_pred_test[i] = test_preds[i]

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        Dice = Dice
        IoU = IoU
        Pre = Pre
        Rre = Rre
        Spe = Spe
        Acc = Acc
        F2 = F2

        mDice = mDice
        mIoU = mIoU
        mPre = mPre
        mRre = mRre
        mSpe = mSpe
        mAcc = mAcc
        mF2 = mF2
    else:
        Dice = Dice / num_val_batches
        IoU = IoU / num_val_batches
        Pre = Pre / num_val_batches
        Rre = Rre / num_val_batches
        Spe = Spe / num_val_batches
        Acc = Acc / num_val_batches
        F2 = F2 / num_val_batches

        mDice = mDice / num_val_batches
        mIoU = mIoU / num_val_batches
        mPre = mPre / num_val_batches
        mRre = mRre / num_val_batches
        mSpe = mSpe / num_val_batches
        mAcc = mAcc / num_val_batches
        mF2 = mF2 / num_val_batches
    return Dice, IoU, Pre, Rre, Spe, Acc, F2, mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2, now_pred_test


def evaluate_ensemble(preloader_test, dataloader_test, preloader_val, time, number_net, bestpop, net_n_classes, device):
    num_val_batches = len(dataloader_test)
    val_preds = [None] * num_val_batches
    test_preds = [None] * num_val_batches
    bestpop = torch.from_numpy(bestpop)
    bestpop = bestpop.to(device=device)
    epoch_eval = int(-1)
    bestdic = 0
    Dice, IoU, Pre, Rre, Spe, Acc, F2 = 0, 0, 0, 0, 0, 0, 0
    mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2 = 0, 0, 0, 0, 0, 0, 0

    for batch_mask_true_test in tqdm(dataloader_test, total=num_val_batches, desc='Ensemble Test round', unit='batch',
                                     leave=False):
        epoch_eval += 1
        mask_true_test = batch_mask_true_test['mask']
        mask_true_test = mask_true_test.to(device=device, dtype=torch.long)
        mask_true_test = F.one_hot(mask_true_test.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre_test = torch.zeros(size=mask_true_test.shape)
        ensemable_mask_pre_test = ensemable_mask_pre_test.to(device=device)
        ensemable_mask_pre_val = torch.zeros(size=mask_true_test.shape)
        ensemable_mask_pre_val = ensemable_mask_pre_val.to(device=device)
        with torch.no_grad():
            for epoch_net in range(number_net):
                ensemable_mask_pre_test += bestpop[epoch_net] * preloader_test[epoch_net][epoch_eval]
                ensemable_mask_pre_val += bestpop[epoch_net] * preloader_val[epoch_net][time][epoch_eval]
                val_preds[epoch_eval] = ensemable_mask_pre_val.cpu()
                test_preds[epoch_eval] = ensemable_mask_pre_test.cpu()
                # IndexError: index 2 is out of bounds for dimension 0 with size 2

            # convert to one-hot format
            if net_n_classes == 1:
                ensemable_mask_pre_test = (F.sigmoid(ensemable_mask_pre_test) > 0.5).float()
                # compute the Dice score
                dice, iou, pre, rec, spe, acc, f2 = comment_coeff(ensemable_mask_pre_test, mask_true_test,
                                                                  reduce_batch_first=False, device=device)
            else:
                ensemable_mask_pre_test = F.one_hot(ensemable_mask_pre_test.argmax(dim=1), net_n_classes).permute(0, 3,
                                                                                                                  1,
                                                                                                                  2).float()
                # compute the Dice score, ignoring background
                mdice, miou, mpre, mrec, mspe, macc, mf2 = multiclass_comment_coeff(ensemable_mask_pre_test[:, 0:, ...],
                                                                                    mask_true_test[:, 0:, ...],
                                                                                    reduce_batch_first=False,
                                                                                    device=device)
                dice, iou, pre, rec, spe, acc, f2 = multiclass_comment_coeff(ensemable_mask_pre_test[:, 1:, ...],
                                                                             mask_true_test[:, 1:, ...],
                                                                             reduce_batch_first=False, device=device)
            Dice += dice
            IoU += iou
            Pre += pre
            Rre += rec
            Spe += spe
            Acc += acc
            F2 += f2

            mDice += mdice
            mIoU += miou
            mPre += mpre
            mRre += mrec
            mSpe += mspe
            mAcc += macc
            mF2 += mf2

    now_pred_val = [None] * (epoch_eval + 1)
    now_pred_test = [None] * (epoch_eval + 1)
    for i in range(epoch_eval + 1):
        now_pred_val[i] = val_preds[i]
        now_pred_test[i] = test_preds[i]

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        Dice = Dice
        IoU = IoU
        Pre = Pre
        Rre = Rre
        Spe = Spe
        Acc = Acc
        F2 = F2

        mDice = mDice
        mIoU = mIoU
        mPre = mPre
        mRre = mRre
        mSpe = mSpe
        mAcc = mAcc
        mF2 = mF2
    else:
        Dice = Dice / num_val_batches
        IoU = IoU / num_val_batches
        Pre = Pre / num_val_batches
        Rre = Rre / num_val_batches
        Spe = Spe / num_val_batches
        Acc = Acc / num_val_batches
        F2 = F2 / num_val_batches

        mDice = mDice / num_val_batches
        mIoU = mIoU / num_val_batches
        mPre = mPre / num_val_batches
        mRre = mRre / num_val_batches
        mSpe = mSpe / num_val_batches
        mAcc = mAcc / num_val_batches
        mF2 = mF2 / num_val_batches
    return Dice, IoU, Pre, Rre, Spe, Acc, F2, mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2, now_pred_val, now_pred_test

def evaluate_ensembleA_valpre(dataloader_val, preloader_val, time, number_net, bestpop, net_n_classes, device):

    num_val_batches = len(dataloader_val)
    val_preds = [None] * num_val_batches
    bestpop = torch.from_numpy(bestpop)
    bestpop = bestpop.to(device=device)
    epoch_eval = int(-1)

    for batch_mask_true_test in tqdm(dataloader_val, total=num_val_batches, desc='Ensemble Val round', unit='batch', leave=False):
        epoch_eval += 1
        mask_true_test = batch_mask_true_test['mask']
        mask_true_test = mask_true_test.to(device=device, dtype=torch.long)
        mask_true_test = F.one_hot(mask_true_test.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre_val = torch.zeros(size=mask_true_test.shape)
        ensemable_mask_pre_val = ensemable_mask_pre_val.to(device=device)
        with torch.no_grad():
            for epoch_net in range(number_net):
                ensemable_mask_pre_val += bestpop[epoch_net] * preloader_val[epoch_net][time][epoch_eval]
                val_preds[epoch_eval] = ensemable_mask_pre_val.cpu()
                # IndexError: index 2 is out of bounds for dimension 0 with size 2

    now_pred_val = [None] * (epoch_eval + 1)
    for i in range(epoch_eval + 1):
        now_pred_val[i] = val_preds[i]

    return now_pred_val

def evaluate_ensembleB_valpre(dataloader_val, preloader_val, time, number_net, label, threshold, bestpop, net_n_classes, device):

    num_val_batches = len(dataloader_val)
    val_preds = [None] * num_val_batches
    bestpop = torch.from_numpy(bestpop)
    bestpop = bestpop.to(device=device)
    epoch_eval = int(-1)

    for batch_mask_true_test in tqdm(dataloader_val, total=num_val_batches, desc='Ensemble Val round', unit='batch', leave=False):
        epoch_eval += 1
        mask_true_test = batch_mask_true_test['mask']
        mask_true_test = mask_true_test.to(device=device, dtype=torch.long)
        mask_true_test = F.one_hot(mask_true_test.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre_val = torch.zeros(size=mask_true_test.shape)
        ensemable_mask_pre_val = ensemable_mask_pre_val.to(device=device)
        with torch.no_grad():
            label_time = int(0)
            for epoch_net in range(number_net):
                # if label[epoch_net] == threshold:
                if label[epoch_net] >= threshold:
                    label_time += 1
                else:
                    # print('the len of pop', len(bestpop))
                    # print('the label is ', label)
                    ensemable_mask_pre_val += bestpop[epoch_net - label_time] * preloader_val[epoch_net][time][epoch_eval]
                    val_preds[epoch_eval] = ensemable_mask_pre_val.cpu()
                # IndexError: index 2 is out of bounds for dimension 0 with size 2

    now_pred_val = [None] * (epoch_eval + 1)
    for i in range(epoch_eval + 1):
        now_pred_val[i] = val_preds[i]

    return now_pred_val

def evaluate_ensemble_l(preloader, dataloader, number_net, bestpop, net_n_classes, device):

    num_val_batches = len(dataloader)
    bestpop = torch.from_numpy(bestpop)
    bestpop = bestpop.to(device=device)
    epoch_eval = int(-1)
    Dice, IoU, Pre, Rre, Spe, Acc, F2 = 0, 0, 0, 0, 0, 0, 0
    mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2 = 0, 0, 0, 0, 0, 0, 0

    # tem_mask_pre = [None] * (number_net)
    # tem_mdice = [None] * (number_net)
    # tem_miou = [None] * (number_net)
    # tem_dice = [None] * (number_net)
    # Tem_mdice = [0.0] * (number_net)
    # Tem_miou = [0.0] * (number_net)
    # Tem_dice = [0.0] * (number_net)

    for batch_mask_true in tqdm(dataloader, total=num_val_batches, desc='Ensemble Validation round', unit='batch', leave=False):
        epoch_eval += 1
        mask_true = batch_mask_true['mask']
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre = torch.zeros(size=mask_true.shape)
        ensemable_mask_pre = ensemable_mask_pre.to(device=device)
        with torch.no_grad():
            for epoch_net in range(number_net):
                ensemable_mask_pre += bestpop[epoch_net] * preloader[epoch_net][epoch_eval]
                # tem_mask_pre[epoch_net] = preloader[epoch_net][epoch_eval]
                # print('\n', 'epoch_net=', epoch_net)
                # print('bestpop[epoch_net]=', bestpop[epoch_net])

            # convert to one-hot format
            if net_n_classes == 1:
                ensemable_mask_pre = (F.sigmoid(ensemable_mask_pre) > 0.5).float()
                # compute the Dice score
                dice, iou, pre, rec, spe, acc, f2 = comment_coeff(ensemable_mask_pre, mask_true, reduce_batch_first=False, device=device)
            else:
                ensemable_mask_pre = F.one_hot(ensemable_mask_pre.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                mdice, miou, mpre, mrec, mspe, macc, mf2 = multiclass_comment_coeff(ensemable_mask_pre[:, 0:, ...],
                                                                               mask_true[:, 0:, ...],
                                                                               reduce_batch_first=False, device=device)
                dice, iou, pre, rec, spe, acc, f2 = multiclass_comment_coeff(ensemable_mask_pre[:, 1:, ...], mask_true[:, 1:, ...],
                                             reduce_batch_first=False, device=device)
            Dice += dice
            IoU += iou
            Pre += pre
            Rre += rec
            Spe += spe
            Acc += acc
            F2 += f2

            mDice += mdice
            mIoU += miou
            mPre += mpre
            mRre += mrec
            mSpe += mspe
            mAcc += macc
            mF2 += mf2

            # for epoch_net in range(number_net):
            #     tem_mask_pre[epoch_net] = F.one_hot(tem_mask_pre[epoch_net].argmax(dim=1), net_n_classes).permute(0, 3, 1,
            #                                                                                             2).float()
            #     # compute the Dice score, ignoring background
            #     tem_mdice[epoch_net], tem_miou[epoch_net], _, _, _, _, _ = multiclass_comment_coeff(tem_mask_pre[epoch_net][:, 0:, ...],
            #                                                                    mask_true[:, 0:, ...],
            #                                                                    reduce_batch_first=False, device=device)
            #     tem_dice[epoch_net] = multiclass_dice_coeff(tem_mask_pre[epoch_net][:, 1:, ...], mask_true[:, 1:, ...],
            #                                  reduce_batch_first=False)
            # Tem_dice[epoch_net] += tem_dice[epoch_net]
            # Tem_mdice[epoch_net] += tem_mdice[epoch_net]
            # Tem_miou[epoch_net] += tem_miou[epoch_net]
            # print('\n', '\n', 'Tem_dice[epoch_net]=', Tem_dice[epoch_net])
            # print('Tem_mdice[epoch_net]=', Tem_mdice[epoch_net])
            # print('Tem_miou[epoch_net]=', Tem_miou[epoch_net])

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        Dice = Dice
        IoU = IoU
        Pre = Pre
        Rre = Rre
        Spe = Spe
        Acc = Acc
        F2 = F2

        mDice = mDice
        mIoU = mIoU
        mPre = mPre
        mRre = mRre
        mSpe = mSpe
        mAcc = mAcc
        mF2 = mF2
    else:
        Dice = Dice / num_val_batches
        IoU = IoU / num_val_batches
        Pre = Pre / num_val_batches
        Rre = Rre / num_val_batches
        Spe = Spe / num_val_batches
        Acc = Acc / num_val_batches
        F2 = F2 / num_val_batches

        mDice = mDice / num_val_batches
        mIoU = mIoU / num_val_batches
        mPre = mPre / num_val_batches
        mRre = mRre / num_val_batches
        mSpe = mSpe / num_val_batches
        mAcc = mAcc / num_val_batches
        mF2 = mF2 / num_val_batches
    return Dice, IoU, Pre, Rre, Spe, Acc, F2, mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2


def evaluate_ensemble_llable(preloader, dataloader, number_net, label, threshold, bestpop, net_n_classes, device):

    num_val_batches = len(dataloader)
    bestpop = torch.from_numpy(bestpop)
    bestpop = bestpop.to(device=device)
    epoch_eval = int(-1)
    Dice, IoU, Pre, Rre, Spe, Acc, F2 = 0, 0, 0, 0, 0, 0, 0
    mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2 = 0, 0, 0, 0, 0, 0, 0

    # tem_mask_pre = [None] * (number_net)
    # tem_mdice = [None] * (number_net)
    # tem_miou = [None] * (number_net)
    # tem_dice = [None] * (number_net)
    # Tem_mdice = [0.0] * (number_net)
    # Tem_miou = [0.0] * (number_net)
    # Tem_dice = [0.0] * (number_net)

    for batch_mask_true in tqdm(dataloader, total=num_val_batches, desc='Ensemble Validation round', unit='batch', leave=False):
        epoch_eval += 1
        mask_true = batch_mask_true['mask']
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre = torch.zeros(size=mask_true.shape)
        ensemable_mask_pre = ensemable_mask_pre.to(device=device)
        with torch.no_grad():
            label_time = int(0)
            for epoch_net in range(number_net):
                if epoch_net == (number_net - 1):
                    ensemable_mask_pre += bestpop[epoch_net - label_time] * preloader[epoch_net][epoch_eval]
                else:
                    if label[epoch_net] >= threshold:
                        label_time += 1
                    else:
                        ensemable_mask_pre += bestpop[epoch_net - label_time] * preloader[epoch_net][epoch_eval]
                # tem_mask_pre[epoch_net] = preloader[epoch_net][epoch_eval]
                # print('\n', 'epoch_net=', epoch_net)
                # print('bestpop[epoch_net]=', bestpop[epoch_net])

            # convert to one-hot format
            if net_n_classes == 1:
                ensemable_mask_pre = (F.sigmoid(ensemable_mask_pre) > 0.5).float()
                # compute the Dice score
                dice, iou, pre, rec, spe, acc, f2 = comment_coeff(ensemable_mask_pre, mask_true, reduce_batch_first=False, device=device)
            else:
                ensemable_mask_pre = F.one_hot(ensemable_mask_pre.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                mdice, miou, mpre, mrec, mspe, macc, mf2 = multiclass_comment_coeff(ensemable_mask_pre[:, 0:, ...],
                                                                               mask_true[:, 0:, ...],
                                                                               reduce_batch_first=False, device=device)
                dice, iou, pre, rec, spe, acc, f2 = multiclass_comment_coeff(ensemable_mask_pre[:, 1:, ...], mask_true[:, 1:, ...],
                                             reduce_batch_first=False, device=device)
            Dice += dice
            IoU += iou
            Pre += pre
            Rre += rec
            Spe += spe
            Acc += acc
            F2 += f2

            mDice += mdice
            mIoU += miou
            mPre += mpre
            mRre += mrec
            mSpe += mspe
            mAcc += macc
            mF2 += mf2

            # for epoch_net in range(number_net):
            #     tem_mask_pre[epoch_net] = F.one_hot(tem_mask_pre[epoch_net].argmax(dim=1), net_n_classes).permute(0, 3, 1,
            #                                                                                             2).float()
            #     # compute the Dice score, ignoring background
            #     tem_mdice[epoch_net], tem_miou[epoch_net], _, _, _, _, _ = multiclass_comment_coeff(tem_mask_pre[epoch_net][:, 0:, ...],
            #                                                                    mask_true[:, 0:, ...],
            #                                                                    reduce_batch_first=False, device=device)
            #     tem_dice[epoch_net] = multiclass_dice_coeff(tem_mask_pre[epoch_net][:, 1:, ...], mask_true[:, 1:, ...],
            #                                  reduce_batch_first=False)
            # Tem_dice[epoch_net] += tem_dice[epoch_net]
            # Tem_mdice[epoch_net] += tem_mdice[epoch_net]
            # Tem_miou[epoch_net] += tem_miou[epoch_net]
            # print('\n', '\n', 'Tem_dice[epoch_net]=', Tem_dice[epoch_net])
            # print('Tem_mdice[epoch_net]=', Tem_mdice[epoch_net])
            # print('Tem_miou[epoch_net]=', Tem_miou[epoch_net])

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        Dice = Dice
        IoU = IoU
        Pre = Pre
        Rre = Rre
        Spe = Spe
        Acc = Acc
        F2 = F2

        mDice = mDice
        mIoU = mIoU
        mPre = mPre
        mRre = mRre
        mSpe = mSpe
        mAcc = mAcc
        mF2 = mF2
    else:
        Dice = Dice / num_val_batches
        IoU = IoU / num_val_batches
        Pre = Pre / num_val_batches
        Rre = Rre / num_val_batches
        Spe = Spe / num_val_batches
        Acc = Acc / num_val_batches
        F2 = F2 / num_val_batches

        mDice = mDice / num_val_batches
        mIoU = mIoU / num_val_batches
        mPre = mPre / num_val_batches
        mRre = mRre / num_val_batches
        mSpe = mSpe / num_val_batches
        mAcc = mAcc / num_val_batches
        mF2 = mF2 / num_val_batches
    return Dice, IoU, Pre, Rre, Spe, Acc, F2, mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2


def evaluate_popfit(preloader, dataloader, number_net, poplist, popsum, net_n_classes, device):

    poplist_val = [float(0)] * popsum
    num_val_batches = len(dataloader)
    epoch_eval = int(-1)
    if num_val_batches == 0:
        fenmu = 1
    else:
        fenmu = num_val_batches

    for batch_mask_true in tqdm(dataloader, total=num_val_batches, desc='popfit round', unit='batch', leave=False):

        epoch_eval += 1
        mask_true = batch_mask_true['mask']
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre = torch.zeros(size=mask_true.shape)
        ensemable_mask_pre = ensemable_mask_pre.to(device=device)
        for pop in range(popsum):
            popone = poplist[pop]
            popone = torch.from_numpy(popone)
            popone = popone.to(device=device)
            with torch.no_grad():
                for time in range(10):
                    for Epoch_Net in range(number_net):
                        ensemable_mask_pre += popone[Epoch_Net] * preloader[Epoch_Net][time][epoch_eval]

                    if net_n_classes == 1:
                        ensemable_mask_pre = (F.sigmoid(ensemable_mask_pre) > 0.5).float()
                        # compute the Dice score
                        dice_score = dice_coeff(ensemable_mask_pre, mask_true, reduce_batch_first=False)
                        dice_score = (dice_score / fenmu) / 10
                    else:
                        ensemable_mask_pre = F.one_hot(ensemable_mask_pre.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
                        # compute the Dice score, ignoring background
                        dice_score = multiclass_dice_coeff(ensemable_mask_pre[:, 0:, ...], mask_true[:, 0:, ...],
                                                                reduce_batch_first=False)
                        dice_score = (dice_score / fenmu) / 10

                    dic = dice_score.data.cpu().numpy()
                    poplist_val[pop] += dic

    return poplist_val


def evaluate_popfit_label(preloader, dataloader, number_net, label, threshold, poplist, popsum, net_n_classes, device):

    poplist_val = [float(0)] * popsum
    num_val_batches = len(dataloader)
    epoch_eval = int(-1)
    if num_val_batches == 0:
        fenmu = 1
    else:
        fenmu = num_val_batches

    for batch_mask_true in tqdm(dataloader, total=num_val_batches, desc='popfit round', unit='batch', leave=False):

        epoch_eval += 1
        mask_true = batch_mask_true['mask']
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre = torch.zeros(size=mask_true.shape)
        ensemable_mask_pre = ensemable_mask_pre.to(device=device)
        for pop in range(popsum):
            popone = poplist[pop]
            popone = torch.from_numpy(popone)
            popone = popone.to(device=device)
            with torch.no_grad():
                for time in range(10):
                    label_time = int(0)
                    for Epoch_Net in range(number_net):
                        if label[Epoch_Net] >= threshold:
                            label_time += 1
                        else:
                            ensemable_mask_pre += popone[Epoch_Net-label_time] * preloader[Epoch_Net][time][epoch_eval]

                    if net_n_classes == 1:
                        ensemable_mask_pre = (F.sigmoid(ensemable_mask_pre) > 0.5).float()
                        # compute the Dice score
                        dice_score = dice_coeff(ensemable_mask_pre, mask_true, reduce_batch_first=False)
                        dice_score = (dice_score / fenmu) / 10
                    else:
                        ensemable_mask_pre = F.one_hot(ensemable_mask_pre.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
                        # compute the Dice score, ignoring background
                        dice_score = multiclass_dice_coeff(ensemable_mask_pre[:, 0:, ...], mask_true[:, 0:, ...],
                                                                reduce_batch_first=False)
                        dice_score = (dice_score / fenmu) / 10

                    dic = dice_score.data.cpu().numpy()
                    poplist_val[pop] += dic

    return poplist_val


def evaluate_popfit_pl(preloader, dataloader, number_net, poplist, popsum, net_n_classes, device):

    poplist_val = [float(0)] * popsum
    num_val_batches = len(dataloader)
    epoch_eval = int(-1)
    if num_val_batches == 0:
        fenmu = 1
    else:
        fenmu = num_val_batches

    for batch_mask_true in tqdm(dataloader, total=num_val_batches, desc='popfit round', unit='batch', leave=False):

        epoch_eval += 1
        mask_true = batch_mask_true['mask']
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre = torch.zeros(size=mask_true.shape)
        ensemable_mask_pre = ensemable_mask_pre.to(device=device)
        for pop in range(popsum):
            popone = poplist[pop]
            popone = torch.from_numpy(popone)
            popone = popone.to(device=device)
            with torch.no_grad():
                for Epoch_Net in range(number_net):
                    ensemable_mask_pre += popone[Epoch_Net] * preloader[Epoch_Net][8][epoch_eval]

                if net_n_classes == 1:
                    ensemable_mask_pre = (F.sigmoid(ensemable_mask_pre) > 0.5).float()
                    # compute the Dice score
                    dice_score = dice_coeff(ensemable_mask_pre, mask_true, reduce_batch_first=False)
                    dice_score = dice_score / fenmu
                else:
                    ensemable_mask_pre = F.one_hot(ensemable_mask_pre.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score = multiclass_dice_coeff(ensemable_mask_pre[:, 0:, ...], mask_true[:, 0:, ...],
                                                            reduce_batch_first=False)
                    dice_score = dice_score / fenmu

                dic = dice_score.data.cpu().numpy()
                poplist_val[pop] += dic

    return poplist_val

def evaluate_popfit_pbl(preloader, dataloader, number_net, poplist, label, threshold, popsum, net_n_classes, device):

    poplist_val = [float(0)] * popsum
    num_val_batches = len(dataloader)
    epoch_eval = int(-1)
    if num_val_batches == 0:
        fenmu = 1
    else:
        fenmu = num_val_batches

    for batch_mask_true in tqdm(dataloader, total=num_val_batches, desc='popfit round', unit='batch', leave=False):

        epoch_eval += 1
        mask_true = batch_mask_true['mask']
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre = torch.zeros(size=mask_true.shape)
        ensemable_mask_pre = ensemable_mask_pre.to(device=device)
        for pop in range(popsum):
            popone = poplist[pop]
            popone = torch.from_numpy(popone)
            popone = popone.to(device=device)
            with torch.no_grad():
                label_time = int(0)
                for Epoch_Net in range(number_net):
                    if Epoch_Net == (number_net - 1):
                        ensemable_mask_pre += popone[Epoch_Net - label_time] * preloader[Epoch_Net + 1][8][epoch_eval]
                    else:
                        if label[Epoch_Net] >= threshold:
                            label_time += 1
                        else:
                            ensemable_mask_pre += popone[Epoch_Net - label_time] * preloader[Epoch_Net][8][epoch_eval]

                if net_n_classes == 1:
                    ensemable_mask_pre = (F.sigmoid(ensemable_mask_pre) > 0.5).float()
                    # compute the Dice score
                    dice_score = dice_coeff(ensemable_mask_pre, mask_true, reduce_batch_first=False)
                    dice_score = dice_score / fenmu
                else:
                    ensemable_mask_pre = F.one_hot(ensemable_mask_pre.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score = multiclass_dice_coeff(ensemable_mask_pre[:, 0:, ...], mask_true[:, 0:, ...],
                                                            reduce_batch_first=False)
                    dice_score = dice_score / fenmu

                dic = dice_score.data.cpu().numpy()
                poplist_val[pop] += dic

    return poplist_val

# def evaluate_popfit(dataloader, nets, number_net, poplist, popsum, net_n_classes, device, w_ustm, w_ust, w_m, w_us, w_t):
#     print(poplist[0])
#     num_val_batches = len(dataloader)
#     poplist_val = [float(0)] * popsum
#
#     if num_val_batches == 0:
#         fenmu = 1
#     else:
#         fenmu = num_val_batches
#
#     for Epoch_Net in range(number_net):
#         nets[Epoch_Net].eval()
#
#     for batch in dataloader:
#         images = batch['image']
#         true_masks = batch['mask']
#         images = images.to(device=device, dtype=torch.float32)
#         true_masks = true_masks.to(device=device, dtype=torch.long)
#         true_masks = F.one_hot(true_masks.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
#         for pop in range(popsum):
#             popone = poplist[pop]
#             popone = torch.from_numpy(popone)
#             popone = popone.to(device=device)
#             with torch.no_grad():
#                 esmb_pred = torch.zeros(size=true_masks.shape)
#                 esmb_pred = esmb_pred.to(device=device)
#                 for Epoch_Net in range(number_net):
#                     deep_out = nets[Epoch_Net](images)
#                     mask_pred = deep_out[0] * w_ustm[Epoch_Net] + deep_out[1] * w_ust[Epoch_Net] + \
#                                 deep_out[2] * w_m[Epoch_Net] + deep_out[3] * w_us[Epoch_Net] + deep_out[4] * w_t[Epoch_Net]
#
#                     esmb_pred += popone[Epoch_Net] * mask_pred
#
#                 if net_n_classes == 1:
#                     esmb_pred = (F.sigmoid(esmb_pred) > 0.5).float()
#                     # compute the Dice score
#                     dice_score = dice_coeff(esmb_pred, true_masks, reduce_batch_first=False)
#                     dice_score = dice_score / fenmu
#                 else:
#                     esmb_pred = F.one_hot(esmb_pred.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
#                     # compute the Dice score, ignoring background
#                     dice_score = multiclass_dice_coeff(esmb_pred[:, 0:, ...], true_masks[:, 0:, ...],
#                                                             reduce_batch_first=False)
#                     dice_score = dice_score / fenmu
#
#             dic = dice_score.data.cpu().numpy()
#             poplist_val[pop] += dic
#
#
#     for Epoch_Net in range(number_net):
#         nets[Epoch_Net].train()
#
#
#     return poplist_val

def evaluate_onepopfit(dataloader, nets, number_net, onepop, net_n_classes, device):

    num_val_batches = len(dataloader)
    onepop = torch.from_numpy(onepop)
    onepop = onepop.to(device=device)
    dice_score = 0

    for Epoch_Net in range(number_net):
        nets[Epoch_Net].eval()

    for batch in tqdm(dataloader, total=num_val_batches, desc='pop Validation round', unit='batch', leave=False):
        images = batch['image']
        mask_true = batch['mask']
        images = images.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre = torch.zeros(size=mask_true.shape)
        ensemable_mask_pre = ensemable_mask_pre.to(device=device)
        with torch.no_grad():
            for Epoch_Net in range(number_net):
                usenet = nets[Epoch_Net]
                mask_pred = usenet(images)
                ensemable_mask_pre += onepop[Epoch_Net] * mask_pred

            output = ensemable_mask_pre
            target = mask_true
            smooth = 1e-5

            if torch.is_tensor(output):
                output = torch.sigmoid(output).data.cpu().numpy()
            if torch.is_tensor(target):
                target = target.data.cpu().numpy()
            output_ = output > 0.5
            target_ = target > 0.5

            # intersection = (output_ & target_).sum()
            # union = (output_ | target_).sum()
            # iou = (intersection + smooth) / (union + smooth)
            # dice = (2 * iou) / (iou + 1)

            output_ = torch.tensor(output_)
            target_ = torch.tensor(target_)
            SE = get_sensitivity(output_, target_, threshold=0.5)
            PC = get_precision(output_, target_, threshold=0.5)
            SP = get_specificity(output_, target_, threshold=0.5)
            ACC = get_accuracy(output_, target_, threshold=0.5)
            F1 = 2 * SE * PC / (SE + PC + 1e-6)

            # convert to one-hot format
            if net_n_classes == 1:
                ensemable_mask_pre = (F.sigmoid(ensemable_mask_pre) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(ensemable_mask_pre, mask_true, reduce_batch_first=False)
            else:
                ensemable_mask_pre = F.one_hot(ensemable_mask_pre.argmax(dim=1), net_n_classes).permute(0, 3, 1,
                                                                                                        2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(ensemable_mask_pre[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)


    if num_val_batches == 0:
        dice = dice_score
    else:
        dice = dice_score / num_val_batches
    iou = dice / (2 - dice)
    return dice, iou, SE, PC, F1, SP, ACC


def evaluate_pop(dataloader, number, onlyone_pop, mask_pred_size, PSO_epochnow, PSO_epochsum, pop_fig, pop_sum, net_n_classes, device):


    onlyone_pop = torch.from_numpy(onlyone_pop)
    onlyone_pop = onlyone_pop.to(device=device)
    esm_pred = torch.zeros(size=mask_pred_size)
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in dataloader:
        mask_pred = batch['PSO_masks_pred']
        mask_true = batch['PSO_true_masks']
        mask_true = F.one_hot(mask_true.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        # move images and labels to correct device and type

        with torch.no_grad():
            esm_pred = torch.zeros(size=mask_true.shape)
            esm_pred = esm_pred.to(device=device)
            for i in range(number):
                mask_pred[i] = mask_pred[i].to(device=device)
                esm_pred += onlyone_pop[i] * mask_pred[i]

            # output = esm_pred
            # target = mask_true

            # if torch.is_tensor(output):
            #     output = torch.sigmoid(output).data.cpu().numpy()
            # if torch.is_tensor(target):
            #     target = target.data.cpu().numpy()

                # convert to one-hot format
            if net_n_classes == 1:
                esm_pred = (F.sigmoid(esm_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(esm_pred, mask_true, reduce_batch_first=False)
            else:
                esm_pred = F.one_hot(esm_pred.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(esm_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                        reduce_batch_first=False)

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        dice = dice_score
    else:
        dice = dice_score / num_val_batches

    return dice






def iou_score(output, target):
    smooth = 1e-5


    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)

    output_ = torch.tensor(output_)
    target_ = torch.tensor(target_)
    SE = get_sensitivity(output_, target_, threshold=0.5)
    PC = get_precision(output_, target_, threshold=0.5)
    SP = get_specificity(output_, target_, threshold=0.5)
    ACC = get_accuracy(output_, target_, threshold=0.5)
    F1 = 2 * SE * PC / (SE + PC + 1e-6)
    return iou, dice, SE, PC, F1, SP, ACC

def Dice(output, target):  # Batch_size>1时
    smooth = 1e-5
    #output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output=torch.sigmoid(output).data.cpu().numpy()
    output[output > 0.5] = 1  #将概率输出变为于标签相匹配的矩阵
    output[output <= 0.5] = 0
    # target = target.view(-1).data.cpu().numpy()
    target = target.data.cpu().numpy()
    dice=0.
    # ipdb.set_trace() # 用于断掉调试
    if len(output)>1:# 如果样本量>1，则逐样本累加
        for i in range(len(output)):
            intersection = (output[i] * target[i]).sum()
            dice += (2. * intersection + smooth)/(output[i].sum() + target[i].sum() + smooth)
    else:
        intersection = (output * target).sum() # 一个数字,=TP
        dice = (2. * intersection + smooth) /(output.sum() + target.sum() + smooth)
    return dice


def Choice_best(val, val_best, bestest, best_test, Epoch):
    assert (len(val)) == (len(val_best) - 1), "len(val) must= len(val_best) - 1."
    for i in range(len(val) - 1):
        if val[i] > val_best[i]:
            val_best[i] = val[i]
    if val[1] == val_best[1]:
        val_best[len(val_best) - 1] = Epoch
        val_best[len(val) - 1] = val[len(val) - 1]
        bestest = best_test
    return val_best, bestest









# def evaluate_val(net, dataloader, device, popfits_masks):
#     net.eval()
#     num_val_batches = len(dataloader)
#     mask_preds = [None] * num_val_batches
#     epoch_eval = int(-1)
#     dice_score = 0
#     dice1 = 0
#     dice2 = 0
#     dice3 = 0
#     dice4 = 0
#     dice5 = 0
#
#     # iterate over the validation set
#     for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
#
#         epoch_eval += 1
#         image, mask_true = batch['image'], batch['mask']
#         # move images and labels to correct device and type
#         image = image.to(device=device, dtype=torch.float32)
#         mask_true = mask_true.to(device=device, dtype=torch.long)
#         mask_true = F.one_hot(mask_true.squeeze_(1), net.n_classes).permute(0, 3, 1, 2).float()
#
#         with torch.no_grad():
#             # predict the mask
#             combine_out = net(image)
#             mask_preds[epoch_eval] = combine_out.cpu()
#
#
#             # convert to one-hot format
#             if net.n_classes == 1:
#                 combine_out = (F.sigmoid(combine_out) > 0.5).float()
#                 # compute the Dice score
#                 dice_score += dice_coeff(combine_out, mask_true, reduce_batch_first=False)
#             else:
#                 combine_out = F.one_hot(combine_out.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
#                 # compute the Dice score, ignoring background
#                 dice_score += multiclass_dice_coeff(combine_out[:, 0:, ...], mask_true[:, 0:, ...],
#                                                     reduce_batch_first=False)
#
#
#     net.train()
#
#     popfits_masks[0] = popfits_masks[1]
#     popfits_masks[1] = popfits_masks[2]
#     popfits_masks[2] = popfits_masks[3]
#     popfits_masks[3] = popfits_masks[4]
#     popfits_masks[4] = popfits_masks[5]
#     popfits_masks[5] = popfits_masks[6]
#     popfits_masks[6] = popfits_masks[7]
#     popfits_masks[7] = popfits_masks[8]
#     popfits_masks[8] = popfits_masks[9]
#     popfits_masks[9] = popfits_masks[10]
#     popfits_masks[10] = popfits_masks[11]
#     popfits_masks[11] = mask_preds
#
#     # popfits_masks[0] = popfits_masks[6]
#     # popfits_masks[1] = popfits_masks[6]
#     # popfits_masks[2] = popfits_masks[6]
#     # popfits_masks[3] = popfits_masks[6]
#     # popfits_masks[4] = popfits_masks[6]
#     # popfits_masks[5] = popfits_masks[6]
#     # popfits_masks[6] = mask_preds
#     # popfits_masks[7] = mask_preds
#     # popfits_masks[8] = mask_preds
#     # popfits_masks[9] = mask_preds
#     # popfits_masks[10] = mask_preds
#     # popfits_masks[11] = mask_preds
#
#     now_pred = [None] * (epoch_eval + 1)
#     for i in range(epoch_eval + 1):
#         now_pred[i] = mask_preds[i]
#
#
#
#     # Fixes a potential division by zero error
#     if num_val_batches == 0:
#         dice = dice_score
#     else:
#         dice = dice_score / num_val_batches
#
#     return dice, popfits_masks, now_pred
#
#
# def evaluate_test(net, dataloader, device):
#     net.eval()
#     num_val_batches = len(dataloader)
#     mask_preds = [None] * num_val_batches
#     epoch_eval = int(-1)
#     Dice, mDice, mIoU, Pre, Rre, Spe, Acc, F2 = 0, 0, 0, 0, 0, 0, 0, 0
#
#     # iterate over the validation set
#     for batch in tqdm(dataloader, total=num_val_batches, desc='Test round', unit='batch', leave=False):
#         epoch_eval += 1
#         image, mask_true = batch['image'], batch['mask']
#         # move images and labels to correct device and type
#         image = image.to(device=device, dtype=torch.float32)
#         mask_true = mask_true.to(device=device, dtype=torch.long)
#         mask_true = F.one_hot(mask_true.squeeze_(1), net.n_classes).permute(0, 3, 1, 2).float()
#
#         with torch.no_grad():
#             # predict the mask
#             mask_pred = net(image)
#             mask_preds[epoch_eval] = mask_pred.cpu()
#
#             # convert to one-hot format
#             if net.n_classes == 1:
#                 mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
#                 # compute the Dice score
#                 _, _, pre, rec, spe, acc, f2 = comment_coeff(mask_pred, mask_true, reduce_batch_first=False)
#                 dice = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
#             else:
#                 mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
#                 # compute the Dice score, ignoring background
#                 mdice, miou, pre, rec, spe, acc, f2 = multiclass_comment_coeff(mask_pred[:, 0:, ...],
#                                                                                mask_true[:, 0:, ...],
#                                                                                reduce_batch_first=False)
#                 dice = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
#                                                    reduce_batch_first=False)
#             Dice += dice
#             mDice += mdice
#             mIoU += miou
#             Pre += pre
#             Rre += rec
#             Spe += spe
#             Acc += acc
#             F2 += f2
#
#
#
#
#     net.train()
#
#     # Fixes a potential division by zero error
#     if num_val_batches == 0:
#         Dice = Dice
#         mDice = mDice
#         mIoU = mIoU
#         Pre = Pre
#         Rre = Rre
#         Spe = Spe
#         Acc = Acc
#         F2 = F2
#     else:
#         Dice = Dice / num_val_batches
#         mDice = mDice / num_val_batches
#         mIoU = mIoU / num_val_batches
#         Pre = Pre / num_val_batches
#         Rre = Rre / num_val_batches
#         Spe = Spe / num_val_batches
#         Acc = Acc / num_val_batches
#         F2 = F2 / num_val_batches
#
#
#     return Dice, mDice, mIoU, Pre, Rre, Spe, Acc, F2, mask_preds


def evaluate_compare_test(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    Dice, IoU, Pre, Rre, Spe, Acc, F2 = 0, 0, 0, 0, 0, 0, 0
    mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2 = 0, 0, 0, 0, 0, 0, 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Test round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net.n_classes).permute(0, 3, 1, 2).float()


        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice, iou, pre, rec, spe, acc, f2 = comment_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                mdice, miou, mpre, mrec, mspe, macc, mf2 = multiclass_comment_coeff(mask_pred[:, 0:, ...],
                                                                               mask_true[:, 0:, ...],
                                                                               reduce_batch_first=False)
                dice, iou, pre, rec, spe, acc, f2 = multiclass_comment_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                   reduce_batch_first=False)
            Dice += dice
            IoU += iou
            Pre += pre
            Rre += rec
            Spe += spe
            Acc += acc
            F2 += f2

            mDice += mdice
            mIoU += miou
            mPre += mpre
            mRre += mrec
            mSpe += mspe
            mAcc += macc
            mF2 += mf2




    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        Dice = Dice
        IoU = IoU
        Pre = Pre
        Rre = Rre
        Spe = Spe
        Acc = Acc
        F2 = F2

        mDice = mDice
        mIoU = mIoU
        mPre = mPre
        mRre = mRre
        mSpe = mSpe
        mAcc = mAcc
        mF2 = mF2
    else:
        Dice = Dice / num_val_batches
        IoU = IoU / num_val_batches
        Pre = Pre / num_val_batches
        Rre = Rre / num_val_batches
        Spe = Spe / num_val_batches
        Acc = Acc / num_val_batches
        F2 = F2 / num_val_batches

        mDice = mDice / num_val_batches
        mIoU = mIoU / num_val_batches
        mPre = mPre / num_val_batches
        mRre = mRre / num_val_batches
        mSpe = mSpe / num_val_batches
        mAcc = mAcc / num_val_batches
        mF2 = mF2 / num_val_batches


    return Dice, IoU, Pre, Rre, Spe, Acc, F2, mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2

def evaluate_compare_val(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

        image, mask_true = batch['image'], batch['mask']
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)


            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 0:, ...], mask_true[:, 0:, ...],
                                                    reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        dice = dice_score
    else:
        dice = dice_score / num_val_batches

    return dice


def Choice_compare_best(val, val_best, Epoch):
    assert (len(val)) == (len(val_best) - 1), "len(val) must= len(val_best) - 1."
    for i in range(len(val) - 1):
        if val[i] > val_best[i]:
            val_best[i] = val[i]
    if val[0] == val_best[0]:
        val_best[len(val_best) - 1] = Epoch
    return val_best

def evaluate_HWC_test(net, dataloader, device, w_ustm, w_ust, w_m, w_us, w_t):
    net.eval()
    num_val_batches = len(dataloader)
    mask_preds = [None] * num_val_batches
    epoch_eval = int(-1)
    dice_score = 0
    mdice_score = 0
    dice_m = 0
    dice_us = 0
    dice_p = 0
    dice_sp = 0
    dice_msp = 0
    mdice_m = 0
    mdice_us = 0
    mdice_p = 0
    mdice_sp = 0
    mdice_msp = 0

    miou_score = 0
    miou_m = 0
    miou_us = 0
    miou_p = 0
    miou_sp = 0
    miou_msp = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

        epoch_eval += 1
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            deep_out = net(image)
            combine_out = deep_out[0] * w_ustm + deep_out[1] * w_ust + deep_out[2] * w_m + deep_out[3] * w_us + deep_out[4] * w_t
            mask_preds[epoch_eval] = combine_out.cpu()


            # convert to one-hot format
            if net.n_classes == 2:
                combine_out = F.one_hot(combine_out.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(combine_out[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)
                mdice_score += multiclass_dice_coeff(combine_out[:, 0:, ...], mask_true[:, 0:, ...],
                                                    reduce_batch_first=False)
                miou_score += multiclass_iou_coeff(combine_out[:, 0:, ...], mask_true[:, 0:, ...],
                                                    reduce_batch_first=False)

            if net.n_classes == 2:

                deep_out_msp = F.one_hot(deep_out[0].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_msp += multiclass_dice_coeff(deep_out_msp[:, 1:, ...], mask_true[:, 1:, ...],
                                                reduce_batch_first=False)
                mdice_msp += multiclass_dice_coeff(deep_out_msp[:, 0:, ...], mask_true[:, 0:, ...],
                                                 reduce_batch_first=False)
                miou_msp += multiclass_iou_coeff(deep_out_msp[:, 0:, ...], mask_true[:, 0:, ...],
                                               reduce_batch_first=False)
                deep_out_sp = F.one_hot(deep_out[1].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_sp += multiclass_dice_coeff(deep_out_sp[:, 1:, ...], mask_true[:, 1:, ...],
                                                reduce_batch_first=False)
                mdice_sp += multiclass_dice_coeff(deep_out_sp[:, 0:, ...], mask_true[:, 0:, ...],
                                                 reduce_batch_first=False)
                miou_sp += multiclass_iou_coeff(deep_out_sp[:, 0:, ...], mask_true[:, 0:, ...],
                                               reduce_batch_first=False)

                deep_out_m = F.one_hot(deep_out[2].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_m += multiclass_dice_coeff(deep_out_m[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)
                mdice_m += multiclass_dice_coeff(deep_out_m[:, 0:, ...], mask_true[:, 0:, ...],
                                                reduce_batch_first=False)
                miou_m += multiclass_iou_coeff(deep_out_m[:, 0:, ...], mask_true[:, 0:, ...],
                                               reduce_batch_first=False)

                deep_out_us = F.one_hot(deep_out[3].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_us += multiclass_dice_coeff(deep_out_us[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)
                mdice_us += multiclass_dice_coeff(deep_out_us[:, 1:, ...], mask_true[:, 1:, ...],
                                                 reduce_batch_first=False)
                miou_us += multiclass_iou_coeff(deep_out_us[:, 0:, ...], mask_true[:, 0:, ...],
                                               reduce_batch_first=False)

                deep_out_p = F.one_hot(deep_out[4].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_p += multiclass_dice_coeff(deep_out_p[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)
                mdice_p += multiclass_dice_coeff(deep_out_p[:, 1:, ...], mask_true[:, 1:, ...],
                                                reduce_batch_first=False)
                miou_p += multiclass_iou_coeff(deep_out_p[:, 0:, ...], mask_true[:, 0:, ...],
                                               reduce_batch_first=False)


    net.train()



    now_pred = [None] * (epoch_eval + 1)
    for i in range(epoch_eval + 1):
        now_pred[i] = mask_preds[i]



    # Fixes a potential division by zero error
    if num_val_batches == 0:
        dice_score = dice_score
        dice_msp = dice_msp
        dice_sp = dice_sp
        dice_m = dice_m
        dice_us = dice_us
        dice_p = dice_p
        mdice_score = mdice_score
        mdice_msp = mdice_msp
        mdice_sp = mdice_sp
        mdice_m = mdice_m
        mdice_us = mdice_us
        mdice_p = mdice_p
        miou_score = miou_score
        miou_msp = miou_msp
        miou_sp = miou_sp
        miou_m = miou_m
        miou_us = miou_us
        miou_p = miou_p
    else:
        dice_score = dice_score / num_val_batches
        dice_msp = dice_msp / num_val_batches
        dice_sp = dice_sp / num_val_batches
        dice_m = dice_m / num_val_batches
        dice_us = dice_us / num_val_batches
        dice_p = dice_p / num_val_batches
        mdice_score = mdice_score / num_val_batches
        mdice_msp = mdice_msp / num_val_batches
        mdice_sp = mdice_sp / num_val_batches
        mdice_m = mdice_m / num_val_batches
        mdice_us = mdice_us / num_val_batches
        mdice_p = mdice_p / num_val_batches
        miou_score = miou_score / num_val_batches
        miou_msp = miou_msp / num_val_batches
        miou_sp = miou_sp / num_val_batches
        miou_m = miou_m / num_val_batches
        miou_us = miou_us / num_val_batches
        miou_p = miou_p / num_val_batches


    return dice_score, dice_msp, dice_sp, dice_m, dice_us, dice_p, mdice_score, mdice_msp, mdice_sp, mdice_m, mdice_us, mdice_p,\
        miou_score, miou_msp, miou_sp, miou_m, miou_us, miou_p

def evaluate_AT_val(net, dataloader, device, w_ustm, w_ust, w_m, w_us, w_t):
    net.eval()
    num_val_batches = len(dataloader)
    epoch_eval = int(-1)
    dice_score = 0


    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

        epoch_eval += 1
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            deep_out, _ = net(image)
            combine_out = deep_out[0] * w_ustm + deep_out[1] * w_ust + deep_out[2] * w_m + deep_out[3] * w_us + deep_out[4] * w_t

            # convert to one-hot format
            if net.n_classes == 1:
                combine_out = (F.sigmoid(combine_out) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(combine_out, mask_true, reduce_batch_first=False)
            else:
                combine_out = F.one_hot(combine_out.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(combine_out[:, 0:, ...], mask_true[:, 0:, ...],
                                                    reduce_batch_first=False)


    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        dice = dice_score
    else:
        dice = dice_score / num_val_batches


    return dice


def evaluate_AT_test(net, dataloader, device, w_ustm, w_ust, w_m, w_us, w_t):
    net.eval()
    num_val_batches = len(dataloader)
    mask_preds = [None] * num_val_batches
    epoch_eval = int(-1)
    dice_score = 0
    dice_m = 0
    dice_us = 0
    dice_p = 0

    miou_score = 0
    miou_m = 0
    miou_us = 0
    miou_p = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

        epoch_eval += 1
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            deep_out, _ = net(image)
            combine_out = deep_out[0] * w_ustm + deep_out[1] * w_ust + deep_out[2] * w_m + deep_out[3] * w_us + deep_out[4] * w_t
            mask_preds[epoch_eval] = combine_out.cpu()


            # convert to one-hot format
            if net.n_classes == 2:
                combine_out = F.one_hot(combine_out.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(combine_out[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)
                miou_score += multiclass_iou_coeff(combine_out[:, 0:, ...], mask_true[:, 0:, ...],
                                                    reduce_batch_first=False)

            if net.n_classes == 2:
                deep_out_m = F.one_hot(deep_out[2].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_m += multiclass_dice_coeff(deep_out_m[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)
                miou_m += multiclass_iou_coeff(deep_out_m[:, 0:, ...], mask_true[:, 0:, ...],
                                               reduce_batch_first=False)

                deep_out_us = F.one_hot(deep_out[3].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_us += multiclass_dice_coeff(deep_out_us[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)
                miou_us += multiclass_iou_coeff(deep_out_us[:, 0:, ...], mask_true[:, 0:, ...],
                                               reduce_batch_first=False)

                deep_out_p = F.one_hot(deep_out[4].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_p += multiclass_dice_coeff(deep_out_p[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)
                miou_p += multiclass_iou_coeff(deep_out_p[:, 0:, ...], mask_true[:, 0:, ...],
                                               reduce_batch_first=False)


    net.train()



    now_pred = [None] * (epoch_eval + 1)
    for i in range(epoch_eval + 1):
        now_pred[i] = mask_preds[i]



    # Fixes a potential division by zero error
    if num_val_batches == 0:
        dice_score = dice_score
        dice_m = dice_m
        dice_us = dice_us
        dice_p = dice_p
        miou_score = miou_score
        miou_m = miou_m
        miou_us = miou_us
        miou_p = miou_p
    else:
        dice_score = dice_score / num_val_batches
        dice_m = dice_m / num_val_batches
        dice_us = dice_us / num_val_batches
        dice_p = dice_p / num_val_batches
        miou_score = miou_score / num_val_batches
        miou_m = miou_m / num_val_batches
        miou_us = miou_us / num_val_batches
        miou_p = miou_p / num_val_batches


    return dice_score, dice_p, dice_m, dice_us, miou_score, miou_p, miou_m, miou_us

def evaluate_HWC_val(net, dataloader, device, w_ustm, w_ust, w_m, w_us, w_t):
    net.eval()
    num_val_batches = len(dataloader)
    epoch_eval = int(-1)
    dice_score = 0


    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

        epoch_eval += 1
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            deep_out = net(image)
            combine_out = deep_out[0] * w_ustm + deep_out[1] * w_ust + deep_out[2] * w_m + deep_out[3] * w_us + deep_out[4] * w_t

            # convert to one-hot format
            if net.n_classes == 1:
                combine_out = (F.sigmoid(combine_out) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(combine_out, mask_true, reduce_batch_first=False)
            else:
                combine_out = F.one_hot(combine_out.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(combine_out[:, 0:, ...], mask_true[:, 0:, ...],
                                                    reduce_batch_first=False)


    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        dice = dice_score
    else:
        dice = dice_score / num_val_batches


    return dice







def evaluate_test_std(net, dataloader, device, w_ustm, w_ust, w_m, w_us, w_t, ornot):
    net.eval()
    num_val_batches = len(dataloader)
    mask_preds = [None] * num_val_batches
    epoch_eval = int(-1)
    Dice, IoU, Pre, Rre, Spe, Acc, F2 = 0, 0, 0, 0, 0, 0, 0
    mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2 = 0, 0, 0, 0, 0, 0, 0
    mDice_lists, mIoU_lists, mPP_lists, mRR_lists, mSS_lists, mAA_lists, mFF_lists, mHD95_lists = [], [], [], [], [], [], [], []
    Dice_lists, IoU_lists, PP_lists, RR_lists, SS_lists, AA_lists, FF_lists, HD95_lists = [], [], [], [], [], [], [], []

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Test round', unit='batch', leave=False):
        epoch_eval += 1
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        # print('mask_true.shape=', mask_true.shape)
        mask_true = F.one_hot(mask_true.squeeze_(1), net.n_classes).permute(0, 3, 1, 2).float()


        with torch.no_grad():
            # predict the mask
            if ornot:
                deep_out = net(image)
                mask_pred = deep_out[0] * w_ustm + deep_out[1] * w_ust + deep_out[2] * w_m + deep_out[3] * w_us + deep_out[4] * w_t
            else:
                mask_pred = net(image)
            mask_preds[epoch_eval] = mask_pred.cpu()

            # convert to one-hot format
            if net.n_classes == 2:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                Dice_list, IoU_list, PP_list, RR_list, SS_list, AA_list, FF_list, HD95_list, mDice_list, mIoU_list, mPP_list, \
                    mRR_list, mSS_list, mAA_list, mFF_list, mHD95_list = multiclass_comment_coeff_onepic(mask_pred[:, 0:, ...],
                                                                                             mask_true[:, 0:, ...],
                                                                                             reduce_batch_first=False,
                                                                                             device=device)
            Dice_list_cpu = [t.cpu() for t in Dice_list]
            IoU_list_cpu = [t.cpu() for t in IoU_list]
            PP_list_cpu = [t.cpu() for t in PP_list]
            RR_list_cpu = [t.cpu() for t in RR_list]
            SS_list_cpu = [t.cpu() for t in SS_list]
            AA_list_cpu = [t.cpu() for t in AA_list]
            FF_list_cpu = [t.cpu() for t in FF_list]
            HD95_list_cpu = [t.cpu() for t in HD95_list]
            mDice_list_cpu = [t.cpu() for t in mDice_list]
            mIoU_list_cpu = [t.cpu() for t in mIoU_list]
            mPP_list_cpu = [t.cpu() for t in mPP_list]
            mRR_list_cpu = [t.cpu() for t in mRR_list]
            mSS_list_cpu = [t.cpu() for t in mSS_list]
            mAA_list_cpu = [t.cpu() for t in mAA_list]
            mFF_list_cpu = [t.cpu() for t in mFF_list]
            mHD95_list_cpu = [t.cpu() for t in mHD95_list]

            Dice_lists.append(Dice_list_cpu)
            IoU_lists.append(IoU_list_cpu)
            PP_lists.append(PP_list_cpu)
            RR_lists.append(RR_list_cpu)
            SS_lists.append(SS_list_cpu)
            AA_lists.append(AA_list_cpu)
            FF_lists.append(FF_list_cpu)
            HD95_lists.append(HD95_list_cpu)

            mDice_lists.append(mDice_list_cpu)
            mIoU_lists.append(mIoU_list_cpu)
            mPP_lists.append(mPP_list_cpu)
            mRR_lists.append(mRR_list_cpu)
            mSS_lists.append(mSS_list_cpu)
            mAA_lists.append(mAA_list_cpu)
            mFF_lists.append(mFF_list_cpu)
            mHD95_list.append(mHD95_list_cpu)



    mDice_mean = np.mean(mDice_lists)
    mIoU_mean = np.mean(mIoU_lists)
    mPP_mean = np.mean(mPP_lists)
    mRR_mean = np.mean(mRR_lists)
    mSS_mean = np.mean(mSS_lists)
    mAA_mean = np.mean(mAA_lists)
    mFF_mean = np.mean(mFF_lists)
    mHD95_mean = np.mean(mHD95_lists)

    mDice_std = np.std(mDice_lists)
    mIoU_std = np.std(mIoU_lists)
    mPP_std = np.std(mPP_lists)
    mRR_std = np.std(mRR_lists)
    mSS_std = np.std(mSS_lists)
    mAA_std = np.std(mAA_lists)
    mFF_std = np.std(mFF_lists)
    mHD95_std = np.std(mHD95_lists)

    Dice_mean = np.mean(Dice_lists)
    IoU_mean = np.mean(IoU_lists)
    PP_mean = np.mean(PP_lists)
    RR_mean = np.mean(RR_lists)
    SS_mean = np.mean(SS_lists)
    AA_mean = np.mean(AA_lists)
    FF_mean = np.mean(FF_lists)
    HD95_mean = np.mean(HD95_lists)

    Dice_std = np.std(Dice_lists)
    IoU_std = np.std(IoU_lists)
    PP_std = np.std(PP_lists)
    RR_std = np.std(RR_lists)
    SS_std = np.std(SS_lists)
    AA_std = np.std(AA_lists)
    FF_std = np.std(FF_lists)
    HD95_std = np.std(HD95_lists)

    # Mean = [Dice_mean, IoU_mean, PP_mean, RR_mean, SS_mean, AA_mean, FF_mean,
    #         mDice_mean, mIoU_mean, mPP_mean, mRR_mean, mSS_mean, mAA_mean, mFF_mean]
    # Std = [Dice_std, IoU_std, PP_std, RR_std, SS_std, AA_std, FF_std,
    #         mDice_std, mIoU_std, mPP_std, mRR_std, mSS_std, mAA_std, mFF_std]

    net.train()

    now_pred = [None] * (epoch_eval + 1)
    for i in range(epoch_eval + 1):
        now_pred[i] = mask_preds[i]

    # Fixes a potential division by zero error

    return Dice_mean, IoU_mean, PP_mean, RR_mean, SS_mean, AA_mean, FF_mean, HD95_mean, mDice_mean, mIoU_mean, mPP_mean, mRR_mean, mSS_mean, mAA_mean, mFF_mean, mHD95_mean, Dice_std, IoU_std, PP_std, RR_std, SS_std, AA_std, FF_std, HD95_std, mDice_std, mIoU_std, mPP_std, mRR_std, mSS_std, mAA_std, mFF_std, mHD95_std, mask_preds, now_pred


def evaluate_test_list(net, dataloader, device, w_ustm, w_ust, w_m, w_us, w_t, ornot):
    net.eval()
    num_val_batches = len(dataloader)
    mask_preds = [None] * num_val_batches
    epoch_eval = int(-1)
    Dice, IoU, Pre, Rre, Spe, Acc, F2 = 0, 0, 0, 0, 0, 0, 0
    mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2 = 0, 0, 0, 0, 0, 0, 0
    mDice_lists, mIoU_lists, mPP_lists, mRR_lists, mSS_lists, mAA_lists, mFF_lists, mHD95_lists = [], [], [], [], [], [], [], []
    Dice_lists, IoU_lists, PP_lists, RR_lists, SS_lists, AA_lists, FF_lists, HD95_lists = [], [], [], [], [], [], [], []

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Test round', unit='batch', leave=False):
        epoch_eval += 1
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        # print('mask_true.shape=', mask_true.shape)
        mask_true = F.one_hot(mask_true.squeeze_(1), net.n_classes).permute(0, 3, 1, 2).float()


        with torch.no_grad():
            # predict the mask
            if ornot:
                deep_out = net(image)
                mask_pred = deep_out[0] * w_ustm + deep_out[1] * w_ust + deep_out[2] * w_m + deep_out[3] * w_us + deep_out[4] * w_t
            else:
                mask_pred = net(image)
            mask_preds[epoch_eval] = mask_pred.cpu()

            # convert to one-hot format
            if net.n_classes == 2:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                Dice_list, IoU_list, PP_list, RR_list, SS_list, AA_list, FF_list, HD95_list, mDice_list, mIoU_list, mPP_list, \
                    mRR_list, mSS_list, mAA_list, mFF_list, mHD95_list = multiclass_comment_coeff_onepic(mask_pred[:, 0:, ...],
                                                                                             mask_true[:, 0:, ...],
                                                                                             reduce_batch_first=False,
                                                                                             device=device)

            Dice_lists.extend([v.item() for v in Dice_list])
            IoU_lists.extend([v.item() for v in IoU_list])
            PP_lists.extend([v.item() for v in PP_list])
            RR_lists.extend([v.item() for v in RR_list])
            SS_lists.extend([v.item() for v in SS_list])
            AA_lists.extend([v.item() for v in AA_list])
            FF_lists.extend([v.item() for v in FF_list])
            HD95_lists.extend([v.item() for v in HD95_list])

            mDice_lists.extend([v.item() for v in mDice_list])
            mIoU_lists.extend([v.item() for v in mIoU_list])
            mPP_lists.extend([v.item() for v in mPP_list])
            mRR_lists.extend([v.item() for v in mRR_list])
            mSS_lists.extend([v.item() for v in mSS_list])
            mAA_lists.extend([v.item() for v in mAA_list])
            mFF_lists.extend([v.item() for v in mFF_list])
            mHD95_lists.extend([v.item() for v in mHD95_list])

    net.train()

    now_pred = [None] * (epoch_eval + 1)
    for i in range(epoch_eval + 1):
        now_pred[i] = mask_preds[i]

    # Fixes a potential division by zero error

    return Dice_lists, IoU_lists, PP_lists, RR_lists, SS_lists, AA_lists, FF_lists, HD95_lists, mDice_lists, mIoU_lists, mPP_lists, mRR_lists, mSS_lists, mAA_lists, mFF_lists, mHD95_lists, mask_preds, now_pred


def evaluate_ensembleB_test_std(preloader_test, dataloader_test, number_net, label, threshold,  bestpop, net_n_classes, device):

    num_val_batches = len(dataloader_test)
    test_preds = [None] * num_val_batches
    bestpop = torch.from_numpy(bestpop)
    bestpop = bestpop.to(device=device)
    epoch_eval = int(-1)
    bestdic = 0
    Dice, IoU, Pre, Rre, Spe, Acc, F2 = 0, 0, 0, 0, 0, 0, 0
    mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2 = 0, 0, 0, 0, 0, 0, 0
    mDice_lists, mIoU_lists, mPP_lists, mRR_lists, mSS_lists, mAA_lists, mFF_lists, mHD95_lists = [], [], [], [], [], [], [], []
    Dice_lists, IoU_lists, PP_lists, RR_lists, SS_lists, AA_lists, FF_lists, HD95_lists = [], [], [], [], [], [], [], []

    for batch_mask_true_test in tqdm(dataloader_test, total=num_val_batches, desc='Ensemble Test round', unit='batch', leave=False):
        epoch_eval += 1
        mask_true_test = batch_mask_true_test['mask']
        mask_true_test = mask_true_test.to(device=device, dtype=torch.long)
        mask_true_test = F.one_hot(mask_true_test.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre_test = torch.zeros(size=mask_true_test.shape)
        ensemable_mask_pre_test = ensemable_mask_pre_test.to(device=device)
        with torch.no_grad():
            label_time = int(0)
            for epoch_net in range(number_net):
                if label[epoch_net] >= threshold:
                    label_time += 1
                else:
                    ensemable_mask_pre_test += bestpop[epoch_net - label_time] * preloader_test[epoch_net][epoch_eval]
                    test_preds[epoch_eval] = ensemable_mask_pre_test.cpu()
                # IndexError: index 2 is out of bounds for dimension 0 with size 2

            # convert to one-hot format
            if net_n_classes == 2:
                ensemable_mask_pre_test = F.one_hot(ensemable_mask_pre_test.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                Dice_list, IoU_list, PP_list, RR_list, SS_list, AA_list, FF_list, HD95_list, mDice_list, mIoU_list, mPP_list, \
                    mRR_list, mSS_list, mAA_list, mFF_list, mHD95_list = multiclass_comment_coeff_onepic(ensemable_mask_pre_test[:, 0:, ...],
                                                                                             mask_true_test[:, 0:, ...],
                                                                                             reduce_batch_first=False, device=device)
            Dice_lists.append(Dice_list)
            IoU_lists.append(IoU_list)
            PP_lists.append(PP_list)
            RR_lists.append(RR_list)
            SS_lists.append(SS_list)
            AA_lists.append(AA_list)
            FF_lists.append(FF_list)
            HD95_lists.append(HD95_list)

            mDice_lists.append(mDice_list)
            mIoU_lists.append(mIoU_list)
            mPP_lists.append(mPP_list)
            mRR_lists.append(mRR_list)
            mSS_lists.append(mSS_list)
            mAA_lists.append(mAA_list)
            mFF_lists.append(mFF_list)
            mHD95_lists.append(mHD95_list)

    mDice_mean = np.mean(mDice_lists)
    mIoU_mean = np.mean(mIoU_lists)
    mPP_mean = np.mean(mPP_lists)
    mRR_mean = np.mean(mRR_lists)
    mSS_mean = np.mean(mSS_lists)
    mAA_mean = np.mean(mAA_lists)
    mFF_mean = np.mean(mFF_lists)
    mHD95_mean = np.mean(mHD95_lists)

    mDice_std = np.std(mDice_lists)
    mIoU_std = np.std(mIoU_lists)
    mPP_std = np.std(mPP_lists)
    mRR_std = np.std(mRR_lists)
    mSS_std = np.std(mSS_lists)
    mAA_std = np.std(mAA_lists)
    mFF_std = np.std(mFF_lists)
    mHD95_std = np.std(mHD95_lists)

    Dice_mean = np.mean(Dice_lists)
    IoU_mean = np.mean(IoU_lists)
    PP_mean = np.mean(PP_lists)
    RR_mean = np.mean(RR_lists)
    SS_mean = np.mean(SS_lists)
    AA_mean = np.mean(AA_lists)
    FF_mean = np.mean(FF_lists)
    HD95_mean = np.mean(HD95_lists)

    Dice_std = np.std(Dice_lists)
    IoU_std = np.std(IoU_lists)
    PP_std = np.std(PP_lists)
    RR_std = np.std(RR_lists)
    SS_std = np.std(SS_lists)
    AA_std = np.std(AA_lists)
    FF_std = np.std(FF_lists)
    HD95_std = np.std(HD95_lists)

    now_pred_test = [None] * (epoch_eval + 1)
    for i in range(epoch_eval + 1):
        now_pred_test[i] = test_preds[i]


    return Dice_mean, IoU_mean, PP_mean, RR_mean, SS_mean, AA_mean, FF_mean, HD95_mean, mDice_mean, mIoU_mean, mPP_mean, mRR_mean, mSS_mean, mAA_mean, mFF_mean, mHD95_mean, Dice_std, IoU_std, PP_std, RR_std, SS_std, AA_std, FF_std, HD95_std, mDice_std, mIoU_std, mPP_std, mRR_std, mSS_std, mAA_std, mFF_std, mHD95_std, now_pred_test


def evaluate_ensembleB_test_list(preloader_test, dataloader_test, number_net, label, threshold,  bestpop, net_n_classes, device):

    num_val_batches = len(dataloader_test)
    test_preds = [None] * num_val_batches
    bestpop = torch.from_numpy(bestpop)
    bestpop = bestpop.to(device=device)
    epoch_eval = int(-1)
    bestdic = 0
    Dice, IoU, Pre, Rre, Spe, Acc, F2 = 0, 0, 0, 0, 0, 0, 0
    mDice, mIoU, mPre, mRre, mSpe, mAcc, mF2 = 0, 0, 0, 0, 0, 0, 0
    mDice_lists, mIoU_lists, mPP_lists, mRR_lists, mSS_lists, mAA_lists, mFF_lists, mHD95_lists = [], [], [], [], [], [], [], []
    Dice_lists, IoU_lists, PP_lists, RR_lists, SS_lists, AA_lists, FF_lists, HD95_lists = [], [], [], [], [], [], [], []

    for batch_mask_true_test in tqdm(dataloader_test, total=num_val_batches, desc='Ensemble Test round', unit='batch', leave=False):
        epoch_eval += 1
        mask_true_test = batch_mask_true_test['mask']
        mask_true_test = mask_true_test.to(device=device, dtype=torch.long)
        mask_true_test = F.one_hot(mask_true_test.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre_test = torch.zeros(size=mask_true_test.shape)
        ensemable_mask_pre_test = ensemable_mask_pre_test.to(device=device)
        with torch.no_grad():
            label_time = int(0)
            for epoch_net in range(number_net):
                if label[epoch_net] >= threshold:
                    label_time += 1
                else:
                    ensemable_mask_pre_test += bestpop[epoch_net - label_time] * preloader_test[epoch_net][epoch_eval]
                    test_preds[epoch_eval] = ensemable_mask_pre_test.cpu()
                # IndexError: index 2 is out of bounds for dimension 0 with size 2

            # convert to one-hot format
            if net_n_classes == 2:
                ensemable_mask_pre_test = F.one_hot(ensemable_mask_pre_test.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                Dice_list, IoU_list, PP_list, RR_list, SS_list, AA_list, FF_list, HD95_list, mDice_list, mIoU_list, mPP_list, \
                    mRR_list, mSS_list, mAA_list, mFF_list, mHD95_list = multiclass_comment_coeff_onepic(ensemable_mask_pre_test[:, 0:, ...],
                                                                                             mask_true_test[:, 0:, ...],
                                                                                             reduce_batch_first=False, device=device)
            Dice_lists.extend([v.item() for v in Dice_list])
            IoU_lists.extend([v.item() for v in IoU_list])
            PP_lists.extend([v.item() for v in PP_list])
            RR_lists.extend([v.item() for v in RR_list])
            SS_lists.extend([v.item() for v in SS_list])
            AA_lists.extend([v.item() for v in AA_list])
            FF_lists.extend([v.item() for v in FF_list])
            HD95_lists.extend([v.item() for v in HD95_list])

            mDice_lists.extend([v.item() for v in mDice_list])
            mIoU_lists.extend([v.item() for v in mIoU_list])
            mPP_lists.extend([v.item() for v in mPP_list])
            mRR_lists.extend([v.item() for v in mRR_list])
            mSS_lists.extend([v.item() for v in mSS_list])
            mAA_lists.extend([v.item() for v in mAA_list])
            mFF_lists.extend([v.item() for v in mFF_list])
            mHD95_lists.extend([v.item() for v in mHD95_list])

    now_pred_test = [None] * (epoch_eval + 1)
    for i in range(epoch_eval + 1):
        now_pred_test[i] = test_preds[i]


    return Dice_lists, IoU_lists, PP_lists, RR_lists, SS_lists, AA_lists, FF_lists, HD95_lists, mDice_lists, mIoU_lists, mPP_lists, mRR_lists, mSS_lists, mAA_lists, mFF_lists, mHD95_lists, now_pred_test



def evaluate_ensemble_list_std(preloader, dataloader, number_net, label, threshold, bestpop, net_n_classes, device):

    num_val_batches = len(dataloader)
    bestpop = torch.from_numpy(bestpop)
    bestpop = bestpop.to(device=device)
    epoch_eval = int(-1)

    mDice_lists, mIoU_lists, mPP_lists, mRR_lists, mSS_lists, mAA_lists, mFF_lists, mHD95_lists = [], [], [], [], [], [], [], []
    Dice_lists, IoU_lists, PP_lists, RR_lists, SS_lists, AA_lists, FF_lists, HD95_lists = [], [], [], [], [], [], [], []

    for batch_mask_true in tqdm(dataloader, total=num_val_batches, desc='Ensemble Validation round', unit='batch', leave=False):
        epoch_eval += 1
        mask_true = batch_mask_true['mask']
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true.squeeze_(1), net_n_classes).permute(0, 3, 1, 2).float()
        ensemable_mask_pre = torch.zeros(size=mask_true.shape)
        ensemable_mask_pre = ensemable_mask_pre.to(device=device)
        with torch.no_grad():
            label_time = int(0)
            for epoch_net in range(number_net):
                if epoch_net == (number_net - 1):
                    ensemable_mask_pre += bestpop[epoch_net - label_time] * preloader[epoch_net][epoch_eval]
                else:
                    if label[epoch_net] >= threshold:
                        label_time += 1
                    else:
                        ensemable_mask_pre += bestpop[epoch_net - label_time] * preloader[epoch_net][epoch_eval]
                # tem_mask_pre[epoch_net] = preloader[epoch_net][epoch_eval]
                # print('\n', 'epoch_net=', epoch_net)
                # print('bestpop[epoch_net]=', bestpop[epoch_net])

            if net_n_classes == 2:
                ensemable_mask_pre = F.one_hot(ensemable_mask_pre.argmax(dim=1), net_n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                Dice_list, IoU_list, PP_list, RR_list, SS_list, AA_list, FF_list, HD95_list, mDice_list, mIoU_list, mPP_list, \
                    mRR_list, mSS_list, mAA_list, mFF_list, mHD95_list = multiclass_comment_coeff_onepic(ensemable_mask_pre[:, 0:, ...],
                                                                                             mask_true[:, 0:, ...],
                                                                                             reduce_batch_first=False, device=device)
            Dice_lists.extend([v.item() for v in Dice_list])
            IoU_lists.extend([v.item() for v in IoU_list])
            PP_lists.extend([v.item() for v in PP_list])
            RR_lists.extend([v.item() for v in RR_list])
            SS_lists.extend([v.item() for v in SS_list])
            AA_lists.extend([v.item() for v in AA_list])
            FF_lists.extend([v.item() for v in FF_list])
            HD95_lists.extend([v.item() for v in HD95_list])

            mDice_lists.extend([v.item() for v in mDice_list])
            mIoU_lists.extend([v.item() for v in mIoU_list])
            mPP_lists.extend([v.item() for v in mPP_list])
            mRR_lists.extend([v.item() for v in mRR_list])
            mSS_lists.extend([v.item() for v in mSS_list])
            mAA_lists.extend([v.item() for v in mAA_list])
            mFF_lists.extend([v.item() for v in mFF_list])
            mHD95_lists.extend([v.item() for v in mHD95_list])

    mDice_mean = np.mean(mDice_lists)
    mIoU_mean = np.mean(mIoU_lists)
    mPP_mean = np.mean(mPP_lists)
    mRR_mean = np.mean(mRR_lists)
    mSS_mean = np.mean(mSS_lists)
    mAA_mean = np.mean(mAA_lists)
    mFF_mean = np.mean(mFF_lists)
    mHD95_mean = np.mean(mHD95_lists)

    mDice_std = np.std(mDice_lists)
    mIoU_std = np.std(mIoU_lists)
    mPP_std = np.std(mPP_lists)
    mRR_std = np.std(mRR_lists)
    mSS_std = np.std(mSS_lists)
    mAA_std = np.std(mAA_lists)
    mFF_std = np.std(mFF_lists)
    mHD95_std = np.std(mHD95_lists)

    Dice_mean = np.mean(Dice_lists)
    IoU_mean = np.mean(IoU_lists)
    PP_mean = np.mean(PP_lists)
    RR_mean = np.mean(RR_lists)
    SS_mean = np.mean(SS_lists)
    AA_mean = np.mean(AA_lists)
    FF_mean = np.mean(FF_lists)
    HD95_mean = np.mean(HD95_lists)

    Dice_std = np.std(Dice_lists)
    IoU_std = np.std(IoU_lists)
    PP_std = np.std(PP_lists)
    RR_std = np.std(RR_lists)
    SS_std = np.std(SS_lists)
    AA_std = np.std(AA_lists)
    FF_std = np.std(FF_lists)
    HD95_std = np.std(HD95_lists)

    return_list = [Dice_lists, IoU_lists, PP_lists, RR_lists, SS_lists, AA_lists, FF_lists, HD95_lists,
                   mDice_lists, mIoU_lists, mPP_lists, mRR_lists, mSS_lists, mAA_lists, mFF_lists, mHD95_lists]
    return_metric = [Dice_mean, IoU_mean, PP_mean, RR_mean, SS_mean, AA_mean, FF_mean,
                     HD95_mean, mDice_mean, mIoU_mean, mPP_mean, mRR_mean, mSS_mean, mAA_mean, mFF_mean, mHD95_mean,
                     Dice_std, IoU_std, PP_std, RR_std, SS_std, AA_std, FF_std, HD95_std,
                     mDice_std, mIoU_std, mPP_std, mRR_std, mSS_std, mAA_std, mFF_std, mHD95_std]

    return Dice_lists, IoU_lists, PP_lists, RR_lists, SS_lists, AA_lists, FF_lists, HD95_lists, mDice_lists, mIoU_lists, mPP_lists, mRR_lists, mSS_lists, mAA_lists, mFF_lists, mHD95_lists, Dice_mean, IoU_mean, PP_mean, RR_mean, SS_mean, AA_mean, FF_mean, HD95_mean, mDice_mean, mIoU_mean, mPP_mean, mRR_mean, mSS_mean, mAA_mean, mFF_mean, mHD95_mean, Dice_std, IoU_std, PP_std, RR_std, SS_std, AA_std, FF_std, HD95_std, mDice_std, mIoU_std, mPP_std, mRR_std, mSS_std, mAA_std, mFF_std, mHD95_std
