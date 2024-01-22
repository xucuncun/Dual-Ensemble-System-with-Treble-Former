import argparse
import logging
import sys
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import shutil
import os
import wandb
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"


from os import listdir
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path

from utils.data_loading import BasicDataset, CarvanaDataset, Dataset_Pro
from utils.dice_score import dice_loss
from evaluate import evaluate_val, evaluate_test, evaluate_ensemble, evaluate_popfit, Choice_best, evaluate_popfit_label, evaluate_ensemble_l, evaluate_ensemble_llable, evaluate_ensemble_val, evaluate_ensembleA_valpre, evaluate_ensembleA_test, evaluate_ensembleB_valpre, evaluate_ensembleB_test
from compare_mode import SemSegNet



def train_net(Ensemable_name,
              set_epochs,
              number_net,
              set_nets_name,
              set_net,
              set_dir_checkpoint,
              device,
              dir_img,
              dir_mask,
              dir_img0,
              dir_mask0,
              dir_img1,
              dir_mask1,
              dir_img2,
              dir_mask2,
              nets_classes,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              test_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              break_delay: int = 5):

    # 1. init ensemable nets

    ornot = [True, True, False, False, False, False]
    number_ensemble = 1
    set_epochs = int(set_epochs)
    Ename = Ensemable_name
    assert len(Ename) == int(4), \
        'the long of Ensemable_name must 4.'

    pso_result_Fall1 = []
    pso_result_Fall2 = []
    pso_result_Fall3 = []
    # pso_result_Fall4 = []

    pso_result_FallB1 = []
    pso_result_FallB2 = []
    pso_result_FallB3 = []
    # pso_result_FallB4 = []

    pso_result_FallC = []

    pso_result_Sall1 = []
    pso_result_Sall2 = []
    pso_result_Sall3 = []
    # pso_result_Sall4 = []

    pso_result_SallB1 = []
    pso_result_SallB2 = []
    pso_result_SallB3 = []
    # pso_result_SallB4 = []

    pso_result_SallC = []
    time_pso = 0
    first = True
    second = True
    third = True
    forth = True
    fiveth = True
    epoch_nets = [int(-1)] * (number_net + 2)
    epochs_nets = [epochs] * (number_net + 2)
    learning_rate_nets = [learning_rate] * number_net
    break_delay_epoch = int(0)
    break_allow = [bool(False)] * number_net
    break_epoch = [int(0)] * number_net
    continue_epoch = [int(0)] * number_net

    vala = [0.0] * (number_net + 2)
    Dicea = [None] * (number_net + 2)
    IoUa = [None] * (number_net + 2)
    SEa = [None] * (number_net + 2)
    PCa = [None] * (number_net + 2)
    F2a = [None] * (number_net + 2)
    SPa = [None] * (number_net + 2)
    ACCa = [None] * (number_net + 2)
    mDicea = [None] * (number_net + 2)
    mIoUa = [None] * (number_net + 2)
    mSEa = [None] * (number_net + 2)
    mPCa = [None] * (number_net + 2)
    mF2a = [None] * (number_net + 2)
    mSPa = [None] * (number_net + 2)
    mACCa = [None] * (number_net + 2)
    val = []
    Dice = []
    IoU = []
    SE = []
    PC = []
    F2 = []
    SP = []
    ACC = []
    mDice = []
    mIoU = []
    mSE = []
    mPC = []
    mF2 = []
    mSP = []
    mACC = []
    for i in range(3):
        val.insert(i, copy.deepcopy(vala))
        Dice.insert(i, copy.deepcopy(Dicea))
        IoU.insert(i, copy.deepcopy(IoUa))
        SE.insert(i, copy.deepcopy(SEa))
        PC.insert(i, copy.deepcopy(PCa))
        F2.insert(i, copy.deepcopy(F2a))
        SP.insert(i, copy.deepcopy(SPa))
        ACC.insert(i, copy.deepcopy(ACCa))
        mDice.insert(i, copy.deepcopy(mDicea))
        mIoU.insert(i, copy.deepcopy(mIoUa))
        mSE.insert(i, copy.deepcopy(mSEa))
        mPC.insert(i, copy.deepcopy(mPCa))
        mF2.insert(i, copy.deepcopy(mF2a))
        mSP.insert(i, copy.deepcopy(mSPa))
        mACC.insert(i, copy.deepcopy(mACCa))
    for i in range(3):
        val[i][number_net] = [None] * number_ensemble
        Dice[i][number_net] = [None] * number_ensemble
        IoU[i][number_net] = [None] * number_ensemble
        SE[i][number_net] = [None] * number_ensemble
        PC[i][number_net] = [None] * number_ensemble
        F2[i][number_net] = [None] * number_ensemble
        SP[i][number_net] = [None] * number_ensemble
        ACC[i][number_net] = [None] * number_ensemble
        mDice[i][number_net] = [None] * number_ensemble
        mIoU[i][number_net] = [None] * number_ensemble
        mSE[i][number_net] = [None] * number_ensemble
        mPC[i][number_net] = [None] * number_ensemble
        mF2[i][number_net] = [None] * number_ensemble
        mSP[i][number_net] = [None] * number_ensemble
        mACC[i][number_net] = [None] * number_ensemble

        val[i][number_net + 1] = [None] * number_ensemble
        Dice[i][number_net + 1] = [None] * number_ensemble
        IoU[i][number_net + 1] = [None] * number_ensemble
        SE[i][number_net + 1] = [None] * number_ensemble
        PC[i][number_net + 1] = [None] * number_ensemble
        F2[i][number_net + 1] = [None] * number_ensemble
        SP[i][number_net + 1] = [None] * number_ensemble
        ACC[i][number_net + 1] = [None] * number_ensemble
        mDice[i][number_net + 1] = [None] * number_ensemble
        mIoU[i][number_net + 1] = [None] * number_ensemble
        mSE[i][number_net + 1] = [None] * number_ensemble
        mPC[i][number_net + 1] = [None] * number_ensemble
        mF2[i][number_net + 1] = [None] * number_ensemble
        mSP[i][number_net + 1] = [None] * number_ensemble
        mACC[i][number_net + 1] = [None] * number_ensemble

    # valb = [0.0] * (number_ensemble)
    # Diceb = [None] * (number_ensemble)
    # IoUb = [None] * (number_ensemble)
    # SEb = [None] * (number_ensemble)
    # PCb = [None] * (number_ensemble)
    # F2b = [None] * (number_ensemble)
    # SPb = [None] * (number_ensemble)
    # ACCb = [None] * (number_ensemble)
    # mDiceb = [None] * (number_ensemble)
    # mIoUb = [None] * (number_ensemble)
    # mSEb = [None] * (number_ensemble)
    # mPCb = [None] * (number_ensemble)
    # mF2b = [None] * (number_ensemble)
    # mSPb = [None] * (number_ensemble)
    # mACCb = [None] * (number_ensemble)
    # valE = []
    # DiceE = []
    # IoUE = []
    # SEE = []
    # PCE = []
    # F2E = []
    # SPE = []
    # ACCE = []
    # mDiceE = []
    # mIoUE = []
    # mSEE = []
    # mPCE = []
    # mF2E = []
    # mSPE = []
    # mACCE = []
    # for i in range(3+3):
    #     valE.insert(i, copy.deepcopy(valb))
    #     DiceE.insert(i, copy.deepcopy(Diceb))
    #     IoUE.insert(i, copy.deepcopy(IoUb))
    #     SEE.insert(i, copy.deepcopy(SEb))
    #     PCE.insert(i, copy.deepcopy(PCb))
    #     F2E.insert(i, copy.deepcopy(F2b))
    #     SPE.insert(i, copy.deepcopy(SPb))
    #     ACCE.insert(i, copy.deepcopy(ACCb))
    #     mDiceE.insert(i, copy.deepcopy(mDiceb))
    #     mIoUE.insert(i, copy.deepcopy(mIoUb))
    #     mSEE.insert(i, copy.deepcopy(mSEb))
    #     mPCE.insert(i, copy.deepcopy(mPCb))
    #     mF2E.insert(i, copy.deepcopy(mF2b))
    #     mSPE.insert(i, copy.deepcopy(mSPb))
    #     mACCE.insert(i, copy.deepcopy(mACCb))


    val_score = [None] * (number_net + 2)
    val_score[number_net] = [None] * number_ensemble
    val_score[number_net + 1] = [None] * number_ensemble

    val_besta = [] * (number_net + 2)
    val_best = []
    valbest = [0] * 17
    midice = [torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])] * number_net
    st = int(8)
    label = [int(0)] * number_net
    label_threshold = int(3)
    number_label = int(0)
    # midice = [midice, midice, midice, midice, midice, midice]
    break_allow_A = False
    break_allow_B = False
    w_ustm = [0.5] * number_net
    w_ust = [0.5] * number_net
    w_m = [0.5] * number_net
    w_us = [0.5] * number_net
    w_t = [0.5] * number_net
    best_vals = [None] * (number_net + 2)
    best_vals[number_net] = [None] * number_ensemble
    best_vals[number_net + 1] = [None] * number_ensemble

    best_testa = [None] * (number_net + 2)
    bestesta = [None] * (number_net + 2)
    best_testa[number_net] = [None] * number_ensemble
    best_testa[number_net + 1] = [None] * number_ensemble
    bestesta[number_net] = [None] * number_ensemble
    bestesta[number_net + 1] = [None] * number_ensemble
    bestest = []
    best_test = []
    for i in range(number_net + 2):
        if i == number_net:
            val_besta.insert(i, [copy.deepcopy(valbest)] * number_ensemble)
        elif i == number_net + 1:
            val_besta.insert(i, [copy.deepcopy(valbest)] * number_ensemble)
        else:
            val_besta.insert(i, copy.deepcopy(valbest))

    for i in range(3):
        best_test.insert(i, copy.deepcopy(best_testa))
        bestest.insert(i, copy.deepcopy(bestesta))

        val_best.insert(i, copy.deepcopy(val_besta))

    gbestpop = [None] * number_ensemble
    gbestpop_label = [None] * number_ensemble


    nets = set_net
    nets_name = set_nets_name
    nets_name_label = [str('hou1'), str('hou2'), str('hou3'), str('hou4'), str('hou5'), str('hou6')]
    dir_checkpoints = set_dir_checkpoint
    the_lr = copy.deepcopy(learning_rate_nets)
    kaiguan = [int(0)] * number_net
    zongkaiguan = False
    Ensemable_popfits_masks = []
    Ensemable_popfits_mask = [None] * 10
    Ensemable_pred_masks = []
    Ensemable_pred_maska = []
    Ensemable_pred_mask = [[None]] * number_net


    Dice_One = [0] * (set_epochs + 1)
    IoU_One = [0] * (set_epochs + 1)
    SE_One = [0] * (set_epochs + 1)
    PC_One = [0] * (set_epochs + 1)
    F2_One = [0] * (set_epochs + 1)
    SP_One = [0] * (set_epochs + 1)
    ACC_One = [0] * (set_epochs + 1)
    mDice_One = [0] * (set_epochs + 1)
    mIoU_One = [0] * (set_epochs + 1)
    mSE_One = [0] * (set_epochs + 1)
    mPC_One = [0] * (set_epochs + 1)
    mF2_One = [0] * (set_epochs + 1)
    mSP_One = [0] * (set_epochs + 1)
    mACC_One = [0] * (set_epochs + 1)
    loss_One = [0] * (set_epochs + 1)
    lr_One = [0] * (set_epochs + 1)
    epoch_One = [0] * (set_epochs + 1)

    Dice_all = [] * (number_net + 2)
    IoU_all = [] * (number_net + 2)
    SE_all = [] * (number_net + 2)
    PC_all = [] * (number_net + 2)
    F2_all = [] * (number_net + 2)
    SP_all = [] * (number_net + 2)
    ACC_all = [] * (number_net + 2)
    mDice_all = [] * (number_net + 2)
    mIoU_all = [] * (number_net + 2)
    mSE_all = [] * (number_net + 2)
    mPC_all = [] * (number_net + 2)
    mF2_all = [] * (number_net + 2)
    mSP_all = [] * (number_net + 2)
    mACC_all = [] * (number_net + 2)

    Dice_All = [] * (3)
    IoU_All = [] * (3)
    SE_All = [] * (3)
    PC_All = [] * (3)
    F2_All = [] * (3)
    SP_All = [] * (3)
    ACC_All = [] * (3)
    mDice_All = [] * (3)
    mIoU_All = [] * (3)
    mSE_All = [] * (3)
    mPC_All = [] * (3)
    mF2_All = [] * (3)
    mSP_All = [] * (3)
    mACC_All = [] * (3)
    lr_All = [] * (number_net + 2)
    loss_All = [] * (number_net + 2)
    epoch_All = [] * (number_net + 2)
    Dice_pbl = [[None] * number_ensemble] * (3)
    IoU_pbl = [[None] * number_ensemble] * (3)
    PC_pbl = [[None] * number_ensemble] * (3)
    SE_pbl = [[None] * number_ensemble] * (3)
    SP_pbl = [[None] * number_ensemble] * (3)
    ACC_pbl = [[None] * number_ensemble] * (3)
    F2_pbl = [[None] * number_ensemble] * (3)
    mDice_pbl = [[None] * number_ensemble] * (3)
    mIoU_pbl = [[None] * number_ensemble] * (3)
    mPC_pbl = [[None] * number_ensemble] * (3)
    mSE_pbl = [[None] * number_ensemble] * (3)
    mSP_pbl = [[None] * number_ensemble] * (3)
    mACC_pbl = [[None] * number_ensemble] * (3)
    mF2_pbl = [[None] * number_ensemble] * (3)
    Dice_pl = [[None] * number_ensemble] * (3)
    IoU_pl = [[None] * number_ensemble] * (3)
    PC_pl = [[None] * number_ensemble] * (3)
    SE_pl = [[None] * number_ensemble] * (3)
    SP_pl = [[None] * number_ensemble] * (3)
    ACC_pl = [[None] * number_ensemble] * (3)
    F2_pl = [[None] * number_ensemble] * (3)
    mDice_pl = [[None] * number_ensemble] * (3)
    mIoU_pl = [[None] * number_ensemble] * (3)
    mPC_pl = [[None] * number_ensemble] * (3)
    mSE_pl = [[None] * number_ensemble] * (3)
    mSP_pl = [[None] * number_ensemble] * (3)
    mACC_pl = [[None] * number_ensemble] * (3)
    mF2_pl = [[None] * number_ensemble] * (3)

    for i in range(number_net + 2):
        Dice_all.insert(i, copy.deepcopy(Dice_One))
        IoU_all.insert(i, copy.deepcopy(IoU_One))
        SE_all.insert(i, copy.deepcopy(SE_One))
        PC_all.insert(i, copy.deepcopy(PC_One))
        F2_all.insert(i, copy.deepcopy(F2_One))
        SP_all.insert(i, copy.deepcopy(SP_One))
        ACC_all.insert(i, copy.deepcopy(ACC_One))
        mDice_all.insert(i, copy.deepcopy(mDice_One))
        mIoU_all.insert(i, copy.deepcopy(mIoU_One))
        mSE_all.insert(i, copy.deepcopy(mSE_One))
        mPC_all.insert(i, copy.deepcopy(mPC_One))
        mF2_all.insert(i, copy.deepcopy(mF2_One))
        mSP_all.insert(i, copy.deepcopy(mSP_One))
        mACC_all.insert(i, copy.deepcopy(mACC_One))
        lr_All.insert(i, copy.deepcopy(lr_One))
        loss_All.insert(i, copy.deepcopy(loss_One))
        epoch_All.insert(i, copy.deepcopy(epoch_One))

    for i in range(3):
        Dice_All.insert(i, copy.deepcopy(Dice_all))
        IoU_All.insert(i, copy.deepcopy(IoU_all))
        SE_All.insert(i, copy.deepcopy(SE_all))
        PC_All.insert(i, copy.deepcopy(PC_all))
        F2_All.insert(i, copy.deepcopy(F2_all))
        SP_All.insert(i, copy.deepcopy(SP_all))
        ACC_All.insert(i, copy.deepcopy(ACC_all))
        mDice_All.insert(i, copy.deepcopy(mDice_all))
        mIoU_All.insert(i, copy.deepcopy(mIoU_all))
        mSE_All.insert(i, copy.deepcopy(mSE_all))
        mPC_All.insert(i, copy.deepcopy(mPC_all))
        mF2_All.insert(i, copy.deepcopy(mF2_all))
        mSP_All.insert(i, copy.deepcopy(mSP_all))
        mACC_All.insert(i, copy.deepcopy(mACC_all))

    for i in range(3):
        for j in range(set_epochs + 1):
            Dice_All[i][number_net][j] = [0] * number_ensemble
            IoU_All[i][number_net][j] = [0] * number_ensemble
            SE_All[i][number_net][j] = [0] * number_ensemble
            PC_All[i][number_net][j] = [0] * number_ensemble
            F2_All[i][number_net][j] = [0] * number_ensemble
            SP_All[i][number_net][j] = [0] * number_ensemble
            ACC_All[i][number_net][j] = [0] * number_ensemble
            mDice_All[i][number_net][j] = [0] * number_ensemble
            mIoU_All[i][number_net][j] = [0] * number_ensemble
            mSE_All[i][number_net][j] = [0] * number_ensemble
            mPC_All[i][number_net][j] = [0] * number_ensemble
            mF2_All[i][number_net][j] = [0] * number_ensemble
            mSP_All[i][number_net][j] = [0] * number_ensemble
            mACC_All[i][number_net][j] = [0] * number_ensemble

            Dice_All[i][number_net + 1][j] = [0] * number_ensemble
            IoU_All[i][number_net + 1][j] = [0] * number_ensemble
            SE_All[i][number_net + 1][j] = [0] * number_ensemble
            PC_All[i][number_net + 1][j] = [0] * number_ensemble
            F2_All[i][number_net + 1][j] = [0] * number_ensemble
            SP_All[i][number_net + 1][j] = [0] * number_ensemble
            ACC_All[i][number_net + 1][j] = [0] * number_ensemble
            mDice_All[i][number_net + 1][j] = [0] * number_ensemble
            mIoU_All[i][number_net + 1][j] = [0] * number_ensemble
            mSE_All[i][number_net + 1][j] = [0] * number_ensemble
            mPC_All[i][number_net + 1][j] = [0] * number_ensemble
            mF2_All[i][number_net + 1][j] = [0] * number_ensemble
            mSP_All[i][number_net + 1][j] = [0] * number_ensemble
            mACC_All[i][number_net + 1][j] = [0] * number_ensemble

    for i in range(10):
        Ensemable_pred_maska.insert(i, copy.deepcopy(Ensemable_pred_mask))
    for i in range(3):
        Ensemable_pred_masks.insert(i, copy.deepcopy(Ensemable_pred_maska))
    for i in range(number_net):
        Ensemable_popfits_masks.insert(i, copy.deepcopy(Ensemable_popfits_mask))

    # 2. Create dataset

    list_img = listdir(dir_img)
    list_mask = listdir(dir_mask)
    list_img_test0 = listdir(dir_img0)
    list_mask_test0 = listdir(dir_mask0)
    list_img_test1 = listdir(dir_img1)
    list_mask_test1 = listdir(dir_mask1)
    list_img_test2 = listdir(dir_img2)
    list_mask_test2 = listdir(dir_mask2)
    n_val = int(len(list_img) * val_percent)
    n_test0 = int(len(list_img_test0))
    n_test1 = int(len(list_img_test1))
    n_test2 = int(len(list_img_test2))
    n_train = int(len(list_img) - n_val)
    print('n_train=', n_train, 'n_val=', n_val, 'n_test0=', n_test0, 'n_test1=', n_test1, 'n_test2=', n_test2)
    train_img_set, val_img_set = random_split(list_img, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_mask_set, val_mask_set = random_split(list_mask, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    test_img_set0, _ = random_split(list_img_test0, [n_test0, 0], generator=torch.Generator().manual_seed(0))
    test_mask_set0, _ = random_split(list_mask_test0, [n_test0, 0], generator=torch.Generator().manual_seed(0))
    test_img_set1, _ = random_split(list_img_test1, [n_test1, 0], generator=torch.Generator().manual_seed(0))
    test_mask_set1, _ = random_split(list_mask_test1, [n_test1, 0], generator=torch.Generator().manual_seed(0))
    test_img_set2, _ = random_split(list_img_test2, [n_test2, 0], generator=torch.Generator().manual_seed(0))
    test_mask_set2, _ = random_split(list_mask_test2, [n_test2, 0], generator=torch.Generator().manual_seed(0))
    train_set = Dataset_Pro(dir_img, dir_mask, train_img_set, train_mask_set, img_scale, augmentations=True)
    val_set = Dataset_Pro(dir_img, dir_mask, val_img_set, val_mask_set, img_scale, augmentations=False)
    test_set0 = Dataset_Pro(dir_img0, dir_mask0, test_img_set0, test_mask_set0, img_scale, augmentations=False)
    test_set1 = Dataset_Pro(dir_img1, dir_mask1, test_img_set1, test_mask_set1, img_scale, augmentations=False)
    test_set2 = Dataset_Pro(dir_img2, dir_mask2, test_img_set2, test_mask_set2, img_scale, augmentations=False)
    # n_val = len(val_img_set)
    # n_test = len(test_img_set)
    # n_train = len(train_img_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    test_loader0 = DataLoader(test_set0, shuffle=False, drop_last=True, **loader_args)
    test_loader1 = DataLoader(test_set1, shuffle=False, drop_last=True, **loader_args)
    test_loader2 = DataLoader(test_set2, shuffle=False, drop_last=True, **loader_args)
    test_loader = [test_loader0, test_loader1, test_loader2]

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. init the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = [None] * number_net
    scheduler = [None] * number_net
    for i in range(number_net):
        optimizer[i] = optim.AdamW(nets[i].parameters(), lr=learning_rate_nets[i], weight_decay=1e-8, betas=(0.9, 0.999))
        scheduler[i] = optim.lr_scheduler.ReduceLROnPlateau(optimizer[i], 'max', factor=0.5, min_lr=1e-6, patience=3)
        # scheduler[i] = optim.lr_scheduler.ReduceLROnPlateau(optimizer[i], 'max', factor=0.5, min_lr=1e-6, patience=10, cooldown=5)
        # scheduler[i] = optim.lr_scheduler.ReduceLROnPlateau(optimizer[i], 'max', factor=0.5, min_lr=1e-6, patience=2)
        # scheduler[i] = optim.lr_scheduler.ReduceLROnPlateau(optimizer1[i], mode='min', factor=0.5, min_lr=1e-6, patience=400)
    grad_scaler = [torch.cuda.amp.GradScaler(enabled=amp)] * number_net
    criterion = [nn.CrossEntropyLoss()] * number_net
    global_step = [int(0)] * (number_net + 2)
    loss = [None] * number_net
    epoch_loss = [float(0)] * number_net
    histograms = [{}] * (number_net + 1)
    experiment0 = [None] * (number_net + 6)
    experiment1 = [None] * (number_net + 6)
    experiment2 = [None] * (number_net + 6)

    experiment0[number_net] = [None] * number_ensemble
    experiment1[number_net] = [None] * number_ensemble
    experiment2[number_net] = [None] * number_ensemble
    experiment0[number_net + 1] = [None] * number_ensemble
    experiment1[number_net + 1] = [None] * number_ensemble
    experiment2[number_net + 1] = [None] * number_ensemble
    experiment0[number_net + 2] = [None] * number_ensemble
    experiment1[number_net + 2] = [None] * number_ensemble
    experiment2[number_net + 2] = [None] * number_ensemble
    experiment0[number_net + 3] = [None] * number_ensemble
    experiment1[number_net + 3] = [None] * number_ensemble
    experiment2[number_net + 3] = [None] * number_ensemble

    traindata_len = 0
    for i in range(len(train_loader)):
        traindata_len += 1

    txt_dice = [open('text_result/net1_dice.txt', 'r+'),
                open('text_result/net2_dice.txt', 'r+'),
                open('text_result/net3_dice.txt', 'r+'),
                open('text_result/net4_dice.txt', 'r+'),
                open('text_result/net5_dice.txt', 'r+'),
                open('text_result/net6_dice.txt', 'r+')
                ]
    txt_w = [open('text_result/net1_w.txt', 'r+'),
             open('text_result/net2_w.txt', 'r+'),
             open('text_result/net3_w.txt', 'r+'),
             open('text_result/net4_w.txt', 'r+'),
             open('text_result/net5_w.txt', 'r+'),
             open('text_result/net6_w.txt', 'r+')
             ]
    txt_ensm = [open('text_result/ensm0.txt', 'r+'),
                open('text_result/ensm1.txt', 'r+'),
                open('text_result/ensm2.txt', 'r+'),
                open('text_result/ensm3.txt', 'r+')
                ]
    txt_last = [open('text_result/last_net.txt', 'r+'),
                open('text_result/last_ensm0.txt', 'r+'),
                open('text_result/last_ensm1.txt', 'r+'),
                open('text_result/last_ensm2.txt', 'r+'),
                open('text_result/last_ensm3.txt', 'r+')
                ]
    txt = [open('text_result/ZZ_net1.txt', 'r+'),
           open('text_result/ZZ_net2.txt', 'r+'),
           open('text_result/ZZ_net3.txt', 'r+'),
           open('text_result/ZZ_net4.txt', 'r+'),
           open('text_result/ZZ_net5.txt', 'r+'),
           open('text_result/ZZ_net6.txt', 'r+'),
           open('text_result/ZZ_EnsA.txt', 'r+'),
           open('text_result/ZZ_EnsB.txt', 'r+'),
           open('text_result/ZZ_other.txt', 'r+')
           ]
    txt_weight = [open('checkpoints/EnsA/weight.txt', 'r+'), open('checkpoints/EnsB/weight.txt', 'r+')]

    # 5  training
    # 5.1.1  training all nets in stage1
    STAG = str('A')
    for TIME in range(int(set_epochs / 10)):
        if STAG == str('A'):
            for Epoch_Net in range(number_net):
                print(f'stage{TIME} train net{Epoch_Net + 1}')
                for epoch_stage in range(10):
                    epoch_nets[Epoch_Net] += 1
                    assert epoch_nets[Epoch_Net] == (epoch_stage + TIME * 10)
                    nets[Epoch_Net].train()
                    with tqdm(total=n_train,
                              desc=f'{nets_name[Epoch_Net]} Epoch {epoch_nets[Epoch_Net] + 1}/{epochs_nets[Epoch_Net]}',
                              unit='img') as pbar:
                        for batch in train_loader:
                            images = batch['image']
                            true_masks = batch['mask']

                            assert images.shape[1] == nets[Epoch_Net].n_channels, \
                                f'Network has been defined with {nets[Epoch_Net].n_channels} input channels, ' \
                                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                                'the images are loaded correctly.'

                            images = images.to(device=device, dtype=torch.float32)
                            true_masks = true_masks.to(device=device, dtype=torch.long)

                            with torch.cuda.amp.autocast(enabled=amp):

                                # w_ustm[Epoch_Net], w_ust[Epoch_Net], w_m[Epoch_Net], w_us[Epoch_Net], w_t[Epoch_Net] = super_combine(midice[Epoch_Net])
                                # masks_pred = w_ustm[Epoch_Net] * deep_out[0] + w_ust[Epoch_Net] * deep_out[1] + w_m[Epoch_Net] * \
                                #              deep_out[2] + w_us[Epoch_Net] * deep_out[3] + w_t[Epoch_Net] * deep_out[4]
                                true_masks = F.one_hot(true_masks.squeeze_(1), nets[Epoch_Net].n_classes).permute(0, 3, 1,
                                                                                                                  2).float()


                                if ornot[Epoch_Net]:
                                    deep_out = nets[Epoch_Net](images)
                                    w_ustm[Epoch_Net], w_ust[Epoch_Net], w_m[Epoch_Net], w_us[Epoch_Net], w_t[
                                        Epoch_Net] = super_combine(midice[Epoch_Net])

                                    masks_pred = w_ustm[Epoch_Net] * deep_out[0] + w_ust[Epoch_Net] * deep_out[1] + w_m[
                                        Epoch_Net] * \
                                                 deep_out[2] + w_us[Epoch_Net] * deep_out[3] + w_t[Epoch_Net] * \
                                                 deep_out[4]

                                    loss[Epoch_Net] = criterion[Epoch_Net](deep_out[0], true_masks) + \
                                                      dice_loss(F.softmax(deep_out[0], dim=1).float(), true_masks,
                                                                multiclass=True) + \
                                                      criterion[Epoch_Net](deep_out[1], true_masks) + \
                                                      dice_loss(F.softmax(deep_out[1], dim=1).float(), true_masks,
                                                                multiclass=True) + \
                                                      criterion[Epoch_Net](deep_out[2], true_masks) + \
                                                      dice_loss(F.softmax(deep_out[2], dim=1).float(), true_masks,
                                                                multiclass=True) + \
                                                      criterion[Epoch_Net](deep_out[3], true_masks) + \
                                                      dice_loss(F.softmax(deep_out[3], dim=1).float(), true_masks,
                                                                multiclass=True) + \
                                                      criterion[Epoch_Net](deep_out[4], true_masks) + \
                                                      dice_loss(F.softmax(deep_out[4], dim=1).float(), true_masks,
                                                                multiclass=True)
                                else:
                                    deep_out = nets[Epoch_Net](images)
                                    masks_pred = deep_out
                                    loss[Epoch_Net] = criterion[Epoch_Net](deep_out, true_masks) + \
                                                      dice_loss(F.softmax(deep_out, dim=1).float(), true_masks,
                                                                multiclass=True)

                                # masks_pred = nets[Epoch_Net](images)
                                # true_masks = F.one_hot(true_masks.squeeze_(1), nets[Epoch_Net].n_classes).permute(0, 3, 1,
                                #                                                                                   2).float()




                            optimizer[Epoch_Net].zero_grad(set_to_none=True)
                            grad_scaler[Epoch_Net].scale(loss[Epoch_Net]).backward()
                            grad_scaler[Epoch_Net].step(optimizer[Epoch_Net])
                            grad_scaler[Epoch_Net].update()
                            pbar.update(images.shape[0])

                            global_step[Epoch_Net] += 1
                            epoch_loss[Epoch_Net] += loss[Epoch_Net].item()
                            if global_step[Epoch_Net] == 1:
                                for i in range(3):
                                    Dice_All[i][Epoch_Net][0] = 0
                                    IoU_All[i][Epoch_Net][0] = 0
                                    SE_All[i][Epoch_Net][0] = 0
                                    PC_All[i][Epoch_Net][0] = 0
                                    F2_All[i][Epoch_Net][0] = 0
                                    SP_All[i][Epoch_Net][0] = 0
                                    ACC_All[i][Epoch_Net][0] = 0
                                    mDice_All[i][Epoch_Net][0] = 0
                                    mIoU_All[i][Epoch_Net][0] = 0
                                    mSE_All[i][Epoch_Net][0] = 0
                                    mPC_All[i][Epoch_Net][0] = 0
                                    mF2_All[i][Epoch_Net][0] = 0
                                    mSP_All[i][Epoch_Net][0] = 0
                                    mACC_All[i][Epoch_Net][0] = 0
                                lr_All[Epoch_Net][0] = learning_rate
                                loss_All[Epoch_Net][0] = 2
                                epoch_All[Epoch_Net][0] = 0
                            pbar.set_postfix(**{f'net{Epoch_Net + 1} loss (batch)': loss[Epoch_Net].item()})

                            # Evaluation round
                            division_step = (n_train // (1 * batch_size))
                            if division_step > 0:
                                if global_step[Epoch_Net] % division_step == 0:
                                    # for tag, value in nets[Epoch_Net].named_parameters():
                                    #     tag = tag.replace('/', '.')
                                    #     histograms[Epoch_Net][f'{Epoch_Net} Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                    #     histograms1['Net1 Gradients/' + tag1] = wandb1.Histogram(value1.grad.data.cpu())

                                    val_score[Epoch_Net], sdice1, sdice2, sdice3, sdice4, sdice5, Ensemable_popfits_masks[
                                        Epoch_Net], best_vals[Epoch_Net] = evaluate_val(
                                        nets[Epoch_Net], val_loader, device, w_ustm[Epoch_Net], w_ust[Epoch_Net],
                                        w_m[Epoch_Net], w_us[Epoch_Net], w_t[Epoch_Net], Ensemable_popfits_masks[Epoch_Net], ornot[Epoch_Net])

                                    for i in range(3):
                                        Dice[i][Epoch_Net], IoU[i][Epoch_Net], PC[i][Epoch_Net], SE[i][Epoch_Net], \
                                            SP[i][Epoch_Net], ACC[i][Epoch_Net], F2[i][Epoch_Net], \
                                            mDice[i][Epoch_Net], mIoU[i][Epoch_Net], mPC[i][Epoch_Net], mSE[i][Epoch_Net], \
                                            mSP[i][Epoch_Net], mACC[i][Epoch_Net], mF2[i][Epoch_Net], \
                                            Ensemable_pred_masks[i][epoch_stage][Epoch_Net],\
                                            best_test[i][Epoch_Net] = evaluate_test(nets[Epoch_Net], test_loader[i], device,
                                                                                    w_ustm[Epoch_Net], w_ust[Epoch_Net],
                                                                                    w_m[Epoch_Net], w_us[Epoch_Net],
                                                                                    w_t[Epoch_Net], ornot[Epoch_Net])
                                    midice[Epoch_Net] = [sdice1, sdice2, sdice3, sdice4, sdice5]

                                    # val_score[Epoch_Net], Ensemable_popfits_masks[Epoch_Net], best_vals[
                                    #     Epoch_Net] = evaluate_val(
                                    #     nets[Epoch_Net], val_loader, device, Ensemable_popfits_masks[Epoch_Net])
                                    #
                                    # Dice[Epoch_Net], mDice[Epoch_Net], mIoU[Epoch_Net], PC[Epoch_Net], SE[
                                    #     Epoch_Net], SP[Epoch_Net], ACC[Epoch_Net], F2[Epoch_Net], \
                                    #     Ensemable_pred_masks[epoch_nets[Epoch_Net]][Epoch_Net] = evaluate_test(nets[Epoch_Net],
                                    #                                                                            test_loader,
                                    #                                                                            device)

                                    if epoch_nets[Epoch_Net] < 10:
                                        pass
                                    else:
                                        scheduler[Epoch_Net].step(val_score[Epoch_Net])

                                    logging.info('Validation Dice score: {}'.format(val_score[Epoch_Net]))
                                    for i in range(3):
                                        Dice_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = Dice[i][Epoch_Net]
                                        IoU_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = IoU[i][Epoch_Net]
                                        SE_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = SE[i][Epoch_Net]
                                        PC_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = PC[i][Epoch_Net]
                                        F2_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = F2[i][Epoch_Net]
                                        SP_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = SP[i][Epoch_Net]
                                        ACC_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = ACC[i][Epoch_Net]
                                        mDice_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = mDice[i][Epoch_Net]
                                        mIoU_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = mIoU[i][Epoch_Net]
                                        mSE_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = mSE[i][Epoch_Net]
                                        mPC_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = mPC[i][Epoch_Net]
                                        mF2_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = mF2[i][Epoch_Net]
                                        mSP_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = mSP[i][Epoch_Net]
                                        mACC_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = mACC[i][Epoch_Net]
                                    lr_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = optimizer[Epoch_Net].param_groups[0]['lr']
                                    loss_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = loss[Epoch_Net].item()
                                    epoch_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = (epoch_nets[Epoch_Net] + 1)

                                    for i in range(3):
                                        val[i][Epoch_Net] = [val_score[Epoch_Net], Dice[i][Epoch_Net], IoU[i][Epoch_Net], SE[i][Epoch_Net],
                                                             PC[i][Epoch_Net], F2[i][Epoch_Net], SP[i][Epoch_Net], ACC[i][Epoch_Net],
                                                          mDice[i][Epoch_Net], mIoU[i][Epoch_Net], mSE[i][Epoch_Net], mPC[i][Epoch_Net],
                                                          mF2[i][Epoch_Net], mSP[i][Epoch_Net], mACC[i][Epoch_Net], best_vals[Epoch_Net]]
                                        val_best[i][Epoch_Net], bestest[i][Epoch_Net] = Choice_best(val=val[i][Epoch_Net],
                                                                                          val_best=val_best[i][Epoch_Net],
                                                                                          bestest=bestest[i][Epoch_Net],
                                                                                          best_test=best_test[i][Epoch_Net],
                                                                                          Epoch=epoch_nets[Epoch_Net])


                                    if optimizer[Epoch_Net].param_groups[0]['lr'] < the_lr[Epoch_Net]:
                                        the_lr[Epoch_Net] = optimizer[Epoch_Net].param_groups[0]['lr']
                                        kaiguan[Epoch_Net] += 1

                                    # print('\n' + 'out_USTM.dice=', midice[Epoch_Net][0].item())
                                    # print('out_UST.dice=', midice[Epoch_Net][1].item())
                                    # print('out_MB.dice=', midice[Epoch_Net][2].item())
                                    # print('out_USB.dice=', midice[Epoch_Net][3].item())
                                    # print('out_TB.dice=', midice[Epoch_Net][4].item())
                                    print('w_ustm=', w_ustm[Epoch_Net], 'w_ust=', w_ust[Epoch_Net], 'w_m=', w_m[Epoch_Net],
                                          'w_us=', w_us[Epoch_Net], 'w_t=', w_t[Epoch_Net])
                                    print('\n', 'label=', label, 'kaiguan=', kaiguan)

                                    # txt_dice[Epoch_Net].write(
                                    #     f'{epoch_nets[Epoch_Net]}' + '\n' +
                                    #     ' out_USTM.dice=' + f'{midice[Epoch_Net][1].item()}' +
                                    #     ', out_UST.dice=' + f'{midice[Epoch_Net][1].item()}' +
                                    #     ', out_MB.dice=' + f'{midice[Epoch_Net][2].item()}' +
                                    #     ', out_USB.dice=' + f'{midice[Epoch_Net][3].item()}' +
                                    #     ', out_TB.dice=' + f'{midice[Epoch_Net][4].item()}')
                                    txt_w[Epoch_Net].write(
                                        f'{epoch_nets[Epoch_Net]}' + '\n' +
                                        ' w_ustm=' + f'{w_ustm[Epoch_Net]}' +
                                        ' w_ust=' + f'{w_ust[Epoch_Net]}' +
                                        ' w_m=' + f'{w_m[Epoch_Net]}' +
                                        ' w_us=' + f'{w_us[Epoch_Net]}' +
                                        ' w_t=' + f'{w_t[Epoch_Net]}')


                    if save_checkpoint:
                        torch.save(nets[Epoch_Net].state_dict(),
                                   str('checkpoints/sub_model/epoch' + '{}'.format(epoch_stage) + 'checkpoint_net{}.pth'.format(
                                       Epoch_Net)))
                        if val[0][Epoch_Net][0] == val_best[0][Epoch_Net][0]:
                            Path(dir_checkpoints[Epoch_Net]).mkdir(parents=True, exist_ok=True)
                            torch.save(nets[Epoch_Net].state_dict(), str(
                                dir_checkpoints[Epoch_Net] / 'best_checkpoint.pth'))
                            logging.info(f'Net Checkpoint {epoch_nets[Epoch_Net] + 1} saved!')
                        elif epoch_nets[Epoch_Net] == (set_epochs - 1):
                            Path(dir_checkpoints[Epoch_Net]).mkdir(parents=True, exist_ok=True)
                            torch.save(nets[Epoch_Net].state_dict(), str(
                                dir_checkpoints[Epoch_Net] / 'last_checkpoint.pth'))
                            logging.info(f'Net Checkpoint {epoch_nets[Epoch_Net] + 1} saved!')
                        else:
                            pass


            # 5.1.2  training ensemble net in stage1
            print('-----------------------------------------------')
            print(f'stage{TIME} PSO Train')
            print('-----------------------------------------------')

            use_popa = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
            use_pop = []
            for i in range(number_ensemble):
                use_pop.insert(i, copy.deepcopy(use_popa))

            if (kaiguan[0] > 0) and (kaiguan[1] > 0) and (kaiguan[2] > 0) and (kaiguan[3] > 0) and (kaiguan[4] > 0) and (kaiguan[5] > 0):
                zongkaiguan = True
                first = True

            pso_lr = (1.0, 1.0)

            if zongkaiguan:
                pso_epoch = int(5)
                pso_sumpop = int(100)
            else:
                pso_epoch = int(3)
                pso_sumpop = int(50)

            # if zongkaiguan:
            #     pso_epoch = int(3)
            # else:
            #     pso_epoch = int(2)
            # pso_sumpop = int(18)

            pso_resultF1, pso_resultS1, gbestpop[0] = SDBHPSO_Ensemble(pso_epoch=pso_epoch, pso_sumpop=pso_sumpop,
                                                              pso_lr=(2.0, 2.0), init=True,
                                                              Ensemable_popfits_masks=Ensemable_popfits_masks,
                                                              val_loader=val_loader, nets_classes=nets_classes,
                                                              number_net=number_net, label=label,
                                                              label_threshold=label_threshold, elimate=False)

            pso_resultF2 = pso_resultF1
            pso_resultS2 = pso_resultS1
            pso_resultF3 = pso_resultF1
            pso_resultS3 = pso_resultS1


            if zongkaiguan:
                for i in range(number_net):
                    kaiguan[i] = kaiguan[i] - 1
                    if gbestpop[0][i] <= 0:
                        label[i] += 1
                        if label[i] >= label_threshold:
                            STAG = str("B")
            txt_ensm[0].write(
                '\n' + '\n' + '\n' + '------------' + '\n' +
                'now is the 1 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS1}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF1}' + '\n' +
                'now is the 2 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS2}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF2}' + '\n' +
                'now is the 3 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS3}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF3}' + '\n' +
                ', label=' + f'{label}' +
                '\n' + '------------' + '\n' + '\n' + '\n')
            txt[number_net + 2].write(
                '\n' + '------------' + '\n' +
                'part A' + f'TIME={TIME}' + '\n' +
                'now is the 1 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS1}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF1}' + '\n' +
                'now is the 2 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS2}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF2}' + '\n' +
                'now is the 3 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS3}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF3}' + '\n' +
                ', label=' + f'{label}' +
                '\n' + '------------' + '\n')
            for i in range(len(pso_resultF1)):
                pso_result_Fall1.insert(time_pso, pso_resultF1[i])
                pso_result_Sall1.insert(time_pso, pso_resultS1[i])

                pso_result_Fall2.insert(time_pso, pso_resultF2[i])
                pso_result_Sall2.insert(time_pso, pso_resultS2[i])

                pso_result_Fall3.insert(time_pso, pso_resultF3[i])
                pso_result_Sall3.insert(time_pso, pso_resultS3[i])
                time_pso += 1
            time_pso_tem = time_pso
            print('label=', label)
            print('pso_resultF1=', pso_resultF1)
            print('pso_resultS1=', pso_resultS1)
            print('pso_resultF2=', pso_resultF2)
            print('pso_resultS2=', pso_resultS2)
            print('pso_resultF3=', pso_resultF3)
            print('pso_resultS3=', pso_resultS3)
            print('PSO one over')
            plt.plot(pso_resultF1)
            plt.plot(pso_resultF2)
            plt.plot(pso_resultF3)
            plt.savefig(f'text_result/{TIME}_A.png')

            print('-----------------------------------------------')
            print(f'stage{TIME} Ensemable Start')
            print('-----------------------------------------------')

            for e in range(10):
                epoch_nets[number_net] += 1
                for j in range(number_ensemble):
                    best_vals[number_net][j] = evaluate_ensembleA_valpre(
                        dataloader_val=val_loader,
                        preloader_val=Ensemable_popfits_masks,
                        time=e,
                        number_net=number_net,
                        bestpop=use_pop[j],
                        net_n_classes=nets_classes,
                        device='cpu')

                for i in range(3):
                    mask_pre = Ensemable_pred_masks[i][e]
                    for j in range(number_ensemble):
                        Dice[i][number_net][j], IoU[i][number_net][j], PC[i][number_net][j], SE[i][number_net][j], \
                            SP[i][number_net][j], ACC[i][number_net][j], F2[i][number_net][j], \
                            mDice[i][number_net][j], mIoU[i][number_net][j], mPC[i][number_net][j], mSE[i][number_net][j], \
                            mSP[i][number_net][j], mACC[i][number_net][j], mF2[i][number_net][j], \
                            best_test[i][number_net][j] = evaluate_ensembleA_test(preloader_test=mask_pre, dataloader_test=test_loader[i],
                                                                              number_net=number_net, bestpop=use_pop[j],
                                                                              net_n_classes=nets_classes, device='cpu')

                for j in range(number_ensemble):
                    val_score[number_net][j] = evaluate_ensemble_val(best_vals[number_net][j], val_loader, nets_classes, device)
                    for i in range(3):
                        val[i][number_net][j] = [val_score[number_net][j], Dice[i][number_net][j], IoU[i][number_net][j], SE[i][number_net][j], PC[i][number_net][j],
                                          F2[i][number_net][j], SP[i][number_net][j], ACC[i][number_net][j],
                                          mDice[i][number_net][j], mIoU[i][number_net][j], mSE[i][number_net][j], mPC[i][number_net][j],
                                          mF2[i][number_net][j], mSP[i][number_net][j], mACC[i][number_net][j], best_vals[number_net][j]]
                        val_best[i][number_net][j], bestest[i][number_net][j] = Choice_best(val=val[i][number_net][j], val_best=val_best[i][number_net][j],
                                                                              bestest=bestest[i][number_net][j],
                                                                              best_test=best_test[i][number_net][j],
                                                                              Epoch=epoch_nets[number_net])
                if save_checkpoint:
                    if val[0][number_net][0][0] == val_best[0][number_net][0][0]:
                        for Epoch_Net in range(number_net):
                            shutil.copy(str('checkpoints/sub_model/epoch' + '{}'.format(e) +
                                            'checkpoint_net{}.pth'.format(Epoch_Net)),
                                        str('checkpoints/EnsA/ensemble_' + 'checkpoint_net{}.pth'.format(Epoch_Net)))
                        txt_weight[0].write('\n' + 'weight=' + f'{use_pop}')
                # val_best[number_net] = Choice_best(val=val[number_net], val_best=val_best[number_net],
                #                                   Epoch=epoch_nets[number_net])


                global_step[number_net] += traindata_len
                for j in range(number_ensemble):
                    use_pop[j] = gbestpop[j]
                if epoch_nets[number_net] == 0:
                    for i in range(3):
                        for j in range(number_ensemble):
                            Dice_All[i][number_net][0][j] = 0
                            IoU_All[i][number_net][0][j] = 0
                            SE_All[i][number_net][0][j] = 0
                            PC_All[i][number_net][0][j] = 0
                            F2_All[i][number_net][0][j] = 0
                            SP_All[i][number_net][0][j] = 0
                            ACC_All[i][number_net][0][j] = 0
                            mDice_All[i][number_net][0][j] = 0
                            mIoU_All[i][number_net][0][j] = 0
                            mSE_All[i][number_net][0][j] = 0
                            mPC_All[i][number_net][0][j] = 0
                            mF2_All[i][number_net][0][j] = 0
                            mSP_All[i][number_net][0][j] = 0
                            mACC_All[i][number_net][0][j] = 0
                    epoch_All[number_net][0] = 0
                for i in range(3):
                    for j in range(number_ensemble):
                        Dice_All[i][number_net][epoch_nets[number_net] + 1][j] = Dice[i][number_net][j]
                        IoU_All[i][number_net][epoch_nets[number_net] + 1][j] = IoU[i][number_net][j]
                        SE_All[i][number_net][epoch_nets[number_net] + 1][j] = SE[i][number_net][j]
                        PC_All[i][number_net][epoch_nets[number_net] + 1][j] = PC[i][number_net][j]
                        F2_All[i][number_net][epoch_nets[number_net] + 1][j] = F2[i][number_net][j]
                        SP_All[i][number_net][epoch_nets[number_net] + 1][j] = SP[i][number_net][j]
                        ACC_All[i][number_net][epoch_nets[number_net] + 1][j] = ACC[i][number_net][j]
                        mDice_All[i][number_net][epoch_nets[number_net] + 1][j] = mDice[i][number_net][j]
                        mIoU_All[i][number_net][epoch_nets[number_net] + 1][j] = mIoU[i][number_net][j]
                        mSE_All[i][number_net][epoch_nets[number_net] + 1][j] = mSE[i][number_net][j]
                        mPC_All[i][number_net][epoch_nets[number_net] + 1][j] = mPC[i][number_net][j]
                        mF2_All[i][number_net][epoch_nets[number_net] + 1][j] = mF2[i][number_net][j]
                        mSP_All[i][number_net][epoch_nets[number_net] + 1][j] = mSP[i][number_net][j]
                        mACC_All[i][number_net][epoch_nets[number_net] + 1][j] = mACC[i][number_net][j]
                epoch_All[number_net][epoch_nets[number_net] + 1] = (epoch_nets[number_net] + 1)

        elif STAG == str('B'):
            for Epoch_Net in range(number_net):

                for epoch_stage in range(10):
                    epoch_nets[Epoch_Net] += 1
                    assert epoch_nets[Epoch_Net] == (epoch_stage + TIME * 10)
                    nets[Epoch_Net].train()
                    with tqdm(total=n_train,
                              desc=f'{nets_name[Epoch_Net]} Epoch {epoch_nets[Epoch_Net] + 1}/{epochs_nets[Epoch_Net]}',
                              unit='img') as pbar:
                        for batch in train_loader:
                            images = batch['image']
                            true_masks = batch['mask']

                            assert images.shape[1] == nets[Epoch_Net].n_channels, \
                                f'Network has been defined with {nets[Epoch_Net].n_channels} input channels, ' \
                                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                                'the images are loaded correctly.'

                            images = images.to(device=device, dtype=torch.float32)
                            true_masks = true_masks.to(device=device, dtype=torch.long)

                            with torch.cuda.amp.autocast(enabled=amp):

                                # w_ustm[Epoch_Net], w_ust[Epoch_Net], w_m[Epoch_Net], w_us[Epoch_Net], w_t[
                                #     Epoch_Net] = super_combine(midice[Epoch_Net])
                                # masks_pred = w_ustm[Epoch_Net] * deep_out[0] + w_ust[Epoch_Net] * deep_out[1] + w_m[
                                #     Epoch_Net] * \
                                #              deep_out[2] + w_us[Epoch_Net] * deep_out[3] + w_t[Epoch_Net] * deep_out[4]
                                true_masks = F.one_hot(true_masks.squeeze_(1), nets[Epoch_Net].n_classes).permute(0, 3,
                                                                                                                  1,
                                                                                                                  2).float()

                                if ornot[Epoch_Net]:
                                    deep_out = nets[Epoch_Net](images)
                                    w_ustm[Epoch_Net], w_ust[Epoch_Net], w_m[Epoch_Net], w_us[Epoch_Net], w_t[
                                        Epoch_Net] = super_combine(midice[Epoch_Net])

                                    masks_pred = w_ustm[Epoch_Net] * deep_out[0] + w_ust[Epoch_Net] * deep_out[1] + w_m[
                                        Epoch_Net] * \
                                                 deep_out[2] + w_us[Epoch_Net] * deep_out[3] + w_t[Epoch_Net] * \
                                                 deep_out[4]

                                    loss[Epoch_Net] = criterion[Epoch_Net](deep_out[0], true_masks) + \
                                                      dice_loss(F.softmax(deep_out[0], dim=1).float(), true_masks,
                                                                multiclass=True) + \
                                                      criterion[Epoch_Net](deep_out[1], true_masks) + \
                                                      dice_loss(F.softmax(deep_out[1], dim=1).float(), true_masks,
                                                                multiclass=True) + \
                                                      criterion[Epoch_Net](deep_out[2], true_masks) + \
                                                      dice_loss(F.softmax(deep_out[2], dim=1).float(), true_masks,
                                                                multiclass=True) + \
                                                      criterion[Epoch_Net](deep_out[3], true_masks) + \
                                                      dice_loss(F.softmax(deep_out[3], dim=1).float(), true_masks,
                                                                multiclass=True) + \
                                                      criterion[Epoch_Net](deep_out[4], true_masks) + \
                                                      dice_loss(F.softmax(deep_out[4], dim=1).float(), true_masks,
                                                                multiclass=True)
                                else:
                                    deep_out = nets[Epoch_Net](images)
                                    masks_pred = deep_out
                                    loss[Epoch_Net] = criterion[Epoch_Net](deep_out, true_masks) + \
                                                      dice_loss(F.softmax(deep_out, dim=1).float(), true_masks,
                                                                multiclass=True)

                            optimizer[Epoch_Net].zero_grad(set_to_none=True)
                            grad_scaler[Epoch_Net].scale(loss[Epoch_Net]).backward()
                            grad_scaler[Epoch_Net].step(optimizer[Epoch_Net])
                            grad_scaler[Epoch_Net].update()
                            pbar.update(images.shape[0])
                            global_step[Epoch_Net] += 1
                            epoch_loss[Epoch_Net] += loss[Epoch_Net].item()
                            pbar.set_postfix(**{f'net{Epoch_Net + 1} loss (batch)': loss[Epoch_Net].item()})

                            # Evaluation round
                            division_step = (n_train // (1 * batch_size))
                            if division_step > 0:
                                if global_step[Epoch_Net] % division_step == 0:
                                    # for tag, value in nets[Epoch_Net].named_parameters():
                                    #     tag = tag.replace('/', '.')
                                    #     histograms[Epoch_Net][f'{Epoch_Net} Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                    #     # histograms1['Net1 Gradients/' + tag1] = wandb1.Histogram(value1.grad.data.cpu())

                                    val_score[Epoch_Net], sdice1, sdice2, sdice3, sdice4, sdice5, \
                                    Ensemable_popfits_masks[
                                        Epoch_Net], best_vals[Epoch_Net] = evaluate_val(
                                        nets[Epoch_Net], val_loader, device, w_ustm[Epoch_Net], w_ust[Epoch_Net],
                                        w_m[Epoch_Net], w_us[Epoch_Net], w_t[Epoch_Net],
                                        Ensemable_popfits_masks[Epoch_Net], ornot[Epoch_Net])

                                    for i in range(3):
                                        Dice[i][Epoch_Net], IoU[i][Epoch_Net], PC[i][Epoch_Net], SE[i][Epoch_Net], \
                                            SP[i][Epoch_Net], ACC[i][Epoch_Net], F2[i][Epoch_Net], \
                                            mDice[i][Epoch_Net], mIoU[i][Epoch_Net], mPC[i][Epoch_Net], mSE[i][Epoch_Net], \
                                            mSP[i][Epoch_Net], mACC[i][Epoch_Net], mF2[i][Epoch_Net], \
                                            Ensemable_pred_masks[i][epoch_stage][Epoch_Net],\
                                            best_test[i][Epoch_Net] = evaluate_test(nets[Epoch_Net], test_loader[i], device,
                                                                                    w_ustm[Epoch_Net], w_ust[Epoch_Net],
                                                                                    w_m[Epoch_Net], w_us[Epoch_Net],
                                                                                    w_t[Epoch_Net], ornot[Epoch_Net])
                                    midice[Epoch_Net] = [sdice1, sdice2, sdice3, sdice4, sdice5]

                                    # val_score[Epoch_Net], Ensemable_popfits_masks[Epoch_Net], best_vals[
                                    #     Epoch_Net] = evaluate_val(
                                    #     nets[Epoch_Net], val_loader, device, Ensemable_popfits_masks[Epoch_Net])
                                    #
                                    # Dice[Epoch_Net], mDice[Epoch_Net], mIoU[Epoch_Net], PC[Epoch_Net], SE[
                                    #     Epoch_Net], SP[Epoch_Net], ACC[Epoch_Net], F2[Epoch_Net], \
                                    #     Ensemable_pred_masks[epoch_nets[Epoch_Net]][Epoch_Net] = evaluate_test(nets[Epoch_Net],
                                    #                                                                            test_loader,
                                    #                                                                            device)

                                    scheduler[Epoch_Net].step(val_score[Epoch_Net])
                                    logging.info('Validation Dice score: {}'.format(val_score[Epoch_Net]))

                                    for i in range(3):
                                        Dice_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = Dice[i][Epoch_Net]
                                        IoU_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = IoU[i][Epoch_Net]
                                        SE_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = SE[i][Epoch_Net]
                                        PC_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = PC[i][Epoch_Net]
                                        F2_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = F2[i][Epoch_Net]
                                        SP_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = SP[i][Epoch_Net]
                                        ACC_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = ACC[i][Epoch_Net]
                                        mDice_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = mDice[i][Epoch_Net]
                                        mIoU_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = mIoU[i][Epoch_Net]
                                        mSE_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = mSE[i][Epoch_Net]
                                        mPC_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = mPC[i][Epoch_Net]
                                        mF2_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = mF2[i][Epoch_Net]
                                        mSP_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = mSP[i][Epoch_Net]
                                        mACC_All[i][Epoch_Net][epoch_nets[Epoch_Net] + 1] = mACC[i][Epoch_Net]
                                    lr_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = optimizer[Epoch_Net].param_groups[0]['lr']
                                    loss_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = loss[Epoch_Net].item()
                                    epoch_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = (epoch_nets[Epoch_Net] + 1)

                                    for i in range(3):
                                        val[i][Epoch_Net] = [val_score[Epoch_Net], Dice[i][Epoch_Net], IoU[i][Epoch_Net], SE[i][Epoch_Net],
                                                             PC[i][Epoch_Net], F2[i][Epoch_Net], SP[i][Epoch_Net], ACC[i][Epoch_Net],
                                                          mDice[i][Epoch_Net], mIoU[i][Epoch_Net], mSE[i][Epoch_Net], mPC[i][Epoch_Net],
                                                          mF2[i][Epoch_Net], mSP[i][Epoch_Net], mACC[i][Epoch_Net], best_vals[Epoch_Net]]
                                        val_best[i][Epoch_Net], bestest[i][Epoch_Net] = Choice_best(val=val[i][Epoch_Net],
                                                                                          val_best=val_best[i][Epoch_Net],
                                                                                          bestest=bestest[i][Epoch_Net],
                                                                                          best_test=best_test[i][Epoch_Net],
                                                                                          Epoch=epoch_nets[Epoch_Net])

                                    # print('\n' + 'out_USTM.dice=', midice[Epoch_Net][0].item())
                                    # print('out_UST.dice=', midice[Epoch_Net][1].item())
                                    # print('out_MB.dice=', midice[Epoch_Net][2].item())
                                    # print('out_USB.dice=', midice[Epoch_Net][3].item())
                                    # print('out_TB.dice=', midice[Epoch_Net][4].item())
                                    print('w_ustm=', w_ustm[Epoch_Net], 'w_ust=', w_ust[Epoch_Net], 'w_m=',
                                          w_m[Epoch_Net],
                                          'w_us=', w_us[Epoch_Net], 'w_t=', w_t[Epoch_Net])

                                    # txt_dice[Epoch_Net].write(
                                    #     f'{epoch_nets[Epoch_Net]}' + '\n' +
                                    #     ' out_USTM.dice=' + f'{midice[Epoch_Net][1].item()}' +
                                    #     ', out_UST.dice=' + f'{midice[Epoch_Net][1].item()}' +
                                    #     ', out_MB.dice=' + f'{midice[Epoch_Net][2].item()}' +
                                    #     ', out_USB.dice=' + f'{midice[Epoch_Net][3].item()}' +
                                    #     ', out_TB.dice=' + f'{midice[Epoch_Net][4].item()}')
                                    txt_w[Epoch_Net].write(
                                        f'{epoch_nets[Epoch_Net]}' + '\n' +
                                        ' w_ustm=' + f'{w_ustm[Epoch_Net]}' +
                                        ' w_ust=' + f'{w_ust[Epoch_Net]}' +
                                        ' w_m=' + f'{w_m[Epoch_Net]}' +
                                        ' w_us=' + f'{w_us[Epoch_Net]}' +
                                        ' w_t=' + f'{w_t[Epoch_Net]}')

                    if save_checkpoint:
                        torch.save(nets[Epoch_Net].state_dict(),
                                   str('checkpoints/sub_model/epoch' + '{}'.format(epoch_stage) + 'checkpoint_net{}.pth'.format(
                                       Epoch_Net)))
                        if val[0][Epoch_Net][0] == val_best[0][Epoch_Net][0]:
                            Path(dir_checkpoints[Epoch_Net]).mkdir(parents=True, exist_ok=True)
                            torch.save(nets[Epoch_Net].state_dict(), str(
                                dir_checkpoints[Epoch_Net] / 'best_checkpoint.pth'))
                            logging.info(f'Net Checkpoint {epoch_nets[Epoch_Net] + 1} saved!')
                        elif epoch_nets[Epoch_Net] == (set_epochs - 1):
                            Path(dir_checkpoints[Epoch_Net]).mkdir(parents=True, exist_ok=True)
                            torch.save(nets[Epoch_Net].state_dict(), str(
                                dir_checkpoints[Epoch_Net] / 'last_checkpoint.pth'))
                            logging.info(f'Net Checkpoint {epoch_nets[Epoch_Net] + 1} saved!')
                        else:
                            pass



            # 5.3.2a  training ensemble net in stage3
            print('-----------------------------------------------')
            print(f'stage{TIME}_A PSO Train')
            print('-----------------------------------------------')

            pso_lr = (1.0, 1.0)

            pso_epoch = int(5)
            pso_sumpop = int(100)

            # pso_epoch = int(2)
            # pso_sumpop = int(18)
            # pso_tt1 = int(2)
            # pso_tt2 = int(1)
            if epoch_nets[0] > 41:
                pso_resultF1, pso_resultS1, gbestpop[0] = SDBHPSO_Ensemble(pso_epoch=pso_epoch, pso_sumpop=pso_sumpop,
                                                                            pso_lr=(2.0, 2.0), init=True,
                                                                            Ensemable_popfits_masks=Ensemable_popfits_masks,
                                                                            val_loader=val_loader,
                                                                            nets_classes=nets_classes,
                                                                            number_net=number_net, label=label,
                                                                            label_threshold=label_threshold,
                                                                            elimate=False)

                pso_resultF2, pso_resultS2 = pso_resultF1, pso_resultS1

                pso_resultF3, pso_resultS3 = pso_resultF1, pso_resultS1
            else:
                pso_resultF1, pso_resultS1, gbestpop[0] = SDBHPSO_Ensemble(pso_epoch=pso_epoch, pso_sumpop=pso_sumpop,
                                                                            pso_lr=(2.0, 2.0), init=True,
                                                                            Ensemable_popfits_masks=Ensemable_popfits_masks,
                                                                            val_loader=val_loader,
                                                                            nets_classes=nets_classes,
                                                                            number_net=number_net, label=label,
                                                                            label_threshold=label_threshold,
                                                                            elimate=False)
                pso_resultF2 = pso_resultF1
                pso_resultS2 = pso_resultS1
                pso_resultF3 = pso_resultF1
                pso_resultS3 = pso_resultS1


            txt_ensm[0].write(
                '\n' + '\n' + '\n' + '------------' + '\n' +
                'now is the 1 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS1}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF1}' + '\n' +
                'now is the 2 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS2}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF2}' + '\n' +
                'now is the 3 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS3}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF3}' + '\n' +
                ', label=' + f'{label}' +
                '\n' + '------------' + '\n' + '\n' + '\n')
            txt[number_net + 2].write(
                '\n' + '------------' + '\n' +
                'part A' + f'TIME={TIME}' + '\n' +
                'now is the 1 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS1}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF1}' + '\n' +
                'now is the 2 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS2}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF2}' + '\n' +
                'now is the 3 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS3}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF3}' + '\n' +
                ', label=' + f'{label}' +
                '\n' + '------------' + '\n')
            for i in range(len(pso_resultF1)):
                pso_result_Fall1.insert(time_pso, pso_resultF1[i])
                pso_result_Sall1.insert(time_pso, pso_resultS1[i])

                pso_result_Fall2.insert(time_pso, pso_resultF2[i])
                pso_result_Sall2.insert(time_pso, pso_resultS2[i])

                pso_result_Fall3.insert(time_pso, pso_resultF3[i])
                pso_result_Sall3.insert(time_pso, pso_resultS3[i])
                time_pso += 1
            print('label=', label)
            print('pso_resultF1=', pso_resultF1)
            print('pso_resultS1=', pso_resultS1)
            print('pso_resultF2=', pso_resultF2)
            print('pso_resultS2=', pso_resultS2)
            print('pso_resultF3=', pso_resultF3)
            print('pso_resultS3=', pso_resultS3)
            print('PSO one over')
            plt.plot(pso_resultF1)
            plt.plot(pso_resultF2)
            plt.plot(pso_resultF3)
            plt.savefig(f'text_result/{TIME}_A.png')


            print('-----------------------------------------------')
            print(f'stage{TIME}_A Ensemable Start')
            print('-----------------------------------------------')

            traindata_len = len(train_loader)
            if fiveth:
                epoch_nets[number_net + 1] = epoch_nets[number_net]
                jiedian = (epoch_nets[number_net] + 1)
                fiveth = False
            global_step[number_net + 1] = global_step[number_net]
            for e in range(10):
                epoch_nets[number_net] += 1
                for j in range(number_ensemble):
                    best_vals[number_net][j] = evaluate_ensembleA_valpre(
                        dataloader_val=val_loader,
                        preloader_val=Ensemable_popfits_masks,
                        time=e,
                        number_net=number_net,
                        bestpop=use_pop[j],
                        net_n_classes=nets_classes,
                        device='cpu')

                for i in range(3):
                    mask_pre = Ensemable_pred_masks[i][e]
                    for j in range(number_ensemble):
                        Dice[i][number_net][j], IoU[i][number_net][j], PC[i][number_net][j], SE[i][number_net][j], \
                            SP[i][number_net][j], ACC[i][number_net][j], F2[i][number_net][j], \
                            mDice[i][number_net][j], mIoU[i][number_net][j], mPC[i][number_net][j], mSE[i][number_net][j], \
                            mSP[i][number_net][j], mACC[i][number_net][j], mF2[i][number_net][j], \
                            best_test[i][number_net][j] = evaluate_ensembleA_test(preloader_test=mask_pre,
                                                                              dataloader_test=test_loader[i],
                                                                              number_net=number_net, bestpop=use_pop[j],
                                                                              net_n_classes=nets_classes, device='cpu')

                for j in range(j):
                    val_score[number_net][j] = evaluate_ensemble_val(best_vals[number_net][j], val_loader, nets_classes, device)
                    for i in range(3):
                        val[i][number_net][j] = [val_score[number_net][j], Dice[i][number_net][j], IoU[i][number_net][j], SE[i][number_net][j], PC[i][number_net][j],
                                          F2[i][number_net][j], SP[i][number_net][j], ACC[i][number_net][j],
                                          mDice[i][number_net][j], mIoU[i][number_net][j], mSE[i][number_net][j], mPC[i][number_net][j],
                                          mF2[i][number_net][j], mSP[i][number_net][j], mACC[i][number_net][j], best_vals[number_net][j]]
                        val_best[i][number_net][j], bestest[i][number_net][j] = Choice_best(val=val[i][number_net][j], val_best=val_best[i][number_net][j],
                                                                              bestest=bestest[i][number_net][j],
                                                                              best_test=best_test[i][number_net][j],
                                                                              Epoch=epoch_nets[number_net])

                if save_checkpoint:
                    if val[0][number_net][0][0] == val_best[0][number_net][0][0]:
                        for Epoch_Net in range(number_net):
                            shutil.copy(str('checkpoints/sub_model/epoch' + '{}'.format(e) +
                                            'checkpoint_net{}.pth'.format(Epoch_Net)),
                                        str('checkpoints/EnsA/ensemble_' + 'checkpoint_net{}.pth'.format(Epoch_Net)))
                        txt_weight[0].write('\n' + 'weight=' + f'{use_pop}')

                global_step[number_net] += traindata_len

                for i in range(3):
                    for j in range(number_ensemble):
                        Dice_All[i][number_net][epoch_nets[number_net] + 1][j] = Dice[i][number_net][j]
                        IoU_All[i][number_net][epoch_nets[number_net] + 1][j] = IoU[i][number_net][j]
                        SE_All[i][number_net][epoch_nets[number_net] + 1][j] = SE[i][number_net][j]
                        PC_All[i][number_net][epoch_nets[number_net] + 1][j] = PC[i][number_net][j]
                        F2_All[i][number_net][epoch_nets[number_net] + 1][j] = F2[i][number_net][j]
                        SP_All[i][number_net][epoch_nets[number_net] + 1][j] = SP[i][number_net][j]
                        ACC_All[i][number_net][epoch_nets[number_net] + 1][j] = ACC[i][number_net][j]
                        mDice_All[i][number_net][epoch_nets[number_net] + 1][j] = mDice[i][number_net][j]
                        mIoU_All[i][number_net][epoch_nets[number_net] + 1][j] = mIoU[i][number_net][j]
                        mSE_All[i][number_net][epoch_nets[number_net] + 1][j] = mSE[i][number_net][j]
                        mPC_All[i][number_net][epoch_nets[number_net] + 1][j] = mPC[i][number_net][j]
                        mF2_All[i][number_net][epoch_nets[number_net] + 1][j] = mF2[i][number_net][j]
                        mSP_All[i][number_net][epoch_nets[number_net] + 1][j] = mSP[i][number_net][j]
                        mACC_All[i][number_net][epoch_nets[number_net] + 1][j] = mACC[i][number_net][j]
                epoch_All[number_net][epoch_nets[number_net] + 1] = (epoch_nets[number_net] + 1)


            # 5.3.2b  training ensemble net in stage3
            print('-----------------------------------------------')
            print(f'stage{TIME}_B PSO Train')
            print('-----------------------------------------------')
            pso_lr_bl = (1.0, 1.0)

            pso_epoch_bl = int(5)
            pso_sumpop_bl = int(100)

            # pso_epoch_bl = int(2)
            # pso_sumpop_bl = int(18)
            # pso_tt1_bl = int(2)
            # pso_tt2_bl = int(1)

            if second:
                for i in range(number_net):
                    if label[i] >= label_threshold:
                        number_label += 1
                number_labelnet = number_net - number_label
                tem_tem = int(0)
                if number_labelnet == 6:
                    use_pop_labela = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
                elif number_labelnet == 5:
                    use_pop_labela = np.array([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5])
                elif number_labelnet == 4:
                    use_pop_labela = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])
                elif number_labelnet == 3:
                    use_pop_labela = np.array([1 / 3, 1 / 3, 1 / 3])
                elif number_labelnet == 2:
                    use_pop_labela = np.array([1 / 2, 1 / 2])
                elif number_labelnet == 1:
                    use_pop_labela = np.array([1.0])

                use_pop_label = []
                for i in range(number_ensemble):
                    use_pop_label.insert(i, copy.deepcopy(use_pop_labela))

                second = False

            if epoch_nets[0] > 41:
                pso_resultF1, pso_resultS1, gbestpop_label[0] = SDBHPSO_Ensemble(pso_epoch=pso_epoch,
                                                                                  pso_sumpop=pso_sumpop,
                                                                                  pso_lr=(2.0, 2.0), init=True,
                                                                                  Ensemable_popfits_masks=Ensemable_popfits_masks,
                                                                                  val_loader=val_loader,
                                                                                  nets_classes=nets_classes,
                                                                                  number_net=number_labelnet,
                                                                                  label=label,
                                                                                  label_threshold=label_threshold,
                                                                                  elimate=True)

                pso_resultF2, pso_resultS2 = pso_resultF1, pso_resultS1

                pso_resultF3, pso_resultS3 = pso_resultF1, pso_resultS1
            else:
                pso_resultF1, pso_resultS1, gbestpop_label[0] = SDBHPSO_Ensemble(pso_epoch=pso_epoch, pso_sumpop=pso_sumpop,
                                                                            pso_lr=(2.0, 2.0), init=True,
                                                                            Ensemable_popfits_masks=Ensemable_popfits_masks,
                                                                            val_loader=val_loader,
                                                                            nets_classes=nets_classes,
                                                                            number_net=number_labelnet, label=label,
                                                                            label_threshold=label_threshold, elimate=True)

                pso_resultF2 = pso_resultF1
                pso_resultS2 = pso_resultS1
                pso_resultF3 = pso_resultF1
                pso_resultS3 = pso_resultS1


            gbestpop_label = gbestpop_label
            if third:
                time_psoB = time_pso_tem
                for i in range(time_psoB):
                    pso_result_FallB1.insert(i, pso_result_Fall1[i])
                    pso_result_FallB2.insert(i, pso_result_Fall2[i])
                    pso_result_FallB3.insert(i, pso_result_Fall3[i])

                    pso_result_SallB1.insert(i, pso_result_Sall1[i])
                    pso_result_SallB2.insert(i, pso_result_Sall2[i])
                    pso_result_SallB3.insert(i, pso_result_Sall3[i])
                third = False
            for i in range(len(pso_resultF1)):
                pso_result_FallB1.insert(time_pso, pso_resultF1[i])
                pso_result_SallB1.insert(time_pso, pso_resultS1[i])

                pso_result_FallB2.insert(time_pso, pso_resultF2[i])
                pso_result_SallB2.insert(time_pso, pso_resultS2[i])

                pso_result_FallB3.insert(time_pso, pso_resultF3[i])
                pso_result_SallB3.insert(time_pso, pso_resultS3[i])
                time_psoB += 1

            txt_ensm[1].write(
                '\n' + '------------' + '\n' +
                'now is the 1 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS1}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF1}' + '\n' +
                'now is the 2 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS2}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF2}' + '\n' +
                'now is the 3 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS3}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF3}' + '\n' +
                '\n' + '------------')
            txt[number_net + 2].write(
                '\n' + '------------' + '\n' +
                'part B EnsB' + f'TIME={TIME}' + '\n' +
                'now is the 1 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS1}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF1}' + '\n' +
                'now is the 2 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS2}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF2}' + '\n' +
                'now is the 3 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS3}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF3}' + '\n' +
                ', label=' + f'{label}' +
                '\n' + '------------' + '\n')

            print('label=', label)
            print('pso_resultF1=', pso_resultF1)
            print('pso_resultS1=', pso_resultS1)
            print('pso_resultF2=', pso_resultF2)
            print('pso_resultS2=', pso_resultS2)
            print('pso_resultF3=', pso_resultF3)
            print('pso_resultS3=', pso_resultS3)
            print('PSO one over')
            plt.plot(pso_resultF1)
            plt.plot(pso_resultF2)
            plt.plot(pso_resultF3)
            plt.savefig(f'text_result/{TIME}_B.png')

            print('-----------------------------------------------')
            print(f'stage{TIME}_B Ensemable Start')
            print('-----------------------------------------------')

            traindata_len = len(train_loader)
            for e in range(10):
                epoch_nets[number_net + 1] += 1
                for j in range(number_ensemble):
                    best_vals[number_net + 1][j] = evaluate_ensembleB_valpre(
                        dataloader_val=val_loader,
                        preloader_val=Ensemable_popfits_masks,
                        time=e,
                        number_net=number_net,
                        label=label,
                        threshold=label_threshold,
                        bestpop=use_pop_label[j],
                        net_n_classes=nets_classes,
                        device='cpu')

                for i in range(3):
                    mask_pre = Ensemable_pred_masks[i][e]
                    for j in range(number_ensemble):
                        Dice[i][number_net + 1][j], IoU[i][number_net + 1][j], PC[i][number_net + 1][j], SE[i][number_net + 1][j], \
                            SP[i][number_net + 1][j], ACC[i][number_net + 1][j], F2[i][number_net + 1][j], \
                            mDice[i][number_net + 1][j], mIoU[i][number_net + 1][j], mPC[i][number_net + 1][j], mSE[i][number_net + 1][j], \
                            mSP[i][number_net + 1][j], mACC[i][number_net + 1][j], mF2[i][number_net + 1][j], \
                            best_test[i][number_net + 1][j] = evaluate_ensembleB_test(preloader_test=mask_pre,
                                                                              dataloader_test=test_loader[i],
                                                                              number_net=number_net, label=label,
                                                                              threshold=label_threshold, bestpop=use_pop_label[j],
                                                                              net_n_classes=nets_classes, device='cpu')

                for j in range(number_ensemble):
                    val_score[number_net + 1][j] = evaluate_ensemble_val(best_vals[number_net + 1][j], val_loader, nets_classes, device)
                    for i in range(3):
                        val[i][number_net + 1][j] = [val_score[number_net + 1][j], Dice[i][number_net + 1][j], IoU[i][number_net + 1][j], SE[i][number_net + 1][j], PC[i][number_net + 1][j],
                                          F2[i][number_net + 1][j], SP[i][number_net + 1][j], ACC[i][number_net + 1][j],
                                          mDice[i][number_net + 1][j], mIoU[i][number_net + 1][j], mSE[i][number_net + 1][j], mPC[i][number_net + 1][j],
                                          mF2[i][number_net + 1][j], mSP[i][number_net + 1][j], mACC[i][number_net + 1][j], best_vals[number_net + 1][j]]
                        val_best[i][number_net + 1][j], bestest[i][number_net + 1][j] = Choice_best(val=val[i][number_net + 1][j], val_best=val_best[i][number_net + 1][j],
                                                                              bestest=bestest[i][number_net + 1][j],
                                                                              best_test=best_test[i][number_net + 1][j],
                                                                              Epoch=epoch_nets[number_net + 1])

                if save_checkpoint:
                    if val[0][number_net + 1][0][0] == val_best[0][number_net + 1][0][0]:
                        label_time = int(0)
                        for Epoch_Net in range(number_net):
                            if label[Epoch_Net] >= label_threshold:
                                label_time += 1
                            else:
                                shutil.copy(str('checkpoints/sub_model/epoch' + '{}'.format(e) +
                                                'checkpoint_net{}.pth'.format(Epoch_Net)),
                                            str('checkpoints/EnsB/ensemble_' + 'checkpoint_net{}.pth'.format(Epoch_Net)))
                        txt_weight[1].write('\n' + 'weight=' + f'{use_pop_label}' + '\n' + 'label=' + f'{label}')

                global_step[number_net + 1] += traindata_len
                use_pop_label = gbestpop_label

                if forth:
                    for ii in range((epoch_nets[number_net + 1] + 1)):
                        for i in range(3):
                            for j in range(number_ensemble):
                                Dice_All[i][number_net + 1][ii][j] = Dice_All[i][number_net][ii][j]
                                IoU_All[i][number_net + 1][ii][j] = IoU_All[i][number_net][ii][j]
                                SE_All[i][number_net + 1][ii][j] = SE_All[i][number_net][ii][j]
                                PC_All[i][number_net + 1][ii][j] = PC_All[i][number_net][ii][j]
                                F2_All[i][number_net + 1][ii][j] = F2_All[i][number_net][ii][j]
                                SP_All[i][number_net + 1][ii][j] = SP_All[i][number_net][ii][j]
                                ACC_All[i][number_net + 1][ii][j] = ACC_All[i][number_net][ii][j]
                                mDice_All[i][number_net + 1][ii][j] = mDice_All[i][number_net][ii][j]
                                mIoU_All[i][number_net + 1][ii][j] = mIoU_All[i][number_net][ii][j]
                                mSE_All[i][number_net + 1][ii][j] = mSE_All[i][number_net][ii][j]
                                mPC_All[i][number_net + 1][ii][j] = mPC_All[i][number_net][ii][j]
                                mF2_All[i][number_net + 1][ii][j] = mF2_All[i][number_net][ii][j]
                                mSP_All[i][number_net + 1][ii] [j]= mSP_All[i][number_net][ii][j]
                                mACC_All[i][number_net + 1][ii][j] = mACC_All[i][number_net][ii][j]
                        epoch_All[number_net + 1][ii] = epoch_All[number_net][ii]
                    forth = False

                for i in range(3):
                    for j in range(number_ensemble):
                        Dice_All[i][number_net + 1][epoch_nets[number_net + 1] + 1][j] = Dice[i][number_net + 1][j]
                        IoU_All[i][number_net + 1][epoch_nets[number_net + 1] + 1][j] = IoU[i][number_net + 1][j]
                        SE_All[i][number_net + 1][epoch_nets[number_net + 1] + 1][j] = SE[i][number_net + 1][j]
                        PC_All[i][number_net + 1][epoch_nets[number_net + 1] + 1][j] = PC[i][number_net + 1][j]
                        F2_All[i][number_net + 1][epoch_nets[number_net + 1] + 1][j] = F2[i][number_net + 1][j]
                        SP_All[i][number_net + 1][epoch_nets[number_net + 1] + 1][j] = SP[i][number_net + 1][j]
                        ACC_All[i][number_net + 1][epoch_nets[number_net + 1] + 1][j] = ACC[i][number_net + 1][j]
                        mDice_All[i][number_net + 1][epoch_nets[number_net + 1] + 1][j] = mDice[i][number_net + 1][j]
                        mIoU_All[i][number_net + 1][epoch_nets[number_net + 1] + 1][j] = mIoU[i][number_net + 1][j]
                        mSE_All[i][number_net + 1][epoch_nets[number_net + 1] + 1][j] = mSE[i][number_net + 1][j]
                        mPC_All[i][number_net + 1][epoch_nets[number_net + 1] + 1][j] = mPC[i][number_net + 1][j]
                        mF2_All[i][number_net + 1][epoch_nets[number_net + 1] + 1][j] = mF2[i][number_net + 1][j]
                        mSP_All[i][number_net + 1][epoch_nets[number_net + 1] + 1][j] = mSP[i][number_net + 1][j]
                        mACC_All[i][number_net + 1][epoch_nets[number_net + 1] + 1][j] = mACC[i][number_net + 1][j]
                epoch_All[number_net + 1][epoch_nets[number_net + 1] + 1] = (epoch_nets[number_net + 1] + 1)

    for Epoch_Net in range(number_net):
        txt_dice[Epoch_Net].close()
        txt_w[Epoch_Net].close()

    txt_ensm[0].close()
    txt_ensm[1].close()

    for Epoch_Net in range(number_net):
        print('-----------------------------------------------')

        print(f'in net{Epoch_Net}:')

        for i in range(3):
            print('now is test', i,
                  'best Dice =', val_best[i][Epoch_Net][1],
                  'best IoU =', val_best[i][Epoch_Net][2],
                  'best Sensitivity =', val_best[i][Epoch_Net][3],
                  'best Precision =', val_best[i][Epoch_Net][4],
                  'best F2 =', val_best[i][Epoch_Net][5],
                  'best Specificity =', val_best[i][Epoch_Net][6],
                  'best Accuracy =', val_best[i][Epoch_Net][7],
                  'best mDice =', val_best[i][Epoch_Net][8],
                  'best mIoU =', val_best[i][Epoch_Net][9],
                  'best mean Sensitivity =', val_best[i][Epoch_Net][10],
                  'best mean Precision =', val_best[i][Epoch_Net][11],
                  'best mean F2 =', val_best[i][Epoch_Net][12],
                  'best mean Specificity =', val_best[i][Epoch_Net][13],
                  'best mean Accuracy =', val_best[i][Epoch_Net][14],
                  'best net of Dice =', val_best[i][Epoch_Net][16]
                  )
        for i in range(3):
            txt_last[0].write(
                '\n' + f'{Epoch_Net}' + '\n' +
                '\n' + f'now in test {i}' + '\n' +
                ' Dice=' + f'{val_best[i][Epoch_Net][1]}' + '\n' +
                ', IoU=' + f'{val_best[i][Epoch_Net][2]}' + '\n' +
                ', Sensitivity=' + f'{val_best[i][Epoch_Net][3]}' + '\n' +
                ', Precision=' + f'{val_best[i][Epoch_Net][4]}' + '\n' +
                ', F2=' + f'{val_best[i][Epoch_Net][5]}' + '\n' +
                ', Specificity=' + f'{val_best[i][Epoch_Net][6]}' + '\n' +
                ', Accuracy=' + f'{val_best[i][Epoch_Net][7]}' + '\n' +
                ', mDice=' + f'{val_best[i][Epoch_Net][8]}' + '\n' +
                ', mIoU=' + f'{val_best[i][Epoch_Net][9]}' + '\n' +
                ', mean Sensitivity=' + f'{val_best[i][Epoch_Net][10]}' + '\n' +
                ', mean Precision=' + f'{val_best[i][Epoch_Net][11]}' + '\n' +
                ', mean F2=' + f'{val_best[i][Epoch_Net][12]}' + '\n' +
                ', mean Specificity=' + f'{val_best[i][Epoch_Net][13]}' + '\n' +
                ', mean Accuracy=' + f'{val_best[i][Epoch_Net][14]}' + '\n' +
                ', best net of Dice=' + f'{val_best[i][Epoch_Net][16]}' + '\n')
    txt_last[0].close()

    print('-----------------------------------------------')
    print(f'in Ensemble Net:')
    for i in range(3):
        print('now is test', i,
              'best Dice =', val_best[i][number_net][0][1],
              'best IoU =', val_best[i][number_net][0][2],
              'best Sensitivity =', val_best[i][number_net][0][3],
              'best Precision =', val_best[i][number_net][0][4],
              'best F2 =', val_best[i][number_net][0][5],
              'best Specificity =', val_best[i][number_net][0][6],
              'best Accuracy =', val_best[i][number_net][0][7],
              'best mDice =', val_best[i][number_net][0][8],
              'best mIoU =', val_best[i][number_net][0][9],
              'best mean Sensitivity =', val_best[i][number_net][0][10],
              'best mean Precision =', val_best[i][number_net][0][11],
              'best mean F2 =', val_best[i][number_net][0][12],
              'best mean Specificity =', val_best[i][number_net][0][13],
              'best mean Accuracy =', val_best[i][number_net][0][14],
              'best net of Dice =', val_best[i][number_net][0][16]
              )
    for i in range(3):
        txt_last[1].write(
            '\n' + 'in Ensemble Net' + '\n' +
            '\n' + f'now in test {i}' + '\n' +
            ' Dice=' + f'{val_best[i][number_net][0][1]}' + '\n' +
            ', IoU=' + f'{val_best[i][number_net][0][2]}' + '\n' +
            ', Sen=' + f'{val_best[i][number_net][0][3]}' + '\n' +
            ', Pre=' + f'{val_best[i][number_net][0][4]}' + '\n' +
            ', F2_=' + f'{val_best[i][number_net][0][5]}' + '\n' +
            ', Spe=' + f'{val_best[i][number_net][0][6]}' + '\n' +
            ', Acc=' + f'{val_best[i][number_net][0][7]}' + '\n' +
            ', mDic' + f'{val_best[i][number_net][0][8]}' + '\n' +
            ', mIoU' + f'{val_best[i][number_net][0][9]}' + '\n' +
            ', mSen' + f'{val_best[i][number_net][0][10]}' + '\n' +
            ', mPre' + f'{val_best[i][number_net][0][11]}' + '\n' +
            ', mF2=' + f'{val_best[i][number_net][0][12]}' + '\n' +
            ', mSpe' + f'{val_best[i][number_net][0][13]}' + '\n' +
            ', mAcc' + f'{val_best[i][number_net][0][14]}' + '\n' +
            ', best net of Dice=' + f'{val_best[i][number_net][0][16]}' + '\n')
    txt_last[1].close()
    print('-----------------------------------------------')

    print('-----------------------------------------------')
    print(f'in ano Ensemble Net:')
    for i in range(3):
        print('now is test', i,
              'best Dice =', val_best[i][number_net + 1][0][1],
              'best IoU =', val_best[i][number_net + 1][0][2],
              'best Sensitivity =', val_best[i][number_net + 1][0][3],
              'best Precision =', val_best[i][number_net + 1][0][4],
              'best F2 =', val_best[i][number_net + 1][0][5],
              'best Specificity =', val_best[i][number_net + 1][0][6],
              'best Accuracy =', val_best[i][number_net + 1][0][7],
              'best mDice =', val_best[i][number_net + 1][0][8],
              'best mIoU =', val_best[i][number_net + 1][0][9],
              'best mean Sensitivity =', val_best[i][number_net + 1][0][10],
              'best mean Precision =', val_best[i][number_net + 1][0][11],
              'best mean F2 =', val_best[i][number_net + 1][0][12],
              'best mean Specificity =', val_best[i][number_net + 1][0][13],
              'best mean Accuracy =', val_best[i][number_net + 1][0][14],
              'best net of Dice =', val_best[i][number_net + 1][0][16]
              )
    for i in range(3):
        txt_last[2].write(
            '\n' + 'in ano Ensemble Net' + '\n' +
            '\n' + f'now in test {i}' + '\n' +
            ' Dice=' + f'{val_best[i][number_net + 1][0][1]}' + '\n' +
            ', IoU=' + f'{val_best[i][number_net + 1][0][2]}' + '\n' +
            ', Sen=' + f'{val_best[i][number_net + 1][0][3]}' + '\n' +
            ', Pre=' + f'{val_best[i][number_net + 1][0][4]}' + '\n' +
            ', F2_=' + f'{val_best[i][number_net + 1][0][5]}' + '\n' +
            ', Spe=' + f'{val_best[i][number_net + 1][0][6]}' + '\n' +
            ', Acc=' + f'{val_best[i][number_net + 1][0][7]}' + '\n' +
            ', mDic' + f'{val_best[i][number_net + 1][0][8]}' + '\n' +
            ', mIoU' + f'{val_best[i][number_net + 1][0][9]}' + '\n' +
            ', mSen' + f'{val_best[i][number_net + 1][0][10]}' + '\n' +
            ', mPre' + f'{val_best[i][number_net + 1][0][11]}' + '\n' +
            ', mF2=' + f'{val_best[i][number_net + 1][0][12]}' + '\n' +
            ', mSpe' + f'{val_best[i][number_net + 1][0][13]}' + '\n' +
            ', mAcc' + f'{val_best[i][number_net + 1][0][14]}' + '\n' +
            ', best net of Dice=' + f'{val_best[i][number_net + 1][0][16]}' + '\n')
    txt_last[2].close()
    print('-----------------------------------------------')



    print('--------------------Peak Ensemble--------------------')
    # # 5.3.2  training ensemble net in stage3
    # print('-----------------------------------------------')
    # print('stage Peak PSO Train STAR!!!')
    # print('-----------------------------------------------')
    #
    # pso_lr_pl = (1.0, 1.0)
    #
    # pso_epoch_pl = int(20)
    # pso_sumpop_pl = int(100)
    # pso_tt1_pl = int(18)
    # pso_tt2_pl = int(2)
    #
    # # pso_epoch_pl = int(3)
    # # pso_sumpop_pl = int(48)
    # # pso_tt1_pl = int(3)
    # # pso_tt2_pl = int(1)
    #
    # pso_tt_site_pl = [0] * (pso_tt1_pl + pso_tt2_pl)
    # pop_fitv_pl = [1.0] * pso_sumpop_pl
    #
    # pop_site_pl = torch.normal(mean=0.1666, std=0.5, size=(pso_sumpop_pl, (number_net + 1)))
    # pop_site_sum_pl = torch.sum(pop_site_pl, dim=1)
    # pop_site_another_pl = torch.ones((number_net + 1))
    # pop_site_other_pl, _ = torch.meshgrid(pop_site_sum_pl, pop_site_another_pl)
    # pop_site_pl = pop_site_pl / pop_site_other_pl
    # pop_site_pl = pop_site_pl.numpy()
    #
    # pop_site_pl[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # pop_site_pl[1] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # pop_site_pl[2] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    # pop_site_pl[3] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    # pop_site_pl[4] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    # pop_site_pl[5] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # pop_site_pl[6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # pop_site_pl[7] = [0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
    # pop_site_pl[8] = [0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.5]
    # pop_site_pl[9] = [0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.5]
    # pop_site_pl[10] = [0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.5]
    # pop_site_pl[11] = [0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.5]
    # pop_site_pl[12] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.5]
    #
    # # pop_site_pl[0] = [1.0, 0.0, 0.0, 0.0]
    # # pop_site_pl[1] = [0.0, 1.0, 0.0, 0.0]
    # # pop_site_pl[2] = [0.0, 0.0, 1.0, 0.0]
    # # pop_site_pl[3] = [0.0, 0.0, 0.0, 1.0]
    #
    # # pop_site_pl[0] = [1.0, 0.0, 0.0]
    # # pop_site_pl[1] = [0.0, 1.0, 0.0]
    # # pop_site_pl[2] = [0.0, 0.0, 1.0]
    # # pop_site_pl[3] = [0.0, 0.5, 0.5]
    # # pop_site_pl[4] = [0.5, 0.0, 0.5]
    # # pop_site_pl[5] = [0.5, 0.5, 0.0]
    #
    # pop_v_pl = np.zeros((pso_sumpop_pl, (number_net + 1)))
    #
    # gbestpop_pl = np.zeros(number_net + 1)
    # pbestpop_pl = np.zeros((pso_sumpop_pl, (number_net + 1)))
    # gbestfitness_pl = float(0)
    # pbestfitness_pl = np.zeros(pso_sumpop_pl)
    # pso_result_pl = np.zeros(pso_epoch_pl)
    # pso_t_pl = 0.75
    # pop_fitness_pl = [float(0)] * pso_sumpop_pl
    #
    # for pso_i_pl in range(pso_epoch_pl):
    #     # 速度更新
    #     re_sit_pl = np.random.dirichlet(np.ones(number_net + 1), size=(pso_tt1_pl + pso_tt2_pl))
    #     # for fig in range(pso_sumpop):
    #     #     pop_score_tensor, _, _, _, _, _, _ = evaluate_pop(net=nets, dataloader=train_loader, device=device, number=number_net + 1,
    #     #                                  onlyone_pop=pop_site[fig], mask_pred_size=masks_pred_size, fig_now=pso_i, pop_now=pso_j, batch_size=batch_size)
    #     #     pop_score_numpy = pop_score_tensor.data.cpu().numpy()
    #     #     pop_fitness.insert(fig, pop_score_numpy)
    #     if not pso_i_pl == 0:
    #         for pso_j_pl in range(pso_sumpop_pl):
    #             pop_v_pl[pso_j_pl] += pso_lr_pl[0] * np.random.rand() * (
    #                     pbestpop_pl[pso_j_pl] - pop_site_pl[pso_j_pl]) + pso_lr_pl[
    #                                       1] * np.random.rand() * (gbestpop_pl - pop_site_pl[pso_j_pl])
    #         # pop_v[pop_v < pso_rangespeed[0]] = pso_rangespeed[0]
    #         # pop_v[pop_v > pso_rangespeed[1]] = pso_rangespeed[1]
    #
    #         # 粒子位置更新
    #         max_fitness_pl = max(pop_fitness_pl)
    #         max_fit_site_pl = pop_fitness_pl.index(max_fitness_pl)
    #         for sit_pl in range(pso_sumpop_pl):
    #             if sit_pl != max_fit_site_pl:
    #                 pop_fitv_pl[sit_pl] = np.linalg.norm(pop_site_pl[max_fit_site_pl] - pop_site_pl[sit_pl])
    #             else:
    #                 pop_fitv_pl[sit_pl] = 1.0
    #         for tt_2_pl in range(pso_tt2_pl):
    #             min_fitv_pl = min(pop_fitv_pl)
    #             pso_tt_site_pl[tt_2_pl] = pop_fitv_pl.index(min_fitv_pl)
    #             pop_fitv_pl[pso_tt_site_pl[tt_2_pl]] = 1.0
    #
    #         for tt_1_pl in range(pso_tt1_pl):
    #             min_fitness_pl = min(pop_fitness_pl)
    #             pso_tt_site_pl[tt_1_pl + pso_tt2_pl] = pop_fitness_pl.index(min_fitness_pl)
    #             pop_fitness_pl[pso_tt_site_pl[tt_1_pl + pso_tt2_pl]] = max(pop_fitness_pl)
    #         for pso_j_pl in range(pso_sumpop_pl):
    #             # pso_pop[pso_j] += 0.5*pso_v[pso_j]
    #             pop_site_pl[pso_j_pl] += pso_t_pl * pop_v_pl[pso_j_pl]
    #
    #         for re_rand_pl, resit_pl in zip(pso_tt_site_pl, range(pso_tt1_pl + pso_tt2_pl)):
    #             pop_site_pl[re_rand_pl] = re_sit_pl[resit_pl]
    #             pbestpop_pl[re_rand_pl] = re_sit_pl[resit_pl]
    #         # pop_size[pop_size < pso_rangepop[0]] = pso_rangepop[0]
    #         # pop_size[pop_size > pso_rangepop[1]] = pso_rangepop[1]
    #
    #     # 适应度更新
    #     pop_fitness_pl = evaluate_popfit_pl(preloader=val_best, dataloader=val_loader, number_net=(number_net + 1),
    #                                         poplist=pop_site_pl,
    #                                         popsum=pso_sumpop_pl, net_n_classes=nets_classes, device='cpu')
    #
    #     print('pso_i', pso_i_pl)
    #     print('pop_fitness', pop_fitness_pl)
    #     print('pop_site', pop_site_pl)
    #
    #     for pso_j_pl in range(pso_sumpop_pl):
    #         if pop_fitness_pl[pso_j_pl] > pbestfitness_pl[pso_j_pl]:
    #             pbestfitness_pl[pso_j_pl] = pop_fitness_pl[pso_j_pl]
    #             pbestpop_pl[pso_j_pl] = copy.deepcopy(pop_site_pl[pso_j_pl])
    #
    #     if pbestfitness_pl.max() > gbestfitness_pl:
    #         gbestfitness_pl = pbestfitness_pl.max()
    #         gbestpop_pl = copy.deepcopy(pop_site_pl[pbestfitness_pl.argmax()])
    #
    #     pso_result_pl[pso_i_pl] = gbestfitness_pl
    #     print('gbestfitness=', gbestfitness_pl)
    #     print('gbestpop=', gbestpop_pl)
    #     txt_ensm[2].write(
    #         f'{pso_i_pl}' + '\n' +
    #         ' pop_fitness=' + f'{pop_fitness_pl}' +
    #         ', pop_site=' + f'{pop_site_pl}' +
    #         ', gbestfitness=' + f'{gbestfitness_pl}' +
    #         ', gbestpop=' + f'{gbestpop_pl}')
    # print('pso_result=', pso_result_pl)
    # print('PSO one over')
    # txt_ensm[2].write(
    #     '\n' + '------------' + '\n' +
    #     ' pso_result=' + f'{pso_result_pl}'
    #     '\n' + '------------')
    # txt[number_net + 2].write(
    #     '\n' + '------------' + '\n' +
    #     'part PK EnsA' + '\n' +
    #     ' pso_result=' + f'{pso_result}' +
    #     '\n' + '------------' + '\n')
    # txt_ensm[2].close()
    # plt.plot(pso_result_pl)
    # plt.savefig(f'text_result/PK.png')
    #
    # print('-----------------------------------------------')
    # print('Peak Ensemable test START!!!')
    # print('-----------------------------------------------')

    gbestpop_pl = np.array([0.5/6, 0.5/6, 0.5/6, 0.5/6, 0.5/6, 0.5/6, 0.5])

    mask_pre_pla = [None, None, None, None, None, None, None]
    mask_pre_plb = []
    mask_pre_pl = []
    for j in range(number_ensemble):
        mask_pre_plb.insert(j, copy.deepcopy(mask_pre_pla))
    for h in range(3):
        mask_pre_pl.insert(h, copy.deepcopy(mask_pre_plb))

    for i in range(3):
        for j in range(number_ensemble):
            mask_pre_pl[i][j] = [bestest[i][0], bestest[i][1], bestest[i][2], bestest[i][3], bestest[i][4], bestest[i][5], bestest[i][6][j]]
            Dice_pl[i][j], IoU_pl[i][j], PC_pl[i][j], SE_pl[i][j], SP_pl[i][j], ACC_pl[i][j], F2_pl[i][j], mDice_pl[i][j], mIoU_pl[i][j], mPC_pl[i][j], \
                mSE_pl[i][j], mSP_pl[i][j], mACC_pl[i][j], mF2_pl[i][j] = evaluate_ensemble_l(preloader=mask_pre_pl[i][j], dataloader=test_loader[i],
                                                                                 number_net=(number_net + 1), bestpop=gbestpop_pl,
                                                                                 net_n_classes=nets_classes, device='cpu')

    print('-----------------------------------------------')
    for i in range(3):
        print('now test =', i,
              'best2 Dice =', Dice_pl[i],
              'best2 IoU =', IoU_pl[i],
              'best2 Sensitivity =', SE_pl[i],
              'best2 Precision =', PC_pl[i],
              'best2 F2 =', F2_pl[i],
              'best2 Specificity =', SP_pl[i],
              'best2 Accuracy =', ACC_pl[i],
              'best2 mDice =', mDice_pl[i],
              'best2 mIoU =', mIoU_pl[i],
              'best2 mean Sensitivity =', mSE_pl[i],
              'best2 mean Precision =', mPC_pl[i],
              'best2 mean F2 =', mF2_pl[i],
              'best2 mean Specificity =', mSP_pl[i],
              'best2 mean Accuracy =', mACC_pl[i]
              )
    print('-----------------------------------------------')
    for i in range(3):
        txt_last[3].write(
            f'in test {i}' +
            ', Dice=' + f'{Dice_pl[i]}' +
            ', IoU=' + f'{IoU_pl[i]}' +
            ', Sensitivity=' + f'{SE_pl[i]}' +
            ', Precision=' + f'{PC_pl[i]}' +
            ', F2=' + f'{F2_pl[i]}' +
            ', Specificity=' + f'{SP_pl[i]}' +
            ', Accuracy=' + f'{ACC_pl[i]}'
            ', mDice=' + f'{mDice_pl[i]}' +
            ', mIoU=' + f'{mIoU_pl[i]}' +
            ', mean Sensitivity=' + f'{mSE_pl[i]}' +
            ', mean Precision=' + f'{mPC_pl[i]}' +
            ', mean F2=' + f'{mF2_pl[i]}' +
            ', mean Specificity=' + f'{mSP_pl[i]}' +
            ', mean Accuracy=' + f'{mACC_pl[i]}')
    txt_last[3].close()

    print('--------------------Peak Eano Ensemble--------------------')
    # # 5.3.2  training ensemble net in stage3
    # print('-----------------------------------------------')
    # print('stage Peak Eano PSO Train STAR!!!')
    # print('-----------------------------------------------')
    #
    # pso_lr_pbl = (1.0, 1.0)
    #
    # pso_epoch_pbl = int(20)
    # pso_sumpop_pbl = int(100)
    # pso_tt1_pbl = int(18)
    # pso_tt2_pbl = int(2)
    #
    # # pso_epoch_pbl = int(3)
    # # pso_sumpop_pbl = int(48)
    # # pso_tt1_pbl = int(3)
    # # pso_tt2_pbl = int(1)
    #
    # pso_tt_site_pbl = [0] * (pso_tt1_pbl + pso_tt2_pbl)
    # pop_fitv_pbl = [1.0] * pso_sumpop_pbl
    #
    # pop_site_pbl = torch.normal(mean=0.1666, std=0.5, size=(pso_sumpop_pbl, (number_labelnet + 1)))
    # pop_site_sum_pbl = torch.sum(pop_site_pbl, dim=1)
    # pop_site_another_pbl = torch.ones((number_labelnet + 1))
    # pop_site_other_pbl, _ = torch.meshgrid(pop_site_sum_pbl, pop_site_another_pbl)
    # pop_site_pbl = pop_site_pbl / pop_site_other_pbl
    # pop_site_pbl = pop_site_pbl.numpy()
    #
    # if number_labelnet == 6:
    #     pop_site_pbl[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     pop_site_pbl[1] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     pop_site_pbl[2] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    #     pop_site_pbl[3] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    #     pop_site_pbl[4] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    #     pop_site_pbl[5] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    #     pop_site_pbl[6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    #     pop_site_pbl[7] = [0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
    #     pop_site_pbl[8] = [0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.5]
    #     pop_site_pbl[9] = [0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.5]
    #     pop_site_pbl[10] = [0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.5]
    #     pop_site_pbl[11] = [0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.5]
    #     pop_site_pbl[12] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.5]
    # elif number_labelnet == 5:
    #     pop_site_pbl[0] = [1.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    #     pop_site_pbl[1] = [0.00, 1.00, 0.00, 0.00, 0.00, 0.00]
    #     pop_site_pbl[2] = [0.00, 0.00, 1.00, 0.00, 0.00, 0.00]
    #     pop_site_pbl[3] = [0.00, 0.00, 0.00, 1.00, 0.00, 0.00]
    #     pop_site_pbl[4] = [0.00, 0.00, 0.00, 0.00, 1.00, 0.00]
    #     pop_site_pbl[5] = [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]
    #     pop_site_pbl[6] = [0.00, 0.125, 0.125, 0.125, 0.125, 0.5]
    #     pop_site_pbl[7] = [0.125, 0.00, 0.125, 0.125, 0.125, 0.5]
    #     pop_site_pbl[8] = [0.125, 0.125, 0.00, 0.125, 0.125, 0.5]
    #     pop_site_pbl[9] = [0.125, 0.125, 0.125, 0.00, 0.125, 0.5]
    #     pop_site_pbl[10] = [0.125, 0.125, 0.125, 0.125, 0.00, 0.5]
    # elif number_labelnet == 4:
    #     pop_site_pbl[0] = [1.0, 0.0, 0.0, 0.0, 0.0]
    #     pop_site_pbl[1] = [0.0, 1.0, 0.0, 0.0, 0.0]
    #     pop_site_pbl[2] = [0.0, 0.0, 1.0, 0.0, 0.0]
    #     pop_site_pbl[3] = [0.0, 0.0, 0.0, 1.0, 0.0]
    #     pop_site_pbl[4] = [0.0, 0.0, 0.0, 0.0, 1.0]
    #     pop_site_pbl[5] = [0.0, 0.2, 0.2, 0.2, 0.4]
    #     pop_site_pbl[6] = [0.2, 0.0, 0.2, 0.2, 0.4]
    #     pop_site_pbl[7] = [0.2, 0.2, 0.0, 0.2, 0.4]
    #     pop_site_pbl[8] = [0.2, 0.2, 0.2, 0.0, 0.4]
    # elif number_labelnet == 3:
    #     pop_site_pbl[0] = [1.0, 0.0, 0.0, 0.0]
    #     pop_site_pbl[1] = [0.0, 1.0, 0.0, 0.0]
    #     pop_site_pbl[2] = [0.0, 0.0, 1.0, 0.0]
    #     pop_site_pbl[3] = [0.0, 0.0, 0.0, 1.0]
    #     pop_site_pbl[4] = [0.0, 0.2, 0.2, 0.6]
    #     pop_site_pbl[5] = [0.2, 0.0, 0.2, 0.6]
    #     pop_site_pbl[6] = [0.2, 0.2, 0.0, 0.6]
    # elif number_labelnet == 2:
    #     pop_site_pbl[0] = [1.0, 0.0, 0.0]
    #     pop_site_pbl[1] = [0.0, 1.0, 0.0]
    #     pop_site_pbl[2] = [0.0, 0.0, 1.0]
    #     pop_site_pbl[3] = [0.2, 0.0, 0.8]
    #     pop_site_pbl[4] = [0.0, 0.2, 0.8]
    #
    # # pop_site_pbl[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # # pop_site_pbl[1] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # # pop_site_pbl[2] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    # # pop_site_pbl[3] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    # # pop_site_pbl[4] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    # # pop_site_pbl[5] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # # pop_site_pbl[6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # # pop_site_pbl[7] = [0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
    # # pop_site_pbl[8] = [0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.5]
    # # pop_site_pbl[9] = [0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.5]
    # # pop_site_pbl[10] = [0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.5]
    # # pop_site_pbl[11] = [0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.5]
    # # pop_site_pbl[12] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.5]
    #
    # # pop_site_pl[0] = [1.0, 0.0, 0.0, 0.0]
    # # pop_site_pl[1] = [0.0, 1.0, 0.0, 0.0]
    # # pop_site_pl[2] = [0.0, 0.0, 1.0, 0.0]
    # # pop_site_pl[3] = [0.0, 0.0, 0.0, 1.0]
    #
    # # pop_site_pl[0] = [1.0, 0.0, 0.0]
    # # pop_site_pl[1] = [0.0, 1.0, 0.0]
    # # pop_site_pl[2] = [0.0, 0.0, 1.0]
    # # pop_site_pl[3] = [0.0, 0.5, 0.5]
    # # pop_site_pl[4] = [0.5, 0.0, 0.5]
    # # pop_site_pl[5] = [0.5, 0.5, 0.0]
    #
    # pop_v_pbl = np.zeros((pso_sumpop_pbl, (number_labelnet + 1)))
    #
    # gbestpop_pbl = np.zeros(number_labelnet + 1)
    # pbestpop_pbl = np.zeros((pso_sumpop_pbl, (number_labelnet + 1)))
    # gbestfitness_pbl = float(0)
    # pbestfitness_pbl = np.zeros(pso_sumpop_pbl)
    # pso_result_pbl = np.zeros(pso_epoch_pbl)
    # pso_t_pbl = 0.75
    # pop_fitness_pbl = [float(0)] * pso_sumpop_pbl
    #
    # for pso_i_pbl in range(pso_epoch_pbl):
    #     # 速度更新
    #     re_sit_pbl = np.random.dirichlet(np.ones(number_labelnet + 1), size=(pso_tt1_pbl + pso_tt2_pbl))
    #     # for fig in range(pso_sumpop):
    #     #     pop_score_tensor, _, _, _, _, _, _ = evaluate_pop(net=nets, dataloader=train_loader, device=device, number=number_labelnet + 1,
    #     #                                  onlyone_pop=pop_site[fig], mask_pred_size=masks_pred_size, fig_now=pso_i, pop_now=pso_j, batch_size=batch_size)
    #     #     pop_score_numpy = pop_score_tensor.data.cpu().numpy()
    #     #     pop_fitness.insert(fig, pop_score_numpy)
    #     if not pso_i_pbl == 0:
    #         for pso_j_pbl in range(pso_sumpop_pbl):
    #             pop_v_pbl[pso_j_pbl] += pso_lr_pbl[0] * np.random.rand() * (
    #                     pbestpop_pbl[pso_j_pbl] - pop_site_pbl[pso_j_pbl]) + pso_lr_pbl[
    #                                       1] * np.random.rand() * (gbestpop_pbl - pop_site_pbl[pso_j_pbl])
    #         # pop_v[pop_v < pso_rangespeed[0]] = pso_rangespeed[0]
    #         # pop_v[pop_v > pso_rangespeed[1]] = pso_rangespeed[1]
    #
    #         # 粒子位置更新
    #         max_fitness_pbl = max(pop_fitness_pbl)
    #         max_fit_site_pbl = pop_fitness_pbl.index(max_fitness_pbl)
    #         for sit_pbl in range(pso_sumpop_pbl):
    #             if sit_pbl != max_fit_site_pbl:
    #                 pop_fitv_pbl[sit_pbl] = np.linalg.norm(pop_site_pbl[max_fit_site_pbl] - pop_site_pbl[sit_pbl])
    #             else:
    #                 pop_fitv_pbl[sit_pbl] = 1.0
    #         for tt_2_pbl in range(pso_tt2_pbl):
    #             min_fitv_pbl = min(pop_fitv_pbl)
    #             pso_tt_site_pbl[tt_2_pbl] = pop_fitv_pbl.index(min_fitv_pbl)
    #             pop_fitv_pbl[pso_tt_site_pbl[tt_2_pbl]] = 1.0
    #
    #         for tt_1_pbl in range(pso_tt1_pbl):
    #             min_fitness_pbl = min(pop_fitness_pbl)
    #             pso_tt_site_pbl[tt_1_pbl + pso_tt2_pbl] = pop_fitness_pbl.index(min_fitness_pbl)
    #             pop_fitness_pbl[pso_tt_site_pbl[tt_1_pbl + pso_tt2_pbl]] = max(pop_fitness_pbl)
    #         for pso_j_pbl in range(pso_sumpop_pbl):
    #             # pso_pop[pso_j] += 0.5*pso_v[pso_j]
    #             pop_site_pbl[pso_j_pbl] += pso_t_pbl * pop_v_pbl[pso_j_pbl]
    #
    #         for re_rand_pbl, resit_pbl in zip(pso_tt_site_pbl, range(pso_tt1_pbl + pso_tt2_pbl)):
    #             pop_site_pbl[re_rand_pbl] = re_sit_pbl[resit_pbl]
    #             pbestpop_pbl[re_rand_pbl] = re_sit_pbl[resit_pbl]
    #         # pop_size[pop_size < pso_rangepop[0]] = pso_rangepop[0]
    #         # pop_size[pop_size > pso_rangepop[1]] = pso_rangepop[1]
    #
    #     # 适应度更新
    #     pop_fitness_pbl = evaluate_popfit_pbl(preloader=val_best, dataloader=val_loader, number_net=(number_net + 1),
    #                                         poplist=pop_site_pbl, label=label, threshold=label_threshold,
    #                                         popsum=pso_sumpop_pbl, net_n_classes=nets_classes, device='cpu')
    #
    #     print('pso_i', pso_i_pbl)
    #     print('pop_fitness', pop_fitness_pbl)
    #     print('pop_site', pop_site_pbl)
    #
    #     for pso_j_pbl in range(pso_sumpop_pbl):
    #         if pop_fitness_pbl[pso_j_pbl] > pbestfitness_pbl[pso_j_pbl]:
    #             pbestfitness_pbl[pso_j_pbl] = pop_fitness_pbl[pso_j_pbl]
    #             pbestpop_pbl[pso_j_pbl] = copy.deepcopy(pop_site_pbl[pso_j_pbl])
    #
    #     if pbestfitness_pbl.max() > gbestfitness_pbl:
    #         gbestfitness_pbl = pbestfitness_pbl.max()
    #         gbestpop_pbl = copy.deepcopy(pop_site_pbl[pbestfitness_pbl.argmax()])
    #
    #     pso_result_pbl[pso_i_pbl] = gbestfitness_pbl
    #     print('gbestfitness=', gbestfitness_pbl)
    #     print('gbestpop=', gbestpop_pbl)
    #     txt_ensm[3].write(
    #         f'{pso_i_pbl}' + '\n' +
    #         ' pop_fitness=' + f'{pop_fitness_pbl}' +
    #         ', pop_site=' + f'{pop_site_pbl}' +
    #         ', gbestfitness=' + f'{gbestfitness_pbl}' +
    #         ', gbestpop=' + f'{gbestpop_pbl}')
    # print('pso_result=', pso_result_pbl)
    # print('PSO one over')
    # txt_ensm[3].write(
    #     '\n' + '------------' + '\n' +
    #     ' pso_result=' + f'{pso_result_pl}'
    #                      '\n' + '------------')
    # txt[number_net + 2].write(
    #     '\n' + '------------' + '\n' +
    #     'part PK EnsB' + '\n' +
    #     ' pso_result=' + f'{pso_result}' + '\n' +
    #     ', label=' + f'{label}' +
    #     '\n' + '------------' + '\n')
    # txt_ensm[3].close()
    # plt.plot(pso_result_pbl)
    # plt.savefig(f'text_result/PK_ano.png')
    #
    # print('-----------------------------------------------')
    # print('Peak Ensemable Eano test START!!!')
    # print('-----------------------------------------------')

    if number_labelnet == 6:
        gbestpop_pbl = np.array([0.5/6, 0.5/6, 0.5/6, 0.5/6, 0.5/6, 0.5/6, 0.5])
    elif number_labelnet == 5:
        gbestpop_pbl = np.array([0.5/5, 0.5/5, 0.5/5, 0.5/5, 0.5/5, 0.5])
    elif number_labelnet == 4:
        gbestpop_pbl = np.array([0.5/4, 0.5/4, 0.5/4, 0.5/4, 0.5])
    elif number_labelnet == 3:
        gbestpop_pbl = np.array([0.5/3, 0.5/3, 0.5/3, 0.5])
    elif number_labelnet == 2:
        gbestpop_pbl = np.array([0.5/2, 0.5/2, 0.5])

    mask_pre_pbla = [None, None, None, None, None, None, None]
    mask_pre_pblb = []
    mask_pre_pbl = []
    for j in range(number_ensemble):
        mask_pre_pblb.insert(j, copy.deepcopy(mask_pre_pbla))
    for h in range(3):
        mask_pre_pbl.insert(h, copy.deepcopy(mask_pre_pblb))

    for i in range(3):
        for j in range(number_ensemble):
            mask_pre_pbl[i][j] = [bestest[i][0], bestest[i][1], bestest[i][2], bestest[i][3], bestest[i][4], bestest[i][5], bestest[i][7][j]]
            Dice_pbl[i][j], IoU_pbl[i][j], PC_pbl[i][j], SE_pbl[i][j], SP_pbl[i][j], ACC_pbl[i][j], F2_pbl[i][j], mDice_pbl[i][j], mIoU_pbl[i][j], \
                mPC_pbl[i][j], mSE_pbl[i][j], mSP_pbl[i][j], mACC_pbl[i][j], mF2_pbl[i][j] = evaluate_ensemble_llable(preloader=mask_pre_pbl[i][j],
                                                                                                  dataloader=test_loader[i],
                                                                                                  number_net=(number_net + 1),
                                                                                                  label=label,
                                                                                                  threshold=label_threshold,
                                                                                                  bestpop=gbestpop_pbl,
                                                                                                  net_n_classes=nets_classes,
                                                                                                  device='cpu')

    print('-----------------------------------------------')
    for i in range(3):
        print('now test =', i,
              'best2 Dice =', Dice_pbl[i],
              'best2 IoU =', IoU_pbl[i],
              'best2 Sensitivity =', SE_pbl[i],
              'best2 Precision =', PC_pbl[i],
              'best2 F2 =', F2_pbl[i],
              'best2 Specificity =', SP_pbl[i],
              'best2 Accuracy =', ACC_pbl[i],
              'best2 mDice =', mDice_pbl[i],
              'best2 mIoU =', mIoU_pbl[i],
              'best2 mean Sensitivity =', mSE_pbl[i],
              'best2 mean Precision =', mPC_pbl[i],
              'best2 mean F2 =', mF2_pbl[i],
              'best2 mean Specificity =', mSP_pbl[i],
              'best2 mean Accuracy =', mACC_pbl[i]
              )
    print('-----------------------------------------------')
    for i in range(3):
        txt_last[4].write(
            f'in test {i}' +
            ' Dice=' + f'{Dice_pbl[i]}' +
            ', IoU=' + f'{IoU_pbl[i]}' +
            ', Sensitivity=' + f'{SE_pbl[i]}' +
            ', Precision=' + f'{PC_pbl[i]}' +
            ', F2=' + f'{F2_pbl[i]}' +
            ', Specificity=' + f'{SP_pbl[i]}' +
            ', Accuracy=' + f'{ACC_pbl[i]}' +
            ', mDice=' + f'{mDice_pbl[i]}' +
            ', mIoU=' + f'{mIoU_pbl[i]}' +
            ', mean Sensitivity=' + f'{mSE_pbl[i]}' +
            ', mean Precision=' + f'{mPC_pbl[i]}' +
            ', mean F2=' + f'{mF2_pbl[i]}' +
            ', mean Specificity=' + f'{mSP_pbl[i]}' +
            ', mean Accuracy=' + f'{mACC_pbl[i]}')
    txt_last[4].close()
    for i in range(3):
        txt[number_net + 2].write(
            '\n' + '------------' + '\n' +
            'part PK EnsA' + '\n' +
            f'now in test {i}' + '\n' +
            ' Dice=' + f'{Dice_pl[i]}' + '\n' +
            ', IoU=' + f'{IoU_pl[i]}' + '\n' +
            ', Sensitivity=' + f'{SE_pl[i]}' + '\n' +
            ', Precision=' + f'{PC_pl[i]}' + '\n' +
            ', F2=' + f'{F2_pl[i]}' + '\n' +
            ', Specificity=' + f'{SP_pl[i]}' + '\n' +
            ', Accuracy=' + f'{ACC_pl[i]}' + '\n' +
            ', mDice=' + f'{mDice_pl[i]}' + '\n' +
            ', mIoU=' + f'{mIoU_pl[i]}' + '\n' +
            ', mean Sensitivity=' + f'{mSE_pl[i]}' + '\n' +
            ', mean Precision=' + f'{mPC_pl[i]}' + '\n' +
            ', mean F2=' + f'{mF2_pl[i]}' + '\n' +
            ', mean Specificity=' + f'{mSP_pl[i]}' + '\n' +
            ', mean Accuracy=' + f'{mACC_pl[i]}' +
            '\n' + '------------' + '\n')

        txt[number_net + 2].write(
            '\n' + '------------' + '\n' +
            'part PK EnsB' + '\n' +
            f'now in test {i}' + '\n' +
            ' Dice=' + f'{Dice_pbl[i]}' + '\n' +
            ', IoU=' + f'{IoU_pbl[i]}' + '\n' +
            ', Sensitivity=' + f'{SE_pbl[i]}' + '\n' +
            ', Precision=' + f'{PC_pbl[i]}' + '\n' +
            ', F2=' + f'{F2_pbl[i]}' + '\n' +
            ', Specificity=' + f'{SP_pbl[i]}' + '\n' +
            ', Accuracy=' + f'{ACC_pbl[i]}' + '\n' +
            ', mDice=' + f'{mDice_pbl[i]}' + '\n' +
            ', mIoU=' + f'{mIoU_pbl[i]}' + '\n' +
            ', mean Sensitivity=' + f'{mSE_pbl[i]}' + '\n' +
            ', mean Precision=' + f'{mPC_pbl[i]}' + '\n' +
            ', mean F2=' + f'{mF2_pbl[i]}' + '\n' +
            ', mean Specificity=' + f'{mSP_pbl[i]}' + '\n' +
            ', mean Accuracy=' + f'{mACC_pbl[i]}' +
            '\n' + '------------' + '\n')

    txt[number_net + 2].close()

    print('-----------------------------------------------')
    print('-----------------------------------------------')
    print('STAR make wandb')
    testi = 0
    for Epoch_Net in range(number_net):
        if label[Epoch_Net] >= label_threshold:
            experiment0[Epoch_Net] = wandb.init(project='Ensemble_TrebleFormer_ETIS-LaribPolypDB', group=Ename,
                                               name=nets_name[Epoch_Net],
                                               resume='allow',
                                               anonymous='must', reinit=True, allow_val_change=True,
                                               id=Ename + nets_name[Epoch_Net])
            experiment0[Epoch_Net].config.update(
                dict(epochs=copy.deepcopy(epochs_nets[Epoch_Net]), batch_size=batch_size,
                     learning_rate=copy.deepcopy(learning_rate_nets[Epoch_Net]),
                     val_percent=val_percent, save_checkpoint=save_checkpoint,
                     img_scale=img_scale,
                     amp=amp), allow_val_change=True)

            for epoch in range((jiedian + 1)):
                experiment0[Epoch_Net].log({
                    'Dice': Dice_All[testi][Epoch_Net][epoch],
                    'IoU': IoU_All[testi][Epoch_Net][epoch],
                    'Sensitivity': SE_All[testi][Epoch_Net][epoch],
                    'Precision': PC_All[testi][Epoch_Net][epoch],
                    'F2-score': F2_All[testi][Epoch_Net][epoch],
                    'Specificity': SP_All[testi][Epoch_Net][epoch],
                    'Accuracy': ACC_All[testi][Epoch_Net][epoch],
                    'mDice': mDice_All[testi][Epoch_Net][epoch],
                    'mIoU': mIoU_All[testi][Epoch_Net][epoch],
                    'mean Sensitivity': mSE_All[testi][Epoch_Net][epoch],
                    'mean Precision': mPC_All[testi][Epoch_Net][epoch],
                    'mean F2-score': mF2_All[testi][Epoch_Net][epoch],
                    'mean Specificity': mSP_All[testi][Epoch_Net][epoch],
                    'mean Accuracy': mACC_All[testi][Epoch_Net][epoch],
                    'learning rate': lr_All[Epoch_Net][epoch],
                    'loss': loss_All[Epoch_Net][epoch],
                    'epoch': epoch_All[Epoch_Net][epoch]
                })

            experiment0[Epoch_Net] = wandb.init(project='Ensemble_TrebleFormer_ETIS-LaribPolypDB', group=Ename,
                                               name=nets_name_label[Epoch_Net],
                                               resume='allow',
                                               anonymous='must', reinit=True, allow_val_change=True,
                                               id=Ename + nets_name_label[Epoch_Net])
            experiment0[Epoch_Net].config.update(
                dict(epochs=copy.deepcopy(epochs_nets[Epoch_Net]), batch_size=batch_size,
                     learning_rate=copy.deepcopy(learning_rate_nets[Epoch_Net]),
                     val_percent=val_percent, save_checkpoint=save_checkpoint,
                     img_scale=img_scale,
                     amp=amp), allow_val_change=True)

            for epoch in range(jiedian, (set_epochs + 1)):

                experiment0[Epoch_Net].log({
                    'Dice': Dice_All[testi][Epoch_Net][epoch],
                    'IoU': IoU_All[testi][Epoch_Net][epoch],
                    'Sensitivity': SE_All[testi][Epoch_Net][epoch],
                    'Precision': PC_All[testi][Epoch_Net][epoch],
                    'F2-score': F2_All[testi][Epoch_Net][epoch],
                    'Specificity': SP_All[testi][Epoch_Net][epoch],
                    'Accuracy': ACC_All[testi][Epoch_Net][epoch],
                    'mDice': mDice_All[testi][Epoch_Net][epoch],
                    'mIoU': mIoU_All[testi][Epoch_Net][epoch],
                    'mean Sensitivity': mSE_All[testi][Epoch_Net][epoch],
                    'mean Precision': mPC_All[testi][Epoch_Net][epoch],
                    'mean F2-score': mF2_All[testi][Epoch_Net][epoch],
                    'mean Specificity': mSP_All[testi][Epoch_Net][epoch],
                    'mean Accuracy': mACC_All[testi][Epoch_Net][epoch],
                    'learning rate': lr_All[Epoch_Net][epoch],
                    'loss': loss_All[Epoch_Net][epoch],
                    'epoch': epoch_All[Epoch_Net][epoch]
                })

        else:
            experiment0[Epoch_Net] = wandb.init(project='Ensemble_TrebleFormer_ETIS-LaribPolypDB', group=Ename,
                                               name=nets_name[Epoch_Net],
                                               resume='allow',
                                               anonymous='must', reinit=True, allow_val_change=True,
                                               id=Ename + nets_name[Epoch_Net])
            experiment0[Epoch_Net].config.update(
                dict(epochs=copy.deepcopy(epochs_nets[Epoch_Net]), batch_size=batch_size,
                     learning_rate=copy.deepcopy(learning_rate_nets[Epoch_Net]),
                     val_percent=val_percent, save_checkpoint=save_checkpoint,
                     img_scale=img_scale,
                     amp=amp), allow_val_change=True)

            for epoch in range((set_epochs + 1)):

                experiment0[Epoch_Net].log({
                    'Dice': Dice_All[testi][Epoch_Net][epoch],
                    'IoU': IoU_All[testi][Epoch_Net][epoch],
                    'Sensitivity': SE_All[testi][Epoch_Net][epoch],
                    'Precision': PC_All[testi][Epoch_Net][epoch],
                    'F2-score': F2_All[testi][Epoch_Net][epoch],
                    'Specificity': SP_All[testi][Epoch_Net][epoch],
                    'Accuracy': ACC_All[testi][Epoch_Net][epoch],
                    'mDice': mDice_All[testi][Epoch_Net][epoch],
                    'mIoU': mIoU_All[testi][Epoch_Net][epoch],
                    'mean Sensitivity': mSE_All[testi][Epoch_Net][epoch],
                    'mean Precision': mPC_All[testi][Epoch_Net][epoch],
                    'mean F2-score': mF2_All[testi][Epoch_Net][epoch],
                    'mean Specificity': mSP_All[testi][Epoch_Net][epoch],
                    'mean Accuracy': mACC_All[testi][Epoch_Net][epoch],
                    'learning rate': lr_All[Epoch_Net][epoch],
                    'loss': loss_All[Epoch_Net][epoch],
                    'epoch': epoch_All[Epoch_Net][epoch]
                })

    for j in range(number_ensemble):
        experiment0[number_net][j] = wandb.init(project='Ensemble_TrebleFormer_ETIS-LaribPolypDB', group=Ename, name='EnA' + str(j),
                                                resume='allow',
                                                anonymous='must', reinit=True, allow_val_change=True, id=Ename + 'EnA' + str(j))
        experiment0[number_net][j].config.update(
            dict(epochs=copy.deepcopy(epochs_nets[number_net]), batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint,
                 img_scale=img_scale, amp=amp), allow_val_change=True)

        for epoch in range((set_epochs + 1)):
            experiment0[number_net][j].log({
                'Dice': Dice_All[testi][number_net][epoch][j],
                'IoU': IoU_All[testi][number_net][epoch][j],
                'Sensitivity': SE_All[testi][number_net][epoch][j],
                'Precision': PC_All[testi][number_net][epoch][j],
                'F2-score': F2_All[testi][number_net][epoch][j],
                'Specificity': SP_All[testi][number_net][epoch][j],
                'Accuracy': ACC_All[testi][number_net][epoch][j],
                'mDice': mDice_All[testi][number_net][epoch][j],
                'mIoU': mIoU_All[testi][number_net][epoch][j],
                'mean Sensitivity': mSE_All[testi][number_net][epoch][j],
                'mean Precision': mPC_All[testi][number_net][epoch][j],
                'mean F2-score': mF2_All[testi][number_net][epoch][j],
                'mean Specificity': mSP_All[testi][number_net][epoch][j],
                'mean Accuracy': mACC_All[testi][number_net][epoch][j],
                'epoch': epoch_All[number_net][epoch]
            })

    for j in range(number_ensemble):
        experiment0[number_net + 1][j] = wandb.init(project='Ensemble_TrebleFormer_ETIS-LaribPolypDB', group=Ename, name='EnB' + str(j),
                                            resume='allow',
                                            anonymous='must', reinit=True, allow_val_change=True, id=Ename + 'EnB' + str(j))
        experiment0[number_net + 1][j].config.update(
            dict(epochs=copy.deepcopy(epochs_nets[number_net]), batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint,
                 img_scale=img_scale, amp=amp), allow_val_change=True)

        for epoch in range((set_epochs + 1)):

            experiment0[number_net + 1][j].log({
                'Dice': Dice_All[testi][number_net + 1][epoch][j],
                'IoU': IoU_All[testi][number_net + 1][epoch][j],
                'Sensitivity': SE_All[testi][number_net + 1][epoch][j],
                'Precision': PC_All[testi][number_net + 1][epoch][j],
                'F2-score': F2_All[testi][number_net + 1][epoch][j],
                'Specificity': SP_All[testi][number_net + 1][epoch][j],
                'Accuracy': ACC_All[testi][number_net + 1][epoch][j],
                'mDice': mDice_All[testi][number_net + 1][epoch][j],
                'mIoU': mIoU_All[testi][number_net + 1][epoch][j],
                'mean Sensitivity': mSE_All[testi][number_net + 1][epoch][j],
                'mean Precision': mPC_All[testi][number_net + 1][epoch][j],
                'mean F2-score': mF2_All[testi][number_net + 1][epoch][j],
                'mean Specificity': mSP_All[testi][number_net + 1][epoch][j],
                'mean Accuracy': mACC_All[testi][number_net + 1][epoch][j],
                'epoch': epoch_All[number_net + 1][epoch]
            })

    for j in range(number_ensemble):
        experiment0[number_net + 2][j] = wandb.init(project='Ensemble_TrebleFormer_ETIS-LaribPolypDB', group=Ename, name='EAP' + str(j),
                                                resume='allow',
                                                anonymous='must', reinit=True, allow_val_change=True, id=Ename + 'EAP' + str(j))
        experiment0[number_net + 2][j].config.update(
            dict(epochs=copy.deepcopy(epochs_nets[number_net]), batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint,
                 img_scale=img_scale, amp=amp), allow_val_change=True)

        experiment0[number_net + 2][j].log({
            'Dice': Dice_pl[testi][j],
            'IoU': IoU_pl[testi][j],
            'Sensitivity': SE_pl[testi][j],
            'Precision': PC_pl[testi][j],
            'F2-score': F2_pl[testi][j],
            'Specificity': SP_pl[testi][j],
            'Accuracy': ACC_pl[testi][j],
            'mDice': mDice_pl[testi][j],
            'mIoU': mIoU_pl[testi][j],
            'mean Sensitivity': mSE_pl[testi][j],
            'mean Precision': mPC_pl[testi][j],
            'mean F2-score': mF2_pl[testi][j],
            'mean Specificity': mSP_pl[testi][j],
            'mean Accuracy': mACC_pl[testi][j],
            'epoch': (set_epochs + 1)
        })

    for j in range(number_ensemble):
        experiment0[number_net + 3][j] = wandb.init(project='Ensemble_TrebleFormer_ETIS-LaribPolypDB', group=Ename, name='EBP' + str(j),
                                                resume='allow',
                                                anonymous='must', reinit=True, allow_val_change=True, id=Ename + 'EBP' + str(j))
        experiment0[number_net + 3][j].config.update(
            dict(epochs=copy.deepcopy(epochs_nets[number_net]), batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint,
                 img_scale=img_scale, amp=amp), allow_val_change=True)

        experiment0[number_net + 3][j].log({
            'Dice': Dice_pbl[testi][j],
            'IoU': IoU_pbl[testi][j],
            'Sensitivity': SE_pbl[testi][j],
            'Precision': PC_pbl[testi][j],
            'F2-score': F2_pbl[testi][j],
            'Specificity': SP_pbl[testi][j],
            'Accuracy': ACC_pbl[testi][j],
            'mDice': mDice_pbl[testi][j],
            'mIoU': mIoU_pbl[testi][j],
            'mean Sensitivity': mSE_pbl[testi][j],
            'mean Precision': mPC_pbl[testi][j],
            'mean F2-score': mF2_pbl[testi][j],
            'mean Specificity': mSP_pbl[testi][j],
            'mean Accuracy': mACC_pbl[testi][j],
            'epoch': (set_epochs + 1)
        })

    testi = 1
    for Epoch_Net in range(number_net):
        if label[Epoch_Net] >= label_threshold:
            experiment1[Epoch_Net] = wandb.init(project='Ensemble_TrebleFormer_CVC-ColonDB', group=Ename,
                                               name=nets_name[Epoch_Net],
                                               resume='allow',
                                               anonymous='must', reinit=True, allow_val_change=True,
                                               id=Ename + nets_name[Epoch_Net])
            experiment1[Epoch_Net].config.update(
                dict(epochs=copy.deepcopy(epochs_nets[Epoch_Net]), batch_size=batch_size,
                     learning_rate=copy.deepcopy(learning_rate_nets[Epoch_Net]),
                     val_percent=val_percent, save_checkpoint=save_checkpoint,
                     img_scale=img_scale,
                     amp=amp), allow_val_change=True)

            for epoch in range((jiedian + 1)):
                experiment1[Epoch_Net].log({
                    'Dice': Dice_All[testi][Epoch_Net][epoch],
                    'IoU': IoU_All[testi][Epoch_Net][epoch],
                    'Sensitivity': SE_All[testi][Epoch_Net][epoch],
                    'Precision': PC_All[testi][Epoch_Net][epoch],
                    'F2-score': F2_All[testi][Epoch_Net][epoch],
                    'Specificity': SP_All[testi][Epoch_Net][epoch],
                    'Accuracy': ACC_All[testi][Epoch_Net][epoch],
                    'mDice': mDice_All[testi][Epoch_Net][epoch],
                    'mIoU': mIoU_All[testi][Epoch_Net][epoch],
                    'mean Sensitivity': mSE_All[testi][Epoch_Net][epoch],
                    'mean Precision': mPC_All[testi][Epoch_Net][epoch],
                    'mean F2-score': mF2_All[testi][Epoch_Net][epoch],
                    'mean Specificity': mSP_All[testi][Epoch_Net][epoch],
                    'mean Accuracy': mACC_All[testi][Epoch_Net][epoch],
                    'learning rate': lr_All[Epoch_Net][epoch],
                    'loss': loss_All[Epoch_Net][epoch],
                    'epoch': epoch_All[Epoch_Net][epoch]
                })

            experiment1[Epoch_Net] = wandb.init(project='Ensemble_TrebleFormer_CVC-ColonDB', group=Ename,
                                               name=nets_name_label[Epoch_Net],
                                               resume='allow',
                                               anonymous='must', reinit=True, allow_val_change=True,
                                               id=Ename + nets_name_label[Epoch_Net])
            experiment1[Epoch_Net].config.update(
                dict(epochs=copy.deepcopy(epochs_nets[Epoch_Net]), batch_size=batch_size,
                     learning_rate=copy.deepcopy(learning_rate_nets[Epoch_Net]),
                     val_percent=val_percent, save_checkpoint=save_checkpoint,
                     img_scale=img_scale,
                     amp=amp), allow_val_change=True)

            for epoch in range(jiedian, (set_epochs + 1)):

                experiment1[Epoch_Net].log({
                    'Dice': Dice_All[testi][Epoch_Net][epoch],
                    'IoU': IoU_All[testi][Epoch_Net][epoch],
                    'Sensitivity': SE_All[testi][Epoch_Net][epoch],
                    'Precision': PC_All[testi][Epoch_Net][epoch],
                    'F2-score': F2_All[testi][Epoch_Net][epoch],
                    'Specificity': SP_All[testi][Epoch_Net][epoch],
                    'Accuracy': ACC_All[testi][Epoch_Net][epoch],
                    'mDice': mDice_All[testi][Epoch_Net][epoch],
                    'mIoU': mIoU_All[testi][Epoch_Net][epoch],
                    'mean Sensitivity': mSE_All[testi][Epoch_Net][epoch],
                    'mean Precision': mPC_All[testi][Epoch_Net][epoch],
                    'mean F2-score': mF2_All[testi][Epoch_Net][epoch],
                    'mean Specificity': mSP_All[testi][Epoch_Net][epoch],
                    'mean Accuracy': mACC_All[testi][Epoch_Net][epoch],
                    'learning rate': lr_All[Epoch_Net][epoch],
                    'loss': loss_All[Epoch_Net][epoch],
                    'epoch': epoch_All[Epoch_Net][epoch]
                })

        else:
            experiment1[Epoch_Net] = wandb.init(project='Ensemble_TrebleFormer_CVC-ColonDB', group=Ename,
                                               name=nets_name[Epoch_Net],
                                               resume='allow',
                                               anonymous='must', reinit=True, allow_val_change=True,
                                               id=Ename + nets_name[Epoch_Net])
            experiment1[Epoch_Net].config.update(
                dict(epochs=copy.deepcopy(epochs_nets[Epoch_Net]), batch_size=batch_size,
                     learning_rate=copy.deepcopy(learning_rate_nets[Epoch_Net]),
                     val_percent=val_percent, save_checkpoint=save_checkpoint,
                     img_scale=img_scale,
                     amp=amp), allow_val_change=True)

            for epoch in range((set_epochs + 1)):

                experiment1[Epoch_Net].log({
                    'Dice': Dice_All[testi][Epoch_Net][epoch],
                    'IoU': IoU_All[testi][Epoch_Net][epoch],
                    'Sensitivity': SE_All[testi][Epoch_Net][epoch],
                    'Precision': PC_All[testi][Epoch_Net][epoch],
                    'F2-score': F2_All[testi][Epoch_Net][epoch],
                    'Specificity': SP_All[testi][Epoch_Net][epoch],
                    'Accuracy': ACC_All[testi][Epoch_Net][epoch],
                    'mDice': mDice_All[testi][Epoch_Net][epoch],
                    'mIoU': mIoU_All[testi][Epoch_Net][epoch],
                    'mean Sensitivity': mSE_All[testi][Epoch_Net][epoch],
                    'mean Precision': mPC_All[testi][Epoch_Net][epoch],
                    'mean F2-score': mF2_All[testi][Epoch_Net][epoch],
                    'mean Specificity': mSP_All[testi][Epoch_Net][epoch],
                    'mean Accuracy': mACC_All[testi][Epoch_Net][epoch],
                    'learning rate': lr_All[Epoch_Net][epoch],
                    'loss': loss_All[Epoch_Net][epoch],
                    'epoch': epoch_All[Epoch_Net][epoch]
                })

    for j in range(number_ensemble):
        experiment1[number_net][j] = wandb.init(project='Ensemble_TrebleFormer_CVC-ColonDB', group=Ename,
                                                name='EnA' + str(j),
                                                resume='allow',
                                                anonymous='must', reinit=True, allow_val_change=True,
                                                id=Ename + 'EnA' + str(j))
        experiment1[number_net][j].config.update(
            dict(epochs=copy.deepcopy(epochs_nets[number_net]), batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint,
                 img_scale=img_scale, amp=amp), allow_val_change=True)

        for epoch in range((set_epochs + 1)):
            experiment1[number_net][j].log({
                'Dice': Dice_All[testi][number_net][epoch][j],
                'IoU': IoU_All[testi][number_net][epoch][j],
                'Sensitivity': SE_All[testi][number_net][epoch][j],
                'Precision': PC_All[testi][number_net][epoch][j],
                'F2-score': F2_All[testi][number_net][epoch][j],
                'Specificity': SP_All[testi][number_net][epoch][j],
                'Accuracy': ACC_All[testi][number_net][epoch][j],
                'mDice': mDice_All[testi][number_net][epoch][j],
                'mIoU': mIoU_All[testi][number_net][epoch][j],
                'mean Sensitivity': mSE_All[testi][number_net][epoch][j],
                'mean Precision': mPC_All[testi][number_net][epoch][j],
                'mean F2-score': mF2_All[testi][number_net][epoch][j],
                'mean Specificity': mSP_All[testi][number_net][epoch][j],
                'mean Accuracy': mACC_All[testi][number_net][epoch][j],
                'epoch': epoch_All[number_net][epoch]
            })

    for j in range(number_ensemble):
        experiment1[number_net + 1][j] = wandb.init(project='Ensemble_TrebleFormer_CVC-ColonDB', group=Ename,
                                                    name='EnB' + str(j),
                                                    resume='allow',
                                                    anonymous='must', reinit=True, allow_val_change=True,
                                                    id=Ename + 'EnB' + str(j))
        experiment1[number_net + 1][j].config.update(
            dict(epochs=copy.deepcopy(epochs_nets[number_net]), batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint,
                 img_scale=img_scale, amp=amp), allow_val_change=True)

        for epoch in range((set_epochs + 1)):
            experiment1[number_net + 1][j].log({
                'Dice': Dice_All[testi][number_net + 1][epoch][j],
                'IoU': IoU_All[testi][number_net + 1][epoch][j],
                'Sensitivity': SE_All[testi][number_net + 1][epoch][j],
                'Precision': PC_All[testi][number_net + 1][epoch][j],
                'F2-score': F2_All[testi][number_net + 1][epoch][j],
                'Specificity': SP_All[testi][number_net + 1][epoch][j],
                'Accuracy': ACC_All[testi][number_net + 1][epoch][j],
                'mDice': mDice_All[testi][number_net + 1][epoch][j],
                'mIoU': mIoU_All[testi][number_net + 1][epoch][j],
                'mean Sensitivity': mSE_All[testi][number_net + 1][epoch][j],
                'mean Precision': mPC_All[testi][number_net + 1][epoch][j],
                'mean F2-score': mF2_All[testi][number_net + 1][epoch][j],
                'mean Specificity': mSP_All[testi][number_net + 1][epoch][j],
                'mean Accuracy': mACC_All[testi][number_net + 1][epoch][j],
                'epoch': epoch_All[number_net + 1][epoch]
            })

    for j in range(number_ensemble):
        experiment1[number_net + 2][j] = wandb.init(project='Ensemble_TrebleFormer_CVC-ColonDB', group=Ename,
                                                    name='EAP' + str(j),
                                                    resume='allow',
                                                    anonymous='must', reinit=True, allow_val_change=True,
                                                    id=Ename + 'EAP' + str(j))
        experiment1[number_net + 2][j].config.update(
            dict(epochs=copy.deepcopy(epochs_nets[number_net]), batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint,
                 img_scale=img_scale, amp=amp), allow_val_change=True)

        experiment1[number_net + 2][j].log({
            'Dice': Dice_pl[testi][j],
            'IoU': IoU_pl[testi][j],
            'Sensitivity': SE_pl[testi][j],
            'Precision': PC_pl[testi][j],
            'F2-score': F2_pl[testi][j],
            'Specificity': SP_pl[testi][j],
            'Accuracy': ACC_pl[testi][j],
            'mDice': mDice_pl[testi][j],
            'mIoU': mIoU_pl[testi][j],
            'mean Sensitivity': mSE_pl[testi][j],
            'mean Precision': mPC_pl[testi][j],
            'mean F2-score': mF2_pl[testi][j],
            'mean Specificity': mSP_pl[testi][j],
            'mean Accuracy': mACC_pl[testi][j],
            'epoch': (set_epochs + 1)
        })

    for j in range(number_ensemble):
        experiment1[number_net + 3][j] = wandb.init(project='Ensemble_TrebleFormer_CVC-ColonDB', group=Ename,
                                                    name='EBP' + str(j),
                                                    resume='allow',
                                                    anonymous='must', reinit=True, allow_val_change=True,
                                                    id=Ename + 'EBP' + str(j))
        experiment1[number_net + 3][j].config.update(
            dict(epochs=copy.deepcopy(epochs_nets[number_net]), batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint,
                 img_scale=img_scale, amp=amp), allow_val_change=True)

        experiment1[number_net + 3][j].log({
            'Dice': Dice_pbl[testi][j],
            'IoU': IoU_pbl[testi][j],
            'Sensitivity': SE_pbl[testi][j],
            'Precision': PC_pbl[testi][j],
            'F2-score': F2_pbl[testi][j],
            'Specificity': SP_pbl[testi][j],
            'Accuracy': ACC_pbl[testi][j],
            'mDice': mDice_pbl[testi][j],
            'mIoU': mIoU_pbl[testi][j],
            'mean Sensitivity': mSE_pbl[testi][j],
            'mean Precision': mPC_pbl[testi][j],
            'mean F2-score': mF2_pbl[testi][j],
            'mean Specificity': mSP_pbl[testi][j],
            'mean Accuracy': mACC_pbl[testi][j],
            'epoch': (set_epochs + 1)
        })

    testi = 2
    for Epoch_Net in range(number_net):
        if label[Epoch_Net] >= label_threshold:
            experiment2[Epoch_Net] = wandb.init(project='Ensemble_TrebleFormer_CVC-300', group=Ename,
                                               name=nets_name[Epoch_Net],
                                               resume='allow',
                                               anonymous='must', reinit=True, allow_val_change=True,
                                               id=Ename + nets_name[Epoch_Net])
            experiment2[Epoch_Net].config.update(
                dict(epochs=copy.deepcopy(epochs_nets[Epoch_Net]), batch_size=batch_size,
                     learning_rate=copy.deepcopy(learning_rate_nets[Epoch_Net]),
                     val_percent=val_percent, save_checkpoint=save_checkpoint,
                     img_scale=img_scale,
                     amp=amp), allow_val_change=True)

            for epoch in range((jiedian + 1)):
                experiment2[Epoch_Net].log({
                    'Dice': Dice_All[testi][Epoch_Net][epoch],
                    'IoU': IoU_All[testi][Epoch_Net][epoch],
                    'Sensitivity': SE_All[testi][Epoch_Net][epoch],
                    'Precision': PC_All[testi][Epoch_Net][epoch],
                    'F2-score': F2_All[testi][Epoch_Net][epoch],
                    'Specificity': SP_All[testi][Epoch_Net][epoch],
                    'Accuracy': ACC_All[testi][Epoch_Net][epoch],
                    'mDice': mDice_All[testi][Epoch_Net][epoch],
                    'mIoU': mIoU_All[testi][Epoch_Net][epoch],
                    'mean Sensitivity': mSE_All[testi][Epoch_Net][epoch],
                    'mean Precision': mPC_All[testi][Epoch_Net][epoch],
                    'mean F2-score': mF2_All[testi][Epoch_Net][epoch],
                    'mean Specificity': mSP_All[testi][Epoch_Net][epoch],
                    'mean Accuracy': mACC_All[testi][Epoch_Net][epoch],
                    'learning rate': lr_All[Epoch_Net][epoch],
                    'loss': loss_All[Epoch_Net][epoch],
                    'epoch': epoch_All[Epoch_Net][epoch]
                })

            experiment2[Epoch_Net] = wandb.init(project='Ensemble_TrebleFormer_CVC-300', group=Ename,
                                               name=nets_name_label[Epoch_Net],
                                               resume='allow',
                                               anonymous='must', reinit=True, allow_val_change=True,
                                               id=Ename + nets_name_label[Epoch_Net])
            experiment2[Epoch_Net].config.update(
                dict(epochs=copy.deepcopy(epochs_nets[Epoch_Net]), batch_size=batch_size,
                     learning_rate=copy.deepcopy(learning_rate_nets[Epoch_Net]),
                     val_percent=val_percent, save_checkpoint=save_checkpoint,
                     img_scale=img_scale,
                     amp=amp), allow_val_change=True)

            for epoch in range(jiedian, (set_epochs + 1)):

                experiment2[Epoch_Net].log({
                    'Dice': Dice_All[testi][Epoch_Net][epoch],
                    'IoU': IoU_All[testi][Epoch_Net][epoch],
                    'Sensitivity': SE_All[testi][Epoch_Net][epoch],
                    'Precision': PC_All[testi][Epoch_Net][epoch],
                    'F2-score': F2_All[testi][Epoch_Net][epoch],
                    'Specificity': SP_All[testi][Epoch_Net][epoch],
                    'Accuracy': ACC_All[testi][Epoch_Net][epoch],
                    'mDice': mDice_All[testi][Epoch_Net][epoch],
                    'mIoU': mIoU_All[testi][Epoch_Net][epoch],
                    'mean Sensitivity': mSE_All[testi][Epoch_Net][epoch],
                    'mean Precision': mPC_All[testi][Epoch_Net][epoch],
                    'mean F2-score': mF2_All[testi][Epoch_Net][epoch],
                    'mean Specificity': mSP_All[testi][Epoch_Net][epoch],
                    'mean Accuracy': mACC_All[testi][Epoch_Net][epoch],
                    'learning rate': lr_All[Epoch_Net][epoch],
                    'loss': loss_All[Epoch_Net][epoch],
                    'epoch': epoch_All[Epoch_Net][epoch]
                })

        else:
            experiment2[Epoch_Net] = wandb.init(project='Ensemble_TrebleFormer_CVC-300', group=Ename,
                                               name=nets_name[Epoch_Net],
                                               resume='allow',
                                               anonymous='must', reinit=True, allow_val_change=True,
                                               id=Ename + nets_name[Epoch_Net])
            experiment2[Epoch_Net].config.update(
                dict(epochs=copy.deepcopy(epochs_nets[Epoch_Net]), batch_size=batch_size,
                     learning_rate=copy.deepcopy(learning_rate_nets[Epoch_Net]),
                     val_percent=val_percent, save_checkpoint=save_checkpoint,
                     img_scale=img_scale,
                     amp=amp), allow_val_change=True)

            for epoch in range((set_epochs + 1)):

                experiment2[Epoch_Net].log({
                    'Dice': Dice_All[testi][Epoch_Net][epoch],
                    'IoU': IoU_All[testi][Epoch_Net][epoch],
                    'Sensitivity': SE_All[testi][Epoch_Net][epoch],
                    'Precision': PC_All[testi][Epoch_Net][epoch],
                    'F2-score': F2_All[testi][Epoch_Net][epoch],
                    'Specificity': SP_All[testi][Epoch_Net][epoch],
                    'Accuracy': ACC_All[testi][Epoch_Net][epoch],
                    'mDice': mDice_All[testi][Epoch_Net][epoch],
                    'mIoU': mIoU_All[testi][Epoch_Net][epoch],
                    'mean Sensitivity': mSE_All[testi][Epoch_Net][epoch],
                    'mean Precision': mPC_All[testi][Epoch_Net][epoch],
                    'mean F2-score': mF2_All[testi][Epoch_Net][epoch],
                    'mean Specificity': mSP_All[testi][Epoch_Net][epoch],
                    'mean Accuracy': mACC_All[testi][Epoch_Net][epoch],
                    'learning rate': lr_All[Epoch_Net][epoch],
                    'loss': loss_All[Epoch_Net][epoch],
                    'epoch': epoch_All[Epoch_Net][epoch]
                })

    for j in range(number_ensemble):
        experiment2[number_net][j] = wandb.init(project='Ensemble_TrebleFormer_CVC-300', group=Ename,
                                                name='EnA' + str(j),
                                                resume='allow',
                                                anonymous='must', reinit=True, allow_val_change=True,
                                                id=Ename + 'EnA' + str(j))
        experiment2[number_net][j].config.update(
            dict(epochs=copy.deepcopy(epochs_nets[number_net]), batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint,
                 img_scale=img_scale, amp=amp), allow_val_change=True)

        for epoch in range((set_epochs + 1)):
            experiment2[number_net][j].log({
                'Dice': Dice_All[testi][number_net][epoch][j],
                'IoU': IoU_All[testi][number_net][epoch][j],
                'Sensitivity': SE_All[testi][number_net][epoch][j],
                'Precision': PC_All[testi][number_net][epoch][j],
                'F2-score': F2_All[testi][number_net][epoch][j],
                'Specificity': SP_All[testi][number_net][epoch][j],
                'Accuracy': ACC_All[testi][number_net][epoch][j],
                'mDice': mDice_All[testi][number_net][epoch][j],
                'mIoU': mIoU_All[testi][number_net][epoch][j],
                'mean Sensitivity': mSE_All[testi][number_net][epoch][j],
                'mean Precision': mPC_All[testi][number_net][epoch][j],
                'mean F2-score': mF2_All[testi][number_net][epoch][j],
                'mean Specificity': mSP_All[testi][number_net][epoch][j],
                'mean Accuracy': mACC_All[testi][number_net][epoch][j],
                'epoch': epoch_All[number_net][epoch]
            })

    for j in range(number_ensemble):
        experiment2[number_net + 1][j] = wandb.init(project='Ensemble_TrebleFormer_CVC-300', group=Ename,
                                                    name='EnB' + str(j),
                                                    resume='allow',
                                                    anonymous='must', reinit=True, allow_val_change=True,
                                                    id=Ename + 'EnB' + str(j))
        experiment2[number_net + 1][j].config.update(
            dict(epochs=copy.deepcopy(epochs_nets[number_net]), batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint,
                 img_scale=img_scale, amp=amp), allow_val_change=True)

        for epoch in range((set_epochs + 1)):
            experiment2[number_net + 1][j].log({
                'Dice': Dice_All[testi][number_net + 1][epoch][j],
                'IoU': IoU_All[testi][number_net + 1][epoch][j],
                'Sensitivity': SE_All[testi][number_net + 1][epoch][j],
                'Precision': PC_All[testi][number_net + 1][epoch][j],
                'F2-score': F2_All[testi][number_net + 1][epoch][j],
                'Specificity': SP_All[testi][number_net + 1][epoch][j],
                'Accuracy': ACC_All[testi][number_net + 1][epoch][j],
                'mDice': mDice_All[testi][number_net + 1][epoch][j],
                'mIoU': mIoU_All[testi][number_net + 1][epoch][j],
                'mean Sensitivity': mSE_All[testi][number_net + 1][epoch][j],
                'mean Precision': mPC_All[testi][number_net + 1][epoch][j],
                'mean F2-score': mF2_All[testi][number_net + 1][epoch][j],
                'mean Specificity': mSP_All[testi][number_net + 1][epoch][j],
                'mean Accuracy': mACC_All[testi][number_net + 1][epoch][j],
                'epoch': epoch_All[number_net + 1][epoch]
            })

    for j in range(number_ensemble):
        experiment2[number_net + 2][j] = wandb.init(project='Ensemble_TrebleFormer_CVC-300', group=Ename,
                                                    name='EAP' + str(j),
                                                    resume='allow',
                                                    anonymous='must', reinit=True, allow_val_change=True,
                                                    id=Ename + 'EAP' + str(j))
        experiment2[number_net + 2][j].config.update(
            dict(epochs=copy.deepcopy(epochs_nets[number_net]), batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint,
                 img_scale=img_scale, amp=amp), allow_val_change=True)

        experiment2[number_net + 2][j].log({
            'Dice': Dice_pl[testi][j],
            'IoU': IoU_pl[testi][j],
            'Sensitivity': SE_pl[testi][j],
            'Precision': PC_pl[testi][j],
            'F2-score': F2_pl[testi][j],
            'Specificity': SP_pl[testi][j],
            'Accuracy': ACC_pl[testi][j],
            'mDice': mDice_pl[testi][j],
            'mIoU': mIoU_pl[testi][j],
            'mean Sensitivity': mSE_pl[testi][j],
            'mean Precision': mPC_pl[testi][j],
            'mean F2-score': mF2_pl[testi][j],
            'mean Specificity': mSP_pl[testi][j],
            'mean Accuracy': mACC_pl[testi][j],
            'epoch': (set_epochs + 1)
        })

    for j in range(number_ensemble):
        experiment2[number_net + 3][j] = wandb.init(project='Ensemble_TrebleFormer_CVC-300', group=Ename,
                                                    name='EBP' + str(j),
                                                    resume='allow',
                                                    anonymous='must', reinit=True, allow_val_change=True,
                                                    id=Ename + 'EBP' + str(j))
        experiment2[number_net + 3][j].config.update(
            dict(epochs=copy.deepcopy(epochs_nets[number_net]), batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint,
                 img_scale=img_scale, amp=amp), allow_val_change=True)

        experiment2[number_net + 3][j].log({
            'Dice': Dice_pbl[testi][j],
            'IoU': IoU_pbl[testi][j],
            'Sensitivity': SE_pbl[testi][j],
            'Precision': PC_pbl[testi][j],
            'F2-score': F2_pbl[testi][j],
            'Specificity': SP_pbl[testi][j],
            'Accuracy': ACC_pbl[testi][j],
            'mDice': mDice_pbl[testi][j],
            'mIoU': mIoU_pbl[testi][j],
            'mean Sensitivity': mSE_pbl[testi][j],
            'mean Precision': mPC_pbl[testi][j],
            'mean F2-score': mF2_pbl[testi][j],
            'mean Specificity': mSP_pbl[testi][j],
            'mean Accuracy': mACC_pbl[testi][j],
            'epoch': (set_epochs + 1)
        })

    for Epoch_Net in range(number_net):
        for epoch in range((set_epochs + 1)):
            for i in range(3):
                txt[Epoch_Net].write(
                    '\n' +
                    ' epoch=' + f'{epoch_All[Epoch_Net][epoch]}' +
                    ' learning rate=' + f'{lr_All[Epoch_Net][epoch]}' +
                    ' loss=' + f'{loss_All[Epoch_Net][epoch]}' +
                    '\n' +
                    f'now in test {i}' +
                    ' Dice=' + f'{Dice_All[i][Epoch_Net][epoch]}' +
                    ', IoU=' + f'{IoU_All[i][Epoch_Net][epoch]}' +
                    ', Sensitivity=' + f'{SE_All[i][Epoch_Net][epoch]}' +
                    ', Precision=' + f'{PC_All[i][Epoch_Net][epoch]}' +
                    ', F2=' + f'{F2_All[i][Epoch_Net][epoch]}' +
                    ', Specificity=' + f'{SP_All[i][Epoch_Net][epoch]}' +
                    ', Accuracy=' + f'{ACC_All[i][Epoch_Net][epoch]}' +
                    ', mDice=' + f'{mDice_All[i][Epoch_Net][epoch]}' +
                    ', mIoU=' + f'{mIoU_All[i][Epoch_Net][epoch]}' +
                    ', mean Sensitivity=' + f'{mSE_All[i][Epoch_Net][epoch]}' +
                    ', mean Precision=' + f'{mPC_All[i][Epoch_Net][epoch]}' +
                    ', mean F2=' + f'{mF2_All[i][Epoch_Net][epoch]}' +
                    ', mean Specificity=' + f'{mSP_All[i][Epoch_Net][epoch]}' +
                    ', mean Accuracy=' + f'{mACC_All[i][Epoch_Net][epoch]}')
        txt[Epoch_Net].close()

    for epoch in range((set_epochs + 1)):
        for i in range(3):
            txt[number_net].write(
                '\n' +
                ' epoch=' + f'{epoch_All[number_net][epoch]}' +
                '\n' +
                f'now in test {i}' +
                ' Dice=' + f'{Dice_All[i][number_net][epoch]}' +
                ', IoU=' + f'{IoU_All[i][number_net][epoch]}' +
                ', Sensitivity=' + f'{SE_All[i][number_net][epoch]}' +
                ', Precision=' + f'{PC_All[i][number_net][epoch]}' +
                ', F2=' + f'{F2_All[i][number_net][epoch]}' +
                ', Specificity=' + f'{SP_All[i][number_net][epoch]}' +
                ', Accuracy=' + f'{ACC_All[i][number_net][epoch]}' +
                ', mDice=' + f'{mDice_All[i][number_net][epoch]}' +
                ', mIoU=' + f'{mIoU_All[i][number_net][epoch]}' +
                ', mean Sensitivity=' + f'{mSE_All[i][number_net][epoch]}' +
                ', mean Precision=' + f'{mPC_All[i][number_net][epoch]}' +
                ', mean F2=' + f'{mF2_All[i][number_net][epoch]}' +
                ', mean Specificity=' + f'{mSP_All[i][number_net][epoch]}' +
                ', mean Accuracy=' + f'{mACC_All[i][number_net][epoch]}')
    txt[number_net].close()

    for epoch in range((set_epochs + 1)):
        for i in range(3):
            txt[number_net + 1].write(
                '\n' +
                ' epoch=' + f'{epoch_All[number_net + 1][epoch]}' +
                '\n' +
                f'now in test {i}' +
                ' Dice=' + f'{Dice_All[i][number_net + 1][epoch]}' +
                ', IoU=' + f'{IoU_All[i][number_net + 1][epoch]}' +
                ', Sensitivity=' + f'{SE_All[i][number_net + 1][epoch]}' +
                ', Precision=' + f'{PC_All[i][number_net + 1][epoch]}' +
                ', F2=' + f'{F2_All[i][number_net + 1][epoch]}' +
                ', Specificity=' + f'{SP_All[i][number_net + 1][epoch]}' +
                ', Accuracy=' + f'{ACC_All[i][number_net + 1][epoch]}' +
                ', mDice=' + f'{mDice_All[i][number_net + 1][epoch]}' +
                ', mIoU=' + f'{mIoU_All[i][number_net + 1][epoch]}' +
                ', mean Sensitivity=' + f'{mSE_All[i][number_net + 1][epoch]}' +
                ', mean Precision=' + f'{mPC_All[i][number_net + 1][epoch]}' +
                ', mean F2=' + f'{mF2_All[i][number_net + 1][epoch]}' +
                ', mean Specificity=' + f'{mSP_All[i][number_net + 1][epoch]}' +
                ', mean Accuracy=' + f'{mACC_All[i][number_net + 1][epoch]}')
    txt[number_net + 1].close()
    print('-----------------------------------------------')
    print('-----------------------------------------------')



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--Ensemable_name', '-Ename', type=str, default=None, help='name long must 4')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=list, default=[352, 352], help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--numb_net', '-nN', type=int, default=6, help='set number of net')
    parser.add_argument('--use_net1', '-n1', type=str, default=None, help='set net1 dir')
    parser.add_argument('--use_net2', '-n2', type=str, default=None, help='set net2 dir')
    parser.add_argument('--use_net3', '-n3', type=str, default=None, help='set net3 dir')
    parser.add_argument('--use_net4', '-n4', type=str, default=None, help='set net4 dir')
    parser.add_argument('--use_net5', '-n5', type=str, default=None, help='set net5 dir')
    parser.add_argument('--use_net6', '-n6', type=str, default=None, help='set net6 dir')
    parser.add_argument('--dir_net1', '-d1', type=str, default='./net1', help='choice use net')
    parser.add_argument('--dir_net2', '-d2', type=str, default='./net2', help='choice use net')
    parser.add_argument('--dir_net3', '-d3', type=str, default='./net3', help='choice use net')
    parser.add_argument('--dir_net4', '-d4', type=str, default='./net4', help='choice use net')
    parser.add_argument('--dir_net5', '-d5', type=str, default='./net5', help='choice use net')
    parser.add_argument('--dir_net6', '-d6', type=str, default='./net6', help='choice use net')
    parser.add_argument('--set_channels', '-in_c', type=int, default=3, help='n_channels use mod')
    parser.add_argument('--set_classes', '-out_c', type=int, default=2, help='n_classes use mod')
    parser.add_argument('--break_delay', '-bd', type=int, default=5, help='break_delay')

    return parser.parse_args()

def super_combine(midice):

    # nor_dice = [midice[0] / 0.950, midice[1] / 0.948, midice[2] / 0.944, midice[3] / 0.912, midice[4] / 0.942]
    nor_dice = [midice[0], midice[1], midice[2], midice[3], midice[4]]

    mid_nor_B = torch.tensor([nor_dice[0], nor_dice[1], nor_dice[2]]).float()
    mid_nor_C = torch.tensor([nor_dice[0], nor_dice[1], nor_dice[2], nor_dice[3], nor_dice[4]]).float()
    B_acc = 1
    C_acc = 1

    Combine_B = torch.tensor([mid_nor_B[0] * B_acc, mid_nor_B[1] * B_acc, mid_nor_B[2] * B_acc]).float()
    Combine_C = torch.tensor([mid_nor_C[0] * C_acc, mid_nor_C[1] * C_acc, mid_nor_C[2] * C_acc,
                            mid_nor_C[3] * C_acc, mid_nor_C[4] * C_acc]).float()

    Weight_A = torch.tensor([1.0])
    Weight_B = F.softmax(Combine_B, dim=0)
    Weight_C = F.softmax(Combine_C, dim=0)

    Weight = torch.tensor([Weight_C[0] + Weight_B[0] + Weight_A[0], Weight_C[1] + Weight_B[1], Weight_C[2] + Weight_B[2], Weight_C[3], Weight_C[4]])

    return Weight[0], Weight[1], Weight[2], Weight[3], Weight[4]


def PSO_Ensemble(pso_epoch, pso_sumpop, pso_lr, init, wv, Ensemable_popfits_masks, val_loader, nets_classes, number_net, label, label_threshold, elimate):

    if init:
        pop_site = torch.normal(mean=0.1666, std=0.5, size=(pso_sumpop, number_net))
        pop_site_sum = torch.sum(pop_site, dim=1)
        pop_site_another = torch.ones((number_net))
        pop_site_other, _ = torch.meshgrid(pop_site_sum, pop_site_another)
        pop_site = pop_site / pop_site_other
        pop_site = pop_site.numpy()

        if number_net == 6:
            pop_site[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            pop_site[1] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            pop_site[2] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            pop_site[3] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            pop_site[4] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            pop_site[5] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            pop_site[6] = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
            pop_site[7] = [0.2, 0.0, 0.2, 0.2, 0.2, 0.2]
            pop_site[8] = [0.2, 0.2, 0.0, 0.2, 0.2, 0.2]
            pop_site[9] = [0.2, 0.2, 0.2, 0.0, 0.2, 0.2]
            pop_site[10] = [0.2, 0.2, 0.2, 0.2, 0.0, 0.2]
            pop_site[11] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
        elif number_net == 5:
            pop_site[0] = [1.00, 0.00, 0.00, 0.00, 0.00]
            pop_site[1] = [0.00, 1.00, 0.00, 0.00, 0.00]
            pop_site[2] = [0.00, 0.00, 1.00, 0.00, 0.00]
            pop_site[3] = [0.00, 0.00, 0.00, 1.00, 0.00]
            pop_site[4] = [0.00, 0.00, 0.00, 0.00, 1.00]
            pop_site[5] = [0.00, 0.25, 0.25, 0.25, 0.25]
            pop_site[6] = [0.25, 0.00, 0.25, 0.25, 0.25]
            pop_site[7] = [0.25, 0.25, 0.00, 0.25, 0.25]
            pop_site[8] = [0.25, 0.25, 0.25, 0.00, 0.25]
            pop_site[9] = [0.25, 0.25, 0.25, 0.25, 0.00]
        elif number_net == 4:
            pop_site[0] = [1.0, 0.0, 0.0, 0.0]
            pop_site[1] = [0.0, 1.0, 0.0, 0.0]
            pop_site[2] = [0.0, 0.0, 1.0, 0.0]
            pop_site[3] = [0.0, 0.0, 0.0, 1.0]
            pop_site[4] = [0.0, 0.33, 0.34, 0.33]
            pop_site[5] = [0.33, 0.0, 0.33, 0.34]
            pop_site[6] = [0.34, 0.33, 0.0, 0.33]
            pop_site[7] = [0.33, 0.34, 0.33, 0.0]
        elif number_net == 3:
            pop_site[0] = [1.0, 0.0, 0.0]
            pop_site[1] = [0.0, 1.0, 0.0]
            pop_site[2] = [0.0, 0.0, 1.0]
            pop_site[3] = [0.0, 0.5, 0.5]
            pop_site[4] = [0.5, 0.0, 0.5]
            pop_site[5] = [0.5, 0.5, 0.0]
        elif number_net == 2:
            pop_site[0] = [1.0, 0.0]
            pop_site[1] = [0.0, 1.0]
            pop_site[2] = [0.5, 0.5]

        pop_v = np.zeros((pso_sumpop, number_net))

        gbestpop = np.zeros(number_net)
        pbestpop = np.zeros((pso_sumpop, number_net))
        gbestfitness = float(0)
        pbestfitness = np.zeros(pso_sumpop)
        pso_resultF = np.zeros(pso_epoch)
        pso_resultS = np.array([np.zeros(number_net)] * pso_epoch)
        pso_t = 0.75
        pop_fitness = [float(0.01)] * pso_sumpop
        init = False

    for pso_i in range(pso_epoch):
        # 速度更新
        # for fig in range(pso_sumpop):
        #     pop_score_tensor, _, _, _, _, _, _ = evaluate_pop(net=nets, dataloader=train_loader, device=device, number=number_net,
        #                                  onlyone_pop=pop_site[fig], mask_pred_size=masks_pred_size, fig_now=pso_i, pop_now=pso_j, batch_size=batch_size)
        #     pop_score_numpy = pop_score_tensor.data.cpu().numpy()
        #     pop_fitness.insert(fig, pop_score_numpy)
        if not pso_i == 0:
            for pso_j in range(pso_sumpop):
                pop_v[pso_j] = wv * pop_v[pso_j] + pso_lr[0] * np.random.rand() * (pbestpop[pso_j] - pop_site[pso_j]) + \
                               pso_lr[1] * np.random.rand() * (gbestpop - pop_site[pso_j])
            # pop_v[pop_v < pso_rangespeed[0]] = pso_rangespeed[0]
            # pop_v[pop_v > pso_rangespeed[1]] = pso_rangespeed[1]

            # 粒子位置更新
            for pso_j in range(pso_sumpop):
                # pso_pop[pso_j] += 0.5*pso_v[pso_j]
                pop_site[pso_j] += pso_t * pop_v[pso_j]
            # pop_size[pop_size < pso_rangepop[0]] = pso_rangepop[0]
            # pop_size[pop_size > pso_rangepop[1]] = pso_rangepop[1]

        # 适应度更新
        if elimate:
            pop_fitness = evaluate_popfit_label(preloader=Ensemable_popfits_masks, dataloader=val_loader,
                                                number_net=number_net, label=label,
                                                poplist=pop_site, threshold=label_threshold,
                                                popsum=pso_sumpop, net_n_classes=nets_classes, device='cpu')
        else:
            pop_fitness = evaluate_popfit(preloader=Ensemable_popfits_masks, dataloader=val_loader,
                                          number_net=number_net,
                                          poplist=pop_site,
                                          popsum=pso_sumpop, net_n_classes=nets_classes, device='cpu')

        print('pso_i', pso_i)
        print('pop_fitness', pop_fitness)
        print('pop_site', pop_site)

        for pso_j in range(pso_sumpop):
            if pop_fitness[pso_j] > pbestfitness[pso_j]:
                pbestfitness[pso_j] = pop_fitness[pso_j]
                pbestpop[pso_j] = copy.deepcopy(pop_site[pso_j])

        if pbestfitness.max() > gbestfitness:
            gbestfitness = pbestfitness.max()
            gbestpop = copy.deepcopy(pop_site[pbestfitness.argmax()])

        pso_resultF[pso_i] = gbestfitness
        pso_resultS[pso_i] = gbestpop
        print('gbestfitness=', gbestfitness)
        print('gbestpop=', gbestpop)
    return pso_resultF, pso_resultS, gbestpop

def SDBHPSO_Ensemble(pso_epoch, pso_sumpop, pso_lr, init, Ensemable_popfits_masks, val_loader, nets_classes, number_net, label, label_threshold, elimate):

    if init:
        wv = 0
        setr = 0.2
        setp = 0
        setl = 1
        thes = np.zeros(number_net)
        theTh = 5
        pop_C = 0
        pop_sita = 1
        theH = 3
        pop_sim = [None] * pso_sumpop
        aaa = 0

        pop_site = torch.normal(mean=0.1666, std=0.5, size=(pso_sumpop, number_net))
        pop_site_sum = torch.sum(pop_site, dim=1)
        pop_site_another = torch.ones((number_net))
        pop_site_other, _ = torch.meshgrid(pop_site_sum, pop_site_another)
        pop_site = pop_site / pop_site_other
        pop_site = pop_site.numpy()

        if number_net == 6:
            pop_site[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            pop_site[1] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            pop_site[2] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            pop_site[3] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            pop_site[4] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            pop_site[5] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            pop_site[6] = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
            pop_site[7] = [0.2, 0.0, 0.2, 0.2, 0.2, 0.2]
            pop_site[8] = [0.2, 0.2, 0.0, 0.2, 0.2, 0.2]
            pop_site[9] = [0.2, 0.2, 0.2, 0.0, 0.2, 0.2]
            pop_site[10] = [0.2, 0.2, 0.2, 0.2, 0.0, 0.2]
            pop_site[11] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
        elif number_net == 5:
            pop_site[0] = [1.00, 0.00, 0.00, 0.00, 0.00]
            pop_site[1] = [0.00, 1.00, 0.00, 0.00, 0.00]
            pop_site[2] = [0.00, 0.00, 1.00, 0.00, 0.00]
            pop_site[3] = [0.00, 0.00, 0.00, 1.00, 0.00]
            pop_site[4] = [0.00, 0.00, 0.00, 0.00, 1.00]
            pop_site[5] = [0.00, 0.25, 0.25, 0.25, 0.25]
            pop_site[6] = [0.25, 0.00, 0.25, 0.25, 0.25]
            pop_site[7] = [0.25, 0.25, 0.00, 0.25, 0.25]
            pop_site[8] = [0.25, 0.25, 0.25, 0.00, 0.25]
            pop_site[9] = [0.25, 0.25, 0.25, 0.25, 0.00]
        elif number_net == 4:
            pop_site[0] = [1.0, 0.0, 0.0, 0.0]
            pop_site[1] = [0.0, 1.0, 0.0, 0.0]
            pop_site[2] = [0.0, 0.0, 1.0, 0.0]
            pop_site[3] = [0.0, 0.0, 0.0, 1.0]
            pop_site[4] = [0.0, 0.33, 0.34, 0.33]
            pop_site[5] = [0.33, 0.0, 0.33, 0.34]
            pop_site[6] = [0.34, 0.33, 0.0, 0.33]
            pop_site[7] = [0.33, 0.34, 0.33, 0.0]
        elif number_net == 3:
            pop_site[0] = [1.0, 0.0, 0.0]
            pop_site[1] = [0.0, 1.0, 0.0]
            pop_site[2] = [0.0, 0.0, 1.0]
            pop_site[3] = [0.0, 0.5, 0.5]
            pop_site[4] = [0.5, 0.0, 0.5]
            pop_site[5] = [0.5, 0.5, 0.0]
        elif number_net == 2:
            pop_site[0] = [1.0, 0.0]
            pop_site[1] = [0.0, 1.0]
            pop_site[2] = [0.5, 0.5]

        pop_v = np.zeros((pso_sumpop, number_net))

        gbestpop = np.zeros(number_net)
        pbestpop = np.zeros((pso_sumpop, number_net))
        gbestfitness = float(0)
        pbestfitness = np.zeros(pso_sumpop)
        pso_resultF = np.zeros(pso_epoch)
        pso_resultS = np.array([np.zeros(number_net)] * pso_epoch)
        pso_t = 1
        pop_fitness = [float(0.01)] * pso_sumpop
        init = False

    for pso_i in range(pso_epoch):
        # 速度更新
        # for fig in range(pso_sumpop):
        #     pop_score_tensor, _, _, _, _, _, _ = evaluate_pop(net=nets, dataloader=train_loader, device=device, number=number_net,
        #                                  onlyone_pop=pop_site[fig], mask_pred_size=masks_pred_size, fig_now=pso_i, pop_now=pso_j, batch_size=batch_size)
        #     pop_score_numpy = pop_score_tensor.data.cpu().numpy()
        #     pop_fitness.insert(fig, pop_score_numpy)
        if not pso_i == 0:
            for pso_j in range(pso_sumpop):
                wv = 0.9 - (0.9 - 0.4) * ((pso_i + 1) / pso_epoch)
                pop_v[pso_j] = wv * pop_v[pso_j] + pso_lr[0] * np.random.rand() * (pbestpop[pso_j] - pop_site[pso_j]) + \
                               pso_lr[
                                   1] * np.random.rand() * (gbestpop - pop_site[pso_j])
            # pop_v[pop_v < pso_rangespeed[0]] = pso_rangespeed[0]
            # pop_v[pop_v > pso_rangespeed[1]] = pso_rangespeed[1]

            # 粒子位置更新
            for pso_j in range(pso_sumpop):
                # pso_pop[pso_j] += 0.5*pso_v[pso_j]
                if theH * pop_sita < pop_C * pop_sim[pso_j]:
                    max_fitness = max(pop_fitness)
                    max_fit_site = pop_fitness.index(max_fitness)
                    # if (pop_site[pso_j][0] == gbestpop[0]) and (pop_site[pso_j][1] == gbestpop[1]) and \
                    #         (pop_site[pso_j][2] == gbestpop[2]) and (pop_site[pso_j][3] == gbestpop[3]) and \
                    #         (pop_site[pso_j][4] == gbestpop[4]) and (pop_site[pso_j][5] == gbestpop[5]):
                    if pso_j == max_fit_site:
                        pop_site[pso_j] = gbestpop
                        aaa += 1
                    else:
                        pop_site[pso_j] = gbestpop + thes
                        pop_v[pso_j] = Revv(pop_v[pso_j], 1, 1.0, 0.1, (pso_i + 1), pso_epoch)
                else:
                    pop_site[pso_j] += pso_t * pop_v[pso_j]

                Rand = np.random.rand()

            # pop_size[pop_size < pso_rangepop[0]] = pso_rangepop[0]
            # pop_size[pop_size > pso_rangepop[1]] = pso_rangepop[1]

        # 适应度更新
        for i in range(pso_sumpop):
            # pop_fitness[i] = 10 - (Odis(pop_site[i], X) + Odis(pop_site[i], Y))
            if elimate:
                pop_fitness = evaluate_popfit_label(preloader=Ensemable_popfits_masks, dataloader=val_loader,
                                                    number_net=number_net, label=label,
                                                    poplist=pop_site, threshold=label_threshold,
                                                    popsum=pso_sumpop, net_n_classes=nets_classes, device='cpu')
            else:
                pop_fitness = evaluate_popfit(preloader=Ensemable_popfits_masks, dataloader=val_loader,
                                              number_net=number_net,
                                              poplist=pop_site,
                                              popsum=pso_sumpop, net_n_classes=nets_classes, device='cpu')

        print('pso_i', pso_i)
        print('pop_fitness', pop_fitness)
        print('pop_site', pop_site)

        for pso_j in range(pso_sumpop):
            if pop_fitness[pso_j] > pbestfitness[pso_j]:
                pbestfitness[pso_j] = pop_fitness[pso_j]
                pbestpop[pso_j] = copy.deepcopy(pop_site[pso_j])

        if pbestfitness.max() > gbestfitness:
            gbestfitness = pbestfitness.max()
            gbestpop = copy.deepcopy(pop_site[pbestfitness.argmax()])

        if number_net == 6:
            thes = np.array([(np.random.rand() * 2 - 1) * setr, (np.random.rand() * 2 - 1) * setr,
                             (np.random.rand() * 2 - 1) * setr, (np.random.rand() * 2 - 1) * setr,
                             (np.random.rand() * 2 - 1) * setr, (np.random.rand() * 2 - 1) * setr])
        elif number_net == 5:
            thes = np.array([(np.random.rand() * 2 - 1) * setr, (np.random.rand() * 2 - 1) * setr,
                             (np.random.rand() * 2 - 1) * setr, (np.random.rand() * 2 - 1) * setr,
                             (np.random.rand() * 2 - 1) * setr])
        elif number_net == 4:
            thes = np.array([(np.random.rand() * 2 - 1) * setr, (np.random.rand() * 2 - 1) * setr,
                             (np.random.rand() * 2 - 1) * setr, (np.random.rand() * 2 - 1) * setr])
        elif number_net == 3:
            thes = np.array([(np.random.rand() * 2 - 1) * setr, (np.random.rand() * 2 - 1) * setr,
                             (np.random.rand() * 2 - 1) * setr])
        elif number_net == 2:
            thes = np.array([(np.random.rand() * 2 - 1) * setr, (np.random.rand() * 2 - 1) * setr])

        for i in range(pso_sumpop):
            pop_sim[i] = sim(pop_site[i], gbestpop)
        pop_C = theC(pop_sim, pso_sumpop)
        pop_sita = sita(pso_epoch, (pso_i + 1), 1, 0.9, 0.4)

        pso_resultF[pso_i] = gbestfitness
        pso_resultS[pso_i] = gbestpop
        print('gbestfitness=', gbestfitness)
        print('gbestpop=', gbestpop)
        print('print(aaa)print(aaa)print(aaa)=========', aaa)

    return pso_resultF, pso_resultS, gbestpop



def sim(A, B):
    return (np.cov(A, B)[0][1]) / ((np.var(A) ** 0.5) * (np.var(B) ** 0.5))

def theC(sim, sum):
    sim_sum = 0
    sim_sum2 = 0
    for i in range(sum):
        sim_sum += sim[i]
    for j in range(sum):
        sim_sum2 += (sim[j] * sim[j]) / sim_sum
    return sim_sum2

def sita(k_max, k, s, ld_max, ld_min):
    return (((k_max - k) / k_max) ** s) * (ld_max - ld_min) + ld_min

def Revv(thex, Pb, sita_max, sita_min, t, T):
    return (np.max(thex) - np.min(thex)) * Pb * np.random.rand() * np.random.normal(loc=0, scale=((sita_max - sita_min) * (t / T)), size=1)

def Resite(thex, Pb, sita_max, sita_min, t, T):
    return (np.max(thex) - np.min(thex)) * Pb * np.random.rand() * np.random.normal(loc=0, scale=((sita_max - sita_min) * (t / T)), size=1)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    set_dir_img = Path('./data/train_and_val/Kvasir_and_CVC-ClinicDB/imgs/')
    set_dir_mask = Path('./data/train_and_val/Kvasir_and_CVC-ClinicDB/masks/')
    set_dir_img0 = Path('./data/test/ETIS-LaribPolypDB/imgs/')
    set_dir_mask0 = Path('./data/test/ETIS-LaribPolypDB/masks/')
    set_dir_img1 = Path('./data/test/CVC-ColonDB/imgs/')
    set_dir_mask1 = Path('./data/test/CVC-ColonDB/masks/')
    set_dir_img2 = Path('./data/test/CVC-300/imgs/')
    set_dir_mask2 = Path('./data/test/CVC-300/masks/')
    number_net = args.numb_net
    set_dir_checkpoint = []
    set_nets_names = []
    set_nets = []
    for i in range(number_net):
        if i == 0:
            dir_net = args.dir_net1
            net_name = 'net1'
            use_net = args.use_net1
        elif i == 1:
            dir_net = args.dir_net2
            net_name = 'net2'
            use_net = args.use_net2
        elif i == 2:
            dir_net = args.dir_net3
            net_name = 'net3'
            use_net = args.use_net3
        elif i == 3:
            dir_net = args.dir_net4
            net_name = 'net4'
            use_net = args.use_net4
        elif i == 4:
            dir_net = args.dir_net5
            net_name = 'net5'
            use_net = args.use_net5
        elif i == 5:
            dir_net = args.dir_net6
            net_name = 'net6'
            use_net = args.use_net6
        else:
            assert i > 5, \
                f'Net must be set >  {i + 1} '

        set_dir_checkpoint.insert(i, Path(f'./checkpoints/{dir_net}/'))
        set_nets_names.insert(i, net_name)
        set_nets.insert(i, SemSegNet(n_channels=args.set_channels,
                                     n_classes=args.set_classes,
                                     what_Net=use_net))
        logging.info(f'Network:\n'
                     f'\t{set_nets[i].n_channels} input channels\n'
                     f'\t{set_nets[i].n_classes} output channels (classes)\n')
        if args.load:
            set_nets[i].load_state_dict(torch.load(os.path.join(set_dir_checkpoint[i], args.load), map_location=device))
            logging.info(f'Model loaded from {os.path.join(set_dir_checkpoint[i], args.load)}')
        set_nets[i].to(device=device)

    try:
        train_net(Ensemable_name=args.Ensemable_name,
                  set_epochs=args.epochs,
                  number_net=number_net,
                  set_nets_name=set_nets_names,
                  set_net=set_nets,
                  set_dir_checkpoint=set_dir_checkpoint,
                  dir_img=set_dir_img,
                  dir_mask=set_dir_mask,
                  dir_img0=set_dir_img0,
                  dir_mask0=set_dir_mask0,
                  dir_img1=set_dir_img1,
                  dir_mask1=set_dir_mask1,
                  dir_img2=set_dir_img2,
                  dir_mask2=set_dir_mask2,
                  nets_classes=args.set_classes,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  break_delay=args.break_delay)
    except KeyboardInterrupt:
        for i in range(number_net):
            torch.save(set_nets[i].state_dict(), set_nets_names[i] + 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
