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
from evaluate import evaluate_val, evaluate_test, evaluate_ensemble, evaluate_popfit, Choice_best, evaluate_ensembleA_valpre, evaluate_ensembleA_test, evaluate_popfit_label, evaluate_ensemble_label, evaluate_ensemble_l, evaluate_popfit_pbl, evaluate_ensemble_llable, evaluate_ensemble_val
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
    # val_percent: float = 0.2
    # test_percent: float = 0.2
    set_epochs = int(set_epochs)
    Ename = Ensemable_name

    number_ensemble = 5
    gbestpop = [None] * number_ensemble
    gbestpop_label = [None] * number_ensemble

    pso_result_Fall1 = []
    pso_result_Fall2 = []
    pso_result_Fall3 = []
    pso_result_Fall4 = []
    pso_result_Fall5 = []

    pso_result_FallB1 = []
    pso_result_FallB2 = []
    pso_result_FallB3 = []
    pso_result_FallB4 = []
    pso_result_FallB5 = []

    pso_result_FallC = []

    pso_result_Sall1 = []
    pso_result_Sall2 = []
    pso_result_Sall3 = []
    pso_result_Sall4 = []
    pso_result_Sall5 = []

    pso_result_SallB1 = []
    pso_result_SallB2 = []
    pso_result_SallB3 = []
    pso_result_SallB4 = []
    pso_result_SallB5 = []

    pso_result_SallC = []

    assert len(Ename) == int(4), \
        'the long of Ensemable_name must 4.'
    pso_result_all = []
    pso_result_allB = []
    pso_result_allC = []
    ornot = [True, True, False, False, False, False]
    time_pso = 0
    first = True
    second = True
    third = True
    forth = True
    fiveth = True
    epoch_nets = [int(-1)] * (number_net + number_ensemble)
    epochs_nets = [epochs] * (number_net + number_ensemble)
    learning_rate_nets = [learning_rate] * number_net
    break_delay_epoch = int(0)
    break_allow = [bool(False)] * number_net
    break_epoch = [int(0)] * number_net
    continue_epoch = [int(0)] * number_net
    val_score = [None] * (number_net + number_ensemble)
    Dice = [None] * (number_net + number_ensemble)
    IoU = [None] * (number_net + number_ensemble)
    SE = [None] * (number_net + number_ensemble)
    PC = [None] * (number_net + number_ensemble)
    F2 = [None] * (number_net + number_ensemble)
    SP = [None] * (number_net + number_ensemble)
    ACC = [None] * (number_net + number_ensemble)
    mDice = [None] * (number_net + number_ensemble)
    mIoU = [None] * (number_net + number_ensemble)
    mSE = [None] * (number_net + number_ensemble)
    mPC = [None] * (number_net + number_ensemble)
    mF2 = [None] * (number_net + number_ensemble)
    mSP = [None] * (number_net + number_ensemble)
    mACC = [None] * (number_net + number_ensemble)
    val = [0.0] * (number_net + number_ensemble)
    val_best = [] * (number_net + number_ensemble)
    valbest = [0] * 17
    midice = [torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])] * number_net
    st = int(8)
    label = [int(0)] * number_net
    label_threshold = int(3)
    number_label = int(0)
    # midice = [midice, midice, midice, midice, midice, midice]
    break_allow_A = False
    break_allow_B = False
    w_ustm = [1.5] * number_net
    w_ust = [1.0] * number_net
    w_m = [1.0] * number_net
    w_us = [0.15] * number_net
    w_t = [0.35] * number_net
    best_vals = [None] * (number_net + number_ensemble)
    best_test = [None] * (number_net + number_ensemble)
    bestest = [None] * (number_net + number_ensemble)
    for i in range(number_net + number_ensemble):
        val_best.insert(i, copy.deepcopy(valbest))
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

    Dice_All = [] * (number_net + number_ensemble)
    IoU_All = [] * (number_net + number_ensemble)
    SE_All = [] * (number_net + number_ensemble)
    PC_All = [] * (number_net + number_ensemble)
    F2_All = [] * (number_net + number_ensemble)
    SP_All = [] * (number_net + number_ensemble)
    ACC_All = [] * (number_net + number_ensemble)
    mDice_All = [] * (number_net + number_ensemble)
    mIoU_All = [] * (number_net + number_ensemble)
    mSE_All = [] * (number_net + number_ensemble)
    mPC_All = [] * (number_net + number_ensemble)
    mF2_All = [] * (number_net + number_ensemble)
    mSP_All = [] * (number_net + number_ensemble)
    mACC_All = [] * (number_net + number_ensemble)
    lr_All = [] * (number_net + number_ensemble)
    loss_All = [] * (number_net + number_ensemble)
    epoch_All = [] * (number_net + number_ensemble)

    for i in range(number_net + number_ensemble):
        Dice_All.insert(i, copy.deepcopy(Dice_One))
        IoU_All.insert(i, copy.deepcopy(IoU_One))
        SE_All.insert(i, copy.deepcopy(SE_One))
        PC_All.insert(i, copy.deepcopy(PC_One))
        F2_All.insert(i, copy.deepcopy(F2_One))
        SP_All.insert(i, copy.deepcopy(SP_One))
        ACC_All.insert(i, copy.deepcopy(ACC_One))
        mDice_All.insert(i, copy.deepcopy(mDice_One))
        mIoU_All.insert(i, copy.deepcopy(mIoU_One))
        mSE_All.insert(i, copy.deepcopy(mSE_One))
        mPC_All.insert(i, copy.deepcopy(mPC_One))
        mF2_All.insert(i, copy.deepcopy(mF2_One))
        mSP_All.insert(i, copy.deepcopy(mSP_One))
        mACC_All.insert(i, copy.deepcopy(mACC_One))
        lr_All.insert(i, copy.deepcopy(lr_One))
        loss_All.insert(i, copy.deepcopy(loss_One))
        epoch_All.insert(i, copy.deepcopy(epoch_One))

    for i in range(10):
        Ensemable_pred_masks.insert(i, copy.deepcopy(Ensemable_pred_mask))
    for i in range(number_net):
        Ensemable_popfits_masks.insert(i, copy.deepcopy(Ensemable_popfits_mask))

    # 2. Create dataset

    list_img = listdir(dir_img)
    list_mask = listdir(dir_mask)
    n_val = int(len(list_img) * val_percent)
    n_test = int(len(list_img) * test_percent)
    n_train = int(len(list_img) - n_val - n_test)
    print('n_train=', n_train, 'n_val=', n_val, 'n_test=', n_test)
    train_img_set, val_img_set, test_img_set = random_split(list_img, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))
    train_mask_set, val_mask_set, test_mask_set = random_split(list_mask, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))
    train_set = Dataset_Pro(dir_img, dir_mask, train_img_set, train_mask_set, img_scale, augmentations=True)
    val_set = Dataset_Pro(dir_img, dir_mask, val_img_set, val_mask_set, img_scale, augmentations=False)
    test_set = Dataset_Pro(dir_img, dir_mask, test_img_set, test_mask_set, img_scale, augmentations=False)
    # n_val = len(val_img_set)
    # n_test = len(test_img_set)
    # n_train = len(train_img_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

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
        scheduler[i] = optim.lr_scheduler.ReduceLROnPlateau(optimizer[i], 'max', factor=0.5, min_lr=1e-6, patience=5)
        # scheduler[i] = optim.lr_scheduler.ReduceLROnPlateau(optimizer[i], 'max', factor=0.5, min_lr=1e-6, patience=10, cooldown=5)
        # scheduler[i] = optim.lr_scheduler.ReduceLROnPlateau(optimizer[i], 'max', factor=0.5, min_lr=1e-6, patience=2)
        # scheduler[i] = optim.lr_scheduler.ReduceLROnPlateau(optimizer1[i], mode='min', factor=0.5, min_lr=1e-6, patience=400)
    grad_scaler = [torch.cuda.amp.GradScaler(enabled=amp)] * number_net
    criterion = [nn.CrossEntropyLoss()] * number_net
    global_step = [int(0)] * (number_net + number_ensemble)
    loss = [None] * number_net
    epoch_loss = [float(0)] * number_net
    histograms = [{}] * (number_net + 1)
    experiment = [None] * (number_net + 6)
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
    for TIME in range(1):
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

                        w_ustm[Epoch_Net], w_ust[Epoch_Net], w_m[Epoch_Net], w_us[Epoch_Net], w_t[
                            Epoch_Net] = super_combine(midice[Epoch_Net])

                        val_score[Epoch_Net], sdice1, sdice2, sdice3, sdice4, sdice5, Ensemable_popfits_masks[
                            Epoch_Net], best_vals[Epoch_Net] = evaluate_val(
                            nets[Epoch_Net], val_loader, device, w_ustm[Epoch_Net], w_ust[Epoch_Net],
                            w_m[Epoch_Net], w_us[Epoch_Net], w_t[Epoch_Net], Ensemable_popfits_masks[Epoch_Net], ornot[Epoch_Net])

                        Dice[Epoch_Net], IoU[Epoch_Net], PC[Epoch_Net], SE[Epoch_Net], \
                            SP[Epoch_Net], ACC[Epoch_Net], F2[Epoch_Net], \
                        mDice[Epoch_Net], mIoU[Epoch_Net], mPC[Epoch_Net], mSE[Epoch_Net], \
                            mSP[Epoch_Net], mACC[Epoch_Net], mF2[Epoch_Net], \
                            Ensemable_pred_masks[epoch_stage][Epoch_Net], best_test[Epoch_Net] = evaluate_test(nets[Epoch_Net],
                                                                                                   test_loader,
                                                                                                   device,
                                                                                                   w_ustm[Epoch_Net],
                                                                                                   w_ust[Epoch_Net],
                                                                                                   w_m[Epoch_Net],
                                                                                                   w_us[Epoch_Net],
                                                                                                   w_t[Epoch_Net],
                                                                                                   ornot[Epoch_Net])
                        midice[Epoch_Net] = [sdice1, sdice2, sdice3, sdice4, sdice5]


                        logging.info('Validation Dice score: {}'.format(val_score[Epoch_Net]))

                        Dice_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = Dice[Epoch_Net]
                        IoU_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = IoU[Epoch_Net]
                        SE_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = SE[Epoch_Net]
                        PC_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = PC[Epoch_Net]
                        F2_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = F2[Epoch_Net]
                        SP_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = SP[Epoch_Net]
                        ACC_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = ACC[Epoch_Net]
                        mDice_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = mDice[Epoch_Net]
                        mIoU_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = mIoU[Epoch_Net]
                        mSE_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = mSE[Epoch_Net]
                        mPC_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = mPC[Epoch_Net]
                        mF2_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = mF2[Epoch_Net]
                        mSP_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = mSP[Epoch_Net]
                        mACC_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = mACC[Epoch_Net]
                        lr_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = optimizer[Epoch_Net].param_groups[0]['lr']
                        epoch_All[Epoch_Net][epoch_nets[Epoch_Net] + 1] = (epoch_nets[Epoch_Net] + 1)

                        val[Epoch_Net] = [val_score[Epoch_Net], Dice[Epoch_Net], IoU[Epoch_Net], SE[Epoch_Net], PC[Epoch_Net],
                                          F2[Epoch_Net], SP[Epoch_Net], ACC[Epoch_Net],
                                          mDice[Epoch_Net], mIoU[Epoch_Net], mSE[Epoch_Net], mPC[Epoch_Net],
                                          mF2[Epoch_Net], mSP[Epoch_Net], mACC[Epoch_Net], best_vals[Epoch_Net]]
                        val_best[Epoch_Net], bestest[Epoch_Net] = Choice_best(val=val[Epoch_Net],
                                                                              val_best=val_best[Epoch_Net],
                                                                              bestest=bestest[Epoch_Net],
                                                                              best_test=best_test[
                                                                                  Epoch_Net],
                                                                              Epoch=epoch_nets[Epoch_Net])

                        print('\n' + 'out_USTM.dice=', midice[Epoch_Net][0])
                        print('out_UST.dice=', midice[Epoch_Net][1])
                        print('out_MB.dice=', midice[Epoch_Net][2])
                        print('out_USB.dice=', midice[Epoch_Net][3])
                        print('out_TB.dice=', midice[Epoch_Net][4])
                        print('w_ustm=', w_ustm[Epoch_Net], 'w_ust=', w_ust[Epoch_Net], 'w_m=', w_m[Epoch_Net],
                              'w_us=', w_us[Epoch_Net], 'w_t=', w_t[Epoch_Net])
                        print('\n', 'val_score=', val_score[Epoch_Net], 'dice=', Dice[Epoch_Net], 'mdice=',
                              mDice[Epoch_Net], 'miou=',
                              mIoU[Epoch_Net])
                        print('\n', 'label=', label, 'kaiguan=', kaiguan)

                        txt_dice[Epoch_Net].write(
                            f'{epoch_nets[Epoch_Net]}' + '\n' +
                            ' out_USTM.dice=' + f'{midice[Epoch_Net][1]}' +
                            ', out_UST.dice=' + f'{midice[Epoch_Net][1]}' +
                            ', out_MB.dice=' + f'{midice[Epoch_Net][2]}' +
                            ', out_USB.dice=' + f'{midice[Epoch_Net][3]}' +
                            ', out_TB.dice=' + f'{midice[Epoch_Net][4]}')
                        txt_w[Epoch_Net].write(
                            f'{epoch_nets[Epoch_Net]}' + '\n' +
                            ' w_ustm=' + f'{w_ustm[Epoch_Net]}' +
                            ' w_ust=' + f'{w_ust[Epoch_Net]}' +
                            ' w_m=' + f'{w_m[Epoch_Net]}' +
                            ' w_us=' + f'{w_us[Epoch_Net]}' +
                            ' w_t=' + f'{w_t[Epoch_Net]}')


            # 5.1.2  training ensemble net in stage1
            print('-----------------------------------------------')
            print(f'stage{TIME} PSO Train')
            print('-----------------------------------------------')

            use_popa = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
            use_pop = [use_popa, use_popa, use_popa, use_popa, use_popa]

            # pso_epoch = int(2)
            # pso_sumpop = int(5)
            pso_epoch = int(5)
            pso_sumpop = int(50)
            pso_resultF1 = 0
            pso_resultS1 = 0
            pso_resultF2 = pso_resultF1
            pso_resultS2 = pso_resultS1
            pso_resultF3 = pso_resultF1
            pso_resultS3 = pso_resultS1
            pso_resultF4 = pso_resultF1
            pso_resultS4 = pso_resultS1
            pso_resultF5 = pso_resultF1
            pso_resultS5 = pso_resultS1

            use_pop[0] = np.array([0.25773975, 0.05107775, 0.3191432, 0.08404478, 0.13840428, 0.14959034])
            use_pop[1] = np.array([0.34339866,  0.22607283,  0.43092674, -0.13763405,  0.45103046, 0.2579245])
            use_pop[2] = np.array([0.28048366,  0.21734571, -0.1033151,  0.24061209,  0.36171758, 0.03888487])
            use_pop[3] = np.array([0.21356659,  0.09075797,  0.16536127,  0.3431381, -0.00235412, 0.18953012])

            # use_pop[0] = np.array([-0.06989957,  0.7410104, -0.22923596,  0.94014084,  0.3596877, -0.7417033])
            # use_pop[1] = np.array([0.44165698,  0.9687021,  0.06118016,  1.2040088, -1.0793492, -0.7364761])
            # use_pop[2] = np.array([-0.3164617,  0.8512927,  0.01444845,  0.52307904,  0.21075408, -0.28311265])
            # use_pop[3] = np.array([0.22905988,  0.6023128, -0.1478048,  0.84962976,  0.07905126, -0.6122489])

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
                'now is the 4 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS4}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF4}' + '\n' +
                'now is the 5 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS5}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF5}' + '\n' +
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
                'now is the 4 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS4}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF4}' + '\n' +
                'now is the 5 :' + '\n' +
                ' pso_result_site=' + f'{pso_resultS5}' + '\n' +
                ' pso_result_fite=' + f'{pso_resultF5}' + '\n' +
                ', label=' + f'{label}' +
                '\n' + '------------' + '\n')

            print('PSO one over')

            print('-----------------------------------------------')
            print(f'stage{TIME} Ensemable Start')
            print('-----------------------------------------------')

            for e in range(10):
                epoch_nets[number_net] += 1
                for j in range(number_ensemble):
                    best_vals[number_net + j] = evaluate_ensembleA_valpre(
                        dataloader_val=val_loader,
                        preloader_val=Ensemable_popfits_masks,
                        time=e,
                        number_net=number_net,
                        bestpop=use_pop[j],
                        net_n_classes=nets_classes,
                        device='cpu')


                mask_pre = Ensemable_pred_masks[e]
                for E in range(number_ensemble):
                    ES = E + number_net
                    Dice[ES], IoU[ES], PC[ES], SE[ES], SP[ES], ACC[ES], F2[ES], mDice[ES], mIoU[ES], mPC[ES], mSE[ES], \
                        mSP[ES], mACC[ES], mF2[ES], best_test[ES] = evaluate_ensembleA_test(preloader_test=mask_pre,
                                                                                            dataloader_test=test_loader,
                                                                                            number_net=number_net,
                                                                                            bestpop=use_pop[E],
                                                                                            net_n_classes=nets_classes,
                                                                                            device='cpu')

                for E in range(number_ensemble):
                    ES = E + number_net
                    val_score[ES] = evaluate_ensemble_val(best_vals[ES], val_loader, nets_classes, device)


                    val[ES] = [val_score[ES], Dice[ES], IoU[ES], SE[ES], PC[ES], F2[ES], SP[ES], ACC[ES], mDice[ES],
                               mIoU[ES], mSE[ES], mPC[ES], mF2[ES], mSP[ES], mACC[ES], best_vals[ES]]
                    val_best[ES], bestest[ES] = Choice_best(val=val[ES], val_best=val_best[ES], bestest=bestest[ES],
                                                            best_test=best_test[ES], Epoch=epoch_nets[ES])

                txt_weight[0].write('\n' + 'weight=' + f'{use_pop}')
                # val_best[number_net] = Choice_best(val=val[number_net], val_best=val_best[number_net],
                #                                   Epoch=epoch_nets[number_net])

                global_step[number_net] += traindata_len

                if epoch_nets[number_net] == 0:
                    for E in range(number_ensemble):
                        ES = E + number_net
                        Dice_All[ES][0] = 0
                        IoU_All[ES][0] = 0
                        SE_All[ES][0] = 0
                        PC_All[ES][0] = 0
                        F2_All[ES][0] = 0
                        SP_All[ES][0] = 0
                        ACC_All[ES][0] = 0
                        mDice_All[ES][0] = 0
                        mIoU_All[ES][0] = 0
                        mSE_All[ES][0] = 0
                        mPC_All[ES][0] = 0
                        mF2_All[ES][0] = 0
                        mSP_All[ES][0] = 0
                        mACC_All[ES][0] = 0
                    epoch_All[ES][0] = 0

                for E in range(number_ensemble):
                    ES = E + number_net
                    Dice_All[ES][epoch_nets[ES] + 1]= Dice[ES]
                    IoU_All[ES][epoch_nets[ES] + 1] = IoU[ES]
                    SE_All[ES][epoch_nets[ES] + 1] = SE[ES]
                    PC_All[ES][epoch_nets[ES] + 1] = PC[ES]
                    F2_All[ES][epoch_nets[ES] + 1] = F2[ES]
                    SP_All[ES][epoch_nets[ES] + 1] = SP[ES]
                    ACC_All[ES][epoch_nets[ES] + 1] = ACC[ES]
                    mDice_All[ES][epoch_nets[ES] + 1] = mDice[ES]
                    mIoU_All[ES][epoch_nets[ES] + 1] = mIoU[ES]
                    mSE_All[ES][epoch_nets[ES] + 1] = mSE[ES]
                    mPC_All[ES][epoch_nets[ES] + 1] = mPC[ES]
                    mF2_All[ES][epoch_nets[ES] + 1] = mF2[ES]
                    mSP_All[ES][epoch_nets[ES] + 1] = mSP[ES]
                    mACC_All[ES][epoch_nets[ES] + 1] = mACC[ES]
                epoch_All[ES][epoch_nets[ES] + 1] = (epoch_nets[ES] + 1)

    for Epoch_Net in range(number_net):
        print('-----------------------------------------------')
        print(f'in net{Epoch_Net}:')
        print('best Dice =', val_best[Epoch_Net][1],
              'best IoU =', val_best[Epoch_Net][2],
              'best Sensitivity =', val_best[Epoch_Net][3],
              'best Precision =', val_best[Epoch_Net][4],
              'best F2 =', val_best[Epoch_Net][5],
              'best Specificity =', val_best[Epoch_Net][6],
              'best Accuracy =', val_best[Epoch_Net][7],
              'best mDice =', val_best[Epoch_Net][8],
              'best mIoU =', val_best[Epoch_Net][9],
              'best mean Sensitivity =', val_best[Epoch_Net][10],
              'best mean Precision =', val_best[Epoch_Net][11],
              'best mean F2 =', val_best[Epoch_Net][12],
              'best mean Specificity =', val_best[Epoch_Net][13],
              'best mean Accuracy =', val_best[Epoch_Net][14],
              'best net of Dice =', val_best[Epoch_Net][16],
              )
        txt_last[0].write(
            '\n' + f'{Epoch_Net}' + '\n' +
            ' Dice=' + f'{val_best[Epoch_Net][1]}' + '\n' +
            ', IoU=' + f'{val_best[Epoch_Net][2]}' + '\n' +
            ', Sensitivity=' + f'{val_best[Epoch_Net][3]}' + '\n' +
            ', Precision=' + f'{val_best[Epoch_Net][4]}' + '\n' +
            ', F2=' + f'{val_best[Epoch_Net][5]}' + '\n' +
            ', Specificity=' + f'{val_best[Epoch_Net][6]}' + '\n' +
            ', Accuracy=' + f'{val_best[Epoch_Net][7]}' + '\n' +
            ', mDice=' + f'{val_best[Epoch_Net][8]}' + '\n' +
            ', mIoU=' + f'{val_best[Epoch_Net][9]}' + '\n' +
            ', mean Sensitivity=' + f'{val_best[Epoch_Net][10]}' + '\n' +
            ', mean Precision=' + f'{val_best[Epoch_Net][11]}' + '\n' +
            ', mean F2=' + f'{val_best[Epoch_Net][12]}' + '\n' +
            ', mean Specificity=' + f'{val_best[Epoch_Net][13]}' + '\n' +
            ', mean Accuracy=' + f'{val_best[Epoch_Net][14]}' + '\n' +
            ', best net of Dice=' + f'{val_best[Epoch_Net][16]}' + '\n')
    txt_last[0].close()

    print('-----------------------------------------------')
    for E in range(number_ensemble):
        ES = E + number_net
        print(f'in PSO {ES}:')
        print('best Dice =', val_best[ES][1],
              'best IoU =', val_best[ES][2],
              'best Sensitivity =', val_best[ES][3],
              'best Precision =', val_best[ES][4],
              'best F2 =', val_best[ES][5],
              'best Specificity =', val_best[ES][6],
              'best Accuracy =', val_best[ES][7],
              'best mDice =', val_best[ES][8],
              'best mIoU =', val_best[ES][9],
              'best mean Sensitivity =', val_best[ES][10],
              'best mean Precision =', val_best[ES][11],
              'best mean F2 =', val_best[ES][12],
              'best mean Specificity =', val_best[ES][13],
              'best mean Accuracy =', val_best[ES][14],
              'best net of Dice =', val_best[ES][16],
              )
        txt_last[1].write(
            '\n' + f'in PSO {ES}' + '\n' +
            ' Dice=' + f'{val_best[ES][1]}' + '\n' +
            ', IoU=' + f'{val_best[ES][2]}' + '\n' +
            ', Sensitivity=' + f'{val_best[ES][3]}' + '\n' +
            ', Precision=' + f'{val_best[ES][4]}' + '\n' +
            ', F2=' + f'{val_best[ES][5]}' + '\n' +
            ', Specificity=' + f'{val_best[ES][6]}' + '\n' +
            ', Accuracy=' + f'{val_best[ES][7]}' + '\n' +
            ', mDice=' + f'{val_best[ES][8]}' + '\n' +
            ', mIoU=' + f'{val_best[ES][9]}' + '\n' +
            ', mean Sensitivity=' + f'{val_best[ES][10]}' + '\n' +
            ', mean Precision=' + f'{val_best[ES][11]}' + '\n' +
            ', mean F2=' + f'{val_best[ES][12]}' + '\n' +
            ', mean Specificity=' + f'{val_best[ES][13]}' + '\n' +
            ', mean Accuracy=' + f'{val_best[ES][14]}' + '\n' +
            ', best net of Dice=' + f'{val_best[ES][16]}' + '\n')

        print('-----------------------------------------------')

    txt_last[1].close()

    for Epoch_Net in range(number_net):
        for epoch in range((set_epochs + 1)):
            txt[Epoch_Net].write(
                '\n' +
                ' epoch=' + f'{epoch_All[Epoch_Net][epoch]}' +
                ' learning rate=' + f'{lr_All[Epoch_Net][epoch]}' +
                '\n' +
                ' Dice=' + f'{Dice_All[Epoch_Net][epoch]}' +
                ', IoU=' + f'{IoU_All[Epoch_Net][epoch]}' +
                ', Sensitivity=' + f'{SE_All[Epoch_Net][epoch]}' +
                ', Precision=' + f'{PC_All[Epoch_Net][epoch]}' +
                ', F2=' + f'{F2_All[Epoch_Net][epoch]}' +
                ', Specificity=' + f'{SP_All[Epoch_Net][epoch]}' +
                ', Accuracy=' + f'{ACC_All[Epoch_Net][epoch]}' +
                ', mDice=' + f'{mDice_All[Epoch_Net][epoch]}' +
                ', mIoU=' + f'{mIoU_All[Epoch_Net][epoch]}' +
                ', mean Sensitivity=' + f'{mSE_All[Epoch_Net][epoch]}' +
                ', mean Precision=' + f'{mPC_All[Epoch_Net][epoch]}' +
                ', mean F2=' + f'{mF2_All[Epoch_Net][epoch]}' +
                ', mean Specificity=' + f'{mSP_All[Epoch_Net][epoch]}' +
                ', mean Accuracy=' + f'{mACC_All[Epoch_Net][epoch]}')

    txt[Epoch_Net].close()

    for E in range(number_ensemble):
        ES = E + number_net
        for epoch in range((set_epochs + 1)):
            txt[number_net].write(
                '\n' +
                ' PSO = ' + f'{ES}' +
                ' epoch=' + f'{epoch_All[ES][epoch]}' +
                '\n' +
                ' Dice=' + f'{Dice_All[ES][epoch]}' +
                ', IoU=' + f'{IoU_All[ES][epoch]}' +
                ', Sensitivity=' + f'{SE_All[ES][epoch]}' +
                ', Precision=' + f'{PC_All[ES][epoch]}' +
                ', F2=' + f'{F2_All[ES][epoch]}' +
                ', Specificity=' + f'{SP_All[ES][epoch]}' +
                ', Accuracy=' + f'{ACC_All[ES][epoch]}' +
                ', mDice=' + f'{mDice_All[ES][epoch]}' +
                ', mIoU=' + f'{mIoU_All[ES][epoch]}' +
                ', mean Sensitivity=' + f'{mSE_All[ES][epoch]}' +
                ', mean Precision=' + f'{mPC_All[ES][epoch]}' +
                ', mean F2=' + f'{mF2_All[ES][epoch]}' +
                ', mean Specificity=' + f'{mSP_All[ES][epoch]}' +
                ', mean Accuracy=' + f'{mACC_All[ES][epoch]}')

    txt[number_net].close()

    for Epoch_Net in range(number_net):
        txt_dice[Epoch_Net].close()
        txt_w[Epoch_Net].close()
    txt_ensm[0].close()
    txt[number_net + 2].close()
    txt_weight[0].close()

    print('-----------------------------------------------')
    print('-----------------------------------------------')



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--Ensemable_name', '-Ename', type=str, default=None, help='name long must 4')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=True, help='Load model from a .pth file')
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

    parser.add_argument('--model1', '-m1', default='checkpoints/net1/best_checkpoint.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--model2', '-m2', default='checkpoints/net2/best_checkpoint.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--model3', '-m3', default='checkpoints/net3/best_checkpoint.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--model4', '-m4', default='checkpoints/net4/best_checkpoint.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--model5', '-m5', default='checkpoints/net5/best_checkpoint.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--model6', '-m6', default='checkpoints/net6/best_checkpoint.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')

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

        # if number_net == 6:
        #     pop_site[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #     pop_site[1] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        #     pop_site[2] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        #     pop_site[3] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        #     pop_site[4] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        #     pop_site[5] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        #     pop_site[6] = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
        #     pop_site[7] = [0.2, 0.0, 0.2, 0.2, 0.2, 0.2]
        #     pop_site[8] = [0.2, 0.2, 0.0, 0.2, 0.2, 0.2]
        #     pop_site[9] = [0.2, 0.2, 0.2, 0.0, 0.2, 0.2]
        #     pop_site[10] = [0.2, 0.2, 0.2, 0.2, 0.0, 0.2]
        #     pop_site[11] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
        # elif number_net == 5:
        #     pop_site[0] = [1.00, 0.00, 0.00, 0.00, 0.00]
        #     pop_site[1] = [0.00, 1.00, 0.00, 0.00, 0.00]
        #     pop_site[2] = [0.00, 0.00, 1.00, 0.00, 0.00]
        #     pop_site[3] = [0.00, 0.00, 0.00, 1.00, 0.00]
        #     pop_site[4] = [0.00, 0.00, 0.00, 0.00, 1.00]
        #     pop_site[5] = [0.00, 0.25, 0.25, 0.25, 0.25]
        #     pop_site[6] = [0.25, 0.00, 0.25, 0.25, 0.25]
        #     pop_site[7] = [0.25, 0.25, 0.00, 0.25, 0.25]
        #     pop_site[8] = [0.25, 0.25, 0.25, 0.00, 0.25]
        #     pop_site[9] = [0.25, 0.25, 0.25, 0.25, 0.00]
        # elif number_net == 4:
        #     pop_site[0] = [1.0, 0.0, 0.0, 0.0]
        #     pop_site[1] = [0.0, 1.0, 0.0, 0.0]
        #     pop_site[2] = [0.0, 0.0, 1.0, 0.0]
        #     pop_site[3] = [0.0, 0.0, 0.0, 1.0]
        #     pop_site[4] = [0.0, 0.33, 0.34, 0.33]
        #     pop_site[5] = [0.33, 0.0, 0.33, 0.34]
        #     pop_site[6] = [0.34, 0.33, 0.0, 0.33]
        #     pop_site[7] = [0.33, 0.34, 0.33, 0.0]
        # elif number_net == 3:
        #     pop_site[0] = [1.0, 0.0, 0.0]
        #     pop_site[1] = [0.0, 1.0, 0.0]
        #     pop_site[2] = [0.0, 0.0, 1.0]
        #     pop_site[3] = [0.0, 0.5, 0.5]
        #     pop_site[4] = [0.5, 0.0, 0.5]
        #     pop_site[5] = [0.5, 0.5, 0.0]
        # elif number_net == 2:
        #     pop_site[0] = [1.0, 0.0]
        #     pop_site[1] = [0.0, 1.0]
        #     pop_site[2] = [0.5, 0.5]

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
                pop_v[pso_j] = pso_lr[0] * np.random.rand() * (pbestpop[pso_j] - pop_site[pso_j]) + \
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

def SCBHPSO_Ensemble(pso_epoch, pso_sumpop, pso_lr, init, Ensemable_popfits_masks, val_loader, nets_classes, number_net, label, label_threshold, elimate):

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

        # if number_net == 6:
        #     pop_site[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #     pop_site[1] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        #     pop_site[2] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        #     pop_site[3] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        #     pop_site[4] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        #     pop_site[5] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        #     pop_site[6] = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
        #     pop_site[7] = [0.2, 0.0, 0.2, 0.2, 0.2, 0.2]
        #     pop_site[8] = [0.2, 0.2, 0.0, 0.2, 0.2, 0.2]
        #     pop_site[9] = [0.2, 0.2, 0.2, 0.0, 0.2, 0.2]
        #     pop_site[10] = [0.2, 0.2, 0.2, 0.2, 0.0, 0.2]
        #     pop_site[11] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
        # elif number_net == 5:
        #     pop_site[0] = [1.00, 0.00, 0.00, 0.00, 0.00]
        #     pop_site[1] = [0.00, 1.00, 0.00, 0.00, 0.00]
        #     pop_site[2] = [0.00, 0.00, 1.00, 0.00, 0.00]
        #     pop_site[3] = [0.00, 0.00, 0.00, 1.00, 0.00]
        #     pop_site[4] = [0.00, 0.00, 0.00, 0.00, 1.00]
        #     pop_site[5] = [0.00, 0.25, 0.25, 0.25, 0.25]
        #     pop_site[6] = [0.25, 0.00, 0.25, 0.25, 0.25]
        #     pop_site[7] = [0.25, 0.25, 0.00, 0.25, 0.25]
        #     pop_site[8] = [0.25, 0.25, 0.25, 0.00, 0.25]
        #     pop_site[9] = [0.25, 0.25, 0.25, 0.25, 0.00]
        # elif number_net == 4:
        #     pop_site[0] = [1.0, 0.0, 0.0, 0.0]
        #     pop_site[1] = [0.0, 1.0, 0.0, 0.0]
        #     pop_site[2] = [0.0, 0.0, 1.0, 0.0]
        #     pop_site[3] = [0.0, 0.0, 0.0, 1.0]
        #     pop_site[4] = [0.0, 0.33, 0.34, 0.33]
        #     pop_site[5] = [0.33, 0.0, 0.33, 0.34]
        #     pop_site[6] = [0.34, 0.33, 0.0, 0.33]
        #     pop_site[7] = [0.33, 0.34, 0.33, 0.0]
        # elif number_net == 3:
        #     pop_site[0] = [1.0, 0.0, 0.0]
        #     pop_site[1] = [0.0, 1.0, 0.0]
        #     pop_site[2] = [0.0, 0.0, 1.0]
        #     pop_site[3] = [0.0, 0.5, 0.5]
        #     pop_site[4] = [0.5, 0.0, 0.5]
        #     pop_site[5] = [0.5, 0.5, 0.0]
        # elif number_net == 2:
        #     pop_site[0] = [1.0, 0.0]
        #     pop_site[1] = [0.0, 1.0]
        #     pop_site[2] = [0.5, 0.5]

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


def RBHPSO_Ensemble(pso_epoch, pso_sumpop, pso_lr, init, Ensemable_popfits_masks, val_loader, nets_classes, number_net, label, label_threshold, elimate):

    if init:
        wv = 0
        setr = 0.2
        setp = 0
        setl = 1
        thes = np.zeros(number_net)
        pop_sim = [None] * pso_sumpop
        aaa = 0

        pop_site = torch.normal(mean=0.1666, std=0.5, size=(pso_sumpop, number_net))
        pop_site_sum = torch.sum(pop_site, dim=1)
        pop_site_another = torch.ones((number_net))
        pop_site_other, _ = torch.meshgrid(pop_site_sum, pop_site_another)
        pop_site = pop_site / pop_site_other
        pop_site = pop_site.numpy()

        # if number_net == 6:
        #     pop_site[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #     pop_site[1] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        #     pop_site[2] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        #     pop_site[3] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        #     pop_site[4] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        #     pop_site[5] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        #     pop_site[6] = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
        #     pop_site[7] = [0.2, 0.0, 0.2, 0.2, 0.2, 0.2]
        #     pop_site[8] = [0.2, 0.2, 0.0, 0.2, 0.2, 0.2]
        #     pop_site[9] = [0.2, 0.2, 0.2, 0.0, 0.2, 0.2]
        #     pop_site[10] = [0.2, 0.2, 0.2, 0.2, 0.0, 0.2]
        #     pop_site[11] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
        # elif number_net == 5:
        #     pop_site[0] = [1.00, 0.00, 0.00, 0.00, 0.00]
        #     pop_site[1] = [0.00, 1.00, 0.00, 0.00, 0.00]
        #     pop_site[2] = [0.00, 0.00, 1.00, 0.00, 0.00]
        #     pop_site[3] = [0.00, 0.00, 0.00, 1.00, 0.00]
        #     pop_site[4] = [0.00, 0.00, 0.00, 0.00, 1.00]
        #     pop_site[5] = [0.00, 0.25, 0.25, 0.25, 0.25]
        #     pop_site[6] = [0.25, 0.00, 0.25, 0.25, 0.25]
        #     pop_site[7] = [0.25, 0.25, 0.00, 0.25, 0.25]
        #     pop_site[8] = [0.25, 0.25, 0.25, 0.00, 0.25]
        #     pop_site[9] = [0.25, 0.25, 0.25, 0.25, 0.00]
        # elif number_net == 4:
        #     pop_site[0] = [1.0, 0.0, 0.0, 0.0]
        #     pop_site[1] = [0.0, 1.0, 0.0, 0.0]
        #     pop_site[2] = [0.0, 0.0, 1.0, 0.0]
        #     pop_site[3] = [0.0, 0.0, 0.0, 1.0]
        #     pop_site[4] = [0.0, 0.33, 0.34, 0.33]
        #     pop_site[5] = [0.33, 0.0, 0.33, 0.34]
        #     pop_site[6] = [0.34, 0.33, 0.0, 0.33]
        #     pop_site[7] = [0.33, 0.34, 0.33, 0.0]
        # elif number_net == 3:
        #     pop_site[0] = [1.0, 0.0, 0.0]
        #     pop_site[1] = [0.0, 1.0, 0.0]
        #     pop_site[2] = [0.0, 0.0, 1.0]
        #     pop_site[3] = [0.0, 0.5, 0.5]
        #     pop_site[4] = [0.5, 0.0, 0.5]
        #     pop_site[5] = [0.5, 0.5, 0.0]
        # elif number_net == 2:
        #     pop_site[0] = [1.0, 0.0]
        #     pop_site[1] = [0.0, 1.0]
        #     pop_site[2] = [0.5, 0.5]

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
                if setl >= setp:
                    pop_site[pso_j] += pso_t * pop_v[pso_j]
                else:
                    pop_site[pso_j] = gbestpop + thes

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
        setp = np.random.rand()
        # setp = 0.1
        for i in range(pso_sumpop):
            setl = np.random.rand()

        pso_resultF[pso_i] = gbestfitness
        pso_resultS[pso_i] = gbestpop
        print('gbestfitness=', gbestfitness)
        print('gbestpop=', gbestpop)
        print('print(aaa)print(aaa)print(aaa)=========', aaa)

    return pso_resultF, pso_resultS, gbestpop

def CSPSO_Ensemble(pso_epoch, pso_sumpop, pso_lr, init, Ensemable_popfits_masks, val_loader, nets_classes, number_net, label, label_threshold, elimate):

    if init:
        wv = 0.5
        theTh = 5
        pop_C = 0
        pop_sita = 1
        theH = 3
        Rand = 100
        pop_sim = [None] * pso_sumpop
        aaa = 0

        pop_site = torch.normal(mean=0.1666, std=0.5, size=(pso_sumpop, number_net))
        pop_site_sum = torch.sum(pop_site, dim=1)
        pop_site_another = torch.ones((number_net))
        pop_site_other, _ = torch.meshgrid(pop_site_sum, pop_site_another)
        pop_site = pop_site / pop_site_other
        pop_site = pop_site.numpy()

        # if number_net == 6:
        #     pop_site[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #     pop_site[1] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        #     pop_site[2] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        #     pop_site[3] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        #     pop_site[4] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        #     pop_site[5] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        #     pop_site[6] = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
        #     pop_site[7] = [0.2, 0.0, 0.2, 0.2, 0.2, 0.2]
        #     pop_site[8] = [0.2, 0.2, 0.0, 0.2, 0.2, 0.2]
        #     pop_site[9] = [0.2, 0.2, 0.2, 0.0, 0.2, 0.2]
        #     pop_site[10] = [0.2, 0.2, 0.2, 0.2, 0.0, 0.2]
        #     pop_site[11] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
        # elif number_net == 5:
        #     pop_site[0] = [1.00, 0.00, 0.00, 0.00, 0.00]
        #     pop_site[1] = [0.00, 1.00, 0.00, 0.00, 0.00]
        #     pop_site[2] = [0.00, 0.00, 1.00, 0.00, 0.00]
        #     pop_site[3] = [0.00, 0.00, 0.00, 1.00, 0.00]
        #     pop_site[4] = [0.00, 0.00, 0.00, 0.00, 1.00]
        #     pop_site[5] = [0.00, 0.25, 0.25, 0.25, 0.25]
        #     pop_site[6] = [0.25, 0.00, 0.25, 0.25, 0.25]
        #     pop_site[7] = [0.25, 0.25, 0.00, 0.25, 0.25]
        #     pop_site[8] = [0.25, 0.25, 0.25, 0.00, 0.25]
        #     pop_site[9] = [0.25, 0.25, 0.25, 0.25, 0.00]
        # elif number_net == 4:
        #     pop_site[0] = [1.0, 0.0, 0.0, 0.0]
        #     pop_site[1] = [0.0, 1.0, 0.0, 0.0]
        #     pop_site[2] = [0.0, 0.0, 1.0, 0.0]
        #     pop_site[3] = [0.0, 0.0, 0.0, 1.0]
        #     pop_site[4] = [0.0, 0.33, 0.34, 0.33]
        #     pop_site[5] = [0.33, 0.0, 0.33, 0.34]
        #     pop_site[6] = [0.34, 0.33, 0.0, 0.33]
        #     pop_site[7] = [0.33, 0.34, 0.33, 0.0]
        # elif number_net == 3:
        #     pop_site[0] = [1.0, 0.0, 0.0]
        #     pop_site[1] = [0.0, 1.0, 0.0]
        #     pop_site[2] = [0.0, 0.0, 1.0]
        #     pop_site[3] = [0.0, 0.5, 0.5]
        #     pop_site[4] = [0.5, 0.0, 0.5]
        #     pop_site[5] = [0.5, 0.5, 0.0]
        # elif number_net == 2:
        #     pop_site[0] = [1.0, 0.0]
        #     pop_site[1] = [0.0, 1.0]
        #     pop_site[2] = [0.5, 0.5]

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
                wv = 0.9 - 0.5 * (((pso_i + 1) / pso_epoch) ** 2)
                if pop_C > (pop_sita * theTh):
                    # print('111')
                    pop_v[pso_j] = wv * pop_v[pso_j] - pso_lr[0] * np.random.rand() * (
                                pbestpop[pso_j] - pop_site[pso_j]) - \
                                   pso_lr[1] * np.random.rand() * (gbestpop - pop_site[pso_j])
                else:
                    # print('000')
                    pop_v[pso_j] = wv * pop_v[pso_j] + pso_lr[0] * np.random.rand() * (
                                pbestpop[pso_j] - pop_site[pso_j]) + pso_lr[
                                       1] * np.random.rand() * (gbestpop - pop_site[pso_j])
                # pop_v[pso_j] = wv * pop_v[pso_j] + pso_lr[0] * np.random.rand() * (pbestpop[pso_j] - pop_site[pso_j]) + \
                #                pso_lr[1] * np.random.rand() * (gbestpop - pop_site[pso_j])
            # pop_v[pop_v < pso_rangespeed[0]] = pso_rangespeed[0]
            # pop_v[pop_v > pso_rangespeed[1]] = pso_rangespeed[1]

            # 粒子位置更新
            for pso_j in range(pso_sumpop):
                # pso_pop[pso_j] += 0.5*pso_v[pso_j]
                # pop_site[pso_j] += pso_t * pop_v[pso_j]

                if Rand < theH * pop_C * pop_sim[pso_j]:
                    # print("111")
                    max_fitness = max(pop_fitness)
                    max_fit_site = pop_fitness.index(max_fitness)
                    # if (pop_site[pso_j][0] == gbestpop[0]) and (pop_site[pso_j][1] == gbestpop[1]) and \
                    #         (pop_site[pso_j][2] == gbestpop[2]) and (pop_site[pso_j][3] == gbestpop[3]) and \
                    #         (pop_site[pso_j][4] == gbestpop[4]) and (pop_site[pso_j][5] == gbestpop[5]):
                    if pso_j == max_fit_site:
                        pop_site[pso_j] = gbestpop
                        aaa += 1
                    else:
                        pop_site[pso_j] = Resite(pop_site[pso_j], gbestpop, 1.0, 0.1, (pso_i + 1), pso_epoch)
                else:
                    # print("000")
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

    set_dir_img = Path('./data/imgs/')
    set_dir_mask = Path('./data/masks/')
    number_net = args.numb_net
    set_dir_checkpoint = []
    set_nets_names = []
    set_nets = []

    model_checkpoint = [None] * 6
    model_checkpoint[0] = args.model1
    model_checkpoint[1] = args.model2
    model_checkpoint[2] = args.model3
    model_checkpoint[3] = args.model4
    model_checkpoint[4] = args.model5
    model_checkpoint[5] = args.model6

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
            set_nets[i].load_state_dict(torch.load(model_checkpoint[i], map_location=device))
            logging.info(f'Model loaded from {os.path.join(model_checkpoint[i])}')
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


