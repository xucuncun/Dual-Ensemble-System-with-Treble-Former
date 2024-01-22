import torch
import torch.nn as nn

from Models.Treble_Former import Treble_Former as TF
from Models.Treble_Former_S import Treble_Former as TF_S
from Models.Treble_Former_L import Treble_Former as TF_L
from Models.FCBFormer_S import FCBFormer as FCBForme_S
from Models.FCBFormer_L import FCBFormer as FCBForme_L
from Models.ESFPNet_S import ESFPNetStructure as ESFP_S
from Models.ESFPNet_L import ESFPNetStructure as ESFP_L


class SemSegNet(nn.Module):
    def __init__(self, n_channels, n_classes, what_Net: str = 'None', img_size: tuple = (352, 352)):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.what_Net = what_Net
        self.img_size = img_size

        trebleformer = TF(
            n_channels=3,
            n_classes=2,
            img_bian=352
        )

        trebleformer_S = TF_S(
            n_channels=3,
            n_classes=2,
            img_bian=352
        )
        trebleformer_L = TF_L(
            n_channels=3,
            n_classes=2,
            img_bian=352
        )
        FCBFormer_S = FCBForme_S(
            n_channels=3,
            n_classes=2
        )
        FCBFormer_L = FCBForme_L(
            n_channels=3,
            n_classes=2
        )
        ESFPNet_S = ESFP_S(
            n_channels=3,
            n_classes=2
        )
        ESFPNet_L = ESFP_L(
            n_channels=3,
            n_classes=2
        )


        if self.what_Net == 'None':
            raise ValueError('must choice what_Net in Use_Net')
        elif self.what_Net == 'TrebleFormer':
            self.usenet = trebleformer
        elif self.what_Net == 'TrebleFormer_S':
            self.usenet = trebleformer_S
        elif self.what_Net == 'TrebleFormer_L':
            self.usenet = trebleformer_L
        elif self.what_Net == 'FCBFormer_S':
            self.usenet = FCBFormer_S
        elif self.what_Net == 'FCBFormer_L':
            self.usenet = FCBFormer_L
        elif self.what_Net == 'ESFPNet_S':
            self.usenet = ESFPNet_S
        elif self.what_Net == 'ESFPNet_L':
            self.usenet = ESFPNet_L
        else:
            raise ValueError(f'{what_Net} need to set as a new net')

    def forward(self, x):

        y = self.usenet(x)

        return y
