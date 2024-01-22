
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.cuda.empty_cache()


from Models import mit
from Models import mlp
from mmcv.cnn import ConvModule


class ESFPNetStructure(nn.Module):

    def __init__(self, n_channels=3, n_classes=2, embedding_dim=160, model_type = 'B4'):
        super(ESFPNetStructure, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.model_type = model_type
        # Backbone
        # if self.model_type == 'B0':
        #     self.backbone = mit.mit_b0()
        # if self.model_type == 'B1':
        #     self.backbone = mit.mit_b1()
        # if self.model_type == 'B2':
        #     self.backbone = mit.mit_b2()
        # if self.model_type == 'B3':
        #     self.backbone = mit.mit_b3()
        if self.model_type == 'B4':
            self.backbone = mit.mit_b4()
        # if self.model_type == 'B5':
        #     self.backbone = mit.mit_b5()

        self._init_weights()  # load pretrain

        # LP Header
        self.LP_1 = mlp.LP(input_dim=self.backbone.embed_dims[0], embed_dim=self.backbone.embed_dims[0])
        self.LP_2 = mlp.LP(input_dim=self.backbone.embed_dims[1], embed_dim=self.backbone.embed_dims[1])
        self.LP_3 = mlp.LP(input_dim=self.backbone.embed_dims[2], embed_dim=self.backbone.embed_dims[2])
        self.LP_4 = mlp.LP(input_dim=self.backbone.embed_dims[3], embed_dim=self.backbone.embed_dims[3])

        # Linear Fuse
        self.linear_fuse34 = ConvModule(in_channels=(self.backbone.embed_dims[2] + self.backbone.embed_dims[3]),
                                        out_channels=self.backbone.embed_dims[2], kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse23 = ConvModule(in_channels=(self.backbone.embed_dims[1] + self.backbone.embed_dims[2]),
                                        out_channels=self.backbone.embed_dims[1], kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse12 = ConvModule(in_channels=(self.backbone.embed_dims[0] + self.backbone.embed_dims[1]),
                                        out_channels=self.backbone.embed_dims[0], kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))

        # Fused LP Header
        self.LP_12 = mlp.LP(input_dim=self.backbone.embed_dims[0], embed_dim=self.backbone.embed_dims[0])
        self.LP_23 = mlp.LP(input_dim=self.backbone.embed_dims[1], embed_dim=self.backbone.embed_dims[1])
        self.LP_34 = mlp.LP(input_dim=self.backbone.embed_dims[2], embed_dim=self.backbone.embed_dims[2])

        # Final Linear Prediction
        self.linear_pred = nn.Conv2d((self.backbone.embed_dims[0] + self.backbone.embed_dims[1] +
                                      self.backbone.embed_dims[2] + self.backbone.embed_dims[3]), self.n_classes, kernel_size=1)

    def _init_weights(self):

        # if self.model_type == 'B0':
        #     pretrained_dict = torch.load('mit_b0.pth')
        # if self.model_type == 'B1':
        #     pretrained_dict = torch.load('mit_b1.pth')
        # if self.model_type == 'B2':
        #     pretrained_dict = torch.load('mit_b2.pth')
        # if self.model_type == 'B3':
        #     pretrained_dict = torch.load('mit_b3.pth')
        # if self.model_type == 'B4':
        #     pretrained_dict = torch.load('mit_b4.pth')
        # if self.model_type == 'B5':
        #     pretrained_dict = torch.load('mit_b5.pth')
        # if self.model_type == 'B0':
        #     pretrained_dict = torch.load('Models/mit_b0.pth')
        # if self.model_type == 'B1':
        #     pretrained_dict = torch.load('Models/mit_b1.pth')
        # if self.model_type == 'B2':
        #     pretrained_dict = torch.load('Models/mit_b2.pth')
        # if self.model_type == 'B3':
        #     pretrained_dict = torch.load('Models/mit_b3.pth')
        if self.model_type == 'B4':
            pretrained_dict = torch.load('Models/mit_b4.pth')
        # if self.model_type == 'B5':
        #     pretrained_dict = torch.load('Models/mit_b5.pth')

        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print("successfully loaded!!!!")

    def forward(self, x):

        ##################  Go through backbone ###################

        B = x.shape[0]

        # stage 1
        out_1, H, W = self.backbone.patch_embed1(x)
        for i, blk in enumerate(self.backbone.block1):
            out_1 = blk(out_1, H, W)
        out_1 = self.backbone.norm1(out_1)
        out_1 = out_1.reshape(B, H, W, -1).permute(0, 3, 1,
                                                   2).contiguous()  # (Batch_Size, self.backbone.embed_dims[0], 88, 88)

        # stage 2
        out_2, H, W = self.backbone.patch_embed2(out_1)
        for i, blk in enumerate(self.backbone.block2):
            out_2 = blk(out_2, H, W)
        out_2 = self.backbone.norm2(out_2)
        out_2 = out_2.reshape(B, H, W, -1).permute(0, 3, 1,
                                                   2).contiguous()  # (Batch_Size, self.backbone.embed_dims[1], 44, 44)

        # stage 3
        out_3, H, W = self.backbone.patch_embed3(out_2)
        for i, blk in enumerate(self.backbone.block3):
            out_3 = blk(out_3, H, W)
        out_3 = self.backbone.norm3(out_3)
        out_3 = out_3.reshape(B, H, W, -1).permute(0, 3, 1,
                                                   2).contiguous()  # (Batch_Size, self.backbone.embed_dims[2], 22, 22)

        # stage 4
        out_4, H, W = self.backbone.patch_embed4(out_3)
        for i, blk in enumerate(self.backbone.block4):
            out_4 = blk(out_4, H, W)
        out_4 = self.backbone.norm4(out_4)
        out_4 = out_4.reshape(B, H, W, -1).permute(0, 3, 1,
                                                   2).contiguous()  # (Batch_Size, self.backbone.embed_dims[3], 11, 11)

        # go through LP Header
        lp_1 = self.LP_1(out_1)
        lp_2 = self.LP_2(out_2)
        lp_3 = self.LP_3(out_3)
        lp_4 = self.LP_4(out_4)

        # linear fuse and go pass LP Header
        lp_34 = self.LP_34(self.linear_fuse34(
            torch.cat([lp_3, F.interpolate(lp_4, scale_factor=2, mode='bilinear', align_corners=False)], dim=1)))
        lp_23 = self.LP_23(self.linear_fuse23(
            torch.cat([lp_2, F.interpolate(lp_34, scale_factor=2, mode='bilinear', align_corners=False)], dim=1)))
        lp_12 = self.LP_12(self.linear_fuse12(
            torch.cat([lp_1, F.interpolate(lp_23, scale_factor=2, mode='bilinear', align_corners=False)], dim=1)))

        # get the final output
        lp4_resized = F.interpolate(lp_4, scale_factor=8, mode='bilinear', align_corners=False)
        lp3_resized = F.interpolate(lp_34, scale_factor=4, mode='bilinear', align_corners=False)
        lp2_resized = F.interpolate(lp_23, scale_factor=2, mode='bilinear', align_corners=False)
        lp1_resized = lp_12

        out = self.linear_pred(torch.cat([lp1_resized, lp2_resized, lp3_resized, lp4_resized], dim=1))
        out_resized = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)

        # return out_resized

        return out_resized, [F.interpolate(torch.cat([lp1_resized, lp2_resized, lp3_resized, lp4_resized], dim=1), scale_factor=4, mode='bilinear', align_corners=True),
                             F.interpolate(torch.cat([lp1_resized, lp2_resized, lp3_resized, lp4_resized], dim=1), scale_factor=4, mode='bilinear', align_corners=True)]


# if __name__ == '__main__':
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
#     device = "cpu"
#     mymodule = ESFPNetStructure()
#     input = torch.ones((4, 3, 352, 352))
#     input = input.to(device=device)
#     mymodule = mymodule.to(device=device)
#     output = mymodule(input)
#     print(output.shape)