import torch
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(1234)
class HGNNPLayer(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 W_e: torch.Tensor,
                 Hyperedge,
                 H: torch.Tensor,
                 D_v: torch.Tensor,
                 D_e: torch.Tensor):
        super(HGNNPLayer, self).__init__()
        self.W_e = W_e
        self.e = Hyperedge
        self.H = H
        self.D_v = D_v
        self.D_e = D_e
        self.BN = nn.BatchNorm1d(input_dim)
        # self.BN_W = nn.BatchNorm2d(W_e.shape)
        self.Activition = nn.LeakyReLU()
        # self.HGNNP_liner = nn.Sequential(nn.Linear(input_dim, output_dim))
        self.theta = nn.Parameter(torch.Tensor(input_dim, output_dim))
        # self.drop_prob = 0.3
        # self.drop = nn.Dropout(self.drop_prob)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta.size(1))
        self.theta.data.uniform_(-stdv, stdv)

    def forward(self, X: torch.Tensor):
        # HGNNP
        S = self.BN(X)
        D_e = torch.where(self.D_e != 0, torch.pow(self.D_e, -1), self.D_e)
        D_v = torch.where(self.D_v != 0, torch.pow(self.D_v, -1), self.D_v)
        W = self.W_e
        Y = torch.mm(torch.mm(torch.mm(W, D_e), self.H.T), S)
        output = torch.mm(torch.mm(torch.mm(D_v, self.H), Y), self.theta)
        output =self.Activition(output)
        # # HGNN
        # X = self.BN(X)
        # D_e = torch.where(self.D_e != 0, torch.pow(self.D_e, -1), self.D_e)
        # D_v = torch.where(self.D_v != 0, torch.pow(self.D_v, -0.5), self.D_v)
        # W = self.W_e
        # Y = torch.mm(torch.mm(torch.mm(torch.mm(W, D_e), self.H.T), D_v), X)
        # Y = torch.mm(D_v, torch.mm(self.H, Y))
        # output = torch.mm(Y, self.theta)
        # output = self.Activition(output)
        return output

class CNNConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, k, h, w):
        super(CNNConvBlock, self).__init__()
        self.BN = nn.BatchNorm2d(ch_in)
        self.conv_in = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv_out = nn.Conv2d(ch_out, ch_out, kernel_size=k, padding=k//2, stride=1, groups=ch_out)
        self.pool = nn.AvgPool2d(3, padding=1, stride=1)
        self.act = nn.LeakyReLU()
        # self.act1 = nn.ReLU()

    def forward(self, x):
        x = self.BN(x)
        x = self.act(self.conv_in(x))
        x = self.pool(x)
        x = self.act(self.conv_out(x))
        return x

class HGCN_MHF(nn.Module):
    def __init__(self, height: int, width: int, channel: int, class_count: int, Q: torch.Tensor,
                W_e:torch.Tensor,Hyperedge,H:torch.Tensor,D_v:torch.Tensor,D_e:torch.Tensor):
        super(HGCN_MHF, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = channel
        self.height = height
        self.width = width
        self.Q = Q
        self.H = H #1
        self.Hyperedge=Hyperedge
        self.D_v=D_v
        self.D_e=D_e
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化Q
        self.w=W_e
        layers_count = 2
        mid_channel = 128

        # # Spectral Transformation Sub-Network
        self.CNN_denoise = nn.Sequential()
        self.CNN_denoise.add_module('CNN_denoise_BN1', nn.BatchNorm2d(self.channel))
        self.CNN_denoise.add_module('CNN_denoise_Conv1',
                                    nn.Conv2d(self.channel, mid_channel, kernel_size=(1, 1)))
        self.CNN_denoise.add_module('CNN_denoise_Act1', nn.LeakyReLU())
        self.CNN_denoise.add_module('CNN_denoise_BN2', nn.BatchNorm2d(mid_channel))
        self.CNN_denoise.add_module('CNN_denoise_Conv2', nn.Conv2d(mid_channel, mid_channel, kernel_size=(1, 1)))
        self.CNN_denoise.add_module('CNN_denoise_Act2', nn.LeakyReLU())


        # Pixel-level Convolutional Sub-Network
        mid_channel1 = 64
        self.CNNlayerA1 = CNNConvBlock(mid_channel, mid_channel1, 7, self.height, self.width)
        self.CNNlayerA2 = CNNConvBlock(3*mid_channel1, mid_channel1, 7, self.height, self.width)

        self.CNNlayerB1 = CNNConvBlock(mid_channel, mid_channel1, 5, self.height, self.width)
        self.CNNlayerB2 = CNNConvBlock(3*mid_channel1, mid_channel1, 5, self.height, self.width)

        self.CNNlayerC1 = CNNConvBlock(mid_channel, mid_channel1, 3, self.height, self.width)
        self.CNNlayerC2 = CNNConvBlock(3*mid_channel1, mid_channel1, 3, self.height, self.width)


        # Superpixel-level HyperGraph Convolutional Sub-Network
        self.HGCN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                self.HGCN_Branch.add_module('HGCN_Branch' + str(i), HGNNPLayer(mid_channel, mid_channel, self.w, self.Hyperedge,
                                                                               self.H, self.D_v, self.D_e))
            else:
                self.HGCN_Branch.add_module('HGCN_Branch' + str(i), HGNNPLayer(mid_channel, 3*mid_channel1, self.w, self.Hyperedge,
                                                                               self.H, self.D_v, self.D_e))

        self.gamma1 = nn.Parameter(torch.randn(1))
        self.gamma2 = nn.Parameter(torch.randn(1))

        # Linear And Softmax layer
        self.pre_HGCN = nn.Linear(in_features=3 * mid_channel1, out_features=self.class_count)
        self.pre_CNN = nn.Linear(in_features=3 * mid_channel1, out_features=self.class_count)
        self.concat_linear = nn.Linear(2 * 3 * mid_channel1, self.class_count)
        self.Softmax_linear1 = nn.Linear(3 * mid_channel1, self.class_count)

    def forward(self, x: torch.Tensor):
        '''
        :param x: H*W*C
        :return: probability_map
        '''
        (h, w, c) = x.shape

        # 去噪与光谱变换
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise  # 直连
        #
        clean_x_flatten = clean_x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  # 低频部分
        hx = clean_x

        # CNN
        CNNin = torch.unsqueeze(hx.permute([2, 0, 1]), 0)

        CNNmid1_A = self.CNNlayerA1(CNNin)
        CNNmid1_B = self.CNNlayerB1(CNNin)
        CNNmid1_C = self.CNNlayerC1(CNNin)

        CNNin = torch.cat([CNNmid1_A, CNNmid1_B, CNNmid1_C], dim=1)


        CNNout_A = self.CNNlayerA2(CNNin)
        CNNout_B = self.CNNlayerB2(CNNin)
        CNNout_C = self.CNNlayerC2(CNNin)

        CNNout = torch.cat([CNNout_A, CNNout_B, CNNout_C], dim=1)
        CNN_result = torch.squeeze(CNNout, 0).permute([1, 2, 0]).reshape([h * w, -1])

        # HGCN
        H_hg = superpixels_flatten
        for i in range(len(self.HGCN_Branch)): H_hg = self.HGCN_Branch[i](H_hg)
        HGCN_result = torch.matmul(self.Q, H_hg)  # 这里self.norm_row_Q == self.Q

        # score weighted feature fusion
        gamma1 = torch.sigmoid(self.gamma1)
        gamma2 = torch.sigmoid(self.gamma2)
        score_CNN = F.softmax(self.pre_CNN(CNN_result), -1)
        score_HGCN = F.softmax(self.pre_HGCN(HGCN_result), -1)
        Y = gamma1 * score_CNN + (1-gamma1) * score_HGCN

        Y_1 = torch.cat([HGCN_result, CNN_result], dim=-1)
        Y_1 = F.softmax(self.concat_linear(Y_1), -1)
        Y_2 = gamma2 * Y + (1-gamma2) *Y_1

        # Y = torch.cat([HGCN_result, CNN_result], dim=-1)
        # Y = F.softmax(self.Softmax_linear1(HGCN_result), -1)
        # Y = F.softmax(self.Softmax_linear1(CNN_result), -1)
        # Y = self.Softmax_linear(Y_2)
        return Y_2