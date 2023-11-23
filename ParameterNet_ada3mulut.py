import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class conv1(nn.Module):
    def __init__(self, K=3, S=1):
        super(conv1, self).__init__()
        self.K = K  # 3
        self.S = S  # 1
        self.conv = nn.Conv2d(1, 64, (1, 4),
                              stride=1, padding=0, dilation=1, bias=True)
        self.P = self.K - 1

        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x, mod):
        B, C, H, W = x.shape
        x = F.unfold(x, self.K)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H - self.P) * (W - self.P),
                      self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K
        if mod == "c":
            x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 2],
                       x[:, :, 2, 0], x[:, :, 2, 2]], dim=1)
        elif mod == "s":
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)
        elif mod == "x":
            x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 1],
                           x[:, :, 1, 0], x[:, :, 1, 1]], dim=1)
        x = x.unsqueeze(1).unsqueeze(1)

        x = self.conv(x)  # B*C*L,K,K
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        x = F.fold(x, ((H - self.P) * self.S, (W - self.P) * self.S),
                   self.S, stride=self.S)

        return x
class Conv(nn.Module):
    """ 2D convolution w/ MSRA init. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

class DenseConv(nn.Module):
    """ Dense connected Conv. with activation. """

    def __init__(self, in_nf, nf=64):
        super(DenseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(in_nf, nf, 1)

    def forward(self, x):
        feat = self.act(self.conv1(x))
        out = torch.cat([x, feat], dim=1)
        return out

class AnisotropicGaussianFilter(torch.nn.Module):
    def __init__(self, num_channels):
        super(AnisotropicGaussianFilter, self).__init__()
        self.num_channels = num_channels


    def forward(self, x, sigx, sigy, theta, sigr):
        # 取得图像大小和通道数
        B, ks, ks, HW = x.size()

        # 构造高斯卷积核
        mesh_x, mesh_y = torch.meshgrid(
            torch.arange(-(ks//2), (ks//2)+1), torch.arange(-(ks//2), (ks//2)+1))


        sigx = sigx.view(-1, 1, 1)  # (0, 1)
        sigy = sigy.view(-1, 1, 1)  # (0, 1)
        theta = theta.view(-1, 1, 1)  # (0, 1)
        sigr = sigr.view(-1, 1, 1)
        # --------------
        spatial_kernel = (
                torch.exp(- (mesh_x.cuda() ** 2 / (2 * sigr ** 2) +
                             mesh_y.cuda() ** 2 / (2 * sigr ** 2))))
        # --------------
        multiplier = 1  # 1 / (2*pi*sigma_x*sigma_y*math.sqrt(1-rho**2) + self.eps)
        e_multiplier = -1 / 2  # * 1/(self.max_sigma) # -1 * (1/(2*(1-rho**2)+self.eps))
        # x方向中心像素点的值与其它像素点的值的差值
        disx = torch.abs(x[:, ks // 2, :, :].unsqueeze(1) - x)
        disx = torch.permute(disx, dims=[0, 3, 1, 2]).reshape(-1, ks, ks).cuda()
        # y方向中心像素点的值与其它像素点的值的差值
        disy = torch.abs(x[:, :, ks // 2, :].unsqueeze(2) - x)
        disy = torch.permute(disy, dims=[0, 3, 1, 2]).reshape(-1, ks, ks).cuda()
        # ============
        rho = theta
        x_nominal = (sigx * disx) ** 2
        y_nominal = (sigy * disy) ** 2

        xy_nominal = sigx * disx * sigy * disy
        exp_term = e_multiplier * (x_nominal - 2 * rho * xy_nominal + y_nominal)
        color_kernel = multiplier * torch.exp(exp_term)
        # ============= 方向自由
        kernel = spatial_kernel * color_kernel


        kernel = kernel / torch.sum(kernel, dim=[1, 2], keepdim=True)
        kernel = torch.permute(kernel.view(B, HW, ks, ks), (0, 2, 3, 1))
        # 将convkernal尺寸转换为 [num_channels, 1, 5, 5]
        kernel = kernel.to(x.device)
        res = torch.sum(kernel * x, axis=[1, 2])

        return res

class BilateralNet(nn.Module):
    def __init__(self, KS=5):
        super(BilateralNet, self).__init__()
        nf = 64
        self.KS = KS
        self.upscale = 4
        self.conv1x = nn.Conv2d(1, nf, (2, 2), stride=1, padding=0, dilation=1)
        self.conv1c = nn.Conv2d(1, nf, (2, 2), stride=1, padding=0, dilation=2)

        self.conv2 = DenseConv(nf, nf)
        self.conv3 = DenseConv(nf + nf * 1, nf)
        self.conv4 = DenseConv(nf + nf * 2, nf)
        self.conv5 = DenseConv(nf + nf * 3, nf)

        self.conv6 = Conv(nf * 5, 4, 1)


        self.mods = ['c', 's', 'x']

        self.conv1cs = conv1(3, 1)

        self.ag = AnisotropicGaussianFilter(num_channels=1)

        self.avg_factor = 4
        self.norm = 255
        self.bias = self.norm // 2


    def construct(self, x):
        # B, C, H, W = x.size()
        x = self.conv2(F.relu(x))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        par = self.conv6(x)

        return par

    def Fkernek(self, x, mod):
        B, C, H, W = x.size()
        x = x.reshape(B * C, 1, H, W)
        if mod == "s":
            c = self.conv1cs(x, 's')
        elif mod == "c":
            c = self.conv1c(x)
        else:
            c = self.conv1x(x)

        res = self.construct(c)

        return res

    def ensemble(self, x):

        pad = 1
        x0 = F.pad(x, (0, pad, 0, pad), mode='reflect')
        rot1 = self.Fkernek(x0, 'x')

        x2 = torch.rot90(x, 2, [2, 3])
        x2 = F.pad(x2, (0, pad, 0, pad), mode='reflect')
        rot2 = self.Fkernek(x2, 'x')
        rot2 = torch.rot90(rot2, 2, [2, 3])
        rot1_x = (rot1 + rot2)/2.

        x1 = torch.rot90(x, 1, [2, 3])
        x1 = F.pad(x1, (0, pad, 0, pad), mode='reflect')
        rot2 = self.Fkernek(x1, 'x')
        rot2 = torch.rot90(rot2, 3, [2, 3])

        x3 = torch.rot90(x, 3, [2, 3])
        x3 = F.pad(x3, (0, pad, 0, pad), mode='reflect')
        rot4 = self.Fkernek(x3, 'x')
        rot4 = torch.rot90(rot4, 1, [2, 3])
        rot2_x = (rot2 + rot4)/2.
        # =====================================================
        pad = 2
        x0 = F.pad(x, (0, pad, 0, pad), mode='reflect')
        rot1 = self.Fkernek(x0, 's')

        x2 = torch.rot90(x, 2, [2, 3])
        x2 = F.pad(x2, (0, pad, 0, pad), mode='reflect')
        rot2 = self.Fkernek(x2, 's')
        rot2 = torch.rot90(rot2, 2, [2, 3])
        rot1_s = (rot1 + rot2)/2.

        x1 = torch.rot90(x, 1, [2, 3])
        x1 = F.pad(x1, (0, pad, 0, pad), mode='reflect')
        rot2 = self.Fkernek(x1, 's')
        rot2 = torch.rot90(rot2, 3, [2, 3])

        x3 = torch.rot90(x, 3, [2, 3])
        x3 = F.pad(x3, (0, pad, 0, pad), mode='reflect')
        rot4 = self.Fkernek(x3, 's')
        rot4 = torch.rot90(rot4, 1, [2, 3])
        rot2_s = (rot2 + rot4)/2.
        # =====================================================

        pad = 2
        x0 = F.pad(x, (0, pad, 0, pad), mode='reflect')
        rot1 = self.Fkernek(x0, 'c')

        x2 = torch.rot90(x, 2, [2, 3])
        x2 = F.pad(x2, (0, pad, 0, pad), mode='reflect')
        rot2 = self.Fkernek(x2, 'c')
        rot2 = torch.rot90(rot2, 2, [2, 3])
        rot1_c = (rot1 + rot2)/2.

        x1 = torch.rot90(x, 1, [2, 3])
        x1 = F.pad(x1, (0, pad, 0, pad), mode='reflect')
        rot2 = self.Fkernek(x1, 'c')
        rot2 = torch.rot90(rot2, 3, [2, 3])

        x3 = torch.rot90(x, 3, [2, 3])
        x3 = F.pad(x3, (0, pad, 0, pad), mode='reflect')
        rot4 = self.Fkernek(x3, 'c')
        rot4 = torch.rot90(rot4, 1, [2, 3])
        rot2_c = (rot2 + rot4)/2.
        rot_scx = (rot1_x + rot1_s + rot1_c +
                   rot2_x + rot2_s + rot2_c) / 6.

        return rot_scx



    def forward(self, x_in):
        B, C, H, W = x_in.size()
        par = self.ensemble(x_in)

        sigx = torch.clamp(torch.sigmoid(par[:, 0:1].view(B, -1)) + 1e-6, min=0, max=1)  # [B, 1, H, W] -> [B, H*W]
        sigy = torch.clamp(torch.sigmoid(par[:, 1:2].view(B, -1)) + 1e-6, min=0, max=1)  # [B, 1, H, W] -> [B, H*W]
        theta = torch.clamp(torch.tanh(par[:, 2:3].view(B, -1)), min=-1, max=1)  # [B, 1, H, W] -> [B, H*W]
        sigr = torch.clamp(torch.tanh(par[:, 3:4].view(B, -1)) + 1e-6, min=-1, max=1)  # [B, 1, H, W] -> [B, H*W]


        x_tmp = F.pad(x_in, (self.KS // 2, self.KS // 2, self.KS // 2, self.KS // 2), mode='constant', value=0)
        x_in_unf = F.unfold(x_tmp, kernel_size=(self.KS, self.KS), stride=1, padding=0)

        outs = self.ag(x_in_unf.reshape(B, self.KS, self.KS, -1), sigx*20, sigy*20, theta, sigr * 10 + 10).view(B, C, H, W)


        return outs, sigx, sigy, theta, sigr

