import numpy as np
import glob
import cv2
from tqdm import tqdm
from PIL import Image
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_cal
from skimage.metrics import structural_similarity as ssim_cal
import time

SAMPLING_INTERVAL = 4       # N bit uniform sampling
L = 2 ** (8 - SAMPLING_INTERVAL) + 1
q = 2**SAMPLING_INTERVAL
outsize = 4
mod = ['x', 's', 'c']
class AnisotropicGaussianFilter(torch.nn.Module):
    def __init__(self, num_channels):
        super(AnisotropicGaussianFilter, self).__init__()
        self.num_channels = num_channels


    def forward(self, x, sigx, sigy, theta, sigcx, sigcy):
        B, ks, ks, HW = x.size()
        mesh_x, mesh_y = torch.meshgrid(
            torch.arange(-(ks//2), (ks//2)+1), torch.arange(-(ks//2), (ks//2)+1))


        sigx = sigx.view(-1, 1, 1)  # (0, 1)
        sigy = sigy.view(-1, 1, 1)  # (0, 1)
        theta = theta.view(-1, 1, 1)  # (0, 1)
        sigr = sigcx.view(-1, 1, 1)

        multiplier = 1  # 1 / (2*pi*sigma_x*sigma_y*math.sqrt(1-rho**2) + self.eps)
        e_multiplier = - 1 / 2  # * 1/(self.max_sigma) # -1 * (1/(2*(1-rho**2)+self.eps))
        rho = theta
        x_nominal = (sigx * mesh_x.cuda()) ** 2
        y_nominal = (sigy * mesh_y.cuda()) ** 2
        xy_nominal = sigx * mesh_x.cuda() * sigy * mesh_y.cuda()
        exp_term = e_multiplier * (x_nominal - 2 * rho * xy_nominal + y_nominal)
        spatial_kernel = multiplier * torch.exp(exp_term)
        # --------------
        disx = torch.abs(x[:, ks // 2, :, :].unsqueeze(1) - x)
        disx = torch.permute(disx, dims=[0, 3, 1, 2]).reshape(-1, ks, ks).cuda()
        disy = torch.abs(x[:, :, ks // 2, :].unsqueeze(2) - x)
        disy = torch.permute(disy, dims=[0, 3, 1, 2]).reshape(-1, ks, ks).cuda()
        # ============ 
        color_kernel = (
                    torch.exp(- (disx.cuda() ** 2 / (2 * sigr ** 2) +
                                 disy.cuda() ** 2 / (2 * sigr ** 2))))

        
        kernel = spatial_kernel * color_kernel


        kernel = kernel / torch.sum(kernel, dim=[1, 2], keepdim=True)
        kernel = torch.permute(kernel.view(B, HW, ks, ks), (0, 2, 3, 1))
        # [num_channels, 1, 5, 5]
        kernel = kernel.to(x.device)
        res = torch.sum(kernel * x, axis=[1, 2])

        return res

def FourSimplexInterpFaster(weight, img_in, h, w, interval, rot, mode='s'):
    q = 2 ** interval
    L = 2 ** (8 - interval) + 1

        if mode == "s":
        # Extract MSBs
        img_a1 = img_in[:, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, 0:0 + h, 1:1 + w] // q
        img_c1 = img_in[:, 1:1 + h, 0:0 + w] // q
        img_d1 = img_in[:, 1:1 + h, 1:1 + w] // q

        # Extract LSBs
        fa = img_in[:, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, 0:0 + h, 1:1 + w] % q
        fc = img_in[:, 1:1 + h, 0:0 + w] % q
        fd = img_in[:, 1:1 + h, 1:1 + w] % q

    elif mode == 'c':
        img_a1 = img_in[:, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, 0:0 + h, 1:1 + w] // q
        img_c1 = img_in[:, 0:0 + h, 2:2 + w] // q
        img_d1 = img_in[:, 0:0 + h, 3:3 + w] // q

        fa = img_in[:, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, 0:0 + h, 1:1 + w] % q
        fc = img_in[:, 0:0 + h, 2:2 + w] % q
        fd = img_in[:, 0:0 + h, 3:3 + w] % q

    elif mode == 'x':
        img_a1 = img_in[:, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, 1:1 + h, 1:1 + w] // q
        img_c1 = img_in[:, 2:2 + h, 2:2 + w] // q
        img_d1 = img_in[:, 3:3 + h, 3:3 + w] // q

        fa = img_in[:, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, 1:1 + h, 1:1 + w] % q
        fc = img_in[:, 2:2 + h, 2:2 + w] % q
        fd = img_in[:, 3:3 + h, 3:3 + w] % q
    else:
        # more sampling modes can be implemented similarly
        raise ValueError("Mode {} not implemented.".format(mode))

    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1


    p0000 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))

    p0001 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))
    p0010 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))
    p0011 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))
    p0100 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))
    p0101 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))
    p0110 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))
    p0111 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))

    p1000 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))
    p1001 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))
    p1010 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))
    p1011 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))
    p1100 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))
    p1101 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))
    p1110 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))
    p1111 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))

    # Output image holder
    out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], outsize))
    sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2]
    out = out.reshape(sz, -1)

    p0000 = p0000.reshape(sz, -1)
    p0100 = p0100.reshape(sz, -1)
    p1000 = p1000.reshape(sz, -1)
    p1100 = p1100.reshape(sz, -1)
    fa = fa.reshape(-1, 1)

    p0001 = p0001.reshape(sz, -1)
    p0101 = p0101.reshape(sz, -1)
    p1001 = p1001.reshape(sz, -1)
    p1101 = p1101.reshape(sz, -1)
    fb = fb.reshape(-1, 1)
    fc = fc.reshape(-1, 1)

    p0010 = p0010.reshape(sz, -1)
    p0110 = p0110.reshape(sz, -1)
    p1010 = p1010.reshape(sz, -1)
    p1110 = p1110.reshape(sz, -1)
    fd = fd.reshape(-1, 1)

    p0011 = p0011.reshape(sz, -1)
    p0111 = p0111.reshape(sz, -1)
    p1011 = p1011.reshape(sz, -1)
    p1111 = p1111.reshape(sz, -1)

    fab = fa > fb;
    fac = fa > fc;
    fad = fa > fd

    fbc = fb > fc;
    fbd = fb > fd;
    fcd = fc > fd

    i1 = i = np.logical_and.reduce((fab, fbc, fcd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i2 = i = np.logical_and.reduce((~i1[:, None], fab, fbc, fbd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i3 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], fab, fbc, fad)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i4 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc)).squeeze(1)

    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]

    i5 = i = np.logical_and.reduce((~(fbc), fab, fac, fbd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i6 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], fab, fac, fcd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i7 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i8 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]

    i9 = i = np.logical_and.reduce((~(fbc), ~(fac), fab, fbd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
    # i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], fab, fcd)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    # i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], fab, fad)).squeeze(1)  # c > a > d > b
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd)).squeeze(1)  # c > d > a > b
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i12 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]

    i13 = i = np.logical_and.reduce((~(fab), fac, fcd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i14 = i = np.logical_and.reduce((~(fab), ~i13[:, None], fac, fad)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i15 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], fac, fbd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i16 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]

    i17 = i = np.logical_and.reduce((~(fab), ~(fac), fbc, fad)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i18 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], fbc, fcd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i19 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i20 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]

    i21 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), fad)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i22 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i23 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i24 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None])).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]

    out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], 6))
    out = np.rot90(out, rot, [1, 2])
    out = out / q
    return out


def process_img(x, LUT_X, LUT_S, LUT_C,h, w):
    pad = 1
    x0 = np.pad(x, ((0, pad), (0, pad), (0, 0)), mode='reflect').transpose((2, 0, 1))
    # _, h, w = x0.shape
    rot0 = FourSimplexInterpFaster(LUT_X, x0, h, w, interval=4, rot=0, mode='x')

    x2 = np.rot90(x, 2)
    x2 = np.pad(x2, ((0, pad), (0, pad), (0, 0)), mode='reflect').transpose((2, 0, 1))
    # _, h, w = x0.shape
    rot2 = FourSimplexInterpFaster(LUT_X, x2, h, w, interval=4, rot=2, mode='x')
    rot1_x = (rot0 + rot2) / 2.
    # # -------------------------------
    x1 = np.rot90(x, 1)
    x1 = np.pad(x1, ((0, pad), (0, pad), (0, 0)), mode='reflect').transpose((2, 0, 1))
    # _, h, w = x0.shape
    rot1 = FourSimplexInterpFaster(LUT_X, x1, w, h, interval=4, rot=3, mode='x')

    x3 = np.rot90(x, 3)
    x3 = np.pad(x3, ((0, pad), (0, pad), (0, 0)), mode='reflect').transpose((2, 0, 1))
    # _, h, w = x0.shape
    rot3 = FourSimplexInterpFaster(LUT_X, x3, w, h, interval=4, rot=1, mode='x')
    rot2_x = (rot1 + rot3) / 2.

    # # =====================================================
    pad = 3
    x0 = np.pad(x, ((0, pad), (0, pad), (0, 0)), mode='reflect').transpose((2, 0, 1))
    # _, h, w = x0.shape
    rot0 = FourSimplexInterpFaster(LUT_S, x0, h, w, interval=4, rot=0, mode='s')

    x2 = np.rot90(x, 2)
    x2 = np.pad(x2, ((0, pad), (0, pad), (0, 0)), mode='reflect').transpose((2, 0, 1))
    # _, h, w = x0.shape
    rot2 = FourSimplexInterpFaster(LUT_S, x2, h, w, interval=4, rot=2, mode='s')
    rot1_s = (rot0 + rot2) / 2.
    # # -------------------------------

    x1 = np.rot90(x, 1)
    x1 = np.pad(x1, ((0, pad), (0, pad), (0, 0)), mode='reflect').transpose((2, 0, 1))
    # _, h, w = x0.shape
    rot1 = FourSimplexInterpFaster(LUT_S, x1, w, h, interval=4, rot=3, mode='s')

    x3 = np.rot90(x, 3)
    x3 = np.pad(x3, ((0, pad), (0, pad), (0, 0)), mode='reflect').transpose((2, 0, 1))
    # _, h, w = x0.shape
    rot3 = FourSimplexInterpFaster(LUT_S, x3, w, h, interval=4, rot=1, mode='s')
    rot2_s = (rot1 + rot3) / 2.
    # # =====================================================

    pad = 3
    x0 = np.pad(x, ((0, pad), (0, pad), (0, 0)), mode='reflect').transpose((2, 0, 1))
    # _, h, w = x0.shape
    rot0 = FourSimplexInterpFaster(LUT_C, x0, h, w, interval=4, rot=0, mode='c')

    x2 = np.rot90(x, 2)
    x2 = np.pad(x2, ((0, pad), (0, pad), (0, 0)), mode='reflect').transpose((2, 0, 1))
    # _, h, w = x0.shape
    rot2 = FourSimplexInterpFaster(LUT_C, x2, h, w, interval=4, rot=2, mode='c')
    rot1_c = (rot0 + rot2) / 2.

    x1 = np.rot90(x, 1)
    x1 = np.pad(x1, ((0, pad), (0, pad), (0, 0)), mode='reflect').transpose((2, 0, 1))
    # _, h, w = x0.shape
    rot1 = FourSimplexInterpFaster(LUT_C, x1, w, h, interval=4, rot=3, mode='c')

    x3 = np.rot90(x, 3)
    x3 = np.pad(x3, ((0, pad), (0, pad), (0, 0)), mode='reflect').transpose((2, 0, 1))
    # _, h, w = x0.shape
    rot3 = FourSimplexInterpFaster(LUT_C, x3, w, h, interval=4, rot=1, mode='c')
    rot2_c = (rot1 + rot3) / 2.

    # =========================================
    rot_scx = (rot1_x + rot1_s + rot1_c +
               rot2_x + rot2_s + rot2_c) / 6.
    par = torch.Tensor(rot_scx.transpose((0, 3, 1, 2))).cuda()
    return par




# TEST_DIR = 'D://Dataset/BSD68_color/'      # Test images
TEST_DIR = 'D://Dataset/Set5/'      # Test images
# TEST_DIR = 'D://Dataset/manga109/'      # Test images
# TEST_DIR = 'D://Dataset/BSDS100/'      # Test images
# TEST_DIR = 'D://Dataset/Urban100/'      # Test images\


LUT_PATH_X = "./LUTs/sample_{}_LUTs_{}.npy".format(SAMPLING_INTERVAL, 'x')
LUT_X = np.load(LUT_PATH_X).astype(np.float32).reshape(-1, 6)

LUT_PATH_S = "./LUTs/sample_{}_LUTs_{}.npy".format(SAMPLING_INTERVAL, 's')
LUT_S = np.load(LUT_PATH_S).astype(np.float32).reshape(-1, 6)

LUT_PATH_C = "./LUTs/sample_{}_LUTs_{}.npy".format(SAMPLING_INTERVAL, 'c')
LUT_C = np.load(LUT_PATH_C).astype(np.float32).reshape(-1, 6)

# Test clean images
files_gt = glob.glob(TEST_DIR + '*')
files_gt.sort()
len = len(files_gt)
# ---------------
val_pnsr = 0.
val_ssim = 0.
val_time = 0.

ag = AnisotropicGaussianFilter(num_channels=1)
KS = 5


# ---------------
for ti, fn in enumerate(tqdm(files_gt)):
    # Load noise image and gt
    img_gt = np.array(Image.open(files_gt[ti])).astype(np.uint)
    if img_gt.ndim == 2:
        img_gt = np.expand_dims(img_gt, axis=2)
    h, w, c = img_gt.shape  # (481, 321)
    noise = np.random.normal(0, 25, [h, w, 1])
    x = np.clip(img_gt + noise, a_min=0, a_max=255)
    cv2.imshow('x', x[:,:,::-1].astype(np.uint8))

    par = process_img(x, LUT_X, LUT_S, LUT_C, h, w)

    x_in = torch.Tensor(x.transpose((2, 0, 1))).unsqueeze(1)/255.
    B, C, H, W = x_in.size()

    t1 = time.time()
    sigx = torch.clamp(torch.sigmoid(par[:, 0:1].view(B, -1)) + 1e-6, min=0, max=1)  # [B, 1, H, W] -> [B, H*W]
    sigy = torch.clamp(torch.sigmoid(par[:, 1:2].view(B, -1)) + 1e-6, min=0, max=1)  # [B, 1, H, W] -> [B, H*W]
    theta = torch.clamp(torch.tanh(par[:, 2:3].view(B, -1)) + 1e-6, min=-1, max=1)  # [B, 1, H, W] -> [B, H*W]

    sigr = torch.clamp(torch.sigmoid(par[:, 3:4].view(B, -1)) + 1e-6, min=0, max=1)  # [B, 1, H, W] -> [B, H*W]



    x_tmp = F.pad(x_in, (KS // 2, KS // 2, KS // 2, KS // 2), mode='constant', value=0)
    x_in_unf = F.unfold(x_tmp, kernel_size=(KS, KS), stride=1, padding=0)

    outs = ag(x_in_unf.reshape(B, KS, KS, -1), sigx*10, sigy*10, theta, sigr * 10).view(1, B, H, W)

    # --------------
    t2 = time.time()
    val_time += t2 - t1
    
    cv2.imshow("label", img_gt[:,:,::-1].astype(np.uint8))
    cv2.imshow("outs", (outs.cpu().numpy().squeeze().transpose(1, 2, 0)[:,:,::-1]*255).astype(np.uint8))
    # cv2.imwrite('./Set5_tst/res_{}'.format(fn.split('\\')[-1]), (outs.cpu().numpy().squeeze().transpose(1, 2, 0)[:,:,::-1]*255).astype(np.uint8))
    # cv2.imwrite('./Set5_tst/ins_{}'.format(fn.split('\\')[-1]), x[:,:,::-1].astype(np.uint8))

    cv2.waitKey(1)

    val_pnsr += psnr_cal(outs.cpu().numpy().squeeze().transpose(1, 2, 0), img_gt/255., data_range=1.)
    val_ssim += ssim_cal(outs.cpu().numpy().squeeze().transpose(1, 2, 0), img_gt/255., data_range=1.,
                         multichannel=True)
    # val_pnsr += psnr_cal(outs.cpu().numpy().squeeze(), img_gt.squeeze() / 255., data_range=1.)
    # val_ssim += ssim_cal(outs.cpu().numpy().squeeze(), img_gt.squeeze() / 255., data_range=1.,
    #                      multichannel=False)


print(len)
print("PSNR: {:4f}".format(val_pnsr/len))
print("SSIM: {:4f}".format(val_ssim/len))




















