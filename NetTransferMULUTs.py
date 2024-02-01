
import torch
import numpy as np
import matplotlib.pyplot as plt

from ParameterNet_ada3mulut import BilateralNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAMPLING_INTERVAL = 4

def paint_amap(acmap):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    plt.imshow(image, vmin=1, vmax=9)
    plt.colorbar()
    # plt.pause(1)
    plt.show()
    # plt.close('all')



model = BilateralNet().to(device)
model.eval()
model.load_state_dict(torch.load("./LUTsModel/30000_0.1687.pth"))
print("-----------------")

mod = ['x', 's', 'c']
with torch.no_grad():
    model.eval()
    # 1D input
    # base = torch.arange(0, 257, 1)
    base = torch.arange(0, 257, 2 ** SAMPLING_INTERVAL)  # 0-256 像素值范围，下采样，只采样2**4=16个种类像素值
    base[-1] -= 1
    L = base.size(0)
    # [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]

    # 2D input
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)  # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    second = base.cuda().repeat(L)  # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D input
    third = base.cuda().unsqueeze(1).repeat(1, L * L).reshape(-1)  # 256*256*256   0 x65536|1 x65536|...|255 x65536
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)  # [256*256*256, 3]

    # 4D input
    fourth = base.cuda().unsqueeze(1).repeat(1, L * L * L).reshape(
        -1)  # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebythree = onebythree.repeat(L, 1)
    onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)  # [256*256*256*256, 4]

    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1, 1, 2, 2).float()
    print("Input size: ", input_tensor.size())
    # -----------------------------------------------
    #

    for ks in mod:
        LUT = []
                if ks == 'x':
            intputs = torch.zeros((input_tensor.size(0), 1, 4, 4))
            intputs[:, :, 0, 0] = input_tensor[:, :, 0, 0]
            intputs[:, :, 1, 1] = input_tensor[:, :, 0, 1]
            intputs[:, :, 2, 2] = input_tensor[:, :, 1, 0]
            intputs[:, :, 3, 3] = input_tensor[:, :, 1, 1]
        elif ks == 'c':
            intputs = torch.zeros((input_tensor.size(0), 1, 4, 4))
            intputs[:, :, 0, 0] = input_tensor[:, :, 0, 0]
            intputs[:, :, 0, 1] = input_tensor[:, :, 0, 1]
            intputs[:, :, 0, 2] = input_tensor[:, :, 1, 0]
            intputs[:, :, 0, 3] = input_tensor[:, :, 1, 1]
        elif ks == 's':
            intputs = torch.zeros((input_tensor.size(0), 1, 2, 2))
            intputs[:, :, 0, 0] = input_tensor[:, :, 0, 0]
            intputs[:, :, 0, 1] = input_tensor[:, :, 0, 1]
            intputs[:, :, 1, 0] = input_tensor[:, :, 1, 0]
            intputs[:, :, 1, 1] = input_tensor[:, :, 1, 1]

        NUM = 1000 # 采样>=5时，调整为10
        # Split input to not over GPU memory
        B = input_tensor.size(0) // NUM

        for b in range(NUM):
            print("Processing: ", b)
            if b == NUM-1:
                raw_x = intputs[b*B:].numpy() / 255.
            else:
                raw_x = intputs[b*B:(b+1)*B].numpy() / 255.

            res = model.Fkernek(torch.FloatTensor(raw_x).cuda(), ks).detach().cpu().numpy()

            LUT += [res]

        LUTs = np.concatenate(LUT, 0)
        print("Resulting LUT size: ", LUTs.shape)
        np.save("./LUTs/sample_{}_LUTs_{}".format(SAMPLING_INTERVAL, ks), LUTs)
