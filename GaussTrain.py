import copy
import cv2
from tqdm import tqdm
import numpy as np

from ParameterNet_ada3mulut import BilateralNet
import torch.optim as optim
from mini_batch_loader import MiniBatchLoader

from utils import *
TRAINING_DATA_PATH = "train.txt"
IMAGE_DIR_PATH = "..//"

corp_size = 64
BATCH_SIZE = 32
totalIter = 300000
LR0 = 0.001


torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    model = init_net(BilateralNet(KS=5).to(device), 'kaiming', gpu_ids=[])
    optimizer = optim.Adam(model.parameters(), lr=LR0)
    i_index = 0

    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        IMAGE_DIR_PATH,
        corp_size)

    # train dataset
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)

    mse = torch.nn.MSELoss()
    for n_epi in tqdm(range(0, totalIter), ncols=70, initial=0):
        r = indices[i_index: i_index + BATCH_SIZE]
        raw_x = mini_batch_loader.load_training_data(r)
        label = copy.deepcopy(raw_x)
        if n_epi % 10 == 0:
            #     # cv2.imwrite('./test_img/'+'ori%2d' % (t+c)+'.jpg', current_state.image[20].transpose(1, 2, 0) * 255)
            image = np.asanyarray(raw_x[0].transpose(1, 2, 0) * 255, dtype=np.uint8)
            image = np.squeeze(image)
            cv2.imshow("l", image)
            cv2.waitKey(1)
        raw_n = np.random.normal(0, 0.1, label.shape).astype(label.dtype)
        ins = np.clip(label + raw_n, a_min=0., a_max=1.)
        label = np.clip(label, a_min=0., a_max=1.)


        if n_epi % 10 == 0:
            image = np.asanyarray(ins[0].transpose(1, 2, 0) * 255, dtype=np.uint8)
            image = np.squeeze(image)
            cv2.imshow("X", image)
            cv2.waitKey(1)

        l = torch.from_numpy(label).float().cuda()
        x = torch.from_numpy(ins).float().cuda()
        out, sigx, sigy, theta, sigr = model(x)

        if n_epi % 10 == 0:
            image = np.asanyarray(out[0].detach().cpu().numpy().transpose(1, 2, 0)*255 , dtype=np.uint8)
            image = np.squeeze(image)
            cv2.imshow("temp", image)
            cv2.waitKey(1)

        loss = mse(out, l)*100
        print("epoch: {}, loss: {}".format(n_epi, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for p in optimizer.param_groups:
            p['lr'] = LR0 * ((1 - n_epi / totalIter) ** 0.9)

        # if n_epi % 100 == 0:
        #     torch.save(model.state_dict(), "./LUTsModel_12000/{}_{:.4f}.pth".format(n_epi, loss.item()))

        if i_index + BATCH_SIZE >= train_data_size:
            i_index = 0
            indices = np.random.permutation(train_data_size)
        else:
            i_index += BATCH_SIZE

        if i_index + 2 * BATCH_SIZE >= train_data_size:
            i_index = train_data_size - BATCH_SIZE




if __name__ == '__main__':
    main()

