import os
import numpy as np
import cv2


class MiniBatchLoader(object):

    def __init__(self, train_path, image_dir_path, crop_size):

        # load data paths
        self.training_path_infos = self.read_paths(train_path, image_dir_path)
        # self.testing_path_infos = self.read_paths(test_path, image_dir_path)
        # self.val_path_infos = self.read_paths(val_path, image_dir_path)

        self.crop_size = crop_size

    # test ok
    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            src_full_path = os.path.join(src_path, line)
            if os.path.isfile(src_full_path):
                yield src_full_path

    # test ok
    @staticmethod
    def count_paths(path):
        c = 0
        for _ in open(path):
            c += 1
        return c

    # test ok
    @staticmethod
    def read_paths(txt_path, src_path):
        cs = []
        for pair in MiniBatchLoader.path_label_generator(txt_path, src_path):
            cs.append(pair)
        return cs

    def load_training_data(self, indices):
        return self.load_data(self.training_path_infos, indices, augment=True, color=False)

    def load_val_data(self, indices):
        return self.load_data(self.val_path_infos, indices, validation=True, color=False)

    # def load_testing_data(self, indices):
    #     return self.load_data(self.testing_path_infos, indices, color=False)

    # test ok
    def load_data(self, path_infos, indices, augment=False, validation=False, color=False):
        mini_batch_size = len(indices)
        if color:
            in_channels = 3
        else:
            in_channels = 1

        if augment:
            xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)

            for i, index in enumerate(indices):
                path = path_infos[index]
                if color:
                    img = cv2.imread(path, 1)
                    h, w, c = img.shape
                else:
                    img = cv2.imread(path, 0)
                    h, w = img.shape
                    c = 1
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))

                if h < self.crop_size or w < self.crop_size:
                    continue
                if np.random.rand() > 0.5:
                    img = np.fliplr(img)

                if np.random.rand() > 0.5:
                    angle = 10 * np.random.rand()
                    if np.random.rand() > 0.5:
                        angle *= -1
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                    img = cv2.warpAffine(img, M, (w, h))

                rand_range_h = h - self.crop_size
                rand_range_w = w - self.crop_size

                x_offset = np.random.randint(rand_range_w)
                y_offset = np.random.randint(rand_range_h)
                if c == 3:
                    img = img[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]
                    xs[i, :, :, :] = (img / 255.).transpose(2, 0, 1).astype(np.float32)
                else:
                    img = img[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size]
                    xs[i, 0, :, :] = (img / 255.).astype(np.float32)
                # if color:
                #     xs[i, :, :, :] = (img / 255.).astype(np.float32)
                # else:
                #     xs[i, :, :, :] = (img / 255.).astype(np.float32)


        elif validation:
            xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)

            for i, index in enumerate(indices):
                path = path_infos[index]

                if color:
                    img = cv2.imread(path, 1)
                else:
                    img = np.expand_dims(cv2.imread(path, 0), 2)
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))
                w, h, c = img.shape
                # print(img.shape)
                if c == 3:
                    xs[i, :, :, :] = cv2.resize(np.asarray(img),
                                    (self.crop_size, self.crop_size),
                                    interpolation=cv2.INTER_AREA).transpose(2, 0, 1) / 255.
                else:
                    xs[i, 0, :, :] = cv2.resize(np.asarray(img).squeeze(),
                                    (self.crop_size, self.crop_size),
                                    interpolation=cv2.INTER_AREA) / 255.

        elif mini_batch_size == 1:
            for i, index in enumerate(indices):
                path = path_infos[index]

                img = cv2.imread(path, 0)
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))

            h, w = img.shape
            xs = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
            xs[0, 0, :, :] = (img / 255.).astype(np.float32)

        else:
            raise RuntimeError("mini batch size must be 1 when testing")

        return xs
