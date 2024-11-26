import torch
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np



class DatasetText(torch.utils.data.Dataset):
    def __init__(self, path_folder, path_labels, transform, set_chars, max_len):
        super().__init__()
        self.path_labels = path_labels
        self.path_folder = path_folder
        self.transform = transform
        self.set_chars = set_chars
        self.max_len = max_len

        self.get_labels()
        print(f'Count data: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        img_path = os.path.join(self.path_folder, data[0])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img / 255.0
        # img = img.transpose([2, 0, 1]) # HWC -> CHW
        img = cv2.resize(img, (256, 128), interpolation=cv2.INTER_AREA)
        img = self.transform(img)
        # img = torch.from_numpy(img)

        return {'img': img, 'label': data[1]}

    def get_labels(self):
        f = open(self.path_labels, 'r')
        temp = f.read().split('\n')
        f.close()
        data = list()

        for row in temp:
            row = row.split('\t')
            flag = True

            if len(row[1]) > self.max_len:
                flag = False

            if flag:
                for i in row[1]:
                    if i not in self.set_chars:
                        flag = False

            if flag:
                data.append((row[0], row[1]))

        self.data = data

    def imshow(self, index):
        data = self.data[index]
        img_path = os.path.join(self.path_folder, data[0])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = img.transpose([0, 1, 2])
        plt.figure(figsize=(3, 5))

        print(data[1])
        plt.imshow(img);

