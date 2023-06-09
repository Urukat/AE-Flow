from typing import Any

import torch
import torch.nn as nn
from torchvision.datasets import VisionDataset
from torchvision import transforms
import os
import os.path as osp
from PIL import Image
from tqdm import tqdm


class AEFlowDataset(VisionDataset):
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Any:
        if self.train:
            label = 0
        else:
            label = self.labels[index]

        img = self.images[index]
        img = img.expand(3, -1, -1)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __init__(self, root, train, transform=None):
        super().__init__(root, transform)
        self.pre_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        self._process()
        self.train = train

        if train:
            self.images = torch.load(self.processed_names[0])
        else:
            normal = torch.load(self.processed_names[1])
            anomaly = torch.load(self.processed_names[2])

            normal_labels = torch.load(self.processed_names[3])
            anomaly_labels = torch.load(self.processed_names[4])

            self.images = torch.cat((normal, anomaly), dim=0)
            self.labels = torch.cat((normal_labels, anomaly_labels), dim=0)

    @property
    def raw_dir(self):
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, "processed")

    @property
    def raw_folders(self):
        folders_names = ["train/NORMAL", "test/NORMAL", "test/PNEUMONIA"]
        return [osp.join(self.raw_dir, name) for name in folders_names]

    @property
    def processed_names(self):
        files_names = ["train_NORMAL.pt", "test_NORMAL.pt", "test_ANOMALY.pt",
                       "test_NORMAL_labels.pt", "test_ANOMALY_labels.pt"]
        return [osp.join(self.processed_dir, name) for name in files_names]

    def _process(self):
        """
        Process dataset if processed files don't exist yet.
        :return: None
        """
        if folderandfiles_exist(self.processed_dir, self.processed_names):
            return

        if self.processed_dir is None:
            os.makedirs(self.processed_dir)
        self.process()
        print("Images pre-transformed !")

    def process(self):
        """
        Process dataset by pre_transforming images, creating labels and creating id to file name mapping.
        Save pytorch tensors containing processed datasets (train + test)

        :return: None
        """
        for i, folder in enumerate(self.raw_folders):
            data_list = []
            print(f"Processing {folder}")
            for f in tqdm(os.listdir(folder)):
                # in case some imges are saved as RGB format
                img = Image.open(osp.join(folder, f)).convert("L")
                if self.pre_transform is not None:
                    img = self.pre_transform(img)
                data_list.append(img.unsqueeze(0))
            torch.save(torch.cat(data_list, dim=0), self.processed_names[i])
            if i == 1:
                torch.save(torch.zeros(len(data_list)), self.processed_names[3])
            if i == 2:
                torch.save(torch.ones(len(data_list)), self.processed_names[4])


def folderandfiles_exist(folder, files):
    return folder is not None\
           and len(files) != 0 \
           and all([osp.exists(f) for f in files]) \
           and osp.exists(folder)
