import os
import torch
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision import transforms

class ChestXrayDataset(VisionDataset):
    def __init__(self, root, name='chest_xray', split='train', label="NORMAL"):
        super().__init__(root)
        self.root = root
        self.name = name
        self.split = split
        self.label = label

        # We follow the author's step, there is no normalization
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
        ])
        

        self.filename = "{}_{}.pt".format(self.split, self.label)
        if(not os.path.exists(os.path.join(root, name, 'processed'))):
            os.makedirs(os.path.join(root, name, 'processed'))
            raw_data_path = os.path.join(root, name, 'raw')
            for split in os.listdir(raw_data_path):
                for label in os.listdir(os.path.join(raw_data_path, split)):
                    filename = "{}_{}.pt".format(split, label)
                    print(filename)
                    data_list = []
                    for image_name in tqdm(os.listdir(os.path.join(raw_data_path, split, label))):
                        img = Image.open(os.path.join(raw_data_path, split, label, image_name)).convert("L")
                        img = self.transform(img)
                        data_list.append(img)
                    torch.save(torch.cat(data_list, dim=0), os.path.join(root, name, 'processed', filename))
                    # print(len(data_list))
        
        self.image_list = torch.load(os.path.join(root, name, "processed", "{}_{}.pt".format(self.split, self.label)))
        
        if(self.label == "NORMAL"):
            self.label = 0
        else:
            self.label = 1
        # for train_normal.pt it should be 1341
        print(len(self.image_list))
    
    def __getitem__(self, index):
        img = self.image_list[index]
        label = self.label
        return img, label
    
    def __len__(self):
        return 