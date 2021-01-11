import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os,torch
import torch.nn as nn
from NO_PRE import image_utils
import argparse,random
import matplotlib.pyplot as plt
from tqdm import tqdm


# 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
# [1290, 281, 717, 4772, 1982, 705, 2524]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/home/siasun/zwp/Self-Cure-Network-master/raf_src/datasets/raf-basic', help='Raf-DB dataset path.')
    parser.add_argument('--train_label_path', type=str, default='EmoLabel/train_label.txt',help='train label path')
    parser.add_argument('--val_label_path', type=str, default='EmoLabel/test_label.txt',help='test label path')

    parser.add_argument('--classes', type=int, default=7, help='label nums')

    parser.add_argument('--checkpoint', type=str, default='model_RAF_pre_Wlogsm1_d0.5', help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=True, help='Pretrained weights')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='Drop out rate.')

    return parser.parse_args()


class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:,
                     LABEL_COLUMN].values - 1

        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx



def compute_dataset_weight(train_label_path, classes, device):
    dataset = pd.read_csv(train_label_path, sep=' ', header=None)
    train_label = dataset.iloc[:,1].values - 1
    ld = dict()
    for i in range(classes):
        ld[i] = 0
    for label in train_label:
        ld[label] += 1
    ll = ld.values()

    weight = torch.tensor(min(ll)/np.array(list(ll)))
    return ll, weight

def run():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    args = parse_args()


    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    train_dataset = RafDataSet(args.raf_path, phase='train', transform=data_transforms, basic_aug=True)

    print('Train set size:', train_dataset.__len__())

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    val_dataset = RafDataSet(args.raf_path, phase='test', transform=data_transforms_val)
    print('Validation set size:', val_dataset.__len__())


    device = torch.device("cpu")
    train_num, train_weight = compute_dataset_weight(train_label_path=os.path.join(args.raf_path, args.train_label_path),
                                    classes=args.classes, device=device)
    val_num, val_weight = compute_dataset_weight(train_label_path=os.path.join(args.raf_path, args.val_label_path),
                                    classes=args.classes, device=device)
    print("train_num:{}, train_weight:{}".format(train_num, train_weight))
    print("val_num:{}, val_weight:{}".format(val_num, val_weight))

if __name__ == "__main__":
    run()
