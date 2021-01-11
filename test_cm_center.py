import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms
import cv2
import random
import torch.nn as nn
from torchvision import models
from main import image_utils
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
# from main import device

parser = argparse.ArgumentParser()
parser.add_argument('--raf_path',default='/home/siasun/zwp/Self-Cure-Network-master/raf_src/datasets/raf-basic', help='path of test result')

parser.add_argument('--model', default='../PRE/i_model_RAF_pre_wSmLce3_05_01/epoch41_acc0.8611.pth',help='the path of trained model')
parser.add_argument('--test_result_path',default='../PRE/i_model_RAF_pre_wSmLce3_05_01/test_result',help='path of test result')
parser.add_argument('--csv_name',default='RAF_pre_wSmLce3_05_01.csv',help='name of csv')
parser.add_argument('--save_img_name',default='RAF_pre_wSmLce3_05_01.jpg',help='name of save img')

parser.add_argument('--batch_size',default=64,type=int)
parser.add_argument('--workers',default=4,type=int)
parser.add_argument('--pretrained', type=str, default=True, help='Pretrained weights')
parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')

args=parser.parse_args()
number_7expression = {0:'Surprise',1:'Fear',2:'Disgust',3:'Happiness',4:'Sadness',5:'Anger',6:'Neutral'}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                     LABEL_COLUMN].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

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

class Res18Feature(nn.Module):
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0):
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1*1

        fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimension 512

        self.fc = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7
        self.register_buffer('centers', (
                torch.rand((num_classes, fc_in_dim)).to(device) - 0.5) * 2)  # -1~1
    def forward(self, x):
        x = self.features(x)

        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def compute_acc(predict, target):
    return np.around(sum(1 for x, y in zip(predict, target) if x == y)/len(predict), decimals=4)

def draw_cm(predict, target, label, img_name='cm.jpg'):
    acc = compute_acc(predict, target)

    classes = sorted(set(target))
    cm = confusion_matrix(target, predict)
    s = np.sum(cm, 1, keepdims=True)
    cm = np.round(cm / s, decimals=2)

    plt.imshow(cm,  interpolation='nearest', cmap=plt.cm.Blues)
    indices = range(len(classes))
    if len(indices) != len(label):
        print('the label number not equal to target set number')
        return

    plt.title(img_name.split('.')[0]+'_acc_{}'.format(acc))
    plt.xlabel('predict')
    plt.ylabel('target')
    plt.xticks(indices, label, rotation=10)
    plt.yticks(indices, label)
    plt.ylim(len(classes) - 0.5, -0.5)
    plt.colorbar()
    for fi in range(len(cm)):
        for si in range(len(cm[fi])):
            plt.text(fi, si, cm[si, fi], ha='center')
    plt.savefig(os.path.join(args.test_result_path, img_name))


def pd_write_csv(target, predict, csv_name='pd.csv'):
    data = {'target':target, 'predict':predict}
    df = pd.DataFrame(data)
    df.to_csv(csv_name)

def pd_read_csv(csv_name):
    df = pd.read_csv(csv_name)
    target = list(df.target)
    predict = list(df.predict)
    return target, predict


def test():
    if not os.path.exists(args.test_result_path):
        os.makedirs(args.test_result_path)
    res18 = Res18Feature()
    if args.model:
        print('Loading model from ', args.model)
        trained = torch.load(args.model)
        model_state_dict = trained['model_state_dict']
        res18.load_state_dict(model_state_dict)
        res18.eval()
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    val_dataset = RafDataSet(args.raf_path, phase='test', transform=data_transforms_val)
    print('Validation set size:', val_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)
    res18 = res18.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        res18.eval()
        # torch.manual_seed(42)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(42)
        res18.apply(apply_dropout)
        predict_label = []
        target_label = []
        for batch_i, (imgs, targets, _) in enumerate(val_loader):
            outputs = res18(imgs.cuda())
            targets = targets.cuda()
            loss = criterion(outputs, targets)
            running_loss += loss
            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets)
            bingo_cnt += correct_num.sum()
            # print("predicts:{}".format(predicts))
            # print("predicts:{}".format(predicts.cpu()))


            ## draw weight on image


            ## 保存判断错误图像
            # for index in error_position:
            #     error_num += 1
            #     img_name = str(error_num)+'_target('+number_7expression[np.array(targets)[index]]+')_predict('+number_7expression[np.array(predicts)[index]]+').jpg'
            #     original_img = cv2.imread(val_loader.dataset.file_paths[indexes[index]])
            #     cv2.imwrite(os.path.join(args.test_result_path,img_name),original_img)

            # predict_label.extend([number_7expression[label] for label in np.array(predicts)])
            # target_label.extend([number_7expression[label] for label in np.array(targets)])
            predict_label.extend(np.array(predicts.cpu()))
            target_label.extend(np.array(targets.cpu()))

        data = {'target':target_label, 'predict':predict_label}
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(args.test_result_path, args.csv_name))

        labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
        draw_cm(predict_label, target_label, labels, img_name=args.save_img_name)

        running_loss = running_loss / iter_cnt
        acc = bingo_cnt.float() / float(val_dataset.__len__())
        print("Validation accuracy:%.4f. Loss:%.3f" % (acc, running_loss))
if __name__ == '__main__':
    test()