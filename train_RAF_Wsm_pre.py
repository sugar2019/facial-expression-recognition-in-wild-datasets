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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/home/siasun/zwp/Self-Cure-Network-master/raf_src/datasets/raf-basic', help='Raf-DB dataset path.')
    parser.add_argument('--train_label_path', type=str, default='EmoLabel/train_label.txt',help='train label path')
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

    def forward(self, x):
        x = self.features(x)

        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

def initialize_weight_goog(m, n=''):
    # weight init as per Tensorflow Official impl
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py

    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()

def plt_train_val_loss(train_losses, val_losses, checkpoint, name='train and val loss'):
    l1, = plt.plot(train_losses, color='blue', linestyle='-', linewidth=1.0, label='train_loss')
    l2, = plt.plot(val_losses, color='red', linestyle='-', linewidth=1.0, label='val_loss')
    plt.legend(handles=[l1, l2, ], loc='best')

    plt.title(name)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.xlim(0, num_epoch)
    plt.savefig(os.path.join(checkpoint, name+'.png'))
    plt.close()
def plt_train_val_acc(train_acces, val_acces, checkpoint, name='train and val acc'):
    l1, = plt.plot(train_acces, color='blue', linestyle='-', linewidth=1.0, label='train_acc')
    l2, = plt.plot(val_acces, color='red', linestyle='-', linewidth=1.0, label='val_acc')
    plt.legend(handles=[l1, l2, ], loc='best')

    plt.title(name)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    # plt.xlim(0, num_epoch)
    plt.savefig(os.path.join(checkpoint, name+'.png'))
    plt.close()

def compute_dataset_weight(train_label_path, classes, device):
    dataset = pd.read_csv(train_label_path, sep=' ', header=None)
    train_label = dataset.iloc[:,1].values - 1
    ld = dict()
    for i in range(classes):
        ld[i]=0
    for label in train_label:
        ld[label] += 1
    ll = ld.values()

    weight = torch.tensor(min(ll)/np.array(list(ll))).cuda(device)
    return weight

def weighted_softmax_loss(outputs,targets,weight, classes, device):
    softmax_outputs = torch.exp(outputs)/torch.sum(torch.exp(outputs), 1, keepdim=True)
    batch_size = len(outputs)
    targets = targets.unsqueeze(dim=1).cuda(device)
    one_hot = torch.zeros(batch_size, classes).cuda(device).scatter_(1, targets, 1)
    result = torch.sum(weight * torch.log(softmax_outputs) *one_hot)
    return -result



def run_training():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    args = parse_args()

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    res18 = Res18Feature(pretrained=args.pretrained, drop_rate=args.drop_rate)
    if not args.pretrained:
        for m in res18.modules():
            initialize_weight_goog(m)



    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    train_dataset = RafDataSet(args.raf_path, phase='train', transform=data_transforms, basic_aug=True)

    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

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

    params = res18.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)  # 默认lr=0.001
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=1e-4)
    else:
        raise ValueError("Optimizer not supported.")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    res18 = res18.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = compute_dataset_weight(train_label_path=os.path.join(args.raf_path, args.train_label_path),
                                    classes=args.classes, device=device)


    best_eval_acc = 0
    train_loss_file = open(os.path.join(args.checkpoint, 'train_loss_acc_file.txt'), 'w')
    test_loss_file = open(os.path.join(args.checkpoint, 'test_loss_acc_file.txt'), 'w')

    train_losses = []
    train_acces = []
    val_losses = []
    val_acces = []

    for i in range(1, args.epochs + 1):
        print('epoch:{}, lr is {}'.format(i, optimizer.state_dict()['param_groups'][0]['lr']))
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        res18.train()
        for imgs, targets, indexes in tqdm(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            outputs = res18(imgs)

            targets = targets.cuda()
            loss = weighted_softmax_loss(outputs, targets, weight, args.classes,device)
            loss.backward()
            optimizer.step()

            running_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        scheduler.step()
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt
        train_losses.append(running_loss)
        train_acces.append(acc)
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))
        train_loss_file.write('loss:%.3f' % running_loss + '' + 'acc:%.4f' % acc + '\n')

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            res18.eval()
            for batch_i, (imgs, targets, _) in enumerate(val_loader):
                outputs = res18(imgs.cuda())
                targets = targets.cuda()
                loss = weighted_softmax_loss(outputs, targets, weight, args.classes,device)
                running_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum()

            running_loss = running_loss / iter_cnt
            acc = bingo_cnt.float() / float(val_dataset.__len__())
            val_losses.append(running_loss)
            val_acces.append(acc)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, acc, running_loss))
            test_loss_file.write('loss:%.3f' % running_loss + '' + 'acc:%.4f' % acc + '\n')

            if acc > best_eval_acc:
                best_eval_acc = acc
                torch.save({'epoch': i,
                            'model_state_dict': res18.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join(args.checkpoint, "epoch" + str(i) + "_acc%.4f" % acc + ".pth"))
                print('Model saved.')
        plt_train_val_loss(train_losses, val_losses, args.checkpoint)
        plt_train_val_acc(train_acces, val_acces, args.checkpoint)

if __name__ == "__main__":
    run_training()
