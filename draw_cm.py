from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--test_result_path',default='PRE/model_RAF_pre_Self_sm1/test_result',help='path of test result')
parser.add_argument('--csv_file',default='RAF_pre_Self_sm1.csv',help='path of csv')
parser.add_argument('--save_img_name',default='RAF_pre_Self_sm1.jpg',help='name of save img')

args=parser.parse_args()

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

if __name__ == '__main__':
    target, predict = pd_read_csv(os.path.join(args.test_result_path, args.csv_file))

    labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

    draw_cm(predict, target, labels, img_name=args.save_img_name)
