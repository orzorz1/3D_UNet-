import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
from data.LoadData import *
from torch.utils.data import DataLoader
import torch
from models.UNet_3D import UNet_3D
from modules.functions import dice_loss
import math
from commons.plot import save_nii
import torchsummary
from commons.log import make_print_to_file
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

def train():
    batch_size = 1
    model = UNet_3D()
    try:
        model.load_state_dict(torch.load("UNet_3D-15.pth", map_location='cpu'))
    except FileNotFoundError:
        print("模型不存在")
    else:
        print("加载模型成功")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for index in range(16,17):
        val_data = load_dataset(index)
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size)

        model.eval()
        a = []
        pre = []
        for batch, (batch_x, batch_y) in enumerate(val_loader):
            batch_x, batch_y = torch.autograd.Variable(batch_x.to(device)), torch.autograd.Variable(batch_y.to(device))
            out = model(batch_x)
            out = nn.Sigmoid()(out)
            # if l != 0:
            #     # print("√", l, end="  ")
            #         save_nii(batch_x.cpu().numpy().astype(np.int16)[n][0],
            #                  '{i}-X-{l}'.format(i=i, l=loss))
            #         save_nii(batch_y.cpu().numpy().astype(np.int16)[n][0],
            #                  '{i}-Y-{l}'.format(i=i, l=loss))
            #         save_nii(out.cpu().detach().numpy().astype(np.int16)[n][0],
            #                  '{i}-Out-{l}'.format(i=i, l=loss))

            pre.append(out.cpu().detach().numpy().astype(np.int64)[0][0].tolist())

        pre = np.array(pre)
        print(pre.shape)
        deep = []
        for i in range(16):
            d = pre[4*i]
            for j in range(1, 4):
                d = np.concatenate((d, pre[4*i+j]), axis=2)
            deep.append(d.tolist())

        deep = np.array(deep)
        height = []
        for i in range(4):
            h = deep[4*i]
            for j in range(1, 4):
                h = np.concatenate((h, deep[4*i+j]), axis=1)
            height.append(h.tolist())

        height = np.array(height)
        predict = height[0]
        for i in range(1, 4):
            predict = np.concatenate((predict, height[i]), axis=0)

        print(predict.shape)

        save_nii(predict.astype(np.int16),"{index}-predict1".format(index = index))



if __name__ == '__main__':
    # make_print_to_file("./")
    torch.cuda.empty_cache()
    train()
    # os.system("shutdown")