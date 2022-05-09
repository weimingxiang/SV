import os
import math
from pytorch_lightning.core.hooks import CheckpointHooks
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from ray import tune
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch.distributions.multivariate_normal as mn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import torchvision
import math
from utilities import IdentifyDataset
from pudb import set_trace
from multiprocessing import Pool, cpu_count
import random

class classfication(nn.Module):
    def __init__(self, full_dim):
        super(classfication, self).__init__()
        dim1 = full_dim[:-1]
        dim2 = full_dim[1:]
        self.layers=nn.ModuleList(
            nn.Sequential(
                nn.Linear(k, m),
                nn.ReLU(),
            ) for k, m in zip(dim1, dim2)
        )
    def forward(self,x):
        out=x
        for i,layer in enumerate(self.layers):
            out=layer(out)
        return out

class attention(nn.Module):
    def __init__(self, dim, out_dim):
        super(attention, self).__init__()
        self.Q = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.Sigmoid(),
        )
        self.K = nn.Sequential(
            nn.Linear(dim, out_dim),
        )

    def forward(self,x):
        q = self.Q(x)
        k = self.K(x)
        out = torch.mul(q, k)
        return out

class attention_classfication(nn.Module):
    def __init__(self, full_dim):
        super(attention_classfication, self).__init__()
        dim1 = full_dim[:-1]
        dim2 = full_dim[1:]
        self.layers=nn.ModuleList(
            nn.Sequential(
                attention(k, m),
                nn.Linear(m, m),
                nn.ReLU(inplace=True),
            ) for k, m in zip(dim1, dim2)
        )
    def forward(self,x):
        out=x
        for i,layer in enumerate(self.layers):
            out=layer(out)
        return out

class conv2ds_sequential(nn.Module):
    def __init__(self, full_dim):
        super(conv2ds_sequential, self).__init__()
        dim1 = full_dim[:-1]
        dim2 = full_dim[1:]
        self.layers=nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=k, out_channels=m, kernel_size=3, stride=1, padding=1), # (m, 224, 224)
                nn.BatchNorm2d(m),
                nn.ReLU(inplace=True),
            ) for k, m in zip(dim1, dim2)
        )
    def forward(self,x):
        out=x
        for i,layer in enumerate(self.layers):
            out=layer(out)
        return out

class IDENet(pl.LightningModule):

    def __init__(self, positive_img, negative_img, config):
        super(IDENet, self).__init__()

        self.lr = config["lr"]
        self.beta1 = config['beta1']
        self.batch_size = config["batch_size"]
        self.conv2d_dim_stride = config["conv2d_dim_stride"]  # [1, 3]
        self.classfication_dim_stride = config["classfication_dim_stride"] #[1, 997]

        self.positive_img = positive_img
        self.negative_img = negative_img

        # self.conv2ds = nn.Sequential(
        #     nn.Conv2d(in_channels=9, out_channels=8, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(in_channels=8, out_channels=7, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(in_channels=7, out_channels=6, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(in_channels=6, out_channels=5, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(in_channels=5, out_channels=4, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1),
        # )
        conv2d_dim = list(range(11, 3, -self.conv2d_dim_stride))
        conv2d_dim.append(3) # 6 -> 3
        self.conv2ds = conv2ds_sequential(conv2d_dim)

        self.resnet_model = torchvision.models.resnet50(pretrained=True) # [224, 224] -> 1000

        # self.attention = attention(1000, 500)

        # full_dim = [1000, 500, 250, 125, 62, 31, 15, 7]
        full_dim = range(1000, 2, -self.classfication_dim_stride) # 1000 -> 2
        full_dim = [1000, 500, 50, 10]
        self.classfication = attention_classfication(full_dim)

        self.softmax = nn.Sequential(
            nn.Linear(full_dim[-1], 2),
            nn.Softmax(1)
        )

        self.criterion = nn.CrossEntropyLoss()


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_t = torch.empty(0, 2)
        for y_item in y:
            if y_item == 0:
                y_t = torch.cat((y_t, torch.tensor([1, 0]).unsqueeze(0)),0)
            else:
                y_t = torch.cat((y_t, torch.tensor([0, 1]).unsqueeze(0)),0)
        x = self.conv2ds(x)
        y_hat = self.resnet_model(x)
        y_hat = self.classfication(y_hat)
        y_hat = self.softmax(y_hat)
        loss = self.criterion(y_hat, y_t.cuda())

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # set_trace()
        return {'loss': loss, 'pred': torch.mean((y == (y_hat[:, 0] < y_hat[:, 1])).float())}

    def training_epoch_end(self, output):
        # set_trace()
        prediction = []
        for out in output:
            prediction.append(out['pred'])
        self.log('train_mean', torch.mean(torch.tensor(prediction)), on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_t = torch.empty(0, 2)
        for y_item in y:
            if y_item == 0:
                y_t = torch.cat((y_t, torch.tensor([1, 0]).unsqueeze(0)),0)
            else:
                y_t = torch.cat((y_t, torch.tensor([0, 1]).unsqueeze(0)),0)
        x = self.conv2ds(x)
        y_hat = self.resnet_model(x)
        y_hat = self.classfication(y_hat)
        y_hat = self.softmax(y_hat)
        loss = self.criterion(y_hat, y_t.cuda())

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # set_trace()
        return torch.mean((y == (y_hat[:, 0] < y_hat[:, 1])).float())

    def validation_epoch_end(self, output):
        self.log('validation_mean', torch.mean(torch.tensor(output)), on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, output):
        N = []
        for i in range(self.class_num):
            label_mean = torch.mean(self.Z[i][1:], dim=0).float()
            label_cov = torch.from_numpy(
                np.cov(self.Z[i][1:].numpy(), rowvar=False)).float()
            m = mn.MultivariateNormal(label_mean, label_cov)
            N.append(m)
        torch.save({'distribution': N}, os.path.join(
            self.data_dir, 'gan_code/class_distribution') + str(data_type) + '.dt')

    def prepare_data(self):

        train_proportion = 0.8
        input_data = IdentifyDataset(self.positive_img, self.negative_img)
        dataset_size = len(input_data)
        indices = list(range(dataset_size))
        split = int(np.floor(train_proportion * dataset_size))
        random.seed(10)
        random.shuffle(indices)
        train_indices, test_indices = indices[:split], indices[split:]
        self.train_dataset= Subset(input_data, train_indices)
        self.test_dataset= Subset(input_data, test_indices)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=int(cpu_count()/4), shuffle=True) # sampler=self.wsampler)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=int(cpu_count()/4))

    def test_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=int(cpu_count()/4))

    # @property
    # def automatic_optimization(self):
    #     return False

    def configure_optimizers(self):
        opt_e = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        # opt_d = torch.optim.Adam(
        #     self.line.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        return [opt_e]
