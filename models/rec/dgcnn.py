
import numpy as np
import jittor as jt
from jittor import nn
from jittor.contrib import concat

from lib import fops, mops
from models.rec.rec_model import RecModule

def get_graph_feature(x, knn=None, k=None, idx=None):
    batch_size = x.shape[0]
    num_points = x.shape[2]
    x = x.reshape(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x,x)   # (batch_size, num_points, k)
        idx = idx.permute(0, 2, 1)
    idx_base = jt.array(np.arange(0, batch_size)).reshape(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.reshape(-1)

    _, num_dims, _ = x.shape

    x = x.transpose(0, 2, 1)   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.reshape(batch_size*num_points, -1)[idx, :]
    feature = feature.reshape(batch_size, num_points, k, num_dims)
    x = x.reshape(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = concat((feature-x, x), dim=3).transpose(0, 3, 1, 2)
    return feature


class DGCNN(RecModule):
    def __init__(self, args, margs):
        super(DGCNN, self).__init__(args,margs)
        self.k = 20
        self.knn = mops.KNN(self.k)
        self.bn1 = nn.BatchNorm(64)
        self.bn2 = nn.BatchNorm(64)
        self.bn3 = nn.BatchNorm(128)
        self.bn4 = nn.BatchNorm(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(scale=0.2))
        self.conv2 = nn.Sequential(nn.Conv(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(scale=0.2))
        self.conv3 = nn.Sequential(nn.Conv(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(scale=0.2))
        self.conv4 = nn.Sequential(nn.Conv(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(scale=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(scale=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 40)

    @classmethod
    def build_model(cls,args,margs):
        return DGCNN(args,margs)

    def execute(self, x):
        x = x.permute(0,2,1)
        batch_size = x.shape[0]

        x = get_graph_feature(x, knn=self.knn, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdims=False)
        x = get_graph_feature(x1, knn=self.knn, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdims=False)
        x = get_graph_feature(x2, knn=self.knn, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdims=False)
        x = get_graph_feature(x3, knn=self.knn, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdims=False)
        x = concat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x) #  ############ this line
        x1 = x.max(dim=2).reshape(batch_size, -1)
        x2 = x.mean(dim=2).reshape(batch_size, -1)
        x = concat((x1, x2), 1)
        x = nn.leaky_relu(self.bn6(self.linear1(x)), scale=0.2)
        x = self.dp1(x)
        x = nn.leaky_relu(self.bn7(self.linear2(x)), scale=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

def get(module=None):
    return DGCNN
