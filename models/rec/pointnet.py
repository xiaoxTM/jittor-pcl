import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
from ops import fops, mops
from models.rec.rec_model import RecModule


class PointNet(RecModule):
    def __init__(self, args,margs):
        super(PointNet, self).__init__(args,margs)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 40)
        self.relu = nn.ReLU()

    @classmethod
    def build_model(cls,args,margs):
        return PointNet(args,margs)

    def execute(self, x):
        x = x.permute(0,2,1)
        b, c, n = x.size()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = jt.max(x, 2)
        x = x.reshape(b, -1)
        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x

def get():
    return PointNet
