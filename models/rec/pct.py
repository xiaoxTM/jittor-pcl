import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
from lib import fops
from lib.layers import Permute
from models.rec.rec_model import RecModule


class KNNEmbed(nn.Module):
    def __init__(self,cin,cout,num_points,num_neighbors):
        super(KNNEmbed,self).__init__()
        self.num_points = num_points
        self.num_neighbors = num_neighbors
        self.embeds = nn.Sequential(nn.Conv1d(cin,cout,kernel_size=1,bias=False),
                                                                    nn.BatchNorm1d(cout),
                                                                    nn.ReLU(),
                                                                    nn.Conv1d(cout,cout,kernel_size=1,bias=False),
                                                                    nn.BatchNorm1d(cout),
                                                                    nn.ReLU())

    def execute(self,x):
        # points: [batch-size, num-points, num-dims]
        # features: [batch-size, num-points, num-dims]
        features,points = x
        # features: [batch-size, num-points, num-neighbors, num-dims]
        points, features = fops.sample_and_group(self.num_points,self.num_neighbors,points,features)
        batch_size, num_points, num_neighbors, num_dims = features.size()
        # features: [batch-size*num_points, num-dims, num_neighbors]
        features = features.permute(0,1,3,2).reshape(-1,num_dims,num_neighbors)
        features = jt.max(self.embeds(features),-1).view(batch_size,num_points,num_dims)
        return features,points


class PointCloudTransformer(RecModule):
    def __init__(self,args,margs):
        super(PointCloudTransformer, self).__init__(args,margs)
        self.input_embeds = nn.Sequential(Permute(0,2,1),
                                                                                  nn.Conv1d(3, 64,kernel_size=1,bias=False),
                                                                                  nn.BatchNorm1d(64),
                                                                                  nn.ReLU(),
                                                                                  nn.Conv1d(64, 64,kernel_size=1,bias=False),
                                                                                  nn.BatchNorm1d(64),
                                                                                  nn.ReLU(),
                                                                                  Permute(0,2,1))
        self.knn_embeds = nn.Sequential(KNNEmbed(128,128,512,32),KNNEmbed(256,256,256,32))
        self.transformer = PointTransformer()
        self.classifier = nn.Sequential(nn.Linear(1024, 512),
                                                                      nn.BatchNorm1d(512),
                                                                      nn.ReLU(),
                                                                      nn.Dropout(p=0.5),
                                                                      nn.Linear(512, 256),
                                                                      nn.BatchNorm1d(256),
                                                                      nn.Dropout(p=0.5),
                                                                      nn.Linear(256, 40))

    @classmethod
    def build_model(cls,args,margs):
        return PointCloudTransformer(args,margs)

    def execute(self, points):
        # points: [batch-size, num-points, num-dims]
        batch_size = points.size(0)
        features = self.input_embeds(points)
        features,points = self.knn_embeds((features,points))
        # [batch-size, num-points, num-dims] -> [batch-size, num-dims, num-points]
        x = self.transformer(features,points)
        x= jt.max(x,2).view(batch_size,-1)
        return self.classifier(x)


class PointTransformer(nn.Module):
    def __init__(self, channels=256):
        super(PointTransformer, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(channels,channels,kernel_size=1,bias=False),
                                                             nn.BatchNorm1d(channels),
                                                             nn.ReLU(),
                                                             nn.Conv1d(channels,channels,kernel_size=1,bias=False),
                                                             nn.BatchNorm1d(channels),
                                                             nn.ReLU())

        self.pos_conv = nn.Sequential(Permute(0,2,1),
                                                                       nn.Conv1d(3, channels,kernel_size=1,bias=False))

        self.oa1 = OffsetAttention(channels)
        self.oa2 = OffsetAttention(channels)
        self.oa3 = OffsetAttention(channels)
        self.oa4 = OffsetAttention(channels)

        self.fuse_conv = nn.Sequential(nn.Conv1d(1280, 1024,kernel_size=1,bias=False),
                                                                         nn.BatchNorm1d(1024),
                                                                         nn.LeakyReLU(scale=0.2,))

    def execute(self, features, points=None):
        # batch-size, num-points, num-dims -> batch-size, num-dims, num-points
        batch_size = features.size(0)
        if points is not None:
            # add position embedding
            # points: batch-size, num-points, num-dims -> batch-size, num-dims, num-points
            points = self.pos_conv(points)
            # end
        # features: batch-size, num-points, num-dims -> batch-size, num-dims, num-points
        features = features.permute(0,2,1)
        x = self.conv(features) # B, D, N
        x1 = self.oa1(x, points)
        x2 = self.oa2(x1, points)
        x3 = self.oa3(x2, points)
        x4 = self.oa4(x3, points)
        return self.fuse_conv(concat((features,x1,x2,x3,x4), dim=1))


class OffsetAttention(nn.Module):
    def __init__(self, channels):
        super(OffsetAttention, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4,kernel_size=1,bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4,kernel_size=1,bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels,kernel_size=1,bias=False)
        self.trans_conv = nn.Conv1d(channels, channels,kernel_size=1,bias=False)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def execute(self, features, points):
        if points is not None:
            features = features + points
        q = self.q_conv(features).permute(0, 2, 1) # b, n, c
        k = self.k_conv(features)# b, c, n
        v = self.v_conv(features)
        energy = nn.bmm(q, k) # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        r = nn.bmm(v, attention) # b, c, n
        r = self.act(self.after_norm(self.trans_conv(features - r)))
        features = features + r
        return features


def get():
    return PointCloudTransformer
