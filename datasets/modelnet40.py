import os
import numpy as np
import h5py
import glob
from jittor.dataset.dataset import Dataset
from . import augments as augs


def download(base):
    if not os.path.exists(base):
        os.mkdir(base)
    if not os.path.exists(os.path.join(base, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], base))
        os.system('rm %s' % (zipfile))


def load_data(partition, base='/home/xiaox/studio/db/modelnet40',with_normal=True):
    download(base)
    all_data = []
    all_label = []
    all_normal = []
    for h5_name in glob.glob(os.path.join(base, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        assert os.path.exists(h5_name),'file: {} not exists'.format(h5_name)
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        normal = f['normal'][:].astype('float32')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_normal.append(normal)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_normal = np.concatenate(all_normal, axis=0)
    if with_normal:
        return all_data, all_label, all_normal
    return all_data,all_label


class ModelNet40(Dataset):
    def __init__(self, data_root, num_points:int=1024, split:str='train',with_normal=False,**kwargs):
        super(ModelNet40,self).__init__(**kwargs)
        assert num_points <= 2048
        self.num_points = num_points
        self.split = split
        data = load_data(partition=split,with_normal=with_normal,base=data_root)
        if with_normal:
           self.data,self.labels,self.normals = data
        else:
            self.data,self.labels = data
            self.normals = None
        self.set_attrs(total_len=len(self.labels))

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.labels[item]
        index = np.arange(2048,dtype=np.int64)
        if self.split == 'train':
            np.random.shuffle(index)
            pointcloud = augs.dropout_pointcloud(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = augs.translate_pointcloud(pointcloud)
        pointcloud = pointcloud[index[:self.num_points]]
        if self.normals is not None:
            return pointcloud, label, self.normals[index[:self.num_points]]
        return pointcloud, label

    def collect_batch(self, batch):
        pts = np.stack([b[0] for b in batch], axis=0)
        cls = np.stack([b[1] for b in batch])
        if self.normals is not None:
            pts, cls, normals = np.stack([b[2] for b in batch], axis=0)
        return pts, cls
