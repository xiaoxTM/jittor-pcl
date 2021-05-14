import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize_pointcloud(pointcloud):
    pointcloud = pointcloud - pointcloud.mean(axis=0)
    scale = np.sqrt((pointcloud ** 2).sum(axis=1).max())
    pointcloud = pointcloud / scale
    return pointcloud


def dropout_pointcloud(pointcloud, max_dropout_ratio=0.875):
    ''' pointcloud: Nx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((pointcloud.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        pointcloud[drop_idx,:] = pointcloud[0,:] # set to the first point
    return pointcloud


def dropout_pointclouds(pointclouds, max_dropout_ratio=0.875):
    assert isinstance(pointclouds, (list,tuple)), 'point clouds must be instance of list/tuple, given {}'.format(type(pointclouds))
    ''' pointcloud: Nx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((pointcloud.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        if isinstance(pointclouds, np.ndarray):
            pointclouds[drop_idx,:] = pointclouds[0,:] # set to the first point
        elif isinstance(pointclouds, (list,tuple)):
            for i in range(len(pointclouds)):
                pointclouds[i,drop_idx,:] = pointclouds[i,0,:]
    return pointclouds


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud_xyz(pointcloud):
    theta = np.pi*2*np.random.rand()
    axis = np.random.randint(0,3)
    if axis==0: # random rotate along x, i.e., (y, z)
        rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        axes = [1,2]
    elif axis==1: # random rotate along y, i.e., (x, z)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        axes = [0,2]
    else: # random rotate along z, i.e., (y, z)
        rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        axes = [0,1]
    pointcloud[:,axes] = pointcloud[:,axes].dot(rotation_matrix)
    return pointcloud


def rotate_pointclouds_xyz(pointclouds):
    assert isinstance(pointclouds, (list,tuple)), 'point clouds must be instance of list/tuple, given {}'.format(type(pointclouds))
    theta = np.pi*2*np.random.rand()
    axis = np.random.randint(0,3)
    if axis==0: # random rotate along x, i.e., (y, z)
        rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        axes = [1,2]
    elif axis==1: # random rotate along y, i.e., (x, z)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        axes = [0,2]
    else: # random rotate along z, i.e., (y, z)
        rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        axes = [0,1]
    for i in range(len(pointclouds)):
        pointclouds[i,:,axes] = pointclouds[i,:,axes].dot(rotation_matrix)
    return pointclouds


def rotate_pointcloud(pointcloud):
    # pointcloud: [num-points, 3]
    r = R.from_quat(np.random.uniform(0,1,size=4))
    # m: [3, 3]
    m = r.as_matrix()
    # (AB^)^ = BA^
    # (rP^)^ = Pr^
    return np.matmul(pointcloud,m.transpose(1,0)).astype(np.float32)


def rotate_pointclouds(pointclouds):
    assert isinstance(pointclouds, (list,tuple)), 'point clouds must be instance of list/tuple, given {}'.format(type(pointclouds))
    # pointcloud: [num-points, 3]
    r = R.from_quat(np.random.uniform(0,1,size=4))
    # m: [3, 3]
    m = r.as_matrix()
    # (AB^)^ = BA^
    # (rP^)^ = Pr^
    for i in range(len(pointclouds)):
        pointclouds[i,:,axes] = np.matmul(pointclouds[i],m.transpose(1,0)).astype(np.float32)
    return pointclouds
