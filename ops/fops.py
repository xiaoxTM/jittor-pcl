import jittor as jt
from sklearn.neighbors import NearestNeighbors
import numpy as np
from jittor.contrib import concat

def topk(input, k, dim=None, largest=True, sorted=True):
    if dim is None:
        dim = -1
    if dim<0:
        dim+=input.ndim
    transpose_dims = [i for i in range(input.ndim)]
    transpose_dims[0] = dim
    transpose_dims[dim] = 0
    input = input.transpose(transpose_dims)
    index,values = jt.argsort(input,dim=0,descending=largest)
    indices = index[:k]
    values = values[:k]
    indices = indices.transpose(transpose_dims)
    values = values.transpose(transpose_dims)
    return [values,indices]


def square_distance(tensor1, tensor2):
    """
    Calculate Euclid distance between each two points.
    tensor1^T * tensor2 = xn * xm + yn * ym + zn * zm；
    sum(tensor1^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(tensor2^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(tensor1**2,dim=-1)+sum(tensor2**2,dim=-1)-2*tensor1^T*dst
    Input:
        tensor1: source points, [B, N, C]
        tensor2: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    # print (src.size(), dst.size())
    B, N, _ = tensor1.shape
    _, M, _ = tensor2.shape
    dist = -2 * jt.matmul(tensor1, tensor2.permute(0, 2, 1))
    dist += jt.sum(tensor1 ** 2, -1).view(B, N, 1)
    dist += jt.sum(tensor2 ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = np.arange(B, dtype='l')
    batch_indices = jt.array(batch_indices).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_points(k, points, centroids):
    """
    Input:
        k: max sample number in local region
        points: all points, [B, N, C]
        centroids: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, k]
    """
    sqrdists = square_distance(centroids, points)
    _, group_idx = topk(sqrdists, k, dim = -1, largest=False, sorted=False)
    return group_idx


def knn (x, k):
    inner = -2 * jt.nn.bmm(x.transpose(0, 2, 1), x)
    xx = jt.sum(x ** 2, dim = 1, keepdims=True)
    distance = -xx - inner - xx.transpose(0, 2, 1)
    idx = topk(distance ,k=k, dim=-1)[1]
    return idx


def farthest_point_sample(points, num_point):
    """
    Input:
        points: pointcloud data, [B, N, C]
        num_point: number of samples
    Return:
        centroids: sampled pointcloud index, [B, num_point]
    """
    B, N, C = points.shape
    centroids = jt.zeros((B, num_point))
    distance = jt.ones((B, N)) * 1e10

    farthest = np.random.randint(0, N, B, dtype='l')
    batch_indices = np.arange(B, dtype='l')
    farthest = jt.array(farthest)
    batch_indices = jt.array(batch_indices)
    # jt.sync_all(True)
    for i in range(num_point):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :]
        centroid = centroid.view(B, 1, 3)

        dist = jt.sum((points - centroid.repeat(1, N, 1)) ** 2, 2)
        mask = dist < distance
        # distance = mask.ternary(distance, dist)
        # print (mask.size())

        if mask.sum().data[0] > 0:
            distance[mask] = dist[mask] # bug if mask.sum() == 0

        farthest = jt.argmax(distance, 1)[0]
        # print (farthest)
        # print (farthest.shape)
    # B, N, C = xyz.size()
    # sample_list = random.sample(range(0, N), npoint)
    # centroids = jt.zeros((1, npoint))
    # centroids[0,:] = jt.array(sample_list)
    # centroids = centroids.view(1, -1).repeat(B, 1)
    # x_center = x[:,sample_list, :]
    return centroids


def knn_indices_func_cpu(rep_pts,  # (N, pts, dim)
                         pts,      # (N, x, dim)
                         K : int,
                         D : int):
    """
    CPU-based Indexing function based on K-Nearest Neighbors search.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    rep_pts = rep_pts.data
    pts = pts.data
    region_idx = []

    for n, p in enumerate(rep_pts):
        P_particular = pts[n]
        nbrs = NearestNeighbors(D*K + 1, algorithm = "ball_tree").fit(P_particular)
        indices = nbrs.kneighbors(p)[1]
        region_idx.append(indices[:,1::D])

    region_idx = jt.array(np.stack(region_idx, axis = 0))
    return region_idx


def knn_indices_func_gpu(rep_pts,  # (N, pts, dim)
                         pts,      # (N, x, dim)
                         k : int, d : int ): # (N, pts, K)
    """
    GPU-based Indexing function based on K-Nearest Neighbors search.
    Very memory intensive, and thus unoptimal for large numbers of points.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    region_idx = []
    batch_size = rep_pts.shape[0]
    for idx in range (batch_size):
        qry = rep_pts[idx]
        ref = pts[idx]
        n, d = ref.shape
        m, d = qry.shape
        mref = ref.view(1, n, d).repeat(m, 1, 1)
        mqry = qry.view(m, 1, d).repeat(1, n, 1)

        dist2 = jt.sum((mqry - mref)**2, 2) # pytorch has squeeze
        _, inds = topk(dist2, k*d + 1, dim = 1, largest = False)

        region_idx.append(inds[:,1::d])

    region_idx = jt.stack(region_idx, dim = 0)

    return region_idx


def expand(x,shape):
    r'''
    Returns a new view of the self tensor with singleton dimensions expanded to a larger size.
    Tensor can be also expanded to a larger number of dimensions, and the new ones will be appended at the front.
    Args:
       x: the input tensor.
       shape: the shape of expanded tensor.
    '''
    x_shape = x.shape
    x_l = len(x_shape)
    rest_shape=shape[:-x_l]
    expand_shape = shape[-x_l:]
    indexs=[]
    ii = len(rest_shape)
    for i,j in zip(expand_shape,x_shape):
        if i!=j:
            assert j==1
        indexs.append(f'i{ii}' if j>1 else f'0')
        ii+=1
    return x.reindex(shape,indexs)


def sample_and_group(num_points, num_neighbors, points, features=None):
    """ sample `num-points` from points via furthest point sampling algorithm
           and consider them sampled points as centroids to construct KNN graphs
           on points where k=`num-neighbors`
    """
    B, _, C = points.shape
    S = num_points
    fps_idx = farthest_point_sample(points,num_points)
    # _, fps_idx = sampler(points) # [B, npoint]

    sampled_points = index_points(points, fps_idx)
    point_knn_idx = knn_points(num_neighbors, points, sampled_points)
    grouped_points = index_points(points, point_knn_idx) # [B, num-point, num-neighbos, C]

    if features is None:
        reshaped_points = sampled_points.view(B, S, 1, C)
        grouped_points_relative = grouped_points - reshaped_points
        return sampled_points, concat([grouped_points_relative,reshaped_points.repeat(1,1,num_neighbors,1)],dim=-1)

    sampled_features = index_points(features, fps_idx)
    grouped_features = index_points(features, point_knn_idx)
    reshaped_features = sampled_features.view(B, S, 1, -1)
    grouped_feature_relative = grouped_features - reshaped_features
    return sampled_points, concat([grouped_feature_relative, reshaped_features.repeat(1, 1, num_neighbors, 1)], dim=-1)


def sample_and_group_with_density_scale(npoint, nsample, xyz, points, density_scale = None):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    # jt.sync_all(True)
    # print ('11111111111111111')
    new_xyz = index_points(xyz, fps_idx)
    # jt.sync_all(True)
    # print ('2222222222222222222')
    idx = knn_points(nsample, xyz, new_xyz)
    # jt.sync_all(True)
    # print ('333333333333333333')
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = concat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    # jt.sync_all(True)
    # print ('44444444444444444444444')

    if density_scale is None:
        return new_xyz, new_points, grouped_xyz_norm, idx
    else:
        grouped_density = index_points(density_scale, idx)
        return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density


def compute_density(points, bandwidth):
    '''
    points: input points position data, [B, N, C]
    '''
    #import ipdb; ipdb.set_trace()
    B, N, C = points.shape
    sqrdists = square_distance(points, points)
    gaussion_density = jt.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    points_density = gaussion_density.mean(dim = -1)

    return points_density
