# -*- coding: future_fstrings -*-

import random
import torch
import torch.utils.data
import numpy as np
from scipy.linalg import expm, norm
import lib.transforms as t
import MinkowskiEngine as ME
import open3d as o3d
import xlrd


def read_xlrd(excelFile):
  data = xlrd.open_workbook(excelFile)
  table = data.sheet_by_index(0)
  dataFile = []
  for rowNum in range(table.nrows):
    if rowNum > 0:
      dataFile.append(table.row_values(rowNum))
  return dataFile


# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
  T = np.eye(4)
  R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  T[:3, :3] = R
  T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
  return T


class DatasetConfig(torch.utils.data.Dataset):
  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    self.config = config
    self.phase = phase
    self.files = []
    self.transform = transform
    self.voxel_size = self.config.voxel_size

    self.random_scale = random_scale
    self.min_scale = self.config.min_scale
    self.max_scale = self.config.max_scale
    self.random_rotation = random_rotation
    self.rotation_range = self.config.rotation_range
    self.randg = np.random.RandomState()
    if manual_seed:
      self.reset_seed()

  def reset_seed(self, seed=0):
    self.randg.seed(seed)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

  def __len__(self):
    return len(self.files)



class ResSCNNDataset(DatasetConfig):
  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    DatasetConfig.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)

    self.data_files = {
      'train': self.config.train_file,
      'test': self.config.test_file,
      'train_path': self.config.train_path,
      'test_path': self.config.test_path
    }

    self.files = read_xlrd(self.data_files[phase])
    files_path_ = read_xlrd(self.data_files[phase+'_path'])
    self.files_path = dict(files_path_)
    pass


  def __getitem__(self, idx):
    plyname = self.files[idx][0]
    pd_path = self.files_path[plyname]
    data = o3d.io.read_point_cloud(pd_path)
    xyz = np.array(data.points)
    color0 = np.array(data.colors)

    if self.random_rotation:
      T0 = sample_random_trans(xyz, self.randg, self.rotation_range)
      xyz = self.apply_transform(xyz, T0)

    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      xyz = scale * xyz

    # rgb -> feature
    feats = []
    feats.append(color0 - 0.5)
    feats = np.hstack(feats)

    coords = np.floor(xyz / self.voxel_size)

    xyz = torch.from_numpy(xyz)

    sel = ME.utils.sparse_quantize(xyz / self.voxel_size, return_index=True)
    sel = sel[1]
    coords = coords[sel]
    coords = ME.utils.batched_coordinates([coords])

    feats = feats[sel]

    feats = torch.as_tensor(feats, dtype=torch.float32)
    coords = torch.as_tensor(coords, dtype=torch.int32)


    if self.transform:
      coords, feats = self.transform(coords, feats)

    MOSlabel = self.files[idx][1]
    MOSlabel = torch.from_numpy(np.array(MOSlabel))

    return (feats, coords, MOSlabel, plyname)




def make_data_loader(config, phase, batch_size, num_threads=0, shuffle=None):
  if shuffle is None:
    shuffle = phase != 'test'

  Dataset = ResSCNNDataset

  use_random_scale = False
  use_random_rotation = False
  transforms = []
  if phase in ['train']:
    use_random_rotation = config.use_random_rotation
    use_random_scale = config.use_random_scale
    transforms += [t.Jitter()]

  dset = Dataset(
      phase,
      transform=t.Compose(transforms),
      random_scale=use_random_scale,
      random_rotation=use_random_rotation,
      config=config)

  loader = torch.utils.data.DataLoader(
      dset,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=num_threads,
      pin_memory=False,
      drop_last=True)

  return loader
