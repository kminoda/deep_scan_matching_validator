import torch
import time
import os
import yaml
import pickle
import bisect

import dataclasses
import numpy as np
import open3d as o3d

from tqdm import tqdm
from pathlib import Path
from typing import List

@dataclasses.dataclass
class ScanData:
  map_file: str
  pts: np.ndarray

@dataclasses.dataclass
class PCDMap:
  x_list: List[float]
  y_list: List[float]
  mp: List[List[bool]]

def generate_pcd_map(points_map, dx):
  min_x = np.min(points_map[:, 0])
  min_y = np.min(points_map[:, 1])
  max_x = np.max(points_map[:, 0])
  max_y = np.max(points_map[:, 1])
  x_list = np.arange(min_x - min_x % dx, max_x + dx * 2, dx)
  y_list = np.arange(min_y - min_y % dx, max_y + dx * 2, dx)
  boolean_map = [[False for _ in range(len(y_list))] for _ in range(len(x_list))]

  for point in tqdm(points_map):
    x_idx = bisect.bisect_left(x_list, point[0])
    y_idx = bisect.bisect_left(y_list, point[1])
    boolean_map[x_idx][y_idx] = True

  pcd_map = PCDMap(
    list(x_list), list(y_list), boolean_map
  )
  return pcd_map


class ScanDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path: str, img_size=(512, 512), dx=0.4, error_range=(1, 5)):
    self.dx = dx
    self.img_size = img_size
    self.error_range = error_range

    dataset_path = Path(dataset_path)
    self.scan_data_list = []
    map_file_list = []
    for p in dataset_path.iterdir():
      # load metadata
      with open(dataset_path / p / 'metadata.yaml') as f:
        metadata = yaml.safe_load(f)
      pointcloud_map_path = metadata['pointcloud_map_path']
      if not pointcloud_map_path in map_file_list:
        map_file_list.append(pointcloud_map_path)

      # load scan pickle files
      scan_file_list = list((dataset_path / p / 'scan_data').glob('*.pickle'))
      for scan_file in scan_file_list:
        with open(scan_file, 'rb') as f:
          drive_data = pickle.load(f)
        scan_data = ScanData(pointcloud_map_path,
                             drive_data['scan'])
        self.scan_data_list.append(scan_data)

    # load pointcloud maps
    self.map_data = {}
    for map_file in map_file_list:
      print('Processing {}...'.format(map_file))
      points_map = np.asarray(o3d.io.read_point_cloud(map_file).points)
      self.map_data[map_file] = generate_pcd_map(points_map, dx)

  def __len__(self):
    return len(self.scan_data_list)

  def __getitem__(self, index, label='valid'):
    scan_data = self.scan_data_list[index]

    if label=='':
      is_valid_scan = np.random.rand() > 0.5
    elif label=='valid':
      is_valid_scan = True
    elif label=='invalid':
      is_valid_scan = False
    else:
      raise NotImplementedError

    if not is_valid_scan:
      delta_x = np.random.rand() * (self.error_range[1] - self.error_range[0]) + self.error_range[0]
      delta_y = np.random.rand() * (self.error_range[1] - self.error_range[0]) + self.error_range[0]
      if np.random.rand() > 0.5:
        delta_x *= -1
      if np.random.rand() > 0.5:
        delta_y *= -1
      
      scan_data.pts[:, 0] += delta_x
      scan_data.pts[:, 1] += delta_y

    scan_pts = scan_data.pts
    map_data = self.map_data[scan_data.map_file]
    return self._generate_image(scan_pts, map_data), torch.Tensor([is_valid_scan * 1])

  def _generate_image(self, scan_pts, pcd_map: PCDMap):    
    center = np.mean(scan_pts[:, :2], axis=0)
    min_x = center[0] - self.img_size[0] / 2 * self.dx 
    min_y = center[1] - self.img_size[1] / 2 * self.dx 
    min_x_idx = bisect.bisect_left(pcd_map.x_list, min_x)
    min_y_idx = bisect.bisect_left(pcd_map.y_list, min_y)

    # start = time.time()
    tmp_data = pcd_map.mp[min_x_idx: min_x_idx + self.img_size[0]]
    data = np.array([v[min_y_idx: min_y_idx + self.img_size[1]] for v in tmp_data]) * 0.1
    x_list = pcd_map.x_list[min_x_idx: min_x_idx + self.img_size[0]]
    y_list = pcd_map.y_list[min_y_idx: min_y_idx + self.img_size[1]]
    # end = time.time()
    # print(f'Map: {end - start} [s]')
    # start = end

    for point in scan_pts:
      # TODO: this is skewin image. Maybe should I convert without skew?
      x_idx = bisect.bisect_left(x_list, point[0])
      y_idx = bisect.bisect_left(y_list, point[1])
      if x_idx >= self.img_size[0] or y_idx >= self.img_size[1]:
        continue
      data[x_idx][y_idx] = 0.9
    # end = time.time()
    # print(f'Scan: {end - start} [s]')

    return torch.Tensor(data).unsqueeze(0).repeat(3, 1, 1)

if __name__ == '__main__':
  import time
  import torchvision.transforms as T
  from PIL import Image

  dataset = ScanDataset('/home/minoda/git/deep_scan_matching_validator/dataset/train')
  
  start = time.time()
  image, label = dataset[1500]
  end = time.time()
  print(f'Duration: {end - start} [s]')
  print(image.shape)

  transform = T.ToPILImage()
  img = transform(image)
  img.save("./outputs/figures/sample.png")
