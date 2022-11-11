import torch
import os
import yaml
import pickle
import dataclasses
import numpy as np
import open3d as o3d

from pathlib import Path

@dataclasses.dataclass
class ScanData:
  map_file: str
  pts: np.ndarray

class ScanDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path: str, img_size=(512, 512)):
    self.img_size = img_size

    dataset_path = '/home/minoda/git/deep_scan_matching_validator/dataset/'
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
      self.map_data[map_file] = np.asarray(o3d.io.read_point_cloud(map_file).points)
      self.map_data[map_file] = self.map_data[map_file][:, [0, 1]] # IGNORE Z

  def __len__(self):
    return len(self.scan_data_list)

  def __getitem__(self, index, label=''):
    scan_data = self.scan_data_list[index]

    if label=='':
      is_valid_scan = np.random.rand() > 0.5
    elif label=='valid':
      is_valid_scan = True
    elif label=='invalid':
      is_valid_scan = False
    else:
      raise NotImplementedError

    scan_pts = scan_data.pts
    map_pts = self.map_data[scan_data.map_file]
    return self._generate_image(scan_pts, map_pts)

  def _generate_image(self, scan_pts, map_pts):
    data = [[0 for _ in range(self.img_size[0])] for _ in range(self.img_size[1])]
    min_x = np.min(scan_pts[:, 0])
    min_y = np.min(scan_pts[:, 1])
    max_x = np.max(scan_pts[:, 0])
    max_y = np.max(scan_pts[:, 1])
    dx = max_x - min_x
    dy = max_y - min_y

    bound_x = np.logical_and(map_pts[:, 0] > min_x, map_pts[:, 0] < max_x)
    bound_y = np.logical_and(map_pts[:, 1] > min_y, map_pts[:, 1] < max_y)
    map_pts_within_bbox = map_pts[np.logical_and(bound_x, bound_y)]

    for point in map_pts_within_bbox:
      # TODO: this is skewin image. Maybe should I convert without skew?
      x_idx = int((point[0] - min_x) / dx * self.img_size[0])
      y_idx = int((point[1] - min_y) / dy * self.img_size[1])
      if x_idx == self.img_size[0] or y_idx == self.img_size[1]:
        continue
      data[x_idx][y_idx] = 0.1

    for point in scan_pts:
      # TODO: this is skewin image. Maybe should I convert without skew?
      x_idx = int((point[0] - min_x) / dx * self.img_size[0])
      y_idx = int((point[1] - min_y) / dy * self.img_size[1])
      if x_idx == self.img_size[0] or y_idx == self.img_size[1]:
        continue
      data[x_idx][y_idx] = 1

    return torch.Tensor(data)

if __name__ == '__main__':
  import torchvision.transforms as T
  from PIL import Image

  dataloader = ScanDataset('/home/minoda/git/deep_scan_matching_validator/dataset')
  data = dataloader[4500]

  transform = T.ToPILImage()
  img = transform(data)
  img.save("./sample.png")
