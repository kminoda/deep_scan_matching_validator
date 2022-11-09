import torch
import os
import yaml
import numpy as np

from pathlib import Path

class ScanDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path: str):
    dataset_path = '/home/minoda/git/deep_scan_matching_validator/dataset/'
    dataset_path = Path(dataset_path)
    for p in dataset_path.iterdir():
      with open(dataset_path / p / 'metadata.yaml') as f:
        metadata = yaml.safe_load(f)
      pointcloud_map_path = metadata['pointcloud_map_path']

      # Should I load pcd file here?

      self.scan_data = list((dataset_path / p / 'scan_data').glob('*.pickle'))

  def __len__(self):
    return len(self.scan_data)

  def __getitem__(self, index, label=''):
    scan = self.scan_data[index]

    if label=='':
      is_valid_scan = np.random.rand() > 0.5
    elif label=='valid':
      is_valid_scan = True
    elif label=='invalid':
      is_valid_scan = False
    else:
      raise NotImplementedError

    scan_pts = None
    map_pts = None
    return _generate_image(scan_pts, map_pts)

  def _generate_image(self, scan_pts, map_pts):
    return torch.Tensor()      

if __name__ == '__main__':
  dataloader = ScanDataset('/home/minoda/git/deep_scan_matching_validator/dataset')