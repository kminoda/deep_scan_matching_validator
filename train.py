import argparse
import yaml

from typing import Tuple
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from torch import autograd

from dataset import ScanDataset
from model import get_model
from logger import Logger

DEVICE = torch.device('cuda:0')

def train_epoch(model, train_data_loader: DataLoader, logger: Logger, optimizer, calculate_loss):
  model.train()
  for images, labels in tqdm(train_data_loader):
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    # print(labels, labels_pred)
    with autograd.detect_anomaly():
      labels_pred = model(images)

      loss = calculate_loss(labels, labels_pred)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    loss_detached = loss.cpu().detach().numpy()
    logger.send_train_loss(loss_detached)
    print(loss_detached)

def exec_train(num_epoch: int, train_data_loader: DataLoader, valid_data_loader: DataLoader, output_dir: str):
  device = torch.device(DEVICE)
  model = get_model().to(device)

  optimizer = Adam(model.parameters(), lr=0.001)
  loss = BCELoss()
  logger = Logger(output_dir)

  # save sample images
  count = 0
  for images, labels in tqdm(train_data_loader):
    logger.send_image(images[0], count, 0)
    count += 1
    if count > 3:
      break

  for epoch in range(0, num_epoch):
    train_epoch(model, train_data_loader, logger, optimizer, loss)

def get_loader(dataset_path: str) -> Tuple[DataLoader, DataLoader]:
  train_dataset = ScanDataset(Path(dataset_path) / 'train')
  valid_dataset = ScanDataset(Path(dataset_path) / 'valid')

  train_data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
  valid_data_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)
  return train_data_loader, valid_data_loader

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('config_path')
  args = parser.parse_args()

  with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)

  train_data_loader, valid_data_loader = get_loader(config['dataset_path'])
  exec_train(config['num_epoch'],
             train_data_loader,
             valid_data_loader,
             Path(config['output_path']) / config['experiment_name'])
