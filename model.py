from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import Module
from torch.nn import Linear
import torch.nn.functional as F

class MyModel(Module):
  def __init__(self):
    super().__init__()
    self.model = resnet18()
    self.model.fc = Linear(512, 1)
  
  def forward(self, x):
    x = self.model.forward(x)
    # x = self.fc(F.relu(x))
    return x

def get_model():
  # model = MyModel()
  model = resnet18()
  model.fc = Linear(512, 1)
  return model

if __name__ == '__main__':
  model = get_model()
  print(model)
  
  # from dataset import ScanDataset
  # dataset = ScanDataset('/home/minoda/git/deep_scan_matching_validator/dataset/train')
  # image, label = dataset[1500]
  # images = image.unsqueeze(0)

  # label_pred = model(images)
  # print(label_pred)
  # print(model)
