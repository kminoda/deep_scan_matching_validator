from torch.utils.tensorboard import SummaryWriter

class Logger:
  def __init__(self, output_path: str):
    self.output_path = output_path
    self.writer = SummaryWriter(log_dir=output_path)

  def send_image(self, image, id, count):
    self.writer.add_image("images/{}".format(id), image, count)

  def send_train_loss(self, loss):
    self.writer.add_scalar('train_loss', loss)

  def send_valid_loss(self, loss):
    self.writer.add_scalar('valid_loss', loss)
