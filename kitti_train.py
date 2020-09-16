import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from kitti_data import KITTI
from prednet import PredNet

from debug import info

import pytorch_lightning as pl
from pytorch_lightning import Trainer

num_epochs = 30
batch_size = 2
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
lr = 0.001 # if epoch < 75 else 0.0001
nt = 10 # num of time steps

layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]).cuda())
time_loss_weights = 1./(nt - 1) * torch.ones(nt, 1)
time_loss_weights[0] = 0
time_loss_weights = Variable(time_loss_weights.cuda())

DATA_DIR = '/home/jeped/pytorch-prednet/kitti_hkl/'

train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')


kitti_train = KITTI(train_file, train_sources, nt)
kitti_val = KITTI(val_file, val_sources, nt)

train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True, num_workers=12)
val_loader = DataLoader(kitti_val, batch_size=batch_size, shuffle=True, num_workers=12)

class PredNetClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = PredNet(R_channels, A_channels, output_mode='error')

    def forward(self, x):
        inputs = x.permute(0, 1, 4, 2, 3) # batch x time_steps x channel x width x height
        return self.model(inputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def training_step(self, batch, batch_idx):
        errors = self(batch)
        loc_batch = errors.size(0)
        errors = torch.mm(errors.view(-1, nt), time_loss_weights) # batch*n_layers x 1
        errors = torch.mm(errors.view(loc_batch, -1), layer_loss_weights)
        errors = torch.mean(errors)

        result = pl.TrainResult(errors)
        result.log('train_loss', errors, on_epoch=True)
    
    # def validation_step(self, batch, batch_idx):
    #     errors = self(batch)
    #     loc_batch = errors.size(0)
    #     errors = torch.mm(errors.view(-1, nt), time_loss_weights) # batch*n_layers x 1
    #     errors = torch.mm(errors.view(loc_batch, -1), layer_loss_weights)
    #     errors = torch.mean(errors)

    #     result = pl.EvalResult(checkpoint_on=errors)
    #     result.log('val_loss', errors, on_epoch=True)

# def lr_scheduler(optimizer, epoch):
#     if epoch < num_epochs //2:
#         return optimizer
#     else:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 0.0001
#         return optimizer

model = PredNetClassifier()
trainer = Trainer(gpus=4, distributed_backend='ddp')
trainer.fit(model, train_loader, val_loader)
torch.save(model.model.state_dict(), 'training.pt')
