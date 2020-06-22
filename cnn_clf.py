import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

# model
class CNN_classifier(LightningModule):
    """
    A CNN sentence/text classification model based on the paper: Convolutional Neural Networks for Sentence Classification.
    Implementation is a generalization of https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
    and it is modififed according to PyTorch Lightning.
    """
    def __init__(self, **kwargs):
        super(CNN_classifier, self).__init__()
        # store paramters from kwargs
        self.batch_size = kwargs['batch_size']
        self.C_out = kwargs['kernel_num']
        self.embed_dim = kwargs['embed_dim']
        self.kernel_filters = kwargs['kernel_filters']
        self.lr = kwargs['learning_rate']
        self.datasets = kwargs['datasets']

        # define convolutional layers
        self.conv_layers = nn.ModuleDict({
            f'conv{i+1}': self._init_conv_layer(filter_size) for i, filter_size in enumerate(self.kernel_filters) 
        })
        
        # define FC layer
        self.fc = nn.Linear(
            in_features=len(self.conv_layers) * self.C_out,
            out_features=kwargs['num_classes']
        )
        # define drop-out layer
        self.dropout = nn.Dropout(kwargs['dropout'])

    def forward(self, x):
        """
        :param x: input tensor of size [batch_size, seq_len, embed_dim ]
        """
        x = x.unsqueeze(1) # [batch_size, in_channel, seq_len, embed_dim]              
        x = self.dropout(
                torch.cat(
                    tuple(self._conv_forward(x, i+1) for i,_ in enumerate(self.conv_layers.keys())),
                    dim=1
                )
        )
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(torch.FloatTensor(data))
        loss = F.cross_entropy(output, target.long())
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(torch.FloatTensor(data))
        loss = F.cross_entropy(output, target.long())
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {
            'avg_val_loss': avg_loss,
            'log': tensorboard_logs
        }
    
    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(torch.FloatTensor(data))
        loss = F.cross_entropy(output, target.long())
        return {'test_loss': loss}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {
            'avg_test_loss': avg_loss,
            'log': tensorboard_logs
        }
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def prepare_data(self):
        """
        The funtion prepare_data follos convention of LightningModule

        :param datasets: dictionary of train, val and (test) sets
        """
        try:
            self.train_dataset = self.datasets['train']
            self.val_dataset = self.datasets['val']
            try:
                self.test_dataset = self.datasets['test']
            except:
                pass
        except Exception as e:
            print('Data was not succesfully prepared:', e)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0
        )

    def test_dataloader(self):
        try:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=0
            )
        except Exception as e:
            print(e) # very likely no test set loaded
        

    def _init_conv_layer(self, filter_size):
        """
        Helper function: a function to instantiate 2D convolutional layers

        :param filter_size: an integer determining a number of words kernel size should span over.
        """
        return nn.Conv2d(
            in_channels=1,
            out_channels=self.C_out,
            kernel_size=(filter_size, self.embed_dim)
        )

    def _conv_forward(self, x, i):
        """
        Helper function: Forward for convolutional layer accompanied by relu and max_pool operation

        :param x: input tensor of size [batch_size, in_channel, seq_len, embed_dim]
        :param i: an integer denoting which convolution layer should be used
        """
        x = F.relu(self.conv_layers[f'conv{i}'](x)).squeeze(3) # [batch_size, in_channel, seq_len]
        return F.max_pool1d(x, x.size(2)).squeeze(2) # [batch_size, in_channel]   