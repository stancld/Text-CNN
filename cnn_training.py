# -*- coding: utf-8 -*-
"""cnn_clf.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NI8nv_5_GWtTPRbTz61oDtK0i81ZwItJ

# CNN Classifier - Training

<hr>

This notebook contains a script for training a CNN classifier implemented in cnn_clf.py

Classifier is implemented in PyTorch and utilizes the structure of PyTorch Lighting which also enables to train our model on TPUs.

## **Setup**

- install modules necessary for TPU computation
- install torch==1.5.0 and pytorch-lightnint
- import libraries and mount Google Drive
"""

!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev

# Commented out IPython magic to ensure Python compatibility.
# insall pytorch, pytorch-ligthning
!pip install torch==1.5.0
!pip install pytorch-lightning

# import basic data science lib
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# mport torch-related modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

# mount a google drive
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
# %cd 'drive/My Drive'

time.sleep(5)
from cnn_clf import * # the best practive would be store a script with the CNN class on github and clone the repo

"""## **Load the data**"""

label = 'label_top5'

# helper loading function
def X_loader(dataset):
  return torch.Tensor(
      np.load(f'SeznamResearch/splitted_data/cnn/{dataset}.npy')
  )
def y_loader(dataset):
  return torch.Tensor(
      pd.read_csv(f'SeznamResearch/splitted_data/bert/{dataset}.csv')[label]
  )

# load feature matrices
X_train = X_loader('train')
X_dev = X_loader('dev')
X_test = X_loader('test')

# load labels
y_train = y_loader('train')
y_dev = y_loader('dev')
y_test = y_loader('test')

# complete tuple (X, y)
datasets = {
    'train': TensorDataset(X_train, y_train),
    'val': TensorDataset(X_dev, y_dev),
    'test': TensorDataset(X_test, y_test)
}

"""## **Run training**"""

### editable parameters
dropout_set = [0.1, 0.2, 0.3, 0.5]
kernel_num_set = [64, 128, 256, 512]

# set early stopping rule
early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0,
    patience=5,
    verbose=1,
    mode='min'
)

############# don't edit #############
val_f1 = np.zeros(
    (len(dropout_set), len(kernel_num_set))
)
test_f1 = np.zeros_like(val_f1)

# run training
for i,dropout in enumerate(dropout_set):
  for j,kernel_num in enumerate(kernel_num_set):
    # set parameters and instantiate net
    cnn_parameters = {
        'batch_size': 32,
        'kernel_num': kernel_num,
        'kernel_filters': (3,4,5),
        'embed_dim': 100,
        'learning_rate': 4e-3,
        'dropout': dropout,
        'num_classes': 6,
        'datasets': datasets
    }
    cnn = CNN_classifier(**cnn_parameters)

    # define trainer and run training
    try:
      del trainer
    except:
      pass
    trainer = Trainer(
      tpu_cores=8,
      check_val_every_n_epochs=2,
      early_stop_callback=early_stop_callback,
      max_epochs=50,
      progress_bar_refresh_rate=0
    )
    trainer.fit(cnn)
    
    # store f1-score on val and test set
    val_f1[i,j] = f1_score(
        y_dev.detach().numpy(), # y_true
        cnn(X_dev).argmax(1).detach().numpy(), # y_pred
        average='macro'
    )
    print(
        f'Kernel num = {kernel_num}, Dropout = {dropout}: F1-macro = {val_f1[i,j]:.4f}'
    )
    test_f1[i,j] = f1_score(
        y_test.detach().numpy(), # y_true
        cnn(X_test).argmax(1).detach().numpy(), # y_pred
        average='macro'
    )

# save the results
np.save('val_f1.npy', val_f1) # the best classifier is selected based on val_score; all test score are generated for the purpose of simplicity of sampling later on
np.save('test_f1.npy', test_f1)