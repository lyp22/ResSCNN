# -*- coding: future_fstrings -*-

import os
import os.path as osp
import gc
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import MinkowskiEngine as ME
from model.resscnn import ResSCNN
import torch.nn as nn
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
import xlrd
from scipy.optimize import curve_fit
from lib.Logging import Logger
from tqdm import tqdm


def read_xlrd(excelFile):
  data = xlrd.open_workbook(excelFile)
  table = data.sheet_by_index(0)
  dataFile = []
  for rowNum in range(table.nrows):
    if rowNum > 0:
      dataFile.append(table.row_values(rowNum))
  dataFile = sorted(dataFile)
  return dataFile

def logistic_5_fitting_no_constraint(x, y):
  def func(x, b0, b1, b2, b3, b4):
    logistic_part = 0.5 - np.divide(1.0, 1 + np.exp(b1 * (x - b2)))
    y_hat = b0 * logistic_part + b3 * np.asarray(x) + b4
    return y_hat

  x_axis = np.linspace(np.amin(x), np.amax(x), 100)
  init = np.array([np.max(y), np.min(y), np.mean(x), 0.1, 0.1])
  popt, _ = curve_fit(func, x, y, p0=init, maxfev=int(1e8))
  curve = func(x_axis, *popt)
  fitted = func(x, *popt)

  return x_axis, curve, fitted



class ResSCNNTrainer:

  def __init__(
      self,
      config,
      data_loader,
      test_data_loader=None,
  ):

    # Model initialization
    Model = ResSCNN
    model = Model(bn_momentum=config.bn_momentum, D=3)

    self.logging = Logger(name='output')
    self.logging.info(model)

    self.config = config
    self.model = model
    self.max_epoch = config.max_epoch
    self.test_max_iter = config.test_max_iter
    self.test_epoch_freq = config.test_epoch_freq

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.optimizer = getattr(optim, config.optimizer)(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)


    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

    self.start_epoch = 1
    self.checkpoint_dir = config.out_dir


    self.iter_size = config.iter_size
    self.batch_size = data_loader.batch_size
    self.data_loader = data_loader
    self.test_data_loader = test_data_loader

    self.test_eval = True if self.test_data_loader is not None else False
    self.log_step = int(np.sqrt(self.config.batch_size))
    self.model = self.model.to(self.device)
    self.writer = SummaryWriter(logdir=config.out_dir)

    if config.resume is not None:
      if osp.isfile(config.resume):
        self.logging.info("=> loading checkpoint '{}'".format(config.resume))
        state = torch.load(config.resume)
        self.start_epoch = state['epoch']
        model.load_state_dict(state['state_dict'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.optimizer.load_state_dict(state['optimizer'])

      else:
        raise ValueError(f"=> no checkpoint found at '{config.resume}'")


  def train(self):

    if self.test_eval:
      with torch.no_grad():
        val_dict,plcc,srocc = self._test_epoch(0)

      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    results_srocc = []

    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      self.logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      # self._save_checkpoint(epoch)
      self.scheduler.step()

      if self.test_eval and epoch % self.test_epoch_freq == 0:
        with torch.no_grad():
          val_dict, plcc, srocc = self._test_epoch(epoch)

        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)

        results_srocc.append([plcc, srocc])
    results_srocc_list = pd.DataFrame(columns=['PLCC','SROCC'], data=results_srocc)
    results_srocc_list.to_csv('results_srocc.csv', index=False)


  def _save_checkpoint(self, epoch, filename='test_checkpoint'):
    state = {
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config,
    }
    filename = os.path.join(self.checkpoint_dir, f'{filename+str(epoch)}.pth')
    self.logging.info("Saving checkpoint: {} ...".format(filename))
    torch.save(state, filename)


class Trainer(ResSCNNTrainer):

  def __init__(
      self,
      config,
      data_loader,
      test_data_loader=None,
  ):
    ResSCNNTrainer.__init__(self, config, data_loader, test_data_loader)


  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T


  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    total_loss = 0
    total_num = 0.0

    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()

    iter_size = self.iter_size
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)


    train_loss = []

    # Main training
    for curr_iter in tqdm(range(len(data_loader) // iter_size)):
      self.optimizer.zero_grad()
      batch_loss = 0

      data_time = 0
      for iter_idx in range(iter_size):
        try:
          feats, coords, labelMOS, temp2 = data_loader_iter.next()
        except:
          self.logging.info("Dataset error")
          continue

        stensor = ME.SparseTensor(feats[0].to(self.device), coordinates=coords[0].to(self.device))

        F0 = self.model(stensor)



        loss_function = nn.SmoothL1Loss()
        a = F0.reshape(1,1).double()
        b = labelMOS.to(self.device).reshape(1,1).double()
        loss = loss_function(a, b)

        loss.backward()
        batch_loss += loss.item()

        train_loss.append(loss.detach().cpu())


      self.optimizer.step()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss/iter_size, start_iter + curr_iter)
        self.logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} "
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss/iter_size))
    train_loss = np.mean(train_loss)
    self.logging.info(
      "Train Epoch: {} , Train Loss: {:.6f} "
      .format(epoch, train_loss))

  def _test_epoch(self,epoch):
    # evaluation mode
    self.model.eval()
    self.test_data_loader.dataset.reset_seed(0)
    num_data = 0
    tot_num_data = len(self.test_data_loader.dataset)
    if self.test_max_iter > 0:
      tot_num_data = min(self.test_max_iter, tot_num_data)
    data_loader_iter = self.test_data_loader.__iter__()

    test_loss = []
    results = []

    for batch_idx in tqdm(range(tot_num_data)):
      feats, coords, labelMOS, temp = data_loader_iter.next()

      stensor = ME.SparseTensor(feats[0].to(self.device), coordinates=coords[0].to(self.device))
      F0 = self.model(stensor)

      loss_function = nn.SmoothL1Loss()
      a = F0.reshape(1, 1).double()
      b = labelMOS.to(self.device).reshape(1, 1).double()  # .double()
      loss = loss_function(a, b)

      test_loss.append(loss.detach().cpu())

      num_data += 1
      torch.cuda.empty_cache()

      if batch_idx % 100 == 0 and batch_idx > 0:
        self.logging.info(' '.join([
            f"Testing iter {num_data} / {tot_num_data} : ,",
            f"Loss: {np.mean(test_loss):.3f}, "
        ]))
      results.append([temp, np.squeeze(a.detach().cpu().numpy())])

    self.logging.info(' '.join([
        f"Final Loss: {np.mean(test_loss):.3f}, "
    ]))
    test_loss = np.mean(test_loss)
    self.logging.info(
      "Test Loss: {:.6f} ".format(test_loss))

    # results2 = pd.DataFrame(columns=['plyname','plyscore'],data=results)
    # results2.to_csv(f'results/test_pre_score{str(epoch)}.csv',index=False)

    compare1 = sorted(results)
    compare2 = read_xlrd(self.config.test_file)

    mos = []
    for y in compare2:
      mos.append(y[1])

    pred = []
    for y in compare1:
      pred.append(y[1])

    _, _, pred = logistic_5_fitting_no_constraint(pred, mos)

    plcc, _ = pearsonr(pred, mos)
    srocc, _ = spearmanr(pred, mos)
    krocc, _ = kendalltau(pred, mos)
    rmse = np.sqrt(np.mean((pred - mos) ** 2))

    self.logging.info(' '.join([
      f"PLCC: {plcc:.6f},",
      f"SROCC: {srocc:.6f}, ",
      f"KROCC: {krocc:.6f}, ",
      f"RMSE: {rmse:.6f}, "
    ]))


    return {"loss": np.mean(test_loss),}, plcc, srocc
