# -*- coding: future_fstrings -*-
import json
import torch
from easydict import EasyDict as edict

from lib.data_loaders import make_data_loader
from config import get_config

from lib.trainer import Trainer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


torch.manual_seed(0)
torch.cuda.manual_seed(0)




def get_trainer():
  return Trainer


def main(config, resume=False):
  train_loader = make_data_loader(
      config,
      config.train_phase,
      config.batch_size,
      num_threads=config.train_num_thread)

  if config.test_eval:
    test_loader = make_data_loader(
        config,
        config.test_phase,
        config.test_batch_size,
        num_threads=config.test_num_thread)
  else:
    test_loader = None

  Trainer = get_trainer()
  trainer = Trainer(
      config=config,
      data_loader=train_loader,
      test_data_loader=test_loader,
  )

  trainer.train()


if __name__ == "__main__":

  config = get_config()
  dconfig = vars(config)
  config = edict(dconfig)
  main(config)
