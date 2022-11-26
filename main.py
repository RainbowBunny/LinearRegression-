import sys
import torch

import pandas as pd

from torch import nn, optim

from constant import BATCH_SIZE, LEARNING_RATE, N_EPOCHS, INPUT_SIZE, OUTPUT_SIZE
from data_processor import DataProcessor
from model import Model

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Trainer:
    pass

if __name__ == "__main__":
    o = Trainer()
    o.train()