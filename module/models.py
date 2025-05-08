import torch
import torch.nn as nn
import torch.nn.functional as F
import tools.functions as fn
import copy

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)


