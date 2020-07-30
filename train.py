import os

import argparse
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader,ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR,MultiStepLR

from models.utils.misc import str2bool,Timer,freeze_net_layers,store_labels
from models.ssd.ssd import MatchPrior
from models.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from