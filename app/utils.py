import torch
import typing as t
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import random
import numpy as np
from typing import Dict
from torch import nn
from pathlib import Path
from torch import Tensor
from logging import getLogger

logger = getLogger(__name__)


def init_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed(seed)  # type: ignore
