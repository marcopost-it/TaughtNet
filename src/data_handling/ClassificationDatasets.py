import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available, EvalPrediction

import torch
from torch import nn
from torch.utils.data.dataset import Dataset

from .DataClasses import InputFeatures, SequenceInputFeatures

logger = logging.getLogger(__name__)

class TokenClassificationDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        features,
    ):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

class SequenceClassificationDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[SequenceInputFeatures]

    def __init__(
        self,
        features,
    ):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]