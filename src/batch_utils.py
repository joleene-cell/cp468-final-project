"""
batch_utils.py
Small glue helper for integrating with data_loader.py.
"""

import torch

def lengths_from_padded(padded: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return (padded != pad_idx).sum(dim=1).long()
