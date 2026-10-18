import numpy.typing as npt
import numpy as np
import torch

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    idx = np.random.randint(0, len(dataset) - context_length, batch_size)
    chunk = np.array([dataset[i : i + context_length + 1] for i in idx]).astype(np.int64)
    x = torch.from_numpy(chunk[:, :-1])
    y = torch.from_numpy(chunk[:, 1:])
    return x.to(device), y.to(device)