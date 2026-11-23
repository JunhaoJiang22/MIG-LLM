import random
import numpy as np
import torch


def reproducibility_settings(seed: int = 0):
    r"""Set the random seed for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False  # Settings for 3090
    torch.backends.cudnn.allow_tf32 = False  # Settings for 3090


def round_floats(o):
        if isinstance(o, float):
            return round(o, 6)
        if isinstance(o, dict):
            return {k: round_floats(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [round_floats(x) for x in o]
        return o


if __name__ == '__main__':
    pass
