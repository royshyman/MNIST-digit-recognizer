# Gets the given image to look like a classic MNIST image the mode was trained on

import numpy as np
from PIL import Image,ImageOps
import torch

def preprocess(arr):
    arr = 255 - arr
    
    rows = np.any(arr > 20, axis=1)                    # True for each row that contains ink
    cols = np.any(arr > 20, axis=0)                    # True for each column that contains ink
    if not rows.any():                                 # nothing drawn: return an all-black image
        return torch.full((1, 1, 28, 28), (0 - 0.1307) / 0.3081)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    digit = arr[top:bottom + 1, left:right + 1]         # crop
    h, w = digit.shape                                  # resize so the longer side is 20 px, keeping the aspect ratio
    scale = 20 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    digit = np.array(Image.fromarray(digit).resize((new_w, new_h), Image.LANCZOS))
    # paste into a black 28x28 frame so the digit's center of mass lands at the center, like MNIST
    weights = digit.astype(np.float64)
    rows_idx, cols_idx = np.indices(digit.shape)
    cy = (rows_idx * weights).sum() / weights.sum()     # brightness-weighted average row
    cx = (cols_idx * weights).sum() / weights.sum()     # brightness-weighted average column
    y = int(np.clip(round(14 - cy), 0, 28 - new_h))     # clip so the digit stays inside the frame
    x = int(np.clip(round(14 - cx), 0, 28 - new_w))
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[y:y + new_h, x:x + new_w] = digit
    final_arr = canvas.astype(np.float32) / 255.0   # like ToTensor(): 0-255 -> 0.0-1.0
    final_arr = (final_arr - 0.1307) / 0.3081   #normalizes like the MNIST data
    return torch.tensor(final_arr).view(1, 1, 28, 28)
    


           
