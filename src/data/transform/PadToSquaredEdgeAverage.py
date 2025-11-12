from monai.transforms import MapTransform
import torch
import numpy as np
import torch.nn.functional as F
from typing import Sequence

class PadToSquaredEdgeAverage(MapTransform):
    """
    A transform that pads images to make them square by using edge pixel averages.
    This transform takes non-square images and pads the shorter dimension to match
    the longer dimension, creating a square image. The padding values are computed
    by taking the average of edge pixels for each channel separately.
    Args:
        keys (Sequence[str]): Keys in the data dictionary to apply the transform to.
    Raises:
        TypeError: If the input data is not a torch.Tensor.
    Note:
        - Input tensors must be 3D with shape (C, H, W)
        - If height > width: pads left and right edges using average of leftmost 
          and rightmost column pixels respectively
        - If width > height: pads top and bottom edges using average of topmost 
          and bottommost row pixels respectively  
        - If the image is already square, no padding is applied
        - Edge averages are computed per-channel to preserve color information
    """
    def __init__(self, keys: Sequence[str]):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if not isinstance(img, torch.Tensor):
                raise TypeError(f"Unsupported data type for PadToSquaredEdgeAverage: {type(img)}")

            # img shape: (C, H, W)
            c, h, w = img.shape
            if h == w:
                continue

            diff = abs(h - w)
            
            if h > w:
                # Height larger -> pad width
                left_pad = diff // 2
                right_pad = diff - left_pad

                # Compute per-channel averages for left and right
                left_edge = img[:, :, 0].float().mean(dim=1)  # (C,)
                right_edge = img[:, :, -1].float().mean(dim=1)  # (C,)

                # Expand to (C, 1, left_pad) and (C, 1, right_pad)
                left_pad_tensor = left_edge[:, None, None].expand(-1, h, left_pad)
                right_pad_tensor = right_edge[:, None, None].expand(-1, h, right_pad)

                # Pad manually by concatenation
                img = torch.cat([left_pad_tensor, img, right_pad_tensor], dim=2)

            else:
                # Width larger -> pad height
                top_pad = diff // 2
                bottom_pad = diff - top_pad

                # Compute per-channel averages for top and bottom
                top_edge = img[:, 0, :].float().mean(dim=1)  # (C,)
                bottom_edge = img[:, -1, :].float().mean(dim=1)  # (C,)

                # Expand to (C, top_pad, W) and (C, bottom_pad, W)
                top_pad_tensor = top_edge[:, None, None].expand(-1, top_pad, w)
                bottom_pad_tensor = bottom_edge[:, None, None].expand(-1, bottom_pad, w)

                # Pad manually by concatenation
                img = torch.cat([top_pad_tensor, img, bottom_pad_tensor], dim=1)

            d[key] = img
        return d
