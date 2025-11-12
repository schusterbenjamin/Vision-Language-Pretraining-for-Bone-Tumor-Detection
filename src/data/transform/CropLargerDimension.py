from typing import Sequence
from monai.transforms import MapTransform
import torch

class CropLargerDimension(MapTransform):
    """
    A transform that crops the larger dimension of an image to make it closer to square.
    This transform reduces the size of the larger dimension (height or width) by cropping
    from both ends equally, up to a maximum ratio of the original dimension size.
    Args:
        keys: Keys of the corresponding items to be transformed.
        maximum_crop_ratio: Maximum ratio of the larger dimension that can be cropped.
            Must be between 0 and 1. Default is 0.05 (5%).
    Raises:
        TypeError: If the input data is not a torch.Tensor.
    Note:
        - Input tensor must have shape (C, H, W) where C is channels, H is height, W is width.
        - If the image is already square (H == W), no cropping is performed.
        - Cropping is applied symmetrically from both ends of the larger dimension.
        - The actual crop size may be less than maximum_crop_ratio to ensure the result
          doesn't make the previously larger dimension smaller than the other dimension.
    """
    def __init__(self, keys: Sequence[str], maximum_crop_ratio: float = 0.05):
        super().__init__(keys)
        self.maximum_crop_ratio = maximum_crop_ratio

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if not isinstance(img, torch.Tensor):
                raise TypeError(f"Unsupported data type for CropLargerDimension: {type(img)}")

            # img shape: (C, H, W)
            c, h, w = img.shape
            if h == w:
                continue

            diff = abs(h - w)
            if diff == 0:
                continue
            
            if h > w:
                crop_size = int(h * self.maximum_crop_ratio)
                if h - crop_size < w:
                    crop_size = h - w
                crop_size_each = crop_size // 2
                img = img[:, crop_size_each:h - crop_size_each, :]
            else:
                crop_size = int(w * self.maximum_crop_ratio)
                if w - crop_size < h:
                    crop_size = w - h
                crop_size_each = crop_size // 2
                img = img[:, :, crop_size_each:w - crop_size_each]

            d[key] = img
        return d
        
