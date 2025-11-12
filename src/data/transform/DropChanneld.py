from monai.transforms import MapTransform
import torch

class DropChanneld(MapTransform):
    """
    A transform that removes a specific channel from multi-channel tensors.

    This transform drops a channel at the specified index from 3D tensors with shape (C, H, W).
    If the tensor has fewer channels than the specified drop index, the tensor remains unchanged.

    Args:
        keys: Keys in the data dictionary to apply the transform to.
        channel_to_drop (int): Index of the channel to remove (0-indexed).

    Raises:
        TypeError: If the data for a key is not a torch.Tensor.
        ValueError: If the tensor is not 3-dimensional with shape (C, H, W).
    """
    def __init__(self, keys, channel_to_drop):
        super().__init__(keys)
        self.channel_to_drop = channel_to_drop

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if not isinstance(img, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for key '{key}', but got {type(img)}")
            if img.ndim != 3:
                raise ValueError(f"Expected 3D tensor (C,H,W) for key '{key}', but got shape {img.shape}")
            if img.shape[0] <= self.channel_to_drop:
                # Nothing to drop; keep original
                continue
            d[key] = torch.cat([img[:self.channel_to_drop], img[self.channel_to_drop + 1:]], dim=0)
        return d
