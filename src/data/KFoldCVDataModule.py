from abc import ABC, abstractmethod
from typing import Generator, Tuple
import lightning as L
from torch.utils.data import DataLoader

class KFoldCVDataModule(ABC):
    """
    KFoldCVDataModule is an abstract base class designed to facilitate K-Fold Cross-Validation 
    using PyTorch Lightning's DataModule. It provides an interface for managing data splits 
    and dataloaders for training, validation, and testing.
    The train scripts are based on using this class to run K-Fold Cross-Validation.

    Attributes:
        batch_size (int): The batch size to be used for the dataloaders. Default is 32.
        num_workers (int): The number of worker threads to use for data loading. Default is 2.

    Methods:
        get_cv_splits() -> Generator[L.LightningDataModule, None, None]:
            Abstract method that should yield a LightningDataModule for each fold.

        test_dataloader() -> DataLoader:
            Abstract method that should return one or more test dataloaders.
    """
    def __init__(self, batch_size: int = 32, num_workers: int = 2):
        self.batch_size = batch_size
        self.num_workers = num_workers

    @abstractmethod
    def get_cv_splits(self) -> Generator[Tuple[L.LightningDataModule, Tuple[float, float]], None, None]:
        """
        Should yield an Tuple of a LightningDataModule and the class weights as tuple (first position: class 0, second position: class 1) per fold.
        The LightningDataModule should have set:
        - train_dataloader
        - list of validation dataloaders
        """
        pass

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """
        Should return one or more test dataloaders.
        """
        pass

class DataModuleFolds(L.LightningDataModule):
    def __init__(self, train_dataloader, val_dataloaders: list):
        super().__init__()
        self.train_loader = train_dataloader
        self.val_loaders = val_dataloaders

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loaders