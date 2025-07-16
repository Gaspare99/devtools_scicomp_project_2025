import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class CIFAR10DataModule(pl.LightningDataModule):

    def __init__(self, data_path='./data', batch_size=64):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.Resize((70, 70)), transforms.RandomCrop((64, 64)),
                transforms.ToTensor()])

    def prepare_data(self):
        """Downloads the CIFAR-10 dataset and put it in the data_path folder."""

        datasets.CIFAR10(root=self.data_path, download=True)

    def setup(self, stage):
        """
        The data set is divide into two pieces: training and validation dataset with 45000 and 5000 data each
        """

        train = datasets.CIFAR10(
            root=self.data_path,
            train=True,
            transform=self.transform,
            download=False,
        )
        # Split the training dataset into training (45,000) and validation (5,000)
        self.train, self.valid = random_split(train, lengths=[45000, 5000])
        self.test = datasets.CIFAR10(
            root=self.data_path,
            train=False,
            transform=self.transform,
            download=False,
        )

    def train_dataloader(self):
        """
        Create the training data loader.

        Returns: DataLoader for the training data set
            
        """
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        """
        Returns the data loader for the validation set.

        This method returns a DataLoader for the validation set, with shuffling disabled
        and drop_last set to False to keep all batches.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        valid_loader = DataLoader(
            dataset=self.valid,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )
        return valid_loader

    def test_dataloader(self):
        """
        Returns the data loader for the testing set.

        This method returns a DataLoader for the testing set, with shuffling disabled
        and drop_last set to False to keep all batches.

        Returns:
            DataLoader: DataLoader for the testing dataset.
        """
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )
        return test_loader