import abc
from torch.utils.data import Dataset


class BaseDataset(Dataset, abc.ABC):
    def __init__(self):
        self.audio_data = None
        self.labels = None

    def __getitem__(self, idx):
        if self.audio_data is None:
            raise NotImplementedError("self.audio_data is not initialized")
        if self.labels is None:
            raise NotImplementedError("self.labels is not initialized")
        return self.audio_data[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    @abc.abstractmethod
    def prepare_data(self):
        pass
