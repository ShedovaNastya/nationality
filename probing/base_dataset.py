import abc
from torch.utils.data import Dataset


class BaseDataset(Dataset, abc.ABC):
    def __init__(self):
        self.audio_data = None
        self.labels = None

    def __getitem__(self, idx):
        assert self.audio_data is not None, "audio_data is not initialized"
        assert self.labels is not None, "labels is not initialized"
        return self.audio_data[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    @abc.abstractmethod
    def prepare_data(self):
        pass
