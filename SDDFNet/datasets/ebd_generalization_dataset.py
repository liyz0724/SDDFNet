"""Evaluation-only EBD data module; no xBD training directories are required."""
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from datasets.base_dataset import LabeledDataset
from datasets.supervised_dataset import collect_files


class EBDGeneralizationDataModule(pl.LightningDataModule):
    def __init__(self, test_dirs, labeled_batch_size=1, num_workers=4):
        super().__init__()
        self.test_dirs = test_dirs
        self.labeled_batch_size = labeled_batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage not in (None, "test", "predict"):
            raise ValueError("EBD is an evaluation-only dataset")
        files = collect_files(self.test_dirs, recursive=True)
        self.test_dataset = LabeledDataset(files, list(range(len(files))), train=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.labeled_batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return self.test_dataloader()
