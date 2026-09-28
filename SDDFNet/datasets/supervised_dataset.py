"""Official xBD test split and fixed, event-stratified train/validation split."""
import json
from pathlib import Path
import lightning.pytorch as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets.base_dataset import LabeledDataset, paired_paths


def collect_files(roots, recursive=False):
    files = []
    for directory in roots:
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(root)
        dirs = sorted(root.rglob("images")) if recursive else [root / "images"]
        for folder in dirs:
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            for pre in sorted(folder.glob("*_pre_disaster.png")):
                for path in paired_paths(pre):
                    if not path.is_file():
                        raise FileNotFoundError(path)
                files.append(str(pre.resolve()))
    if not files:
        raise ValueError("No paired samples found")
    if len(files) != len(set(files)):
        raise ValueError("Duplicate sample paths found")
    return files


class SLDataModule(pl.LightningDataModule):
    def __init__(self, data_dirs, test_dirs, labeled_transforms, labeled_batch_size=16,
                 num_workers=4, eval_batch_size=1, split_seed=23):
        super().__init__()
        self.data_dirs, self.test_dirs = data_dirs, test_dirs
        self.labeled_transforms = labeled_transforms
        self.labeled_batch_size, self.eval_batch_size = labeled_batch_size, eval_batch_size
        self.num_workers, self.split_seed = num_workers, split_seed
        self.train_files = self.val_files = self.test_files = None

    def setup(self, stage=None):
        if stage in (None, "fit", "validate") and self.train_files is None:
            files = collect_files(self.data_dirs)
            names = [Path(f).name for f in files]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate image identifiers across development subsets")
            labels = [Path(f).name.split("_")[0] for f in files]
            self.train_files, self.val_files = train_test_split(
                files, test_size=0.1, random_state=self.split_seed, stratify=labels)
            self.train_dataset = self._dataset(self.train_files, True)
            self.val_dataset = self._dataset(self.val_files, False)
        if stage in (None, "test", "predict") and self.test_files is None:
            self.test_files = collect_files(self.test_dirs)
            # Check identifiers, not only absolute paths, across source subsets.
            development = collect_files(self.data_dirs)
            if {Path(f).name for f in development} & {Path(f).name for f in self.test_files}:
                raise ValueError("Development and test sample identifiers overlap")
            self.test_dataset = self._dataset(self.test_files, False)
        if self.trainer is not None and self.trainer.is_global_zero:
            out = Path(self.trainer.default_root_dir) / "splits"
            out.mkdir(parents=True, exist_ok=True)
            for name in ("train", "val", "test"):
                files = getattr(self, name + "_files")
                if files is not None:
                    (out / (name + ".json")).write_text(json.dumps(files, indent=2))

    def _dataset(self, files, train):
        return LabeledDataset(files, list(range(len(files))), self.labeled_transforms, train)

    def _loader(self, dataset, train=False):
        return DataLoader(dataset, batch_size=self.labeled_batch_size if train else self.eval_batch_size,
                          shuffle=train, num_workers=self.num_workers, pin_memory=True)

    def train_dataloader(self): return self._loader(self.train_dataset, True)
    def val_dataloader(self): return self._loader(self.val_dataset)
    def test_dataloader(self): return self._loader(self.test_dataset)
    def predict_dataloader(self): return self._loader(self.test_dataset)
