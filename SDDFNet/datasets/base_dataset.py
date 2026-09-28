"""Paired BGR images and five binary target channels."""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_image(img):
    return img.float() / 127.0 - 1.0


def paired_paths(pre):
    pre = Path(pre)
    mask = pre.parent.parent / "masks" / pre.name
    return (pre, pre.with_name(pre.name.replace("_pre_disaster", "_post_disaster")),
            mask, mask.with_name(mask.name.replace("_pre_disaster", "_post_disaster")))


class LabeledDataset(Dataset):
    def __init__(self, all_files, labeled_idxs, labeled_transforms=None, train=False):
        self.all_files = all_files
        self.labeled_idxs = labeled_idxs
        self.labeled_transforms = labeled_transforms
        self.train = train

    def __len__(self):
        return len(self.labeled_idxs)

    def __getitem__(self, idx):
        fn = self.all_files[self.labeled_idxs[idx]]
        paths = paired_paths(fn)
        arrays = [cv2.imread(str(p), cv2.IMREAD_COLOR if i < 2 else cv2.IMREAD_UNCHANGED)
                  for i, p in enumerate(paths)]
        for p, a in zip(paths, arrays):
            if a is None:
                raise OSError(f"Missing or unreadable sample file: {p}")
        pre, post, loc, dmg = arrays
        if len({a.shape[:2] for a in arrays}) != 1:
            raise ValueError(f"Image and mask dimensions differ: {fn}")
        if loc.ndim != 2 or dmg.ndim != 2:
            raise ValueError(f"Expected single-channel masks: {fn}")
        if not set(np.unique(loc)).issubset({0, 255}):
            raise ValueError(f"Localization mask must use 0/255: {fn}")
        if not set(np.unique(dmg)).issubset({0, 1, 2, 3, 4}):
            raise ValueError(f"Damage mask must use IDs 0..4: {fn}")
        img = torch.from_numpy(np.concatenate([pre, post], axis=2).transpose(2, 0, 1)).float()
        msk = torch.from_numpy(np.stack([loc > 127] + [dmg == c for c in range(1, 5)])).float()
        if self.train and self.labeled_transforms is not None:
            together = self.labeled_transforms(torch.cat([img, msk], dim=0))
            img, msk = together[:6], together[6:]
        return {"img": normalize_image(img), "msk": msk, "fn": str(fn)}
