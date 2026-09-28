"""CPU smoke tests for data integrity, checkpointing, and model interfaces."""
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch
import lightning.pytorch as pl
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from datasets.base_dataset import LabeledDataset
from datasets.supervised_dataset import SLDataModule
from datasets.ebd_generalization_dataset import EBDGeneralizationDataModule
from tools.create_masks import polygon_mask
from shapely.wkt import loads


def make_pair(folder, name):
    folder = Path(folder)
    for child in ["images", "masks"]: (folder / child).mkdir(parents=True, exist_ok=True)
    image = np.full((64, 64, 3), 127, np.uint8)
    loc = np.zeros((64, 64), np.uint8); loc[8:56, 8:56] = 255
    dmg = np.zeros((64, 64), np.uint8)
    for c in range(1, 5): dmg[8:56, 8+(c-1)*12:8+c*12] = c
    for time, mask in [("pre", loc), ("post", dmg)]:
        fn = name + "_" + time + "_disaster.png"
        cv2.imwrite(str(folder / "images" / fn), image)
        cv2.imwrite(str(folder / "masks" / fn), mask)
    return str(folder / "images" / (name + "_pre_disaster.png"))


def configuration():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "conf"), version_base="1.3"):
        return compose(config_name="config", overrides=["network=resnest50_baseline", "network.model.pretrained=false"])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_data_integrity_and_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            fn = make_pair(temp, "event_00000001")
            ds = LabeledDataset([fn], [0])
            sample = ds[0]
            self.assertEqual(tuple(sample["img"].shape), (6, 64, 64))
            self.assertEqual(tuple(sample["msk"].shape), (5, 64, 64))
            self.assertEqual(float(sample["img"].abs().sum()), 0.0)
            Path(fn.replace("_pre_", "_post_")).unlink()
            with self.assertRaises(OSError): ds[0]

    def test_split_and_ebd_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for event in ["event-a", "event-b"]:
                for i in range(10): make_pair(root/"train", f"{event}_{i:08d}")
            make_pair(root/"test", "event-a_99999999")
            dm = SLDataModule([str(root/"train")], [str(root/"test")], None, num_workers=0)
            dm.setup("fit");dm.setup("test")
            self.assertEqual((len(dm.train_dataset),len(dm.val_dataset),len(dm.test_dataset)), (18,2,1))
            ebd = EBDGeneralizationDataModule([str(root)], num_workers=0)
            ebd.setup("test");self.assertEqual(len(ebd.test_dataset),21)
            with self.assertRaises(ValueError): ebd.setup("fit")

    def test_polygon_hole(self):
        geometry = loads("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 7 3, 7 7, 3 7, 3 3))")
        mask = polygon_mask(geometry, (16,16))
        self.assertEqual(mask[1,1],1);self.assertEqual(mask[5,5],0)

    def test_baseline_forward_backward(self):
        from legacy.zoo.models import ResNeSt50_Unet_Double, Res34_Unet_Double
        for constructor in [Res34_Unet_Double, ResNeSt50_Unet_Double]:
            model = constructor(pretrained=False).eval()
            output = model(torch.randn(1,6,64,64))
            self.assertEqual(tuple(output.shape),(1,5,64,64))
            output.mean().backward()
            self.assertIsNotNone(model.res.weight.grad)

    def test_training_checkpoint_resume_and_test(self):
        cfg = configuration()
        with tempfile.TemporaryDirectory() as temp:
            fn = make_pair(temp, "event_00000001")
            loader = DataLoader(LabeledDataset([fn,fn],[0,1]),batch_size=2)
            cfg.network.model = {"_target_":"torch.nn.Conv2d", "in_channels":6,"out_channels":5,"kernel_size":1}
            cfg.network.model_dir = temp
            module = instantiate(cfg.network)
            callback = ModelCheckpoint(dirpath=str(Path(temp)/"ckpts"),monitor="val_xbd_score",mode="max",save_last=True)
            trainer = pl.Trainer(accelerator="cpu",devices=1,max_epochs=1,logger=False,
                callbacks=[callback],enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0)
            trainer.fit(module,train_dataloaders=loader,val_dataloaders=loader)
            self.assertTrue(Path(callback.best_model_path).is_file())
            resume_path = callback.last_model_path
            results = trainer.test(module,dataloaders=loader,ckpt_path=callback.best_model_path)
            self.assertIn("test_xbd_score",results[0])
            resumed = instantiate(cfg.network)
            trainer2 = pl.Trainer(accelerator="cpu",devices=1,max_epochs=2,logger=False,
                enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0)
            trainer2.fit(resumed,train_dataloaders=loader,val_dataloaders=loader,ckpt_path=resume_path)
            self.assertEqual(trainer2.global_step,2)


if __name__ == "__main__": unittest.main()
