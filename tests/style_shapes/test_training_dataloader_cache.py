import unittest
from types import SimpleNamespace
from unittest import mock

from style_shapes.permutations import IndexedDataset, create_permutation_manifest
from style_shapes.training import StyleShapesDataModule, StyleShapesPLModel


class _DictDataset:
    def __init__(self, size):
        self.size = int(size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"value": int(index)}


class _CountingDataModule:
    def __init__(self, batches):
        self.batches = int(batches)
        self.calls = 0

    def train_dataloader(self):
        self.calls += 1
        return [None] * self.batches


class TrainingDataloaderCacheTest(unittest.TestCase):
    def test_train_dataloader_is_constructed_once(self):
        datamodule = object.__new__(StyleShapesDataModule)
        datamodule.permutation_manifest = create_permutation_manifest(11, 2, 1, 2021)
        datamodule.train_dataset = IndexedDataset(_DictDataset(11))
        datamodule.train_batch_size = 4
        datamodule.args = SimpleNamespace(num_workers=0)
        datamodule.collate_fn = lambda batch: batch
        datamodule._style_shapes_train_dataloader = None

        first = datamodule.train_dataloader()
        second = datamodule.train_dataloader()

        self.assertIs(first, second)
        self.assertEqual(len(first), 3)

    def test_num_training_steps_is_cached(self):
        datamodule = _CountingDataModule(7)
        owner = SimpleNamespace(
            trainer=SimpleNamespace(
                max_steps=-1,
                datamodule=datamodule,
                max_epochs=3,
                accumulate_grad_batches=1,
                limit_train_batches=1.0,
            ),
            _style_shapes_num_training_steps=None,
        )

        getter = StyleShapesPLModel.num_training_steps.fget
        self.assertEqual(getter(owner), 21)
        self.assertEqual(getter(owner), 21)
        self.assertEqual(datamodule.calls, 1)

    def test_dual_validation_loaders_are_exactly_rank_sharded(self):
        datamodule = object.__new__(StyleShapesDataModule)
        datamodule.permutation_manifest = create_permutation_manifest(11, 2, 6, 2021)
        datamodule.val_dataset_r10 = _DictDataset(10000)
        datamodule.val_dataset_r20 = _DictDataset(10000)
        datamodule.valtest_batch_size = 1
        datamodule.args = SimpleNamespace(num_workers=0, val_data_path="")
        datamodule.collate_fn = lambda batch: batch
        datamodule._per_epoch_dual_val_loaders = True

        with mock.patch.dict("os.environ", {"LOCAL_RANK": "2"}, clear=False):
            loaders = datamodule.val_dataloader()

        self.assertEqual([len(loader) for loader in loaders], [1667, 1667])
        self.assertEqual(list(loaders[0].sampler)[:3], [2, 8, 14])
        self.assertEqual(list(loaders[0].sampler)[-1], 9998)


if __name__ == "__main__":
    unittest.main()
