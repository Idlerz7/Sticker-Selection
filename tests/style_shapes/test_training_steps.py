import unittest

from style_shapes.training import _sharded_training_steps


class ShardedTrainingStepsTest(unittest.TestCase):
    def test_does_not_divide_rank_sharded_loader_by_world_size_again(self):
        # DSTC: ceil(ceil(211575 / 4) / 16) batches per rank.
        self.assertEqual(_sharded_training_steps(3306, 10), 33060)
        # StickerChat with six visible GPUs:
        # ceil(ceil(320168 / 6) / 16) batches per rank.
        self.assertEqual(_sharded_training_steps(3336, 10), 33360)

    def test_accumulation_and_limits(self):
        self.assertEqual(_sharded_training_steps(11, 3, 2), 18)
        self.assertEqual(_sharded_training_steps(11, 3, 1, 5), 15)
        self.assertEqual(_sharded_training_steps(20, 2, 1, 0.25), 10)


if __name__ == "__main__":
    unittest.main()
