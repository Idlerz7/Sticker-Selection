import unittest

from style_shapes.training import StyleShapesPLModel


class TrainingStepsPropertyTest(unittest.TestCase):
    def test_lightning_optimizer_reads_numeric_property(self):
        self.assertIsInstance(StyleShapesPLModel.num_training_steps, property)


if __name__ == "__main__":
    unittest.main()
