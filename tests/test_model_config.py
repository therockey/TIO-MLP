import unittest

import torch

from tio_mlp.config import Config
from tio_mlp.model import MLP, build_model, count_parameters


class ModelTest(unittest.TestCase):
    def test_output_shape(self):
        model = MLP(hidden_sizes=(32, 16))
        output = model(torch.randn(8, 1, 28, 28))
        self.assertEqual(tuple(output.shape), (8, 10))

    def test_batchnorm_and_dropout(self):
        model = MLP(hidden_sizes=(32, 16), dropout=0.2, batchnorm=True)
        output = model(torch.randn(8, 1, 28, 28))
        self.assertEqual(tuple(output.shape), (8, 10))
        self.assertGreater(count_parameters(model), 0)

    def test_all_activations_work(self):
        for activation in ("relu", "leaky_relu", "gelu", "tanh", "sigmoid"):
            model = MLP(hidden_sizes=(8,), activation=activation)
            output = model(torch.randn(4, 1, 28, 28))
            self.assertEqual(tuple(output.shape), (4, 10))

    def test_config_roundtrip_and_validation(self):
        cfg = Config(hidden_sizes=(128, 64), lr=0.001)
        restored = Config.from_dict(cfg.to_dict())
        self.assertEqual(restored.hidden_sizes, (128, 64))
        self.assertEqual(restored.lr, 0.001)
        with self.assertRaises(ValueError):
            Config(dropout=1.5).validate()
        with self.assertRaises(ValueError):
            Config(hidden_sizes=()).validate()

    def test_build_model_uses_config(self):
        cfg = Config(hidden_sizes=(24, 12), activation="tanh", dropout=0.1, batchnorm=True)
        model = build_model(cfg)
        self.assertEqual(model.hidden_sizes, (24, 12))
        self.assertEqual(model.activation_name, "tanh")


if __name__ == "__main__":
    unittest.main()
