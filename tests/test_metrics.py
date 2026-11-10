import unittest

import numpy as np

from tio_mlp.metrics import classification_metrics, confusion_matrix


class MetricsTest(unittest.TestCase):
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 2, 1, 0])
        result = classification_metrics(y_true, y_true, num_classes=3)
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 1.0)
        self.assertAlmostEqual(result["macro_precision"], 1.0)
        self.assertAlmostEqual(result["macro_recall"], 1.0)

    def test_confusion_matrix_layout(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        matrix = confusion_matrix(y_true, y_pred, num_classes=2)
        np.testing.assert_array_equal(matrix, np.array([[1, 1], [1, 1]]))

    def test_known_metrics(self):
        # klasa 0: TP=2, FP=1, FN=1; klasa 1: TP=1, FP=1, FN=1
        y_true = [0, 0, 0, 1, 1]
        y_pred = [0, 0, 1, 1, 0]
        result = classification_metrics(y_true, y_pred, num_classes=2)
        self.assertAlmostEqual(result["accuracy"], 0.6)
        self.assertAlmostEqual(result["per_class"]["0"]["precision"], 2 / 3)
        self.assertAlmostEqual(result["per_class"]["0"]["recall"], 2 / 3)
        self.assertAlmostEqual(result["per_class"]["0"]["f1"], 2 / 3)
        self.assertAlmostEqual(result["per_class"]["1"]["precision"], 0.5)
        self.assertAlmostEqual(result["per_class"]["1"]["recall"], 0.5)
        self.assertAlmostEqual(result["per_class"]["1"]["f1"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 0.5) / 2)

    def test_absent_class_is_ignored_in_macro(self):
        y_true = [0, 0, 1]
        y_pred = [0, 0, 1]
        result = classification_metrics(y_true, y_pred, num_classes=3)
        self.assertAlmostEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["per_class"]["2"]["support"], 0)


if __name__ == "__main__":
    unittest.main()
