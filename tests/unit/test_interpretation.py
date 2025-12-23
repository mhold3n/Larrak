import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.infra.interpretation import DOEAnalyzer


class TestInterpretation(unittest.TestCase):
    def setUp(self):
        # Create synthetic data: y = 2*x1 + 0.5*x2^2 + 5.0
        # x1 in [0, 10], x2 in [-5, 5]
        self.df = pd.DataFrame(
            {
                "x1": np.linspace(0, 10, 20),
                "x2": np.linspace(-5, 5, 20),
            }
        )
        # Add some noise? No, let's keep it exact for check
        self.df["y"] = 2.0 * self.df["x1"] + 0.5 * self.df["x2"] ** 2 + 5.0

        self.analyzer = DOEAnalyzer(df=self.df)

    def test_fit_and_predict(self):
        inputs = ["x1", "x2"]
        response = "y"
        self.analyzer.fit_model(inputs, response, degree=2, interactions=False)

        # Check prediction at known point
        # x1=5, x2=0 => y = 10 + 0 + 5 = 15
        test_pt = pd.DataFrame({"x1": [5.0], "x2": [0.0]})
        pred = self.analyzer.predict(test_pt)[0]
        self.assertAlmostEqual(pred, 15.0, places=4)

    def test_anova(self):
        inputs = ["x1", "x2"]
        response = "y"
        self.analyzer.fit_model(inputs, response, degree=2)
        anova = self.analyzer.run_anova()
        # Should have rows for x1, x2 (or I(x2**2))
        # statsmodels format varies, just check it returns a DF
        self.assertIsInstance(anova, pd.DataFrame)
        print("\nANOVA:\n", anova)

    def test_sensitivity(self):
        inputs = ["x1", "x2"]
        response = "y"
        self.analyzer.fit_model(inputs, response, degree=2)
        bounds = {"x1": (0, 10), "x2": (-5, 5)}
        sens = self.analyzer.compute_sensitivity(bounds, n_samples=100)
        self.assertIsInstance(sens, pd.DataFrame)
        print("\nSensitivity:\n", sens)


if __name__ == "__main__":
    unittest.main()
